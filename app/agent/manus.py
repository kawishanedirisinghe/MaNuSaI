from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.token_trimmer import TokenTrimmer


class Manus(ToolCallAgent):

    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks using multiple tools including MCP-based tools"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 500

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            AskHuman(),
            Terminate(),
            TokenTrimmer(),
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None

    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command
    _initialized: bool = False

    # Advanced API key management
    api_key_manager: Optional[object] = Field(default=None, exclude=True)
    current_key_config: Optional[Dict] = Field(default=None, exclude=True)
    retry_count: int = Field(default=0, exclude=True)

    # Stop signal tracking
    task_id: Optional[str] = Field(default=None, exclude=True)
    running_tasks: Optional[Dict] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        """Initialize basic components synchronously."""
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    @classmethod
    async def create(cls, api_key_manager=None, task_id=None, running_tasks=None, **kwargs) -> "Manus":
        """Factory method to create and properly initialize a Manus instance."""
        # Use the advanced API key manager if provided
        if api_key_manager:
            # Get an available API key from the manager
            result = api_key_manager.get_available_api_key(use_random=True)
            if result:
                api_key, key_config = result
                kwargs['api_key'] = api_key
                kwargs['api_key_manager'] = api_key_manager
                kwargs['current_key_config'] = key_config

        # Add task tracking for stop functionality
        kwargs['task_id'] = task_id
        kwargs['running_tasks'] = running_tasks

        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"Connected to MCP server {server_id} at {server_config.url}"
                        )
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(
                            f"Connected to MCP server {server_id} using command {server_config.command}"
                        )
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
    ) -> None:
        """Connect to an MCP server and add its tools."""
        if use_stdio:
            await self.mcp_clients.connect_stdio(
                server_url, stdio_args or [], server_id
            )
            self.connected_servers[server_id or server_url] = server_url
        else:
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url

        # Update available tools with only the new tools from this server
        new_tools = [
            tool for tool in self.mcp_clients.tools if tool.server_id == server_id
        ]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """Disconnect from an MCP server and remove its tools."""
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # Rebuild available tools without the disconnected server's tools
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        """Clean up Manus agent resources."""
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        # Disconnect from all MCP servers only if we were initialized
        if self._initialized:
            await self.disconnect_mcp_server()
            self._initialized = False

    async def think(self) -> bool:
        """Process current state and decide next actions with advanced API key rotation."""
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        # Enhanced error handling with API key rotation
        max_retries = 3 if self.api_key_manager else 1

        for attempt in range(max_retries):
            try:
                # Record successful request if using API key manager
                if self.api_key_manager and self.current_key_config:
                    current_api_key = self.current_key_config['api_key']

                result = await super().think()

                # Record successful request
                if self.api_key_manager and self.current_key_config:
                    self.api_key_manager.record_successful_request(current_api_key)
                    logger.info(f"Successful request with key: {self.current_key_config['name']}")

                # Restore original prompt and return result
                self.next_step_prompt = original_prompt
                return result

            except Exception as e:
                error_str = str(e).lower()

                # Handle API key rotation if manager is available
                if self.api_key_manager and self.current_key_config:
                    current_api_key = self.current_key_config['api_key']
                    key_name = self.current_key_config['name']

                    # Categorize the error and handle accordingly
                    if any(keyword in error_str for keyword in ["rate limit", "quota", "too many requests", "resource_exhausted"]):
                        logger.warning(f"Rate limit error with key {key_name}: {e}")
                        self.api_key_manager.record_rate_limit_error(current_api_key, key_name)
                    elif any(keyword in error_str for keyword in ["authentication", "invalid api key", "unauthorized"]):
                        logger.error(f"Authentication error with key {key_name}: {e}")
                        self.api_key_manager.record_failure(current_api_key, key_name, "auth_error")
                    elif any(keyword in error_str for keyword in ["timeout", "connection"]):
                        logger.warning(f"Connection error with key {key_name}: {e}")
                        self.api_key_manager.record_failure(current_api_key, key_name, "connection_error")
                    else:
                        logger.error(f"Unexpected error with key {key_name}: {e}")
                        self.api_key_manager.record_failure(current_api_key, key_name, "unknown_error")

                    # Try to get a different API key for retry
                    if attempt < max_retries - 1:
                        logger.info(f"Attempting to rotate API key (attempt {attempt + 1}/{max_retries})")

                        # Get next available API key
                        result = self.api_key_manager.get_available_api_key(use_random=True)
                        if result:
                            new_api_key, new_key_config = result
                            if new_api_key != current_api_key:  # Only rotate if different key
                                logger.info(f"Rotating to API key: {new_key_config['name']}")

                                # Update the LLM client with new API key
                                if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'api_key'):
                                    self.llm.client.api_key = new_api_key
                                elif hasattr(self.llm, 'api_key'):
                                    self.llm.api_key = new_api_key

                                # Update current key config
                                self.current_key_config = new_key_config

                                # Add small delay before retry
                                import asyncio
                                await asyncio.sleep(1)
                                continue

                        logger.warning("No alternative API key available for rotation")

                # If this is the last attempt or no API key manager, raise the error
                if attempt == max_retries - 1:
                    # Log rotation stats if available
                    if self.api_key_manager:
                        rotation_stats = self.api_key_manager.get_keys_status()
                        logger.info(f"Final API Key Status: {len([k for k in rotation_stats if k['available']])} available keys")

                    # Restore original prompt before raising
                    self.next_step_prompt = original_prompt
                    raise e

        # This should not be reached, but restore prompt just in case
        self.next_step_prompt = original_prompt
        return False
