import asyncio
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.api_manager import APIKeyManager, APIKeyConfig, create_api_key_manager_from_config
from app.api_key_manager import api_key_manager
from app.bedrock import BedrockClient
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)


REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key  # Keep for backward compatibility
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            # Initialize API key manager for multiple keys
            self.api_key_manager = None
            if llm_config.api_keys:
                # Register keys with the global API key manager
                api_key_manager.register_keys(llm_config.api_keys)
                self.api_key_manager = api_key_manager
                logger.info(f"Registered {len(llm_config.api_keys)} API keys with manager")

            # Add token counting related attributes
            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            # Initialize tokenizer
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # If the model is not in tiktoken's presets, use cl100k_base as default
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            # Initialize client - will be updated per request if using multiple keys
            self._init_client()
            self.token_counter = TokenCounter(self.tokenizer)

    def _init_client(self, api_key: Optional[str] = None):
        """Initialize the client with given API key or default"""
        current_api_key = api_key or self.api_key

        if self.api_type == "azure":
            self.client = AsyncAzureOpenAI(
                base_url=self.base_url,
                api_key=current_api_key,
                api_version=self.api_version,
            )
        elif self.api_type == "aws":
            self.client = BedrockClient()
        else:
            self.client = AsyncOpenAI(api_key=current_api_key, base_url=self.base_url)

    async def _get_available_api_key(self) -> Optional[Tuple[str, Any]]:
        """Get an available API key from the manager with configuration"""
        if not self.api_key_manager:
            return self.api_key, None

        # Get the current LLM config to access API keys
        from app.config import config
        llm_config = config.llm.get("default", config.llm["default"])

        if not llm_config.api_keys:
            return self.api_key, None

        # Try to get available key using random selection
        result = api_key_manager.get_available_key(llm_config.api_keys, use_random_selection=True)
        if result:
            api_key, key_config = result
            return api_key, key_config

        # If no keys available immediately, wait with exponential backoff
        max_wait_time = 300  # 5 minutes
        wait_time = 5  # Start with 5 seconds
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            logger.info(f"No API keys available, waiting {wait_time} seconds before retry...")
            await asyncio.sleep(wait_time)

            result = api_key_manager.get_available_key(llm_config.api_keys, use_random_selection=True)
            if result:
                api_key, key_config = result
                logger.info(f"API key became available: {key_config.name or 'Unnamed'}")
                return api_key, key_config

            # Exponential backoff, max 60 seconds
            wait_time = min(wait_time * 2, 60)

        logger.error("No API keys available after waiting")
        return None, None

    def _record_api_request(self, api_key: str):
        """Record an API request for rate limiting"""
        api_key_manager.record_request(api_key)

    def _handle_api_error(self, api_key: str, error: Exception):
        """Handle API errors and record them with the API key manager"""
        if isinstance(error, RateLimitError):
            logger.warning(f"Rate limit hit for key {api_key[:8]}...")
            # Extract reset time from error if available
            reset_time = None
            if hasattr(error, 'response') and error.response:
                # Try to extract reset time from headers
                headers = getattr(error.response, 'headers', {})
                if 'retry-after' in headers:
                    try:
                        retry_after = int(headers['retry-after'])
                        reset_time = datetime.now() + timedelta(seconds=retry_after)
                    except (ValueError, TypeError):
                        pass

            api_key_manager.record_rate_limit_error(api_key, reset_time)

        elif isinstance(error, AuthenticationError):
            logger.error(f"Authentication failed for key {api_key[:8]}...")
            api_key_manager.record_failure(api_key, "auth_error")
        else:
            logger.warning(f"API error for key {api_key[:8]}...: {str(error)}")
            api_key_manager.record_failure(api_key, "api_error")

    def get_api_keys_status(self) -> List[Dict]:
        """Get status of all API keys"""
        from app.config import config
        llm_config = config.llm.get("default", config.llm["default"])

        if not llm_config.api_keys:
            return [{"message": "Single API key mode - no rate limiting"}]

        status_list = []
        for key_config in llm_config.api_keys:
            stats = api_key_manager.get_usage_stats(key_config.api_key)

            # Check if key can make request
            current_time = datetime.now()
            tracker = api_key_manager._usage_trackers.get(key_config.api_key)
            can_make_request = False
            next_available_in = None

            if tracker:
                can_make_request = api_key_manager._is_key_within_limits(key_config, tracker, current_time)
                if not can_make_request and tracker.rate_limit_reset_time:
                    next_available_in = int((tracker.rate_limit_reset_time - current_time).total_seconds())
            else:
                can_make_request = True

            status = {
                "name": key_config.name or f"Key {key_config.api_key[:8]}...",
                "enabled": key_config.enabled,
                "can_make_request": can_make_request,
                "rate_limits": {
                    "requests_this_minute": stats.get("requests_this_minute", 0),
                    "requests_this_hour": stats.get("requests_this_hour", 0),
                    "requests_this_day": stats.get("requests_this_day", 0),
                    "max_per_minute": key_config.max_requests_per_minute,
                    "max_per_hour": key_config.max_requests_per_hour,
                    "max_per_day": key_config.max_requests_per_day,
                },
                "consecutive_failures": stats.get("consecutive_failures", 0),
                "last_used": str(stats.get("last_used", "Never")) if stats.get("last_used") else "Never"
            }

            if next_available_in and next_available_in > 0:
                status["next_available_in"] = next_available_in

            status_list.append(status)

        return status_list

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # If max_input_tokens is not set, always return True
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response with automatic API key fallback.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Get available API key with configuration
                result = await self._get_available_api_key()
                if not result[0]:
                    raise ValueError("No API keys available")

                current_api_key, key_config = result

                # Update client with current API key
                self._init_client(current_api_key)

                # Check if the model supports images
                supports_images = self.model in MULTIMODAL_MODELS

                # Format system and user messages with image support check
                if system_msgs:
                    system_msgs = self.format_messages(system_msgs, supports_images)
                    messages = system_msgs + self.format_messages(messages, supports_images)
                else:
                    messages = self.format_messages(messages, supports_images)

                # Calculate input token count
                input_tokens = self.count_message_tokens(messages)

                # Check if token limits are exceeded
                if not self.check_token_limit(input_tokens):
                    error_message = self.get_limit_error_message(input_tokens)
                    raise TokenLimitExceeded(error_message)

                params = {
                    "model": self.model,
                    "messages": messages,
                }

                if self.model in REASONING_MODELS:
                    params["max_completion_tokens"] = self.max_tokens
                else:
                    params["max_tokens"] = self.max_tokens
                    params["temperature"] = (
                        temperature if temperature is not None else self.temperature
                    )

                # Record the API request for rate limiting
                self._record_api_request(current_api_key)

                if not stream:
                    # Non-streaming request
                    response = await self.client.chat.completions.create(
                        **params, stream=False
                    )

                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty or invalid response from LLM")

                    # Update token counts
                    self.update_token_count(
                        response.usage.prompt_tokens, response.usage.completion_tokens
                    )

                    return response.choices[0].message.content

                # Streaming request
                self.update_token_count(input_tokens)

                response = await self.client.chat.completions.create(**params, stream=True)

                collected_messages = []
                completion_text = ""
                async for chunk in response:
                    chunk_message = chunk.choices[0].delta.content or ""
                    collected_messages.append(chunk_message)
                    completion_text += chunk_message
                    print(chunk_message, end="", flush=True)

                print()  # Newline after streaming
                full_response = "".join(collected_messages).strip()
                if not full_response:
                    raise ValueError("Empty response from streaming LLM")

                # estimate completion tokens for streaming response
                completion_tokens = self.count_tokens(completion_text)
                logger.info(
                    f"Estimated completion tokens for streaming response: {completion_tokens}"
                )
                self.total_completion_tokens += completion_tokens

                return full_response

            except TokenLimitExceeded:
                # Don't retry token limit errors
                raise
            except (RateLimitError, AuthenticationError) as e:
                # Handle API key errors and try next key
                self._handle_api_error(current_api_key, e)
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"All retries exhausted for API request")
                    raise
                logger.info(f"Retrying with different API key (attempt {retry_count + 1}/{max_retries})")
                continue
            except ValueError:
                logger.exception(f"Validation error")
                raise
            except OpenAIError as oe:
                logger.exception(f"OpenAI API error")
                if isinstance(oe, AuthenticationError):
                    logger.error("Authentication failed. Check API key.")
                    self._handle_api_error(current_api_key, oe)
                elif isinstance(oe, RateLimitError):
                    logger.error("Rate limit exceeded. Trying next API key.")
                    self._handle_api_error(current_api_key, oe)
                elif isinstance(oe, APIError):
                    logger.error(f"API error: {oe}")

                retry_count += 1
                if retry_count >= max_retries:
                    raise
                continue
            except Exception:
                logger.exception(f"Unexpected error in ask")
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                continue

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Set up API parameters
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # Add model-specific parameters
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Handle non-streaming request
            if not stream:
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # Handle streaming request
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Set up the completion request
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            params["stream"] = False  # Always use non-streaming for tool requests
            response: ChatCompletion = await self.client.chat.completions.create(
                **params
            )

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                print(response)
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # Update token counts
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

            return response.choices[0].message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error("""<span style="color: red; font-weight: bold;">OpenAI API error</span>""")
            if isinstance(oe, AuthenticationError):
                logger.error("""<span style="color: red; font-weight: bold;">Authentication failed. Check API key.</span>""")
            elif isinstance(oe, RateLimitError):
                logger.error("""<span style="color: red; font-weight: bold;">Rate limit exceeded. Consider increasing retry attempts.</span>""")
            elif isinstance(oe, APIError):
                logger.error(f"""<span style="color: red; font-weight: bold;">API error</span>""")
            raise
        except Exception as e:
            logger.error(f"""<span>Unexpected error in ask_tool</span>""")
            raise