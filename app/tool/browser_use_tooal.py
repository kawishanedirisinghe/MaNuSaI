        import asyncio
        import base64
        import json
        from typing import Generic, Optional, TypeVar

        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import Select
        from pydantic import Field, field_validator
        from pydantic_core.core_schema import ValidationInfo

        from app.config import config
        from app.llm import LLM
        from app.tool.base import BaseTool, ToolResult
        from app.tool.web_search import WebSearch

        _BROWSER_DESCRIPTION = """\
        A powerful browser automation tool that allows interaction with web pages through various actions.
        * This tool provides commands for controlling a browser session, navigating web pages, and extracting information
        * It maintains state across calls, keeping the browser session alive until explicitly closed
        * Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
        * Each action requires specific parameters as defined in the tool's dependencies

        Key capabilities include:
        * Navigation: Go to specific URLs, go back, search the web, or refresh pages
        * Interaction: Click elements, input text, select from dropdowns, send keyboard commands
        * Scrolling: Scroll up/down by pixel amount or scroll to specific text
        * Content extraction: Extract and analyze content from web pages based on specific goals
        * Tab management: Switch between tabs, open new tabs, or close tabs

        Note: When using element indices, refer to the numbered elements shown in the current browser state.
        """

        Context = TypeVar("Context")

        class BrowserUseTool(BaseTool, Generic[Context]):
            name: str = "browser_use"
            description: str = _BROWSER_DESCRIPTION
            parameters: dict = {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "go_to_url",
                            "click_element",
                            "input_text",
                            "scroll_down",
                            "scroll_up",
                            "scroll_to_text",
                            "send_keys",
                            "get_dropdown_options",
                            "select_dropdown_option",
                            "go_back",
                            "web_search",
                            "wait",
                            "extract_content",
                            "switch_tab",
                            "open_tab",
                            "close_tab",
                        ],
                        "description": "The browser action to perform",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL for 'go_to_url' or 'open_tab' actions",
                    },
                    "index": {
                        "type": "integer",
                        "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
                    },
                    "scroll_amount": {
                        "type": "integer",
                        "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
                    },
                    "tab_id": {
                        "type": "integer",
                        "description": "Tab ID for 'switch_tab' action",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for 'web_search' action",
                    },
                    "goal": {
                        "type": "string",
                        "description": "Extraction goal for 'extract_content' action",
                    },
                    "keys": {
                        "type": "string",
                        "description": "Keys to send for 'send_keys' action",
                    },
                    "seconds": {
                        "type": "integer",
                        "description": "Seconds to wait for 'wait' action",
                    },
                },
                "required": ["action"],
                "dependencies": {
                    "go_to_url": ["url"],
                    "click_element": ["index"],
                    "input_text": ["index", "text"],
                    "switch_tab": ["tab_id"],
                    "open_tab": ["url"],
                    "scroll_down": ["scroll_amount"],
                    "scroll_up": ["scroll_amount"],
                    "scroll_to_text": ["text"],
                    "send_keys": ["keys"],
                    "get_dropdown_options": ["index"],
                    "select_dropdown_option": ["index", "text"],
                    "go_back": [],
                    "web_search": ["query"],
                    "wait": ["seconds"],
                    "extract_content": ["goal"],
                },
            }

            lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
            driver: Optional[webdriver.Chrome] = Field(default=None, exclude=True)
            web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)
            tool_context: Optional[Context] = Field(default=None, exclude=True)
            llm: Optional[LLM] = Field(default_factory=LLM)

            @field_validator("parameters", mode="before")
            def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
                if not v:
                    raise ValueError("Parameters cannot be empty")
                return v

            async def _ensure_browser_initialized(self) -> webdriver.Chrome:
                """Ensure browser is initialized with Selenium."""
                if self.driver is None:
                    chrome_options = Options()
                    chrome_options.add_argument('--no-sandbox')
                    chrome_options.add_argument('--disable-dev-shm-usage')

                    if config.browser_config:
                        if getattr(config.browser_config, "headless", False):
                            chrome_options.add_argument('--headless')
                        if getattr(config.browser_config, "disable_security", False):
                            chrome_options.add_argument('--disable-web-security')
                        if getattr(config.browser_config, "proxy", None) and config.browser_config.proxy.server:
                            chrome_options.add_argument(f'--proxy-server={config.browser_config.proxy.server}')
                        if getattr(config.browser_config, "extra_chromium_args", None):
                            for arg in config.browser_config.extra_chromium_args:
                                chrome_options.add_argument(arg)

                    self.driver = webdriver.Chrome(options=chrome_options)
                return self.driver

            async def execute(
                self,
                action: str,
                url: Optional[str] = None,
                index: Optional[int] = None,
                text: Optional[str] = None,
                scroll_amount: Optional[int] = None,
                tab_id: Optional[int] = None,
                query: Optional[str] = None,
                goal: Optional[str] = None,
                keys: Optional[str] = None,
                seconds: Optional[int] = None,
                **kwargs,
            ) -> ToolResult:
                """
                Execute a specified browser action using Selenium.

                Args:
                    action: The browser action to perform
                    url: URL for navigation or new tab
                    index: Element index for click or input actions
                    text: Text for input action or search query
                    scroll_amount: Pixels to scroll for scroll action
                    tab_id: Tab ID for switch_tab action
                    query: Search query for Google search
                    goal: Extraction goal for content extraction
                    keys: Keys to send for keyboard actions
                    seconds: Seconds to wait
                    **kwargs: Additional arguments

                Returns:
                    ToolResult with the action's output or error
                """
                async with self.lock:
                    try:
                        driver = await self._ensure_browser_initialized()
                        wait = WebDriverWait(driver, 10)  # 10-second timeout for waits

                        max_content_length = getattr(config.browser_config, "max_content_length", 2000)

                        # Navigation actions
                        if action == "go_to_url":
                            if not url:
                                return ToolResult(error="URL is required for 'go_to_url' action")
                            driver.get(url)
                            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                            return ToolResult(output=f"Navigated to {url}")

                        elif action == "go_back":
                            driver.back()
                            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                            return ToolResult(output="Navigated back")

                        elif action == "refresh":
                            driver.refresh()
                            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                            return ToolResult(output="Refreshed current page")

                        elif action == "web_search":
                            if not query:
                                return ToolResult(error="Query is required for 'web_search' action")
                            search_response = await self.web_search_tool.execute(query=query, fetch_content=True, num_results=1)
                            first_search_result = search_response.results[0]
                            url_to_navigate = first_search_result.url
                            driver.get(url_to_navigate)
                            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                            return search_response

                        # Element interaction actions
                        elif action == "click_element":
                            if index is None:
                                return ToolResult(error="Index is required for 'click_element' action")
                            elements = driver.find_elements(By.XPATH, "//*[self::a or self::button or self::input[@type='submit' or @type='button']]")
                            if index >= len(elements):
                                return ToolResult(error=f"Element with index {index} not found")
                            elements[index].click()
                            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                            return ToolResult(output=f"Clicked element at index {index}")

                        elif action == "input_text":
                            if index is None or not text:
                                return ToolResult(error="Index and text are required for 'input_text' action")
                            elements = driver.find_elements(By.XPATH, "//input[@type='text' or @type='search' or @type='email' or @type='password' or @type='tel' or @type='url'] | //textarea")
                            if index >= len(elements):
                                return ToolResult(error=f"Element with index {index} not found")
                            elements[index].clear()
                            elements[index].send_keys(text)
                            return ToolResult(output=f"Input '{text}' into element at index {index}")

                        elif action == "scroll_down" or action == "scroll_up":
                            direction = 1 if action == "scroll_down" else -1
                            amount = scroll_amount if scroll_amount is not None else 1080  # Default to 1080px (common viewport height)
                            driver.execute_script(f"window.scrollBy(0, {direction * amount});")
                            return ToolResult(output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels")

                        elif action == "scroll_to_text":
                            if not text:
                                return ToolResult(error="Text is required for 'scroll_to_text' action")
                            try:
                                element = driver.find_element(By.XPATH, f"//*[contains(text(), '{text}')]")
                                driver.execute_script("arguments[0].scrollIntoView(true);", element)
                                return ToolResult(output=f"Scrolled to text: '{text}'")
                            except Exception as e:
                                return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                        elif action == "send_keys":
                            if not keys:
                                return ToolResult(error="Keys are required for 'send_keys' action")
                            from selenium.webdriver.common.action_chains import ActionChains
                            actions = ActionChains(driver)
                            actions.send_keys(keys).perform()
                            return ToolResult(output=f"Sent keys: {keys}")

                        elif action == "get_dropdown_options":
                            if index is None:
                                return ToolResult(error="Index is required for 'get_dropdown_options' action")
                            elements = driver.find_elements(By.TAG_NAME, "select")
                            if index >= len(elements):
                                return ToolResult(error=f"Element with index {index} not found")
                            select = Select(elements[index])
                            options = [{"text": opt.text, "value": opt.get_attribute("value"), "index": idx} for idx, opt in enumerate(select.options)]
                            return ToolResult(output=f"Dropdown options: {options}")

                        elif action == "select_dropdown_option":
                            if index is None or not text:
                                return ToolResult(error="Index and text are required for 'select_dropdown_option' action")
                            elements = driver.find_elements(By.TAG_NAME, "select")
                            if index >= len(elements):
                                return ToolResult(error=f"Element with index {index} not found")
                            select = Select(elements[index])
                            select.select_by_visible_text(text)
                            return ToolResult(output=f"Selected option '{text}' from dropdown at index {index}")

                        # Content extraction actions
                        elif action == "extract_content":
                            if not goal:
                                return ToolResult(error="Goal is required for 'extract_content' action")
                            import markdownify
                            content = markdownify.markdownify(driver.page_source)
                            prompt = f"""\
        Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.
        Extraction goal: {goal}

        Page content:
        {content[:max_content_length]}
        """
                            messages = [{"role": "system", "content": prompt}]
                            extraction_function = {
                                "type": "function",
                                "function": {
                                    "name": "extract_content",
                                    "description": "Extract specific information from a webpage based on a goal",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "extracted_content": {
                                                "type": "object",
                                                "description": "The content extracted from the page according to the goal",
                                                "properties": {
                                                    "text": {
                                                        "type": "string",
                                                        "description": "Text content extracted from the page",
                                                    },
                                                    "metadata": {
                                                        "type": "object",
                                                        "description": "Additional metadata about the extracted content",
                                                        "properties": {
                                                            "source": {
                                                                "type": "string",
                                                                "description": "Source of the extracted content",
                                                            }
                                                        },
                                                    },
                                                },
                                            }
                                        },
                                        "required": ["extracted_content"],
                                    },
                                },
                            }
                            response = await self.llm.ask_tool(messages, tools=[extraction_function], tool_choice="required")
                            if response and response.tool_calls:
                                args = json.loads(response.tool_calls[0].function.arguments)
                                extracted_content = args.get("extracted_content", {})
                                return ToolResult(output=f"Extracted from page:\n{extracted_content}\n")
                            return ToolResult(output="No content was extracted from the page.")

                        # Tab management actions
                        elif action == "switch_tab":
                            if tab_id is None:
                                return ToolResult(error="Tab ID is required for 'switch_tab' action")
                            if tab_id >= len(driver.window_handles):
                                return ToolResult(error=f"Tab with ID {tab_id} not found")
                            driver.switch_to.window(driver.window_handles[tab_id])
                            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                            return ToolResult(output=f"Switched to tab {tab_id}")

                        elif action == "open_tab":
                            if not url:
                                return ToolResult(error="URL is required for 'open_tab' action")
                            driver.execute_script(f"window.open('{url}');")
                            driver.switch_to.window(driver.window_handles[-1])
                            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                            return ToolResult(output=f"Opened new tab with {url}")

                        elif action == "close_tab":
                            if len(driver.window_handles) > 1:
                                driver.close()
                                driver.switch_to.window(driver.window_handles[-1])
                                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                                return ToolResult(output="Closed current tab")
                            return ToolResult(error="Cannot close the last tab")

                        # Utility actions
                        elif action == "wait":
                            seconds_to_wait = seconds if seconds is not None else 3
                            await asyncio.sleep(seconds_to_wait)
                            return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                        else:
                            return ToolResult(error=f"Unknown action: {action}")

                    except Exception as e:
                        return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

            async def get_current_state(self) -> ToolResult:
                """
                Get the current browser state as a ToolResult.
                """
                try:
                    driver = await self._ensure_browser_initialized()
                    wait = WebDriverWait(driver, 10)

                    # Get clickable elements
                    clickable_elements = driver.find_elements(By.XPATH, "//*[self::a or self::button or self::input[@type='submit' or @type='button']]")
                    clickable_elements_str = "\n".join([f"[{i}] {el.get_attribute('outerHTML')[:100]}" for i, el in enumerate(clickable_elements)])

                    # Get scroll info
                    scroll_y = driver.execute_script("return window.scrollY;")
                    total_height = driver.execute_script("return document.body.scrollHeight;")
                    viewport_height = driver.execute_script("return window.innerHeight;")
                    pixels_above = scroll_y
                    pixels_below = total_height - scroll_y - viewport_height

                    # Take screenshot
                    screenshot = driver.get_screenshot_as_base64()

                    state_info = {
                        "url": driver.current_url,
                        "title": driver.title,
                        "tabs": [{"id": i, "url": driver.current_url if i == driver.window_handles.index(driver.current_window_handle) else ""} for i in range(len(driver.window_handles))],
                        "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                        "interactive_elements": clickable_elements_str,
                        "scroll_info": {
                            "pixels_above": pixels_above,
                            "pixels_below": pixels_below,
                            "total_height": total_height,
                        },
                        "viewport_height": viewport_height,
                    }

                    return ToolResult(output=json.dumps(state_info, indent=4, ensure_ascii=False), base64_image=screenshot)
                except Exception as e:
                    return ToolResult(error=f"Failed to get browser state: {str(e)}")

            async def cleanup(self):
                """Clean up browser resources."""
                async with self.lock:
                    if self.driver is not None:
                        try:
                            self.driver.quit()
                        finally:
                            self.driver = None

            def __del__(self):
                """Ensure cleanup when object is destroyed."""
                if self.driver is not None:
                    try:
                        asyncio.run(self.cleanup())
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(self.cleanup())
                        loop.close()

            @classmethod
            def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
                """Factory method to create a BrowserUseTool with a specific context."""
                tool = cls()
                tool.tool_context = context
                return tool