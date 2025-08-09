"""
LLM Client Wrapper with Multi-API Key Support

This module provides a wrapper around LLM clients that automatically handles
API key rotation, rate limiting, and error recovery.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional, Callable

from .config import LLMSettings, APIKeySettings
from .api_key_manager import api_key_manager
from .exceptions import RateLimitError, NoAvailableKeysError, AuthenticationError # fixed import and added AuthenticationError

logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """
    Wrapper for LLM clients that handles multi-API key rotation and rate limiting.

    This wrapper automatically:
    - Selects the best available API key based on rate limits and priority
    - Rotates to backup keys when primary keys hit rate limits
    - Tracks usage and failures for each key
    - Provides fallback mechanisms
    """

    def __init__(self, llm_settings: LLMSettings, client_factory: Callable[[str], Any]):
        """
        Initialize the wrapper.

        Args:
            llm_settings: LLM configuration with API keys
            client_factory: Function that creates an LLM client given an API key
        """
        self.llm_settings = llm_settings
        self.client_factory = client_factory
        self._current_client = None
        self._current_api_key = None

        # Register API keys with the manager
        if llm_settings.api_keys:
            api_key_manager.register_keys(llm_settings.api_keys)

        # Initialize with the first available key
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the client with the best available API key"""
        try:
            api_key, key_config = self._get_next_available_key()
            self._current_api_key = api_key
            self._current_client = self.client_factory(api_key)
            logger.info(f"Initialized LLM client with key: {key_config.name or 'Unnamed'}")
        except NoAvailableKeysError:
            logger.error("No available API keys for LLM client initialization")
            self._current_client = None
            self._current_api_key = None

    def _get_next_available_key(self) -> tuple[str, APIKeySettings]:
        """Get the next available API key"""
        # Use multi-key configuration if available
        if self.llm_settings.api_keys:
            result = api_key_manager.get_available_key(self.llm_settings.api_keys)
            if result:
                return result

        # Fallback to single key (legacy support)
        if self.llm_settings.api_key:
            # Create a temporary APIKeySettings for the single key
            single_key_config = APIKeySettings(
                api_key=self.llm_settings.api_key,
                name="Legacy Single Key",
                max_requests_per_minute=60,
                max_requests_per_hour=3600,
                max_requests_per_day=86400,
                priority=1,
                enabled=True
            )
            return self.llm_settings.api_key, single_key_config

        raise NoAvailableKeysError("No API keys available")

    def _rotate_key(self) -> bool:
        """
        Rotate to the next available API key.

        Returns:
            True if rotation was successful, False otherwise
        """
        try:
            api_key, key_config = self._get_next_available_key()
            if api_key != self._current_api_key:
                self._current_api_key = api_key
                self._current_client = self.client_factory(api_key)
                logger.info(f"Rotated to API key: {key_config.name or 'Unnamed'}")
                return True
        except NoAvailableKeysError:
            logger.error("No available keys for rotation")

        return False

    def make_request(self, request_func: Callable, *args, max_retries: int = 3, **kwargs) -> Any:
        """
        Make an LLM request with automatic key rotation on rate limits.

        Args:
            request_func: Function to call on the LLM client (e.g., client.chat.completions.create)
            *args: Arguments to pass to request_func
            max_retries: Maximum number of retry attempts
            **kwargs: Keyword arguments to pass to request_func

        Returns:
            Response from the LLM API

        Raises:
            RateLimitError: If all keys are rate limited
            NoAvailableKeysError: If no keys are available
            Exception: Other API errors
        """
        if not self._current_client:
            self._initialize_client()
            if not self._current_client:
                raise NoAvailableKeysError("No API keys available")

        last_exception = None

        for attempt in range(max_retries):
            try:
                # Make the request
                start_time = time.time()
                response = request_func(self._current_client, *args, **kwargs)

                # Record successful request
                if self._current_api_key:
                    api_key_manager.record_request(self._current_api_key)

                logger.debug(f"Request successful in {time.time() - start_time:.2f}s")
                return response

            except Exception as e:
                last_exception = e
                error_message = str(e).lower()

                # Handle rate limit errors
                if any(phrase in error_message for phrase in [
                    'rate limit', 'too many requests', '429', 'quota exceeded',
                    'rate_limit_exceeded', 'requests per minute'
                ]):
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")

                    if self._current_api_key:
                        # Extract reset time if available (implementation depends on API)
                        reset_time = self._extract_reset_time(e)
                        api_key_manager.record_rate_limit_error(self._current_api_key, reset_time)

                    # Try to rotate to next key
                    if not self._rotate_key():
                        # No more keys available
                        raise RateLimitError("All API keys are rate limited")

                    continue

                # Handle other API errors
                elif any(phrase in error_message for phrase in [
                    'invalid api key', 'unauthorized', '401', 'forbidden', '403'
                ]):
                    logger.error(f"Authentication error: {e}")

                    if self._current_api_key:
                        api_key_manager.record_failure(self._current_api_key, "auth_error")

                    # Try next key
                    if not self._rotate_key():
                        raise AuthenticationError(e) # Raise AuthenticationError

                    continue

                # Handle server errors (retry with same key)
                elif any(phrase in error_message for phrase in [
                    'server error', '500', '502', '503', '504', 'timeout'
                ]):
                    logger.warning(f"Server error on attempt {attempt + 1}: {e}")

                    if self._current_api_key:
                        api_key_manager.record_failure(self._current_api_key, "server_error")

                    # Wait before retry
                    time.sleep(min(2 ** attempt, 10))  # Exponential backoff, max 10s
                    continue

                # Other errors - don't retry
                else:
                    logger.error(f"Unhandled error: {e}")
                    if self._current_api_key:
                        api_key_manager.record_failure(self._current_api_key, "unknown_error")
                    raise e

        # All retries exhausted
        raise last_exception or Exception("Max retries exceeded")

    def _extract_reset_time(self, exception: Exception) -> Optional[datetime]:
        """
        Extract rate limit reset time from exception.
        This is API-specific and should be customized based on the API being used.
        """
        # This is a placeholder implementation
        # Different APIs provide reset time in different ways:
        # - HTTP headers (Retry-After, X-RateLimit-Reset)
        # - Exception details
        # - Response body

        # For now, return None and let the manager use default timing
        return None

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all registered API keys"""
        stats = {}

        if self.llm_settings.api_keys:
            for key_config in self.llm_settings.api_keys:
                key_stats = api_key_manager.get_usage_stats(key_config.api_key)
                stats[key_config.name or key_config.api_key[:8] + "..."] = key_stats

        elif self.llm_settings.api_key:
            key_stats = api_key_manager.get_usage_stats(self.llm_settings.api_key)
            stats["Legacy Key"] = key_stats

        return stats

    def get_current_key_info(self) -> Optional[str]:
        """Get information about the currently active API key"""
        if not self._current_api_key:
            return None

        if self.llm_settings.api_keys:
            for key_config in self.llm_settings.api_keys:
                if key_config.api_key == self._current_api_key:
                    return key_config.name or "Unnamed Key"

        return "Legacy Key"


def create_llm_wrapper(llm_settings: LLMSettings, client_factory: Callable[[str], Any]) -> LLMClientWrapper:
    """
    Factory function to create an LLM client wrapper.

    Args:
        llm_settings: LLM configuration
        client_factory: Function that creates an LLM client given an API key

    Returns:
        LLMClientWrapper instance
    """
    return LLMClientWrapper(llm_settings, client_factory)