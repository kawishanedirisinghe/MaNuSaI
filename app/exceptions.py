class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


class OpenManusError(Exception):
    """Base exception for all OpenManus errors"""


class TokenLimitExceeded(OpenManusError):
    """Exception raised when the token limit is exceeded"""
class RateLimitError(Exception):
    """Exception raised when all API keys are rate limited"""
    pass


class AuthenticationError(Exception):
    """Exception raised when API key authentication fails"""
    pass
