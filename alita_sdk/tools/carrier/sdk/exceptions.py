from typing import Optional


class CarrierAPIError(Exception):
    """Custom exception for Carrier API errors, containing rich context."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        self.status_code = status_code
        self.response_text = response_text
        full_message = f"{message}"
        if status_code:
            full_message += f" | Status Code: {status_code}"
        if response_text:
            full_message += f" | Response: {response_text[:250]}..."
        super().__init__(full_message)
