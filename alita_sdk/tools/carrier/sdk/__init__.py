"""
Carrier SDK Package

This package provides a client for interacting with the Carrier Performance platform API.
"""

from .client import CarrierClient
from .data_models import CarrierCredentials
from .exceptions import CarrierAPIError

# This controls what is available when someone does `from carrier.sdk import *`
__all__ = [
    "CarrierClient",
    "CarrierCredentials",
    "CarrierAPIError"
]