"""
JAM Protocol Implementation

A Python implementation of the JAM (Just Another Mining) protocol.
This package provides core functionality for blockchain consensus and mining operations.
"""

__version__ = "1.0.0"
__author__ = "JAM Protocol Team"

from .core.safrole_manager import SafroleManager
from .protocols.fallback_condition import calculate_fallback_gamma_s

__all__ = [
    "SafroleManager",
    "calculate_fallback_gamma_s",
] 