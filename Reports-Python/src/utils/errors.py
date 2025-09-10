"""
Custom error classes for the JAM Reports Component.
"""

class ValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.name = "ValidationError"

class AuthorizationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.name = "AuthorizationError"

class PVMExecutionError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.name = "PVMExecutionError"

class ProtocolError(Exception):
    def __init__(self, message: str):
        super().__init__(message)