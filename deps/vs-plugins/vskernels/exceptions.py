from __future__ import annotations

from typing import Any

from vstools import CustomValueError, FuncExceptT

__all__ = [
    'UnknownScalerError',
    'UnknownDescalerError',
    'UnknownKernelError'
]


class UnknownScalerError(CustomValueError):
    """Raised when an unknown scaler is passed."""

    def __init__(
        self, func: FuncExceptT, scaler: str, message: str = 'Unknown concrete scaler "{scaler}"!',
        **kwargs: Any
    ) -> None:
        super().__init__(message, func, scaler=scaler, **kwargs)


class UnknownDescalerError(CustomValueError):
    """Raised when an unknown descaler is passed."""

    def __init__(
        self, func: FuncExceptT, descaler: str, message: str = 'Unknown concrete descaler "{descaler}"!',
        **kwargs: Any
    ) -> None:
        super().__init__(message, func, descaler=descaler, **kwargs)


class UnknownKernelError(CustomValueError):
    """Raised when an unknown kernel is passed."""

    def __init__(
        self, func: FuncExceptT, kernel: str, message: str = 'Unknown concrete kernel "{kernel}"!',
        **kwargs: Any
    ) -> None:
        super().__init__(message, func, kernel=kernel, **kwargs)
