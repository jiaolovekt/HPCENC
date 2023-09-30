from __future__ import annotations

from typing import Any

from vstools import CustomValueError, FuncExceptT

from .backends import PyBackend


class UnavailableBackend(CustomValueError):
    """Raised when trying to initialize an unavailable backend"""

    def __init__(
        self, backend: PyBackend, func: FuncExceptT | None = None,
        message: str = 'This plugin is built on top of the {backend} backend which is unavailable!',
        **kwargs: Any
    ) -> None:
        super().__init__(message, func, backend=backend, **kwargs)
