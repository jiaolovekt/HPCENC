from __future__ import annotations

from typing import Any

from vskernels import Kernel, KernelT
from vstools import CustomValueError, FuncExceptT

__all__ = [
    'CompareSameKernelError'
]


class CompareSameKernelError(CustomValueError):
    """Raised when two of the same kernels are compared to each other."""

    def __init__(
        self, func: FuncExceptT, kernel: KernelT, message: str = 'You may not compare {kernel} with itself!',
        **kwargs: Any
    ) -> None:
        super().__init__(message, func, kernel=Kernel.from_param(kernel, CompareSameKernelError), **kwargs)
