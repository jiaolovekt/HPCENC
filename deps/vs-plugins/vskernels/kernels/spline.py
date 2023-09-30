from __future__ import annotations

from typing import Any

from vstools import core

from .abstract import Kernel
from .fmtconv import FmtConv

__all__ = [
    'Spline',
    'Spline16',
    'Spline36',
    'Spline64',
    'Spline100',
    'Spline144',
    'Spline196',
    'Spline256',
]


class Spline(FmtConv):
    """fmtconv's spline resizer."""

    _kernel = 'spline'

    def __init__(self, taps: int = 2, **kwargs: Any) -> None:
        super().__init__(taps=taps, **kwargs)


class Spline16(Kernel):
    """
    Built-in spline16 resizer.

    Dependencies:

    * VapourSynth-descale
    """

    scale_function = resample_function = core.lazy.resize.Spline16
    descale_function = core.lazy.descale.Despline16


class Spline36(Kernel):
    """
    Built-in spline36 resizer.

    Dependencies:

    * VapourSynth-descale
    """

    scale_function = resample_function = core.lazy.resize.Spline36
    descale_function = core.lazy.descale.Despline36


class Spline64(Kernel):
    """
    Built-in spline64 resizer.

    Dependencies:

    * VapourSynth-descale
    """

    scale_function = resample_function = core.lazy.resize.Spline64
    descale_function = core.lazy.descale.Despline64


class Spline100(Spline):
    """fmtconv's spline kernel with taps=5."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(taps=5, **kwargs)


class Spline144(Spline):
    """fmtconv's spline kernel with taps=6."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(taps=6, **kwargs)


class Spline196(Spline):
    """fmtconv's spline kernel with taps=7."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(taps=7, **kwargs)


class Spline256(Spline):
    """fmtconv's spline kernel with taps=8."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(taps=8, **kwargs)
