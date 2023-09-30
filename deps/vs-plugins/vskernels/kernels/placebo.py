from __future__ import annotations

from typing import Any

from vstools import Transfer, core, vs

from .abstract import Scaler

__all__ = [
    'Placebo'
]


class Placebo(Scaler):
    _kernel: str
    """Name of the placebo kernel"""

    # Kernel settings
    taps: float | None
    b: float | None
    c: float | None

    # Filter settings
    clamp: float
    blur: float
    taper: float

    # Quality settings
    antiring: float
    cutoff: float

    # Other settings
    lut_entries: int = 64

    # Linearize/Sigmoidize settings
    sigmoidize: bool = True
    linearize: bool = True
    sigmoid_center: float = 0.75
    sigmoid_slope: float = 6.5
    curve: Transfer = Transfer.BT709

    def scale_function(self, *args: Any, **kwargs: Any) -> vs.VideoNode:
        # Wrapping it here so it's not a hard-dep
        return core.placebo.Resample(*args, **kwargs)

    def __init__(
        self,
        taps: float | None = None, b: float | None = None, c: float | None = None,
        clamp: float = 0.0, blur: float = 0.0, taper: float = 0.0,
        antiring: float = 0.0, cutoff: float = 0.001,
        **kwargs: Any
    ) -> None:
        self.taps = taps
        self.b = b
        self.c = c
        self.clamp = clamp
        self.blur = blur
        self.taper = taper
        self.antiring = antiring
        self.cutoff = cutoff
        super().__init__(**kwargs)

    def get_scale_args(
        self, clip: vs.VideoNode, shift: tuple[float, float] = (0, 0),
        width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return dict(sx=shift[1], sy=shift[0]) | self.kwargs | self.get_params_args(
            False, clip, width, height, **kwargs
        )

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return dict(
            width=width, height=height, filter=self._kernel,
            radius=self.taps, param1=self.b, param2=self.c,
            clamp=self.clamp, taper=self.taper, blur=self.blur,
            antiring=self.antiring, cutoff=self.cutoff,
            lut_entries=self.lut_entries,
            sigmoidize=self.sigmoidize, linearize=self.linearize,
            sigmoid_center=self.sigmoid_center, sigmoid_slope=self.sigmoid_slope,
            trc=self.curve.value_libplacebo,
        )
