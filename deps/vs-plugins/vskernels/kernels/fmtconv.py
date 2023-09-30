from __future__ import annotations

from typing import Any, Callable, overload

from vstools import VideoFormatT, VSFunction, core, inject_self, vs

from .abstract import Scaler
from .bicubic import Bicubic

__all__ = [
    'FmtConv'
]


call_wrapT = Callable[..., VSFunction]


class FmtConv(Scaler):
    """
    Abstract fmtconv's resizer.

    Dependencies:

    * fmtconv
    """

    def scale_function(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        assert clip.format

        def _check_fmt(fmt: int | VideoFormatT) -> tuple[vs.VideoFormat, bool]:
            fmt = core.get_video_format(fmt)

            return fmt, ((
                fmt.bits_per_sample == 32 and fmt.sample_type == vs.FLOAT
            ) or (
                fmt.bits_per_sample == 16 and fmt.sample_type == vs.INTEGER
            ))

        in_fmt = clip.format
        out_fmt = None

        csp = kwargs.get('csp', None)
        bits = kwargs.get('bits', None)
        flt = kwargs.get('flt', 0)

        if csp:
            out_fmt, valid = _check_fmt(csp)

            if not valid:
                kwargs.pop('csp')
            else:
                out_fmt = None
        else:
            out_fmt = in_fmt

        if bits:
            if not out_fmt:
                out_fmt = in_fmt

            out_fmt = out_fmt.replace(bits_per_sample=bits)

        if not _check_fmt(in_fmt)[1]:
            clip = Bicubic.resample(
                clip, in_fmt.replace(bits_per_sample=16 * (1 + flt), sample_type=vs.SampleType(flt))
            )
        if out_fmt:
            out_fmt, valid = _check_fmt(out_fmt)
            if valid:
                kwargs['csp'] = out_fmt
                out_fmt = None

        filtered = clip.fmtc.resample(**kwargs)

        if not out_fmt:
            return filtered

        assert filtered.format
        return Bicubic.resample(filtered, out_fmt)

    resample_function = scale_function

    _kernel: str
    """Name of the fmtconv kernel"""

    _resize_fmtc_params_lut = {
        'sw': 'src_width', 'sx': 'src_left',
        'sh': 'src_height', 'sy': 'src_top'
    }

    def __init__(self, taps: int = 4, **kwargs: Any) -> None:
        self.taps = taps
        super().__init__(**kwargs)

    def _clean_args(self, **kwargs: Any) -> dict[str, Any]:
        for fmtc_param, res_param in self._resize_fmtc_params_lut.items():
            if res_param in kwargs:
                kwargs[fmtc_param] = kwargs.pop(res_param)

        return kwargs

    def get_scale_args(
        self, clip: vs.VideoNode, shift: tuple[float, float] = (0, 0),
        width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self._clean_args(
            sx=shift[1], sy=shift[0], kernel=self._kernel,
            **self.kwargs, **self.get_params_args(False, clip, width, height, **kwargs)
        )

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        kwargs |= dict(w=width, h=height, taps=self.taps)

        if is_descale:
            kwargs |= dict(sw=width, sh=height)
        return kwargs

    @overload
    @inject_self.cached
    def shift(self, clip: vs.VideoNode, shift: tuple[float, float] = (0, 0), **kwargs: Any) -> vs.VideoNode:
        ...

    @overload
    @inject_self.cached
    def shift(
        self, clip: vs.VideoNode,
        shift_top: float | list[float] = 0.0, shift_left: float | list[float] = 0.0, **kwargs: Any
    ) -> vs.VideoNode:
        ...

    @inject_self.cached  # type: ignore
    def shift(
        self, clip: vs.VideoNode,
        shifts_or_top: float | tuple[float, float] | list[float] | None = None,
        shift_left: float | list[float] | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        assert clip.format

        n_planes = clip.format.num_planes

        def _shift(shift_top: float | list[float] = 0.0, shift_left: float | list[float] = 0.0) -> vs.VideoNode:
            return self.scale_function(
                clip, sy=shift_top, sx=shift_left, kernel=self._kernel, **self.kwargs, **kwargs
            )

        if not shifts_or_top and not shift_left:
            return _shift()
        elif isinstance(shifts_or_top, tuple):
            return _shift(*shifts_or_top)
        elif isinstance(shifts_or_top, float) and isinstance(shift_left, float):
            return _shift(shifts_or_top, shift_left)

        shifts_top = shifts_or_top or 0.0
        if isinstance(shifts_top, list):
            if not shifts_top:
                shifts_top = [0.0] * n_planes
            elif len(shifts_top) > n_planes:
                shifts_top[:n_planes]

        shifts_left = shift_left or 0.0
        if isinstance(shifts_left, list):
            if not shifts_left:
                shifts_left = [0.0] * n_planes
            elif len(shifts_left) > n_planes:
                shifts_left = shifts_left[:n_planes]

        return _shift(shifts_top, shifts_left)
