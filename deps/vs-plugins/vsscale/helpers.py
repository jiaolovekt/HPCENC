from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from math import ceil, floor
from typing import Any, Callable, Protocol

from vsaa import Nnedi3
from vskernels import Catrom, Kernel, KernelT, Scaler, ScalerT
from vstools import F_VD, KwargsT, MatrixT, fallback, get_w, mod2, plane, vs

from .types import Resolution

__all__ = [
    'GenericScaler',
    'scale_var_clip',
    'fdescale_args',
    'descale_args',

    'ScalingArgs'
]


class _GeneriScaleNoShift(Protocol):
    def __call__(self, clip: vs.VideoNode, width: int, height: int, *args: Any, **kwds: Any) -> vs.VideoNode:
        ...


class _GeneriScaleWithShift(Protocol):
    def __call__(
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float],
        *args: Any, **kwds: Any
    ) -> vs.VideoNode:
        ...


@dataclass
class GenericScaler(Scaler):
    """
    Generic Scaler base class.
    Inherit from this to create more complex scalers with built-in utils.
    Instantiate with a callable taking at least a VideoNode, width, and height
    to use that as a Scaler in functions taking that.
    """

    kernel: KernelT | None = field(default=None, kw_only=True)
    """
    Base kernel to be used for certain scaling/shifting/resampling operations.
    Must be specified and defaults to catrom
    """

    scaler: ScalerT | None = field(default=None, kw_only=True)
    """Scaler used for scaling operations. Defaults to kernel."""

    shifter: KernelT | None = field(default=None, kw_only=True)
    """Kernel used for shifting operations. Defaults to kernel."""

    def __post_init__(self) -> None:
        self._kernel = Kernel.ensure_obj(self.kernel or Catrom, self.__class__)
        self._scaler = Scaler.ensure_obj(self.scaler or self._kernel, self.__class__)
        self._shifter = Kernel.ensure_obj(
            self.shifter or (self._scaler if isinstance(self._scaler, Kernel) else Catrom), self.__class__
        )

    def __init__(
        self, func: _GeneriScaleNoShift | _GeneriScaleWithShift | F_VD, **kwargs: Any
    ) -> None:
        self.func = func
        self.kwargs = kwargs

    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        kwargs = self.kwargs | kwargs

        output = None

        if shift != (0, 0):
            try:
                output = self.func(clip, width, height, shift, **kwargs)
            except BaseException:
                try:
                    output = self.func(clip, width=width, height=height, shift=shift, **kwargs)
                except BaseException:
                    pass

        if output is None:
            try:
                output = self.func(clip, width, height, **kwargs)
            except BaseException:
                output = self.func(clip, width=width, height=height, **kwargs)

        return self._finish_scale(output, clip, width, height, shift)

    def _finish_scale(
        self, clip: vs.VideoNode, input_clip: vs.VideoNode, width: int, height: int,
        shift: tuple[float, float] = (0, 0), matrix: MatrixT | None = None,
        copy_props: bool = False
    ) -> vs.VideoNode:
        assert input_clip.format

        if input_clip.format.num_planes == 1:
            clip = plane(clip, 0)

        if (clip.width, clip.height) != (width, height):
            clip = self._scaler.scale(clip, width, height)

        if shift != (0, 0):
            clip = self._shifter.shift(clip, shift)

        assert clip.format

        if clip.format.id != input_clip.format.id:
            clip = self._kernel.resample(clip, input_clip, matrix)

        if copy_props:
            return clip.std.CopyFrameProps(input_clip)

        return clip

    def ensure_scaler(self, scaler: ScalerT) -> Scaler:
        from dataclasses import is_dataclass, replace

        scaler_obj = Scaler.ensure_obj(scaler, self.__class__)

        if is_dataclass(scaler_obj):
            from inspect import Signature

            kwargs = dict()

            init_keys = Signature.from_callable(scaler_obj.__init__).parameters.keys()

            if 'kernel' in init_keys:
                kwargs.update(kernel=self.kernel or scaler_obj.kernel)

            if 'scaler' in init_keys:
                kwargs.update(scaler=self.scaler or scaler_obj.scaler)

            if 'shifter' in init_keys:
                kwargs.update(shifter=self.shifter or scaler_obj.shifter)

            if kwargs:
                scaler_obj = replace(scaler_obj, **kwargs)

        return scaler_obj


def scale_var_clip(
    clip: vs.VideoNode,
    width: int | Callable[[Resolution], int] | None, height: int | Callable[[Resolution], int],
    shift: tuple[float, float] | Callable[[Resolution], tuple[float, float]] = (0, 0),
    scaler: Scaler | Callable[[Resolution], Scaler] = Nnedi3(), debug: bool = False
) -> vs.VideoNode:
    """Scale a variable clip to constant or variable resolution."""
    if not debug:
        try:
            return scaler.scale(clip, width, height, shift)  # type: ignore
        except BaseException:
            pass

    _cached_clips = dict[str, vs.VideoNode]()

    no_accepts_var = list[Scaler]()

    def _eval_scale(f: vs.VideoFrame, n: int) -> vs.VideoNode:
        key = f'{f.width}_{f.height}'

        if key not in _cached_clips:
            res = Resolution(f.width, f.height)

            norm_scaler = scaler(res) if callable(scaler) else scaler
            norm_shift = shift(res) if callable(shift) else shift
            norm_height = height(res) if callable(height) else height

            if width is None:
                norm_width = get_w(norm_height, res.width / res.height)
            else:
                norm_width = width(res) if callable(width) else width

            part_scaler = partial(
                norm_scaler.scale, width=norm_width, height=norm_height, shift=norm_shift
            )

            scaled = clip
            if (scaled.width, scaled.height) != (norm_width, norm_height):
                if norm_scaler not in no_accepts_var:
                    try:
                        scaled = part_scaler(clip)
                    except BaseException:
                        no_accepts_var.append(norm_scaler)

                if norm_scaler in no_accepts_var:
                    const_clip = clip.resize.Point(res.width, res.height)

                    scaled = part_scaler(const_clip)

            if debug:
                scaled = scaled.std.SetFrameProps(var_width=res.width, var_height=res.height)

            _cached_clips[key] = scaled

        return _cached_clips[key]

    if callable(width) or callable(height):
        out_clip = clip
    else:
        out_clip = clip.std.BlankClip(width, height)

    return out_clip.std.FrameEval(_eval_scale, clip, clip)


@dataclass
class ScalingArgs:
    width: int
    height: int
    src_width: float
    src_height: float
    src_top: float
    src_left: float
    mode: str = 'hw'

    base_clip: vs.VideoNode | None = None

    def _do(self) -> tuple[bool, bool]:
        return 'h' in self.mode.lower(), 'w' in self.mode.lower()

    def _up_rate(self, clip: vs.VideoNode | None = None) -> tuple[float, float]:
        if clip is None:
            return 1.0, 1.0

        assert self.base_clip

        do_h, do_w = self._do()

        return (
            (clip.height / self.height) if do_h else 1.0,
            (clip.width / self.width) if do_w else 1.0
        )

    def kwargs(self, clip_or_rate: vs.VideoNode | float | None = None, /) -> KwargsT:
        kwargs = KwargsT()
        do_h, do_w = self._do()
        up_rate_h, up_rate_w = (
            self._up_rate(clip_or_rate)
            if clip_or_rate is None or isinstance(clip_or_rate, vs.VideoNode) else
            (clip_or_rate, clip_or_rate)
        )

        if do_h:
            kwargs.update(
                src_height=self.src_height * up_rate_h,
                src_top=self.src_top * up_rate_h
            )

        if do_w:
            kwargs.update(
                src_width=self.src_width * up_rate_w,
                src_left=self.src_left * up_rate_w
            )

        return kwargs

    def descale(self, kernel: type[Kernel], clip: vs.VideoNode | None = None) -> vs.VideoNode:
        if not clip:
            clip = self.base_clip

        do_h, do_w = self._do()

        return kernel.descale(
            clip,
            self.width if do_w else clip.width,
            self.height if do_h else clip.height,
            **self.kwargs()
        )


def descale_args(
    clip: vs.VideoNode,
    src_height: float, src_width: float | None = None,
    base_height: int | None = None, base_width: int | None = None,
    crop_top: int = 0, crop_bottom: int = 0,
    crop_left: int = 0, crop_right: int = 0,
    mode: str = 'hw'
) -> ScalingArgs:
    base_height = fallback(base_height, mod2(ceil(src_height)))
    base_width = fallback(base_width, get_w(base_height, clip, 2))

    ratio = src_height / (clip.height + crop_top + crop_bottom)
    src_width = fallback(src_width, ratio * (clip.width + crop_left + crop_right))

    margin_left = (base_width - src_width) / 2 + ratio * crop_left
    margin_right = (base_width - src_width) / 2 + ratio * crop_right
    cropped_width = base_width - floor(margin_left) - floor(margin_right)

    margin_top = (base_height - src_height) / 2 + ratio * crop_top
    margin_bottom = (base_height - src_height) / 2 + ratio * crop_bottom
    cropped_height = base_height - floor(margin_top) - floor(margin_bottom)

    cropped_src_width = ratio * clip.width
    cropped_src_left = margin_left - floor(margin_left)

    cropped_src_height = ratio * clip.height
    cropped_src_top = margin_top - floor(margin_top)

    return ScalingArgs(
        cropped_width, cropped_height,
        cropped_src_width, cropped_src_height,
        cropped_src_top, cropped_src_left,
        mode, clip
    )


def fdescale_args(
    clip: vs.VideoNode, src_height: float,
    base_height: int | None = None, base_width: int | None = None,
    src_top: float | None = None, src_left: float | None = None,
    src_width: float | None = None, mode: str = 'hw', up_rate: float = 2.0
) -> tuple[KwargsT, KwargsT]:
    base_height = fallback(base_height, mod2(ceil(src_height)))
    base_width = fallback(base_width, get_w(base_height, clip, 2))

    src_width = fallback(src_width, src_height * clip.width / clip.height)

    cropped_width = base_width - 2 * floor((base_width - src_width) / 2)
    cropped_height = base_height - 2 * floor((base_height - src_height) / 2)

    do_h, do_w = 'h' in mode.lower(), 'w' in mode.lower()

    de_args = dict(
        width=cropped_width if do_w else clip.width,
        height=cropped_height if do_h else clip.height
    )

    up_args = dict()

    src_top = fallback(src_top, (cropped_height - src_height) / 2)
    src_left = fallback(src_left, (cropped_width - src_width) / 2)

    if do_h:
        de_args.update(src_height=src_height, src_top=src_top)
        up_args.update(src_height=src_height * up_rate, src_top=src_top * up_rate)

    if do_w:
        de_args.update(src_width=src_width, src_left=src_left)
        up_args.update(src_width=src_width * up_rate, src_left=src_left * up_rate)

    return de_args, up_args
