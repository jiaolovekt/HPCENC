from __future__ import annotations

from fractions import Fraction
from functools import partial
from math import ceil
from typing import NamedTuple

from vskernels import Catrom, Kernel, KernelT
from vstools import CustomEnum, CustomIntEnum, change_fps, check_variable_format, clamp, vs, InvalidSubsamplingError

from .easing import EasingT, Linear, OnAxis

__all__ = [
    'PanDirection', 'PanFunction', 'PanFunctions',

    'panner'
]


class PanDirection(CustomIntEnum):
    NORMAL = 0
    INVERTED = 1


class PanFunction(NamedTuple):
    direction: PanDirection = PanDirection.NORMAL
    function_x: EasingT = OnAxis
    function_y: EasingT = OnAxis


class PanFunctions(PanFunction, CustomEnum):
    VERTICAL_TTB = PanFunction(function_y=Linear)
    HORIZONTAL_LTR = PanFunction(function_x=Linear)
    VERTICAL_BTT = PanFunction(PanDirection.INVERTED, function_y=Linear)
    HORIZONTAL_RTL = PanFunction(PanDirection.INVERTED, function_x=Linear)


def panner(
    clip: vs.VideoNode, stitched: vs.VideoNode,
    pan_func: PanFunction | PanFunctions = PanFunctions.VERTICAL_TTB,
    fps: Fraction = Fraction(24000, 1001), kernel: KernelT = Catrom
) -> vs.VideoNode:
    assert check_variable_format(clip, panner)
    assert check_variable_format(stitched, panner)

    if (stitched.format.subsampling_h, stitched.format.subsampling_w) != (0, 0):
        raise InvalidSubsamplingError(panner, stitched.format, "Stitched can't be subsampled!", reason='{subsampling}')

    kernelo = Kernel.ensure_obj(kernel, panner)
    clip_cfps = change_fps(clip, fps)

    offset_x, offset_y = (stitched.width - clip.width), (stitched.height - clip.height)

    ease_x = pan_func.function_x(0, offset_x, clip_cfps.num_frames).ease
    ease_y = pan_func.function_y(0, offset_y, clip_cfps.num_frames).ease

    clamp_x = partial(lambda x: int(clamp(x, min_val=0, max_val=offset_x)))
    clamp_y = partial(lambda x: int(clamp(x, min_val=0, max_val=offset_y)))

    def _pan(n: int) -> vs.VideoNode:
        x_e, x_v = divmod(clamp_x(ease_x(n)), 1)
        y_e, y_v = divmod(clamp_y(ease_y(n)), 1)

        if n == clip_cfps.num_frames - 1:
            x_e, y_e = clamp_x(offset_x - 1), clamp_y(offset_y - 1)
            x_v, y_v = int(x_e == offset_x - 1), int(y_e == offset_y - 1)

        x_c, y_c = ceil(x_v), ceil(y_v)

        cropped = stitched.std.CropAbs(
            clip.width + x_c, clip.height + y_c, int(x_e), int(y_e)
        )

        shifted = kernelo.shift(cropped, (y_v, x_v))

        cropped = shifted.std.Crop(bottom=y_c, right=x_c)

        return kernelo.resample(cropped, clip)

    newpan = clip_cfps.std.FrameEval(_pan)

    return newpan[::-1] if pan_func.direction == PanDirection.INVERTED else newpan
