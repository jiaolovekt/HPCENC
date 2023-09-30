from __future__ import annotations

from enum import IntEnum
from itertools import cycle
from math import cos, degrees, pi, sin
from typing import Any, NamedTuple, Sequence, SupportsFloat

from vsexprtools import ExprOp, norm_expr
from vstools import (
    ColorRange, CustomValueError, FrameRangeN, StrList, VSFunction, check_variable, fallback, get_depth,
    get_neutral_value, get_prop, insert_clip, normalize_ranges, scale_value, vs
)
from vstransitions import EasingT, ExponentialEaseIn, crossfade

__all__ = [
    'Tweak',
    'tweak_clip', 'multi_tweak',

    'BalanceMode', 'WeightMode', 'Override',

    'auto_balance'
]


def tweak_clip(
    clip: vs.VideoNode, cont: float = 1.0, sat: float = 1.0,
    bright: float = 0.0, hue: float = 0.0, relative_sat: float | None = None,
    range_in: ColorRange | None = None, range_out: ColorRange | None = None,
    clamp: bool = True, pre: vs.VideoNode | VSFunction | None = None, post: VSFunction | None = None
) -> vs.VideoNode:
    assert clip.format

    bits = get_depth(clip)

    range_in = fallback(range_in, ColorRange(clip))
    range_out = fallback(range_out, range_in)

    sv_args_out = dict[str, Any](
        input_depth=8, output_depth=bits, range_in=range_in, range_out=range_out, scale_offsets=True
    )

    luma_min = scale_value(16, **sv_args_out)
    chroma_min = scale_value(16, **sv_args_out, chroma=True)

    luma_max = scale_value(235, **sv_args_out)
    chroma_max = scale_value(240, **sv_args_out, chroma=True)

    chroma_center = get_neutral_value(clip, True)

    if relative_sat is not None:
        if cont == 1.0 or relative_sat == 1.0:
            sat = cont
        else:
            sat = (cont - 1.0) * relative_sat + 1.0

    cont = max(cont, 0.0)
    sat = max(sat, 0.0)

    if (hue == bright == 0.0) and (sat == cont == 1.0):
        return clip

    pre_clip = pre(clip) if callable(pre) else fallback(pre, clip)

    clips = [pre_clip]

    yexpr = list[Any](['x'])
    cexpr = list[Any](['x'])

    if (hue != 0.0 or sat != 1.0) and clip.format.color_family != vs.GRAY:
        hue *= pi / degrees(pi)

        hue_sin, hue_cos = sin(hue), cos(hue)

        normalize = [chroma_center, ExprOp.SUB]

        cexpr.extend([normalize, hue_cos, sat, ExprOp.MUL * 2])

        if hue != 0:
            clips += [pre_clip.std.ShufflePlanes([0, 2, 1], vs.YUV)]
            cexpr.extend(['y', normalize, hue_sin, sat, ExprOp.MUL * 2, ExprOp.ADD])

        cexpr.extend([chroma_center, ExprOp.ADD])

        if clamp and range_out:
            cexpr.extend(StrList([chroma_min, ExprOp.MAX, chroma_max, ExprOp.MIN]))

    if bright != 0 or cont != 1:
        if luma_min > 0:
            yexpr.extend([luma_min, ExprOp.SUB])

        if cont != 1:
            yexpr.extend([cont, ExprOp.MUL])

        if (luma_min + bright) != 0:
            yexpr.extend([luma_min, bright, ExprOp.ADD * 2])

        if clamp and range_out:
            yexpr.extend(StrList([luma_min, ExprOp.MAX, luma_max, ExprOp.MIN]))

    tclip = norm_expr(clips, (yexpr, cexpr))

    return post(tclip) if callable(post) else tclip


class Tweak(NamedTuple):
    frame: int
    cont: SupportsFloat | None = None
    sat: SupportsFloat | None = None
    bright: SupportsFloat | None = None
    hue: SupportsFloat | None = None
    ease_func: EasingT = ExponentialEaseIn


def multi_tweak(clip: vs.VideoNode, tweaks: list[Tweak], debug: bool = False, **tkargs: dict[str, Any]) -> vs.VideoNode:
    if len(tweaks) < 2:
        raise ValueError("multi_tweak: 'At least two tweaks need to be passed!'")

    for i, tmp_tweaks in enumerate(zip([tweaks[0]] + tweaks, tweaks, cycle(tweaks[1:]))):
        tprev, tweak, tnext = [list(filter(None, x)) for x in tmp_tweaks]

        if len(tweak) == 1 and len(tprev) > 1 and i > 0:
            tweak = tweak[:1] + tprev[1:]

        cefunc, _ = tweak.pop(), tnext.pop()
        start, stop = tweak.pop(0), tnext.pop(0)

        if start == stop:
            continue

        assert isinstance(start, int) and isinstance(stop, int)

        spliced_clip = clip[start:stop]

        if tweak == tnext:
            tweaked_clip = tweak_clip(spliced_clip, *tweak, **tkargs)  # type: ignore
        else:
            clipa, clipb = (tweak_clip(spliced_clip, *args, **tkargs) for args in (tweak, tnext))  # type: ignore

            tweaked_clip = crossfade(clipa, clipb, cefunc, debug)

        clip = insert_clip(clip, tweaked_clip, start)

    return clip


class BalanceMode(IntEnum):
    AUTO = 0
    UNDIMMING = 1
    DIMMING = 2


class WeightMode(IntEnum):
    INTERPOLATE = 0
    MEDIAN = 1
    MEAN = 2
    MAX = 3
    MIN = 4
    NONE = 5


class Override(NamedTuple):
    frame_range: FrameRangeN
    cont: SupportsFloat
    override_mode: WeightMode = WeightMode.INTERPOLATE


def auto_balance(
    clip: vs.VideoNode, target_max: SupportsFloat | None = None, relative_sat: float = 1.0,
    range_in: ColorRange = ColorRange.LIMITED, frame_overrides: Override | Sequence[Override] = [],
    ref: vs.VideoNode | None = None, radius: int = 1, delta_thr: float = 0.4,
    min_thr: float = 1.0, max_thr: float = 5.0,
    min_thr_tr: float = 1.0, max_thr_tr: float = 5.0,
    balance_mode: BalanceMode = BalanceMode.UNDIMMING, weight_mode: WeightMode = WeightMode.MEAN,
    prop: bool = False
) -> vs.VideoNode:
    import numpy as np

    ref_clip = fallback(ref, clip)

    assert check_variable(clip, auto_balance)
    assert check_variable(ref_clip, auto_balance)

    if ref_clip.format.sample_type is vs.FLOAT:
        raise CustomValueError(auto_balance, 'Float auto_balance not implemented yet!')

    zero = scale_value(16, 8, ref_clip, range_in, scale_offsets=True)

    target = float(fallback(
        target_max,
        scale_value(
            235, input_depth=8, output_depth=ref_clip,
            range_in=range_in, scale_offsets=True
        )
    ))

    if weight_mode == WeightMode.NONE:
        raise CustomValueError(auto_balance, 'Global weight mode can\'t be NONE!')

    ref_stats = ref_clip.std.PlaneStats()

    over_mapped = list[tuple[range, float, WeightMode]]()

    if frame_overrides:
        frame_overrides = [frame_overrides] if isinstance(frame_overrides, Override) else list(frame_overrides)

        over_frames, over_conts, over_int_modes = list(zip(*frame_overrides))

        oframes_ranges = [
            range(start, stop + 1)
            for start, stop in normalize_ranges(clip, list(over_frames))
        ]

        over_mapped = list(zip(oframes_ranges, over_conts, over_int_modes))

    clipfrange = range(0, clip.num_frames)

    def _weighted(x: float, y: float, z: float) -> float:
        return max(1e-6, x - z) / max(1e-6, y - z)

    nobalanceclip = clip.std.SetFrameProps(AutoBalance=False) if prop else clip

    def _autobalance(n: int, f: Sequence[vs.VideoFrame]) -> vs.VideoNode:
        override: tuple[range, float, WeightMode] | None = next((x for x in over_mapped if n in x[0]), None)

        psvalues: Any = np.asarray([
            _weighted(target, get_prop(frame.props, 'PlaneStatsMax', int), zero) for frame in f
        ])

        middle_idx = psvalues.size // 2

        mean_value = np.mean(psvalues)

        if not override and not (mean_value >= min_thr_tr and mean_value <= max_thr_tr):
            return nobalanceclip

        curr_value = psvalues[middle_idx]

        if not override and not (curr_value >= min_thr and curr_value <= max_thr):
            return nobalanceclip

        if balance_mode == BalanceMode.UNDIMMING:
            psvalues[psvalues < 1.0] = 1.0
        elif balance_mode == BalanceMode.DIMMING:
            psvalues[psvalues > 1.0] = 1.0

        psvalues[(abs(psvalues - curr_value) > delta_thr)] = curr_value

        def _get_cont(mode: WeightMode, frange: range) -> Any:
            if mode == WeightMode.INTERPOLATE:
                if radius < 1:
                    raise CustomValueError(auto_balance, 'Radius has to be >= 1 with WeightMode.INTERPOLATE!')

                weight = (n - (frange.start - 1)) / (frange.stop - (frange.start - 1))

                weighted_prev = psvalues[middle_idx - 1] * (1 - weight)
                weighted_next = psvalues[middle_idx + 1] * weight

                return weighted_prev + weighted_next

            if mode == WeightMode.MEDIAN:
                return np.median(psvalues)

            if mode == WeightMode.MEAN:
                return psvalues.mean()

            if mode == WeightMode.MAX:
                return psvalues.max()

            if mode == WeightMode.MIN:
                return psvalues.min()

            return psvalues[middle_idx]

        if override:
            frange, cont, override_mode = override

            if override_mode == WeightMode.NONE:
                return nobalanceclip

            if cont is not None:
                psvalues[
                    max(0, middle_idx - (n - frange.start)):
                    min(len(psvalues), middle_idx + (frange.stop - n))
                ] = cont

            if (override_mode != weight_mode):
                cont = _get_cont(override_mode, frange)
        else:
            cont = _get_cont(weight_mode, clipfrange)

        sat = (cont - 1) * relative_sat + 1

        fix = tweak_clip(clip, cont, sat, range_in=range_in)

        if prop:
            return fix.std.SetFrameProps(AutoBalance=True, AutoBalanceCont=cont, AutoBalanceSat=sat)

        return fix

    stats_clips = [
        *(ref_stats[0] * i + ref_stats[:-i] for i in range(1, radius + 1)),
        ref_stats,
        *(ref_stats[i:] + ref_stats[-1] * i for i in range(1, radius + 1)),
    ]

    return clip.std.FrameEval(_autobalance, stats_clips, clip)
