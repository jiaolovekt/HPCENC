from __future__ import annotations

from typing import Sequence, overload

from vsexprtools import ExprOp, ExprVars, complexpr_available, norm_expr
from vsrgtools import box_blur, gauss_blur
from vstools import (
    ColorRange, CustomRuntimeError, DitherType, FuncExceptT, StrList, check_variable, core, depth, fallback,
    get_lowest_value, get_peak_value, get_sample_type, get_y, plane, scale_8bit, scale_value, vs
)

from .edge import MinMax
from .morpho import Morpho

__all__ = [
    'adg_mask',
    'retinex',
    'flat_mask',
    'texture_mask'
]


@overload
def adg_mask(  # type: ignore
    clip: vs.VideoNode, luma_scaling: float = 8.0, relative: bool = False, func: FuncExceptT | None = None
) -> vs.VideoNode:
    ...


@overload
def adg_mask(
    clip: vs.VideoNode, luma_scaling: Sequence[float] = ..., relative: bool = False, func: FuncExceptT | None = None
) -> list[vs.VideoNode]:
    ...


def adg_mask(
    clip: vs.VideoNode, luma_scaling: float | Sequence[float] = 8.0,
    relative: bool = False, func: FuncExceptT | None = None
) -> vs.VideoNode | list[vs.VideoNode]:
    func = func or adg_mask

    assert check_variable(clip, func)

    y = plane(clip, 0).std.PlaneStats(prop='P')

    if not complexpr_available:
        if relative:
            raise CustomRuntimeError(
                "You don't have akarin plugin, you can't use this function!", func, 'relative=True'
            )

        if isinstance(luma_scaling, Sequence):
            return [y.adg.Mask(ls) for ls in luma_scaling]  # type: ignore

        return y.adg.Mask(luma_scaling)  # type: ignore

    assert y.format

    peak = get_peak_value(y)

    is_integer = y.format.sample_type == vs.INTEGER

    x_string, aft_int = (f'x {peak} / ', f' {peak} * 0.5 +') if is_integer else ('x ', '0 1 clamp')

    if relative:
        x_string += 'Y! Y@ 0.5 < x.PMin 0 max 0.5 / log Y@ * x.PMax 1.0 min 0.5 / log Y@ * ? '

    x_string += '0 0.999 clamp X!'

    def _adgfunc(ls: float) -> vs.VideoNode:
        return norm_expr(
            y, f'{x_string} 1 X@ X@ X@ X@ X@ '
            '18.188 * 45.47 - * 36.624 + * 9.466 - * 1.124 + * - '
            f'x.PAverage 2 pow {ls} * pow {aft_int}'
        )

    if isinstance(luma_scaling, Sequence):
        return [_adgfunc(ls) for ls in luma_scaling]

    return _adgfunc(luma_scaling)


def retinex(
    clip: vs.VideoNode, sigma: Sequence[float] = [25, 80, 250],
    lower_thr: float = 0.001, upper_thr: float = 0.001,
    fast: bool | None = None, func: FuncExceptT | None = None
) -> vs.VideoNode:
    func = func or retinex

    assert check_variable(clip, func)

    sigma = sorted(sigma)

    y = get_y(clip)

    if not complexpr_available or not hasattr(core, 'psm'):
        if fast:
            raise CustomRuntimeError(
                "You don't have {missing} plugin, you can't use this function!", func, 'fast=True',
                missing=iter(x for x in ('akarin', 'psm') if not hasattr(core, x))
            )

        return y.retinex.MSRCP(sigma, lower_thr, upper_thr)  # type: ignore
    elif fast is None:
        fast = True

    y = y.std.PlaneStats()
    is_float = get_sample_type(y) is vs.FLOAT

    if is_float:
        luma_float = norm_expr(y, "x x.PlaneStatsMin - x.PlaneStatsMax x.PlaneStatsMin - /")
    else:
        luma_float = norm_expr(y, "1 x.PlaneStatsMax x.PlaneStatsMin - / x x.PlaneStatsMin - *", None, vs.GRAYS)

    slen, slenm = len(sigma), len(sigma) - 1

    expr_msr = StrList([
        f"{x} 0 <= 1 x {x} / 1 + ? "
        for x in ExprVars(1, slen + (not fast))
    ])

    if fast:
        expr_msr.append("x.PlaneStatsMax 0 <= 1 x x.PlaneStatsMax / 1 + ? ")
        sigma = sigma[:-1]

    expr_msr.extend(ExprOp.ADD * slenm)
    expr_msr.append(f"log {slen} /")

    msr = norm_expr([luma_float, (gauss_blur(luma_float, i) for i in sigma)], expr_msr)

    msr_stats = msr.psm.PlaneMinMax(lower_thr, upper_thr)

    expr_balance = "x x.psmMin - x.psmMax x.psmMin - /"

    if not is_float:
        expr_balance = f"{expr_balance} {{ymax}} {{ymin}} - * {{ymin}} + round {{ymin}} {{ymax}} clamp"

    return norm_expr(
        msr_stats, expr_balance, None, y,
        ymin=get_lowest_value(y, False, ColorRange.LIMITED),
        ymax=get_peak_value(y, False, ColorRange.LIMITED)
    )


def flat_mask(src: vs.VideoNode, radius: int = 5, thr: float = 0.011, gauss: bool = False) -> vs.VideoNode:
    luma = get_y(src)

    blur = gauss_blur(luma, radius * 0.361083333) if gauss else box_blur(luma, radius)

    mask = depth(luma, 8).abrz.AdaptiveBinarize(depth(blur, 8), scale_value(thr, 32, 8))

    return depth(mask, luma, dither_type=DitherType.NONE, range_in=ColorRange.FULL, range_out=ColorRange.FULL)


def texture_mask(
    clip: vs.VideoNode, rady: int = 2, radc: int | None = None,
    blur: int | float = 8, thr: float = 0.2,
    stages: list[tuple[int, int]] = [(60, 2), (40, 4), (20, 2)],
    points: list[tuple[bool, float]] = [(False, 1.75), (True, 2.5), (True, 5), (False, 10)]
) -> vs.VideoNode:
    levels = [x for x, _ in points]
    _points = [scale_value(x, 8, clip) for _, x in points]

    qm, peak = len(points), get_peak_value(clip)

    rmask = MinMax(rady, fallback(radc, rady)).edgemask(clip, lthr=0)

    emask = clip.std.Prewitt()

    rm_txt = ExprOp.MIN(rmask, (
        Morpho.minimum(Morpho.binarize(emask, scale_8bit(32, thr), 1.0, 0), iterations=it)
        for thr, it in stages
    ))

    expr = [f'x {_points[0]} < x {_points[-1]} > or 0']

    for x in range(len(_points) - 1):
        if _points[x + 1] < _points[-1]:
            expr.append(f'x {_points[x+1]} <=')

        if levels[x] == levels[x + 1]:
            expr.append(f'{peak if levels[x] else 0}')
        else:
            mean = peak * (levels[x + 1] - levels[x]) / (_points[x + 1] - _points[x])
            expr.append(f'x {_points[x]} - {mean} * {peak * levels[x]} +')

    weighted = norm_expr(rm_txt, [expr, ExprOp.TERN * (qm - 1)])

    if isinstance(blur, float):
        weighted = gauss_blur(weighted, blur)
    else:
        weighted = box_blur(weighted, blur)

    return norm_expr(weighted, f'x {peak * thr} - {1 / (1 - thr)} *')
