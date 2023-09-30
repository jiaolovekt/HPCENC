from __future__ import annotations

from vsexprtools import ExprOp
from vsrgtools import RemoveGrainMode, RemoveGrainModeT, bilateral, gauss_blur, removegrain
from vstools import check_variable, get_y, plane, vs

from .edge import Kirsch, MinMax, Prewitt, PrewittTCanny
from .masks import range_mask
from .morpho import Morpho
from .types import GenericMaskT
from .utils import normalize_mask

__all__ = [
    'detail_mask',
    'detail_mask_neo',
    'simple_detail_mask',
    'multi_detail_mask',
]


def detail_mask(
    clip: vs.VideoNode, brz_mm: float, brz_ed: float,
    minmax: MinMax = MinMax(rady=3, radc=2),
    edge: GenericMaskT = Kirsch
) -> vs.VideoNode:
    assert check_variable(clip, detail_mask)

    range_mask = Morpho.binarize(minmax.edgemask(clip), brz_mm)

    edges = Morpho.binarize(normalize_mask(edge, clip), brz_ed)

    mask = ExprOp.MAX.combine(range_mask, edges)

    mask = removegrain(mask, 22)
    mask = removegrain(mask, 11)

    return mask.std.Limiter()


def detail_mask_neo(
    clip: vs.VideoNode, sigma: float = 1.0, detail_brz: float = 0.05, lines_brz: float = 0.08,
    edgemask: GenericMaskT = Prewitt, rg_mode: RemoveGrainModeT = RemoveGrainMode.MINMAX_MEDIAN_OPP
) -> vs.VideoNode:
    assert check_variable(clip, "detail_mask_neo")

    clip_y = get_y(clip)
    blur_pf = gauss_blur(clip_y, sigma * 0.75)

    blur_pref = bilateral(clip_y, sigma, ref=blur_pf)
    blur_pref_diff = ExprOp.SUB.combine(blur_pref, clip_y).std.Deflate()
    blur_pref = Morpho.inflate(blur_pref_diff, iterations=4)

    prew_mask = normalize_mask(edgemask, clip_y).std.Deflate().std.Inflate()

    if detail_brz > 0:
        blur_pref = Morpho.binarize(blur_pref, detail_brz)

    if lines_brz > 0:
        prew_mask = Morpho.binarize(prew_mask, lines_brz)

    merged = ExprOp.ADD.combine(blur_pref, prew_mask)

    return removegrain(merged, rg_mode).std.Limiter()


def simple_detail_mask(
    clip: vs.VideoNode, sigma: float | None = None, rad: int = 3, brz_a: float = 0.025, brz_b: float = 0.045
) -> vs.VideoNode:
    y = plane(clip, 0)

    blur = gauss_blur(y, sigma) if sigma else y

    mask_a = Morpho.binarize(range_mask(blur, rad=rad), brz_a)

    mask_b = Morpho.binarize(PrewittTCanny.edgemask(blur), brz_b)

    mask = ExprOp.MAX.combine(mask_a, mask_b)

    return removegrain(removegrain(mask, 22), 11).std.Limiter()


def multi_detail_mask(clip: vs.VideoNode, thr: float = 0.015) -> vs.VideoNode:
    general_mask = simple_detail_mask(clip, rad=1, brz_a=1, brz_b=24.3 * thr)

    return ExprOp.MIN.combine(
        ExprOp.MIN.combine(
            simple_detail_mask(clip, brz_a=1, brz_b=2 * thr),
            Morpho.maximum(general_mask, iterations=4) .std.Inflate()
        ), general_mask.std.Maximum()
    )
