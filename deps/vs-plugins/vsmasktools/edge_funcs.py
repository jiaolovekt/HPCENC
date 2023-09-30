from __future__ import annotations

from functools import partial
from typing import Any, Literal, Sequence, cast, overload

from vsexprtools import ExprOp, ExprToken, norm_expr
from vsrgtools import gauss_blur
from vsrgtools.util import wmean_matrix
from vstools import (
    ColorRange, CustomEnum, VSFunctionAllArgs, check_variable, core, depth, get_peak_value, get_y, iterate, plane,
    scale_value, vs
)

from .details import multi_detail_mask
from .edge import FDoGTCanny, Kirsch, MagDirection, Prewitt, PrewittTCanny
from .morpho import Morpho
from .spat_funcs import retinex
from .types import Coordinates, GenericMaskT
from .utils import normalize_mask

__all__ = [
    'ringing_mask',

    'luma_mask', 'luma_credit_mask',

    'tcanny_retinex',

    'limited_linemask',

    'dre_edgemask'
]


def ringing_mask(
    clip: vs.VideoNode,
    rad: int = 2, brz: float = 0.35,
    thmi: float = 0.315, thma: float = 0.5,
    thlimi: float = 0.195, thlima: float = 0.392,
    credit_mask: GenericMaskT = Prewitt, **kwargs: Any
) -> vs.VideoNode:
    assert check_variable(clip, ringing_mask)

    thmi, thma, thlimi, thlima = (
        scale_value(t, 32, clip) for t in [thmi, thma, thlimi, thlima]
    )

    edgemask = normalize_mask(credit_mask, plane(clip, 0), **kwargs).std.Limiter()

    light = norm_expr(edgemask, f'x {thlimi} - {thma - thmi} / {ExprToken.RangeMax} *')

    shrink = Morpho.dilation(light, rad)
    shrink = Morpho.binarize(shrink, brz)
    shrink = Morpho.erosion(shrink, 2)
    shrink = iterate(shrink, partial(core.std.Convolution, matrix=wmean_matrix), 2)

    strong = norm_expr(edgemask, f'x {thmi} - {thlima - thlimi} / {ExprToken.RangeMax} *')
    expand = Morpho.dilation(strong, rad)

    mask = norm_expr([expand, strong, shrink], 'x y z max -')

    return ExprOp.convolution('x', wmean_matrix, premultiply=2, multiply=2, clamp=True)(mask)


def luma_mask(clip: vs.VideoNode, thr_lo: float, thr_hi: float, invert: bool = True) -> vs.VideoNode:
    peak = get_peak_value(clip)

    lo, hi = (peak, 0) if invert else (0, peak)
    inv_pre, inv_post = (peak, '-') if invert else ('', '')

    thr_lo, thr_hi = scale_value(thr_lo, 32, clip), scale_value(thr_hi, 32, clip)

    return norm_expr(
        get_y(clip),
        f'x {thr_lo} < {lo} x {thr_hi} > {hi} {inv_pre} x {thr_lo} - {thr_lo} {thr_hi} - / {peak} * {inv_post} ? ?'
    )


def luma_credit_mask(
    clip: vs.VideoNode, thr: float = 0.9, edgemask: GenericMaskT = FDoGTCanny, draft: bool = False, **kwargs: Any
) -> vs.VideoNode:
    y = plane(clip, 0)

    edge_mask = normalize_mask(edgemask, y, **kwargs)

    credit_mask = norm_expr([edge_mask, y], f'y {scale_value(thr, 32, y)} > y 0 ? x min')

    if not draft:
        credit_mask = Morpho.maximum(credit_mask, iterations=4)
        credit_mask = Morpho.inflate(credit_mask, iterations=2)

    return credit_mask


def tcanny_retinex(
    clip: vs.VideoNode, thr: float, sigma: Sequence[float] = [50, 200, 350], blur_sigma: float = 1.0
) -> vs.VideoNode:
    blur = gauss_blur(clip, blur_sigma)

    msrcp = retinex(blur, sigma, upper_thr=thr, fast=True, func=tcanny_retinex)

    tcunnied = msrcp.tcanny.TCanny(mode=1, sigma=1)

    return Morpho.minimum(tcunnied, None, Coordinates.CORNERS)


def limited_linemask(
    clip: vs.VideoNode,
    sigmas: list[float] = [0.000125, 0.0025, 0.0055],
    detail_sigmas: list[float] = [0.011, 0.013],
    edgemasks: Sequence[GenericMaskT] = [Kirsch],
    **kwargs: Any
) -> vs.VideoNode:
    clip_y = plane(clip, 0)

    return ExprOp.ADD(
        (normalize_mask(edge, clip_y, **kwargs) for edge in edgemasks),
        (tcanny_retinex(clip_y, s) for s in sigmas),
        (multi_detail_mask(clip_y, s) for s in detail_sigmas)
    )


class _dre_edgemask(CustomEnum):
    """Edgemask with dynamic range enhancement prefiltering."""

    RETINEX = cast(VSFunctionAllArgs, object())
    CLAHE = cast(VSFunctionAllArgs, object())

    def _prefilter(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        if self is self.RETINEX:  # type: ignore
            sigmas = kwargs.get('sigmas', [50, 200, 350])

            return retinex(clip, sigmas, 0.001, 0.005)

        if self is self.CLAHE:  # type: ignore
            limit, tile = kwargs.get('limit', 0.0305), kwargs.get('tile', 5)

            return depth(depth(clip, 16).ehist.CLAHE(scale_value(limit, 32, 16), tile), clip)

        return clip

    @overload
    def __call__(  # type: ignore
        self: Literal[RETINEX], src: vs.VideoNode, tsigma: float = 1, brz: float = 0.122,
        *, sigmas: Sequence[float] = [50, 200, 350]
    ) -> vs.VideoNode:
        ...

    @overload
    def __call__(  # type: ignore
        self: Literal[CLAHE], src: vs.VideoNode, tsigma: float = 1, brz: float = 0.122,
        *, limit: float = 0.0305, tile: int = 5
    ) -> vs.VideoNode:
        ...

    def __call__(self, src: vs.VideoNode, tsigma: float = 1, brz: float = 0.122, **kwargs: Any) -> vs.VideoNode:
        luma = get_y(src)

        dreluma = self._prefilter(luma, **kwargs)

        tcanny = PrewittTCanny.edgemask(dreluma, sigma=tsigma, scale=1)
        tcanny = Morpho.minimum(tcanny, coords=Coordinates.CORNERS)

        kirsch = Kirsch(MagDirection.N | MagDirection.EAST).edgemask(luma)

        add_clip = ExprOp.ADD(tcanny, kirsch)

        if brz > 0:
            add_clip = Morpho.binarize(add_clip, brz)

        return ColorRange.FULL.apply(add_clip)


dre_edgemask = _dre_edgemask.RETINEX
