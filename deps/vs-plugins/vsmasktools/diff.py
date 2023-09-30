from __future__ import annotations

from typing import Any

from vsexprtools import ExprOp
from vskernels import Bicubic, Catrom, Kernel, KernelT
from vsrgtools import RemoveGrainMode, bilateral, gauss_blur, removegrain
from vstools import (
    CustomValueError, FuncExceptT, KwargsT, check_variable, depth, expect_bits, get_w, get_y, insert_clip, iterate, vs
)

from .edge import ExLaplacian4
from .morpho import Morpho
from .types import XxpandMode

__all__ = [
    'diff_rescale',
    'diff_creditless',
    'diff_creditless_oped',
    'credit_mask',
]


def diff_rescale(
    clip: vs.VideoNode, height: int, kernel: KernelT = Catrom,
    thr: float = 0.216, expand: int = 2, func: FuncExceptT | None = None
) -> vs.VideoNode:
    func = func or diff_rescale

    assert check_variable(clip, func)

    kernel = Kernel.ensure_obj(kernel, func)

    y = get_y(clip)

    pre, bits = expect_bits(y, 32)

    descale = kernel.descale(pre, get_w(height), height)
    rescale = kernel.scale(descale, y.width, y.height)

    diff = ExprOp.mae(y)(pre, rescale)

    mask = iterate(diff, removegrain, 2, RemoveGrainMode.MINMAX_AROUND2)
    mask = mask.std.Expr(f'x 2 4 pow * {thr} < 0 1 ?')

    mask = Morpho.expand(mask, 2 + expand, mode=XxpandMode.ELLIPSE).std.Deflate()

    return depth(mask, bits)


def diff_creditless(
    credit_clip: vs.VideoNode, nc_clip: vs.VideoNode, thr: float = 0.01,
    start_frame: int = 0, expand: int = 2, *, prefilter: bool | int = False,
    ep_clip: vs.VideoNode | None = None, func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    func = func or diff_creditless

    assert check_variable(credit_clip, func)

    clips = [credit_clip, nc_clip]

    if prefilter:
        sigma = 5 if prefilter is True else prefilter
        kwargs |= KwargsT(sigmaS=((sigma ** 2 - 1) / 12) ** 0.5, sigmaR=sigma / 10) | kwargs
        clips = [bilateral(c, **kwargs) for c in clips]

    dst_fmt = credit_clip.format.replace(subsampling_w=0, subsampling_h=0)
    diff_fmt = dst_fmt.replace(color_family=vs.GRAY)

    diff = ExprOp.mae(dst_fmt)(
        (Bicubic.resample(c, dst_fmt) for c in clips),
        format=diff_fmt, split_planes=True
    )

    mask = ExLaplacian4.edgemask(diff, lthr=thr, hthr=thr)
    mask = Morpho.expand(mask, 2 + expand, mode=XxpandMode.ELLIPSE)

    if not ep_clip or ep_clip.num_frames == mask.num_frames:
        return mask

    blank = ep_clip.std.BlankClip(format=diff_fmt.id, keep=True)

    return insert_clip(blank, mask, start_frame)


def diff_creditless_oped(
    ep: vs.VideoNode, ncop: vs.VideoNode, nced: vs.VideoNode, thr: float = 0.1,
    opstart: int | None = None, opend: int | None = None,
    edstart: int | None = None, edend: int | None = None,
    func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    func = func or diff_creditless_oped

    op_mask = ed_mask = None

    kwargs |= KwargsT(expand=4, prefilter=False, func=func, ep_clip=ep) | kwargs

    if opstart is not None and opend is not None:
        op_mask = diff_creditless(ep[opstart:opend + 1], ncop[:opend - opstart + 1], thr, opstart, **kwargs)

    if edstart is not None and edend is not None:
        ed_mask = diff_creditless(ep[edstart:edend + 1], nced[:edend - edstart + 1], thr, edstart, **kwargs)

    if op_mask and ed_mask:
        return ExprOp.ADD.combine(op_mask, ed_mask)
    elif op_mask or ed_mask:
        return op_mask or ed_mask  # type: ignore

    raise CustomValueError(
        'You must specify one or both of ("opstart", "opend"), ("edstart", "edend")', func
    )


def credit_mask(
    clip: vs.VideoNode, ref: vs.VideoNode, thr: float,
    blur: float | None = 1.65, prefilter: bool | int = 5,
    expand: int = 8
) -> vs.VideoNode:
    if blur:
        clip = gauss_blur(clip, blur)
        ref = gauss_blur(ref, blur)

    ed_mask = diff_creditless(clip, ref, thr, prefilter=prefilter)

    credit_mask, bits = expect_bits(ed_mask)
    credit_mask = Morpho.erosion(credit_mask, 6)
    credit_mask = iterate(credit_mask, lambda x: x.std.Minimum().std.Maximum(), 8)

    if expand:
        credit_mask = Morpho.dilation(credit_mask, expand)

    credit_mask = Morpho.inflate(credit_mask, 3)

    return depth(credit_mask, bits)
