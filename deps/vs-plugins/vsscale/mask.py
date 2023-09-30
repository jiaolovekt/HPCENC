from __future__ import annotations

from vsexprtools import ExprOp, average_merge, norm_expr
from vskernels import Catrom
from vsmasktools import Morpho, XxpandMode
from vsrgtools import box_blur, gauss_blur
from vstools import core, get_y, iterate, shift_clip_multi, split, vs

__all__ = [
    'descale_detail_mask', 'descale_error_mask'
]


def descale_detail_mask(
    clip: vs.VideoNode, rescaled: vs.VideoNode, thr: float = 0.05,
    inflate: int = 2, xxpand: tuple[int, int] = (4, 0)
) -> vs.VideoNode:
    """
    Mask non-native resolution detail to prevent detail loss and artifacting.

    Descaling without masking is very dangerous, as descaling FHD material often leads to
    heavy artifacting and fine detail loss.

    :param clip:        Original clip.
    :param rescaled:    Clip rescaled using the presumed native kernel.
    :param thr:         Binarizing threshold. Lower will catch more.
                        Assumes float bitdepth input.
                        Default: 0.05.
    :param inflate:     Amount of times to ``inflate`` the mask. Default: 2.
    :param xxpand:      Amount of times to ``Maximum`` the clip by.
                        The first ``Maximum`` is done before inflating, the second after.
                        Default: 4 times pre-inflating, 0 times post-inflating.

    :return:            Mask containing all the native FHD detail.
    """
    mask = norm_expr([get_y(clip), get_y(rescaled)], 'x y - abs')

    mask = Morpho.binarize(mask, thr)

    if xxpand[0]:
        mask = iterate(mask, core.std.Maximum if xxpand[0] > 0 else core.std.Minimum, xxpand[0])

    if inflate:
        mask = iterate(mask, core.std.Inflate, inflate)

    if xxpand[1]:
        mask = iterate(mask, core.std.Maximum if xxpand[1] > 0 else core.std.Minimum, xxpand[1])

    return mask.std.Limiter()


def descale_error_mask(
    clip: vs.VideoNode, rescaled: vs.VideoNode,
    thr: float | list[float] = 0.038,
    expands: int | tuple[int, int, int] = (2, 2, 3),
    blur: int | float = 3, bwbias: int = 1, tr: int = 1
) -> vs.VideoNode:
    """
    Create an error mask from the original and rescaled clip.

    :param clip:        Original clip.
    :param rescaled:    Rescaled clip.
    :param thr:         Threshold of the minimum difference.
    :param expands:     Iterations of mask expand at each step (diff, expand, binarize).
    :param blur:        How much to blur the clip. If int, it will be a box_blur, else gauss_blur.
    :param bwbias:      Calculate a bias with the clip's chroma.
    :param tr:          Make the error mask temporally stable with a temporal radius.

    :return:            Descale error mask.
    """
    assert clip.format and rescaled.format

    y, *chroma = split(clip)

    error = norm_expr([y, rescaled], 'x y - abs')

    if bwbias > 1 and chroma:
        chroma_abs = norm_expr(chroma, 'x range_half - abs y range_half - abs max')
        chroma_abs = Catrom.scale(chroma_abs, y.width, y.height)

        bias = norm_expr([y, chroma_abs], f'x ymax >= x ymin <= or y 0 = and {bwbias} 1 ?')
        bias = Morpho.expand(bias, 2)

        error = ExprOp.MUL(error, bias)

    if isinstance(expands, int):
        exp1 = exp2 = exp3 = expands
    else:
        exp1, exp2, exp3 = expands

    if exp1:
        error = Morpho.expand(error, exp1)

    if exp2:
        error = Morpho.expand(error, exp2, mode=XxpandMode.ELLIPSE)

    thrs = [thr] if isinstance(thr, float) else thr

    error = Morpho.binarize(error, thrs[0])

    for scaled_thr in thrs[1:]:
        bin2 = Morpho.binarize(error, scaled_thr)
        error = bin2.misc.Hysteresis(error)

    if exp3:
        error = Morpho.expand(error, exp2, mode=XxpandMode.ELLIPSE)

    if tr > 1:
        avg = Morpho.binarize(average_merge(*shift_clip_multi(error, (-tr, tr))), 0.5)

        error = ExprOp.MIN(error, ExprOp.MAX(shift_clip_multi(ExprOp.MIN(error, avg), (-tr, tr))))

    if isinstance(blur, int):
        error = box_blur(error, blur)
    else:
        error = gauss_blur(error, blur)

    return error.std.Limiter()
