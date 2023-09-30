from __future__ import annotations

from typing import Any, cast

from vsexprtools import complexpr_available, expr_func, norm_expr
from vstools import VSFunction, join, shift_clip, shift_clip_multi, vs

from .combing import vinverse
from .utils import telecine_patterns

__all__ = [
    'deblending_helper',

    'deblend', 'deblend_bob',

    'deblend_fix_kf'
]


def deblending_helper(deblended: vs.VideoNode, fieldmatched: vs.VideoNode, length: int = 5) -> vs.VideoNode:
    """
    Helper function to select a deblended clip pattern from a fieldmatched clip.

    :param deblended:       Deblended clip.
    :param fieldmatched:    Source after field matching, must have field=3 and possibly low cthresh.
    :param length:          Length of the pattern.

    :return: Deblended clip.
    """
    inters = telecine_patterns(fieldmatched, deblended, length)
    inters += [shift_clip(inter, 1) for inter in inters]

    inters.insert(0, fieldmatched)

    prop_srcs = shift_clip_multi(fieldmatched, (0, 1))

    if complexpr_available:
        index_src = expr_func(
            prop_srcs, f'x._Combed N {length} % 1 + y._Combed {length} 0 ? + 0 ?', vs.GRAY8
        )

        return fieldmatched.std.FrameEval(lambda n, f: inters[f[0][0, 0]], index_src)  # type: ignore

    def _deblend_eval(n: int, f: list[vs.VideoFrame]) -> vs.VideoNode:
        idx = 0

        if f[0].props._Combed == 1:
            idx += (n % length) + 1

            if f[1].props._Combed == 1:
                idx += length

        return inters[idx]

    return fieldmatched.std.FrameEval(_deblend_eval, prop_srcs)


def deblend(
    src: vs.VideoNode, fieldmatched: vs.VideoNode | None = None, decomber: VSFunction | None = vinverse, **kwargs: Any
) -> vs.VideoNode:
    """
    Automatically deblends if normal field matching leaves 2 blends every 5 frames. Adopted from jvsfunc.

    :param src:             Input source to fieldmatching.
    :param fieldmatched:    Source after field matching, must have field=3 and possibly low cthresh.
    :param decomber:        Optional post processing decomber after deblending and before pattern matching.

    :return: Deblended clip.
    """

    deblended = norm_expr(shift_clip_multi(src, (-1, 2)), 'z a 2 / - y x 2 / - +')

    if decomber:
        deblended = decomber(deblended, **kwargs)

    if fieldmatched:
        deblended = deblending_helper(fieldmatched, deblended)

    return join(fieldmatched or src, deblended)


def deblend_bob(
    bobbed: vs.VideoNode | tuple[vs.VideoNode, vs.VideoNode],
    fieldmatched: vs.VideoNode | None = None, blend_out: bool = False
) -> vs.VideoNode:
    """
    Stronger version of `deblend` that uses a bobbed clip to deblend. Adopted from jvsfunc.

    :param bobbed:          Bobbed source or a tuple of even/odd fields.
    :param fieldmatched:    Source after field matching, must have field=3 and possibly low cthresh.

    :return: Deblended clip.
    """

    if isinstance(bobbed, tuple):
        bob0, bob1 = bobbed
    else:
        bob0, bob1 = bobbed.std.SelectEvery(2, 0), bobbed.std.SelectEvery(2, 1)

    ab0, bc0, c0 = shift_clip_multi(bob0, (0, 2))
    bc1, ab1, a1 = shift_clip_multi(bob1)

    deblended = norm_expr([a1, ab1, ab0, bc1, bc0, c0], ('b', 'y x - z + b c - a + + 2 /'))

    if fieldmatched:
        return deblending_helper(fieldmatched, deblended)

    return deblended


def deblend_fix_kf(deblended: vs.VideoNode, fieldmatched: vs.VideoNode) -> vs.VideoNode:
    """
    Should be used after deblend/_bob to fix scene changes. Adopted from jvsfunc.

    :param deblended:       Deblended clip.
    :param fieldmatched:    Fieldmatched clip used to debled, must have field=3 and possibly low cthresh.

    :return: Deblended clip with fixed blended keyframes.
    """

    shifted_clips = shift_clip_multi(deblended)
    prop_srcs = shift_clip_multi(fieldmatched, (0, 1))

    if complexpr_available:
        index_src = expr_func(
            prop_srcs, 'x._Combed x.VFMSceneChange and y.VFMSceneChange 2 0 ? 1 ?', vs.GRAY8
        )

        return deblended.std.FrameEval(lambda n, f: shifted_clips[f[0][0, 0]], index_src)  # type: ignore

    def _keyframe_fix(n: int, f: list[vs.VideoFrame]) -> vs.VideoNode:
        keyfm = cast(tuple[int, int], (f[0].props.VFMSceneChange, f[1].props.VFMSceneChange))

        idx = 1
        if f[0].props._Combed == 1:
            if keyfm == (1, 0):
                idx = 0
            elif keyfm == (1, 1):
                idx = 2

        return shifted_clips[idx]

    return deblended.std.FrameEval(_keyframe_fix, prop_srcs)
