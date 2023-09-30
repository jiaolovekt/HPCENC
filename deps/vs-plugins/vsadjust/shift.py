from __future__ import annotations

from typing import Sequence

from vsexprtools import ExprOp
from vstools import CustomIndexError, check_variable, normalize_seq, scale_8bit, vs

__all__ = [
    'shift_tint'
]


def shift_tint(clip: vs.VideoNode, values: int | Sequence[int] = 16) -> vs.VideoNode:
    """
    Forcibly adds pixel values to a clip.

    Can be used to fix green tints in Crunchyroll sources, for example.
    Only use this if you know what you're doing!

    This function accepts a single int8 or a list of int8 values.

    :param clip:            Clip to process.
    :param values:          Value added to every pixel, scales accordingly to your clip's depth (Default: 16).

    :return:                Clip with pixel values added.

    :raises IndexError:     Any value in ``values`` are above 255.
    """

    assert check_variable(clip, "shift_tint")

    val = normalize_seq(values)

    if any(v > 255 or v < -255 for v in val):
        raise CustomIndexError('Every value in "values" must be an 8 bit number!', shift_tint)

    return ExprOp.ADD.combine(clip, suffix=[scale_8bit(clip, v) for v in val])
