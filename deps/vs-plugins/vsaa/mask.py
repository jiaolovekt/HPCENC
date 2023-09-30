from __future__ import annotations

from math import ceil

from vskernels import Box, Point
from vstools import fallback, vs

__all__ = [
    'resize_aa_mask'
]


def resize_aa_mask(mclip: vs.VideoNode, width: int | None = None, height: int | None = None) -> vs.VideoNode:
    iw, ih = mclip.width, mclip.height
    ow, oh = fallback(width, iw), fallback(height, ih)

    if (ow > iw and ow / iw != ow // iw) or (oh > ih and oh / ih != oh // ih):
        mclip = Point.scale(mclip, iw * ceil(ow / iw), ih * ceil(oh / ih))

    return Box(fulls=1, fulld=1).scale(mclip, ow, oh)
