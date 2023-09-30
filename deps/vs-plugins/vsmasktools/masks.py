from __future__ import annotations

from vsexprtools import ExprOp
from vstools import check_variable, join, normalize_seq, split, vs

from .morpho import Morpho
from .types import Coordinates

__all__ = [
    'range_mask'
]


def range_mask(clip: vs.VideoNode, rad: int = 2, radc: int = 0) -> vs.VideoNode:
    assert check_variable(clip, range_mask)

    def _minmax(clip: vs.VideoNode, iters: int, maxx: bool) -> vs.VideoNode:
        func = Morpho.maximum if maxx else Morpho.minimum

        for i in range(1, iters + 1):
            clip = func(clip, coordinates=Coordinates.from_iter(i))

        return clip

    return join([
        ExprOp.SUB.combine(
            _minmax(plane, r, True),
            _minmax(plane, r, False)
        ) for plane, r in zip(
            split(clip), normalize_seq(radc and [rad, radc] or rad, clip.format.num_planes)
        )
    ]).std.Limiter()
