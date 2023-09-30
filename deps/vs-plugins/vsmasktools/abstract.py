from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from vsexprtools import ExprOp, norm_expr
from vsrgtools import box_blur
from vstools import (
    ColorRange, FrameRangeN, FrameRangesN, FramesLengthError, Position, Size, check_variable, depth, get_peak_value,
    normalize_seq, replace_ranges, vs
)

__all__ = [
    'GeneralMask',
    'BoundingBox',
    'DeferredMask'
]


class GeneralMask(ABC):
    @abstractmethod
    def get_mask(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        ...


class BoundingBox(GeneralMask):
    pos: Position
    size: Size
    invert: bool

    def __init__(self, pos: tuple[int, int] | Position, size: tuple[int, int] | Size, invert: bool = False) -> None:
        self.pos, self.size, self.invert = Position(pos), Size(size), invert

    def get_mask(self, ref: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:  # type: ignore[override]
        from .utils import squaremask
        return squaremask(ref, self.size.x, self.size.y, self.pos.x, self.pos.y, self.invert, self.get_mask)


class DeferredMask(GeneralMask):
    ranges: FrameRangesN
    bound: BoundingBox | None
    refframes: list[int | None]
    blur: bool

    def __init__(
        self, ranges: FrameRangeN | FrameRangesN | None = None, bound: BoundingBox | None = None,
        *, blur: bool = False, refframes: int | list[int | None] | None = None
    ) -> None:
        self.ranges = ranges if isinstance(ranges, Sequence) else [(0, None)] if ranges is None else [ranges]
        self.blur = blur
        self.bound = bound

        if refframes is None:
            self.refframes = []
        else:
            self.refframes = refframes if isinstance(refframes, list) else normalize_seq(refframes, len(self.ranges))

        if len(self.refframes) > 0 and len(self.refframes) != len(self.ranges):
            raise FramesLengthError(
                self.__class__, '', 'Received reference frame and range list size mismatch!'
            )

    def get_mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:  # type: ignore[override]
        assert check_variable(clip, self.get_mask)
        assert check_variable(ref, self.get_mask)

        if self.bound:
            bm = self.bound.get_mask(ref, **kwargs)

            if self.blur:
                bm = box_blur(bm, 5, 5)

        if self.refframes:
            hm = ref.std.BlankClip(
                format=ref.format.replace(color_family=vs.GRAY, subsampling_h=0, subsampling_w=0).id, keep=True
            )

            for ran, rf in zip(self.ranges, self.refframes):
                if rf is None:
                    rf = ref.num_frames - 1
                elif rf < 0:
                    rf = ref.num_frames - 1 + rf

                mask = depth(
                    self._mask(clip[rf], ref[rf], **kwargs), clip,
                    range_out=ColorRange.FULL, range_in=ColorRange.FULL
                ).std.Loop(hm.num_frames)

                hm = replace_ranges(hm, ExprOp.MAX.combine(hm, mask), ran)
        else:
            hm = depth(
                self._mask(clip, ref, **kwargs), clip,
                range_out=ColorRange.FULL, range_in=ColorRange.FULL
            )

        if self.bound:
            return norm_expr([hm, bm], f'y {ExprOp.clamp(0, get_peak_value(hm), "x")} 0 ?')

        return hm.std.Limiter()

    @abstractmethod
    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        ...
