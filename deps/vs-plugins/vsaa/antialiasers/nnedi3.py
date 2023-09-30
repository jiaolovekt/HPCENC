from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vstools import core, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, _Antialiaser, _FullInterpolate

__all__ = [
    'Nnedi3', 'Nnedi3DR'
]


@dataclass
class NNEDI3(_FullInterpolate, _Antialiaser):
    nsize: int = 0
    nns: int = 4
    qual: int = 2
    etype: int = 0
    pscrn: int = 1

    opencl: bool | None = None

    def is_full_interpolate_enabled(self, x: bool, y: bool) -> bool:
        return not not (hasattr(core, 'sneedif') if self.opencl is None else self.opencl)

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        assert clip.format
        is_float = clip.format.sample_type == vs.FLOAT
        pscrn = 1 if is_float else self.pscrn
        return dict(nsize=self.nsize, nns=self.nns, qual=self.qual, etype=self.etype, pscrn=pscrn)

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> vs.VideoNode:
        if not (hasattr(core, 'znedi3') or hasattr(core, 'nnedi3')):
            interpolated: vs.VideoNode = self.full_interpolate(
                clip, double_y or not self.drop_fields, False, **kwargs
            )
        else:
            interpolated = getattr(
                core, 'znedi3' if hasattr(core, 'znedi3') else 'nnedi3'
            ).nnedi3(
                clip, self.field, double_y or not self.drop_fields, **kwargs
            )

        return self.shift_interpolate(clip, interpolated, double_y, **kwargs)

    def full_interpolate(self, clip: vs.VideoNode, double_y: bool, double_x: bool, **kwargs: Any) -> vs.VideoNode:
        if hasattr(core, 'sneedif'):
            clip = core.sneedif.NNEDI3(
                clip, self.field, double_y, double_x, transpose_first=self.transpose_first, **kwargs
            )
        else:
            if not self.transpose_first:
                clip = clip.std.Transpose()

            clip = core.nnedi3cl.NNEDI3CL(clip, self.field, double_y, double_x, **kwargs)

            if not self.transpose_first:
                clip = clip.std.Transpose()

        return clip

    _shift = 0.5


class Nnedi3SS(NNEDI3, SuperSampler):
    ...


class Nnedi3SR(NNEDI3, SingleRater):
    ...


class Nnedi3DR(NNEDI3, DoubleRater):
    ...


class Nnedi3(NNEDI3, Antialiaser):
    ...
