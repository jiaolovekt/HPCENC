from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from vstools import core, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, _Antialiaser, _FullInterpolate

__all__ = [
    'Eedi2', 'Eedi2DR'
]


@dataclass
class EEDI2(_FullInterpolate, _Antialiaser):
    mthresh: int = 10
    lthresh: int = 20
    vthresh: int = 20
    estr: int = 2
    dstr: int = 4
    maxd: int = 24
    pp: int = 1

    cuda: bool = dc_field(default=False, kw_only=True)

    def is_full_interpolate_enabled(self, x: bool, y: bool) -> bool:
        return self.cuda and x and y

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return dict(
            mthresh=self.mthresh, lthresh=self.lthresh, vthresh=self.vthresh,
            estr=self.estr, dstr=self.dstr, maxd=self.maxd, pp=self.pp
        )

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> vs.VideoNode:
        inter: vs.VideoNode

        if self.cuda:
            inter = core.eedi2cuda.EEDI2(clip, self.field, **kwargs)  # type: ignore
        else:
            inter = core.eedi2.EEDI2(clip, self.field, **kwargs)  # type: ignore

        if not double_y:
            if self.drop_fields:
                inter = inter.std.SeparateFields(not self.field)[::2]

                inter = self._shifter.shift(inter, (0.5 - 0.75 * self.field, 0))
            else:
                shift = (self._shift * int(not self.field), 0)

                if self._scaler:
                    inter = self._scaler.scale(inter, clip.width, clip.height, shift)
                else:
                    inter = self._shifter.scale(inter, clip.width, clip.height, shift)

        return self._post_interpolate(clip, inter, double_y, **kwargs)

    def full_interpolate(self, clip: vs.VideoNode, double_y: bool, double_x: bool, **kwargs: Any) -> vs.VideoNode:
        return core.eedi2cuda.Enlarge2(clip, **kwargs)  # type: ignore

    _shift = -0.5


class Eedi2SS(EEDI2, SuperSampler):
    ...


class Eedi2SR(EEDI2, SingleRater):
    ...


class Eedi2DR(EEDI2, DoubleRater):
    ...


class Eedi2(EEDI2, Antialiaser):
    ...
