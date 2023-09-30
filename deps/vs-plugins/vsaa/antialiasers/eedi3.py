from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Literal

from vstools import CustomValueError, core, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, _Antialiaser
from . import nnedi3

__all__ = [
    'Eedi3', 'Eedi3DR'
]


@dataclass
class EEDI3(_Antialiaser):
    alpha: float = 0.25
    beta: float = 0.5
    gamma: float = 40
    nrad: int = 2
    mdis: int = 20

    ucubic: bool = True
    cost3: bool = True
    vcheck: int = 2
    vthresh0: float = 32.0
    vthresh1: float = 64.0
    vthresh2: float = 4.0

    device: int = -1
    opencl: bool = dc_field(default=False, kw_only=True)

    mclip: vs.VideoNode | None = None
    sclip_aa: type[Antialiaser] | Antialiaser | Literal[True] | None = dc_field(
        default_factory=lambda: nnedi3.Nnedi3(nsize=4, nns=4, qual=2, etype=1)
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        self._sclip_aa: Antialiaser | Literal[True] | None

        if self.sclip_aa and self.sclip_aa is not True and not isinstance(self.sclip_aa, Antialiaser):
            self._sclip_aa = self.sclip_aa()
        else:
            self._sclip_aa = self.sclip_aa  # type: ignore[assignment]

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        args = dict(
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
            nrad=self.nrad, mdis=self.mdis,
            ucubic=self.ucubic, cost3=self.cost3,
            vcheck=self.vcheck,
            vthresh0=self.vthresh0, vthresh1=self.vthresh1, vthresh2=self.vthresh2
        ) | kwargs

        if self.opencl:
            args |= dict(device=self.device)

        return args

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> vs.VideoNode:
        aa_kwargs = self.get_aa_args(clip, **kwargs)

        if self._sclip_aa and ((('sclip' in kwargs) and not kwargs['sclip']) or 'sclip' not in kwargs):
            if self._sclip_aa is True:
                if double_y:
                    raise CustomValueError("You can't pass sclip_aa=True when supersampling!", self.__class__)
                aa_kwargs.update(sclip=clip)
            else:
                sclip_args = self._sclip_aa.get_aa_args(clip, **(dict(mclip=kwargs.get('mclip', None))))

                if double_y:
                    sclip_args |= self._sclip_aa.get_ss_args(clip)
                else:
                    sclip_args |= self._sclip_aa.get_sr_args(clip)

                aa_kwargs.update(
                    sclip=self._sclip_aa.interpolate(clip, double_y or not self.drop_fields, **sclip_args)
                )

        function = core.eedi3m.EEDI3CL if self.opencl else core.eedi3m.EEDI3

        interpolated = function(  # type: ignore[operator]
            clip, self.field, double_y or not self.drop_fields, **aa_kwargs
        )

        return self.shift_interpolate(clip, interpolated, double_y, **kwargs)

    _shift = 0.5


class Eedi3SS(EEDI3, SuperSampler):
    ...


class Eedi3SR(EEDI3, SingleRater):
    _mclips: tuple[vs.VideoNode, vs.VideoNode] | None = None

    def get_sr_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        if not self.mclip:
            return {}

        if not self._mclips:
            self._mclips = (self.mclip, self.mclip.std.Transpose())

        if self.mclip.width == clip.width and self.mclip.height == clip.height:
            return dict(mclip=self._mclips[0])
        else:
            return dict(mclip=self._mclips[1])

    def __vs_del__(self, core_id: int) -> None:
        self._mclips = None


class Eedi3DR(EEDI3, DoubleRater):
    ...


class Eedi3(Eedi3SR, Antialiaser):
    ...
