from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Concatenate, Literal, cast

from vsaa import Nnedi3
from vsexprtools import ExprOp, combine, norm_expr
from vskernels import Scaler, ScalerT
from vsmasktools import ringing_mask
from vsrgtools import LimitFilterMode, RepairMode, limit_filter, median_clips, repair, unsharp_masked
from vstools import (
    EXPR_VARS, ColorRange, CustomIndexError, CustomOverflowError, P, check_ref_clip, inject_self, scale_8bit,
    scale_value, vs
)

from .helpers import GenericScaler
from .shaders import FSRCNNXShader, FSRCNNXShaderT

__all__ = [
    'MergeScalers',
    'ClampScaler', 'MergedFSRCNNX',
    'UnsharpLimitScaler', 'UnsharpedFSRCNNX'
]


class MergeScalers(GenericScaler):
    def __init__(self, *scalers: ScalerT | tuple[ScalerT, float | None]) -> None:
        """Create a unified Scaler from multiple Scalers with optional weights."""
        if (slen := len(scalers)) < 2:
            raise CustomIndexError('Not enough scalers passed!', self.__class__, slen)
        elif slen > len(EXPR_VARS):
            raise CustomIndexError('Too many scalers passed!', self.__class__, slen)

        if any(isinstance(s, tuple) for s in scalers):
            norm_scalers = [
                scaler if isinstance(scaler, tuple) else (scaler, None) for scaler in scalers
            ]

            curr_sum = 0.0
            n_auto_weight = 0

            for i, (_, weight) in enumerate(norm_scalers):
                if weight is None:
                    n_auto_weight += 1
                elif weight <= 0.0:
                    raise CustomOverflowError(
                        f'Weights have to be positive, >= 0.0! (Scaler index: ({i})', self.__class__
                    )
                else:
                    curr_sum += weight

            if curr_sum > 1.0:
                raise CustomOverflowError(
                    'Sum of the weights should be less or equal than 1.0!', self.__class__
                )

            if n_auto_weight:
                a_wgh = (1.0 - curr_sum) / n_auto_weight

                norm_scalers = [
                    (scaler, a_wgh if weight is None else weight)
                    for scaler, weight in norm_scalers
                ]
        else:
            weight = 1.0 / len(scalers)

            norm_scalers = [(scaler, weight) for scaler in scalers]  # type: ignore

        self.scalers = [
            (self.ensure_scaler(scaler), float(weight if weight else 0))
            for scaler, weight in norm_scalers
        ]

    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        scalers, weights = cast(tuple[list[Scaler], list[float]], zip(*self.scalers))

        return combine(
            [scaler.scale(clip, width, height, shift, **kwargs) for scaler in scalers],
            ExprOp.ADD, zip(weights, ExprOp.MUL), expr_suffix=[sum(weights), ExprOp.DIV]
        )


@dataclass
class ClampScaler(GenericScaler):
    """Clamp a reference Scaler."""

    ref_scaler: ScalerT
    """Scaler to clamp."""

    strength: int = 80
    """Strength of clamping."""

    overshoot: float | None = None
    """Overshoot threshold."""

    undershoot: float | None = None
    """Undershoot threshold."""

    limit: RepairMode | bool = True
    """Whether to use under/overshoot limit (True) or a reference repaired clip for limiting."""

    operator: Literal[ExprOp.MAX, ExprOp.MIN] | None = ExprOp.MIN
    """Whether to take the brightest or darkest pixels in the merge."""

    masked: bool = True
    """Whether to mask with a ringing mask or not."""

    reference: ScalerT | vs.VideoNode = Nnedi3
    """Reference Scaler used to clamp ref_scaler"""

    range_out: ColorRange | None = None
    """Range out for clamping the output. If None it will be fetched from the VideoNode."""

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.strength >= 100:
            raise CustomOverflowError('strength can\'t be more or equal to 100!', self.__class__)
        elif self.strength <= 0:
            raise CustomOverflowError('strength can\'t be less or equal to 0!', self.__class__)

        if self.overshoot is None:
            self.overshoot = self.strength / 100
        if self.undershoot is None:
            self.undershoot = self.overshoot

    @inject_self
    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0),
        *, smooth: vs.VideoNode | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        assert (self.undershoot or self.undershoot == 0) and (self.overshoot or self.overshoot == 0)

        ref = self.ensure_scaler(self.ref_scaler).scale(clip, width, height, shift, **kwargs)

        if isinstance(self.reference, vs.VideoNode):
            smooth = self.reference  # type: ignore

            if shift != (0, 0):
                smooth = self._kernel.shift(smooth, shift)  # type: ignore
        else:
            smooth = self.ensure_scaler(self.reference).scale(clip, width, height, shift)  # type: ignore

        assert smooth

        check_ref_clip(ref, smooth)

        range_out = ColorRange.from_video(clip, False) if self.range_out is None else self.range_out

        merge_weight = self.strength / 100

        if self.limit is True:
            expression = [
                'x {merge_weight} * y {ref_weight} * + up!',
                '{overshoot} O!', '{undershoot} U!',
                'up@ z O@ + > z O@ + up@ ? a U@ - < a U@ - up@ z O@ + > z O@ + up@ ? ?'
            ]

            if range_out is ColorRange.LIMITED:
                expression.append(f'{scale_8bit(clip, 16)} {{clamp_max}} clamp')

            merged = norm_expr(
                [ref, smooth, smooth.std.Maximum(), smooth.std.Minimum()],
                expression, merge_weight=merge_weight, ref_weight=1.0 - merge_weight,
                undershoot=scale_value(self.undershoot, 32, clip),
                overshoot=scale_value(self.overshoot, 32, clip),
                clamp_max=[scale_8bit(clip, 235), scale_8bit(clip, 240)]
            )
        else:
            merged = smooth.std.Merge(ref, merge_weight)

            if isinstance(self.limit, RepairMode):
                merged = repair(merged, smooth, self.limit)

        if self.operator is not None:
            merge2 = combine([smooth, ref], self.operator)

            if self.masked:
                merged = merged.std.MaskedMerge(merge2, ringing_mask(smooth))
            else:
                merged = merge2
        elif self.masked:
            merged.std.MaskedMerge(smooth, ringing_mask(smooth))

        return merged


class UnsharpLimitScaler(GenericScaler):
    """Limit a scaler with a masked unsharping."""

    def __init__(
        self, ref_scaler: ScalerT,
        unsharp_func: Callable[
            Concatenate[vs.VideoNode, P], vs.VideoNode
        ] = partial(unsharp_masked, radius=2, strength=65),
        merge_mode: LimitFilterMode | bool = True,
        reference: ScalerT | vs.VideoNode = Nnedi3(0, opencl=None),
        *args: P.args, **kwargs: P.kwargs
    ) -> None:
        """
        :param ref_scaler:      Scaler of which to limit haloing.
        :param unsharp_func:    Unsharpening function used as reference for limiting.
        :param merge_mode:      Whether to limit with LimitFilterMode,
                                use a median filter (True) or just take the darkest pixels (False).
        :param reference:       Reference scaler used to fill in the haloed parts.
        """

        self.unsharp_func = unsharp_func

        self.merge_mode = merge_mode

        self.reference = reference
        self.ref_scaler = self.ensure_scaler(ref_scaler)

        self.args = args
        self.kwargs = kwargs

    @inject_self
    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0),
        *, smooth: vs.VideoNode | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        fsrcnnx = self.ref_scaler.scale(clip, width, height, shift, **kwargs)

        if isinstance(self.reference, vs.VideoNode):
            smooth = self.reference  # type: ignore

            if shift != (0, 0):
                smooth = self._kernel.shift(smooth, shift)  # type: ignore
        else:
            smooth = self.ensure_scaler(self.reference).scale(clip, width, height, shift)  # type: ignore

        assert smooth

        check_ref_clip(fsrcnnx, smooth)

        smooth_sharp = self.unsharp_func(smooth, *self.args, **self.kwargs)

        if isinstance(self.merge_mode, LimitFilterMode):
            return limit_filter(smooth, fsrcnnx, smooth_sharp, self.merge_mode)

        if self.merge_mode:
            return median_clips(smooth, fsrcnnx, smooth_sharp)

        return combine([smooth, fsrcnnx, smooth_sharp], ExprOp.MIN)


@dataclass
class MergedFSRCNNX(ClampScaler):
    """Clamped FSRCNNX Scaler."""

    ref_scaler: FSRCNNXShaderT = field(default_factory=lambda: FSRCNNXShader.x56, kw_only=True)


class UnsharpedFSRCNNX(UnsharpLimitScaler):
    """Clamped FSRCNNX Scaler with an unsharp mask."""

    def __init__(
        self,
        unsharp_func: Callable[
            Concatenate[vs.VideoNode, P], vs.VideoNode
        ] = partial(unsharp_masked, radius=2, strength=65),
        merge_mode: LimitFilterMode | bool = True,
        reference: ScalerT | vs.VideoNode = Nnedi3(0, opencl=None),
        ref_scaler: ScalerT = FSRCNNXShader.x56,
        *args: P.args, **kwargs: P.kwargs
    ) -> None:
        super().__init__(ref_scaler, unsharp_func, merge_mode, reference, *args, **kwargs)
