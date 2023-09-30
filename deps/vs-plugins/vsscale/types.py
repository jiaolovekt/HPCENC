from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from vsexprtools import expr_func
from vskernels import Kernel
from vstools import (
    ComparatorFunc, CustomIntEnum, CustomNotImplementedError, CustomStrEnum, Resolution, VSMapValue, merge_clip_props,
    vs
)

__all__ = [
    'Resolution', 'DescaleAttempt',
    'DescaleMode', 'DescaleResult', 'PlaneStatsKind',
    'DescaleModeWithInfo'
]


class DescaleAttempt(NamedTuple):
    """Tuple representing a descale attempt."""

    resolution: Resolution
    """The native resolution."""

    descaled: vs.VideoNode
    """Descaled frame in native resolution."""

    rescaled: vs.VideoNode
    """Descaled frame reupscaled with the same kernel."""

    diff: vs.VideoNode
    """The subtractive difference between the original and descaled frame."""

    kernel: Kernel
    """Kernel used for descaling."""

    @classmethod
    def from_args(
        cls, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float],
        kernel: Kernel, mode: DescaleModeWithInfo, **kwargs: VSMapValue
    ) -> DescaleAttempt:
        """Get a DescaleAttempt from args. Calculate difference nodes too."""

        descaled = kernel.descale(clip, width, height, shift)
        descaled = descaled.std.SetFrameProps(**kwargs)

        rescaled = kernel.scale(descaled, clip.width, clip.height)

        diff = expr_func([rescaled, clip], 'x y - abs').std.PlaneStats(
            None, prop=DescaleMode.PlaneDiff.prop_key
        )

        if mode.mode in {DescaleMode.KernelDiff, DescaleMode.KernelDiffMin, DescaleMode.KernelDiffMax}:
            diff_props = rescaled.std.PlaneStats(clip, prop=DescaleMode.KernelDiff.prop_key)

            diff = merge_clip_props(diff, diff_props)

        resolution = Resolution(width, height)

        return DescaleAttempt(resolution, descaled, rescaled, diff, kernel)

    def __hash__(self) -> int:
        return hash(f'{self.resolution}_{self.kernel.__class__.__name__}')


@dataclass
class DescaleResult:
    """Dataclass representing a complete result of vsscale.descale."""

    descaled: vs.VideoNode
    """The descaled clip. Can be a variable resolution."""

    rescaled: vs.VideoNode
    """
    The descaled clip reupscaled to the source resolution using the same kernel used to descale.
    Can be a variable resolution clip.
    """

    upscaled: vs.VideoNode | None
    """The descaled clip reupscaled using the given upscaler."""

    error_mask: vs.VideoNode | None
    """
    The descale error mask. This catches the big differences
    between the source clip and the rescaled clip as a mask.
    If no \"mask\" is passed, this attribute will be None.
    """

    pproc_mask: vs.VideoNode | None
    """
    The post-processing mask. This is the second mask passed to \"mask\".
    If no \"mask\" is passed, this attribute will be None.
    """

    attempts: list[DescaleAttempt]
    """
    Descale attempts made. These are used to determine
    the correct kernel if multiple \"Kernels\" were passed.
    """

    out: vs.VideoNode
    """The final clip that is returned during regular usage with \"result=False\"."""


class PlaneStatsKind(CustomStrEnum):
    """Type of PlaneStats comparing to use."""

    AVG = 'Average'
    MIN = 'Min'
    MAX = 'Max'
    DIFF = 'Diff'


class DescaleMode(CustomIntEnum):
    """Descale modes for vsscale.descale."""

    PlaneDiff = 0
    """Simple PlaneStatsDiff between original and descaled."""

    PlaneDiffMax = 1
    """Get the video with the maximum absolute difference from original."""

    PlaneDiffMin = 2
    """Get the video with the minimum absolute difference from original."""

    KernelDiff = 3
    """Simple PlaneStats between original and descaled kernels differences."""

    KernelDiffMax = 4
    """Get the video descaled with the kernel with the maximum absolute difference from original."""

    KernelDiffMin = 5
    """Get the video descaled with the kernel with the minimum absolute difference from original."""

    def __call__(self, thr: float = 5e-8, op: ComparatorFunc | None = None) -> DescaleModeWithInfo:
        return DescaleModeWithInfo(self, thr) if op is None else DescaleModeWithInfo(self, thr, op)

    @property
    def prop_key(self) -> str:
        """Get the props key for this DescaleMode."""

        if self.is_average:
            return 'PlaneStatsPAvg'
        elif self.is_kernel_diff:
            return 'PlaneStatsKDiff'

        raise CustomNotImplementedError

    @property
    def res_op(self) -> ComparatorFunc:
        """Get the operator for calculating sort operation between two resolutions."""

        if self in {self.PlaneDiff, self.KernelDiff, self.PlaneDiffMax, self.KernelDiffMax}:
            return max

        if self in {self.PlaneDiffMin, self.KernelDiffMin}:
            return min

        raise CustomNotImplementedError

    @property
    def diff_op(self) -> ComparatorFunc:
        """Get the operator for calculating sort operation between two props."""

        if self in {self.PlaneDiff, self.KernelDiff, self.PlaneDiffMin, self.KernelDiffMin}:
            return min

        if self in {self.KernelDiffMax, self.PlaneDiffMax}:
            return max

        raise CustomNotImplementedError

    @property
    def is_average(self) -> bool:
        """Whether this DescaleMode is of PlaneDiff kind."""

        return self in {self.PlaneDiff, self.PlaneDiffMin, self.PlaneDiffMax}

    @property
    def is_kernel_diff(self) -> bool:
        """Whether this DescaleMode is of KernelDiff kind."""

        return self in {self.KernelDiff, self.KernelDiffMin, self.KernelDiffMax}

    def prop_value(self, kind: PlaneStatsKind) -> str:
        """Get props key for getting the value of the PlaneStatsKind."""

        return f'{self.prop_key}{kind.value}'

    def __hash__(self) -> int:
        return hash(self._name_)


@dataclass
class DescaleModeWithInfo:
    mode: DescaleMode
    """Actual descale mode used for descaling."""

    thr: float = field(default=5e-8)
    """Diff threshold."""

    op: ComparatorFunc = field(default_factory=lambda: max)
    """Operator used for generic sorting."""
