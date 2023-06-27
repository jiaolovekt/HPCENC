from __future__ import annotations

from enum import auto
from math import ceil, exp, log2, pi, sqrt
from typing import Generic, Sequence, TypeVar

from vstools import ConvMode, CustomEnum, CustomIntEnum, Nb, PlanesT, vs

__all__ = [
    'LimitFilterMode',
    'RemoveGrainMode', 'RemoveGrainModeT',
    'RepairMode', 'RepairModeT',
    'VerticalCleanerMode', 'VerticalCleanerModeT',
    'BlurMatrix'
]


class LimitFilterModeMeta:
    force_expr = True


class LimitFilterMode(LimitFilterModeMeta, CustomIntEnum):
    """Two sources, one filtered"""
    SIMPLE_MIN = auto()
    SIMPLE_MAX = auto()
    """One source, two filtered"""
    SIMPLE2_MIN = auto()
    SIMPLE2_MAX = auto()
    DIFF_MIN = auto()
    DIFF_MAX = auto()
    """One/Two sources, one filtered"""
    CLAMPING = auto()

    @property
    def op(self) -> str:
        return '<' if 'MIN' in self._name_ else '>'

    def __call__(self, force_expr: bool = True) -> LimitFilterMode:
        self.force_expr = force_expr

        return self


class RemoveGrainMode(CustomIntEnum):
    NONE = 0
    MINMAX_AROUND1 = 1
    MINMAX_AROUND2 = 2
    MINMAX_AROUND3 = 3
    MINMAX_MEDIAN = 4
    EDGE_CLIP_STRONG = 5
    EDGE_CLIP_MODERATE = 6
    EDGE_CLIP_MEDIUM = 7
    EDGE_CLIP_LIGHT = 8
    LINE_CLIP_CLOSE = 9
    MIN_SHARP = 10
    SQUARE_BLUR = 11
    BOB_TOP_CLOSE = 13
    BOB_BOTTOM_CLOSE = 14
    BOB_TOP_INTER = 15
    BOB_BOTTOM_INTER = 16
    MINMAX_MEDIAN_OPP = 17
    LINE_CLIP_OPP = 18
    CIRCLE_BLUR = 19
    BOX_BLUR = 20
    OPP_CLIP_AVG = 21
    OPP_CLIP_AVG_FAST = 22
    EDGE_DEHALO = 23
    EDGE_DEHALO2 = 24
    MIN_SHARP2 = 25
    SMART_RGC = 26
    SMART_RGCL = 27
    SMART_RGCL2 = 28

    def __call__(self, clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
        from .rgtools import removegrain
        from .util import norm_rmode_planes
        return removegrain(clip, norm_rmode_planes(clip, self, planes))


RemoveGrainModeT = int | RemoveGrainMode | Sequence[int | RemoveGrainMode]


class RepairMode(CustomIntEnum):
    NONE = 0
    MINMAX_SQUARE1 = 1
    MINMAX_SQUARE2 = 2
    MINMAX_SQUARE3 = 3
    MINMAX_SQUARE4 = 4
    LINE_CLIP_MIN = 5
    LINE_CLIP_LIGHT = 6
    LINE_CLIP_MEDIUM = 7
    LINE_CLIP_STRONG = 8
    LINE_CLIP_CLOSE = 9
    MINMAX_SQUARE_REF_CLOSE = 10
    MINMAX_SQUARE_REF1 = 11
    MINMAX_SQUARE_REF2 = 12
    MINMAX_SQUARE_REF3 = 13
    MINMAX_SQUARE_REF4 = 14
    CLIP_REF_RG5 = 15
    CLIP_REF_RG6 = 16
    CLIP_REF_RG17 = 17
    CLIP_REF_RG18 = 18
    CLIP_REF_RG19 = 19
    CLIP_REF_RG20 = 20
    CLIP_REF_RG21 = 21
    CLIP_REF_RG22 = 22
    CLIP_REF_RG23 = 23
    CLIP_REF_RG24 = 24
    CLIP_REF_RG26 = 26
    CLIP_REF_RG27 = 27
    CLIP_REF_RG28 = 28

    def __call__(self, clip: vs.VideoNode, repairclip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
        from .rgtools import repair
        from .util import norm_rmode_planes
        return repair(clip, repairclip, norm_rmode_planes(clip, self, planes))


RepairModeT = int | RepairMode | Sequence[int | RepairMode]


class VerticalCleanerMode(CustomIntEnum):
    NONE = 0
    MEDIAN = 1
    PRESERVING = 2

    def __call__(self, clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
        from .rgtools import vertical_cleaner
        from .util import norm_rmode_planes
        return vertical_cleaner(clip, norm_rmode_planes(clip, self, planes))


VerticalCleanerModeT = int | VerticalCleanerMode | Sequence[int | VerticalCleanerMode]


class BaseBlurMatrix(Generic[Nb], list[Nb]):
    def __call__(
        self, clip: vs.VideoNode, planes: PlanesT = None, mode: ConvMode = ConvMode.SQUARE,
        bias: float | None = None, divisor: float | None = None, saturate: int | None = None,
        passes: int = 1
    ) -> vs.VideoNode:
        conv = self.tosizes(*range(3, 26, 2))

        for _ in range(passes):
            clip = clip.std.Convolution(conv, bias, divisor, planes, saturate, mode)

        return clip

    @property
    def asint(self) -> BaseBlurMatrix[int]:
        return BaseBlurMatrix[int](map(int, self))

    @property
    def asfloat(self) -> BaseBlurMatrix[float]:
        return BaseBlurMatrix[float](map(float, self))

    def tosize(self: BBMatrixT, size: int) -> BBMatrixT:
        curlen = len(self)

        diff = curlen - size

        if diff == 0:
            return self

        lh = abs(diff) // 2
        rh = abs(diff) - lh

        if diff < 0:
            return self.__class__(self[:lh] + self + self[-rh:])

        return self.__class__(self[lh:-rh])

    def tosizes(self: BBMatrixT, *_sizes: int) -> BBMatrixT:
        sizes = list(sorted(_sizes, reverse=True))

        curlen = len(self)

        for sizeh, sizel in zip(sizes, sizes[1:]):
            if curlen >= sizeh:
                return self.tosize(sizeh)
            elif curlen == sizel:
                return self.tosize(sizel)
            elif curlen > sizel:
                if (sizeh - curlen) < (curlen - sizel):
                    return self.tosize(sizeh)

                return self.tosize(sizel)

        return self.tosize(sizes[-1])


BBMatrixT = TypeVar('BBMatrixT', bound=BaseBlurMatrix)  # type: ignore


class BlurMatrix(BaseBlurMatrix[int], CustomEnum):
    BOX = [1, 1, 0, 1, 1, 0, 0, 0, 0]
    MEAN = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    WMEAN = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    CIRCLE = [1, 1, 1, 1, 0, 1, 1, 1, 1]

    @classmethod
    def gauss(cls, sigma: float) -> BaseBlurMatrix[float]:
        taps = ceil(sigma * 6 + 1)

        if not taps % 2:
            taps += 1

        half_pisqrt = 1.0 / sqrt(2.0 * pi) * sigma
        doub_qsigma = 2 * sigma * sigma

        high, *kernel = [
            half_pisqrt * exp(-(x * x) / doub_qsigma)
            for x in range(taps // 2)
        ]

        kernel = [x * 1023 / high for x in kernel]
        kernel = [*kernel[::-1], 1023, *kernel]

        return BaseBlurMatrix[float](kernel)

    @classmethod
    def gauss_from_radius(cls, radius: int) -> BaseBlurMatrix[float]:
        return cls.gauss((radius + 1.0) / 3)

    @classmethod
    def log(cls, radius: int = 1, strength: float = 100.0) -> BaseBlurMatrix[float]:
        strength = max(1e-6, min(log2(3) * strength / 100, log2(3)))

        weight = 0.5 ** strength / ((1 - 0.5 ** strength) * 0.5)

        matrix = [1.0]

        for _ in range(radius):
            matrix.append(matrix[-1] / weight)

        kernel = [*matrix[::-1], *matrix[1:]]

        return BaseBlurMatrix[float](kernel)
