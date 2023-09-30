from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, IntFlag, auto
from typing import Any, ClassVar, NoReturn, Sequence, TypeAlias

from vsexprtools import ExprOp, ExprToken, norm_expr
from vstools import (
    ColorRange, CustomRuntimeError, CustomValueError, FuncExceptT, KwargsT, PlanesT, T, check_variable, core,
    get_lowest_values, get_peak_value, get_peak_values, get_subclasses, inject_self, join, normalize_planes, plane,
    scale_value, vs
)

from ..exceptions import UnknownEdgeDetectError, UnknownRidgeDetectError

__all__ = [
    'EdgeDetect', 'EdgeDetectT',
    'RidgeDetect', 'RidgeDetectT',

    'MatrixEdgeDetect', 'SingleMatrix', 'EuclideanDistance', 'MagnitudeMatrix',

    'Max',

    'MagDirection',

    'get_all_edge_detects',
    'get_all_ridge_detect',
]


class _Feature(Enum):
    EDGE = auto()
    RIDGE = auto()


class MagDirection(IntFlag):
    """Direction of magnitude calculation."""

    N = auto()
    NW = auto()
    W = auto()
    SW = auto()
    S = auto()
    SE = auto()
    E = auto()
    NE = auto()

    ALL = N | NW | W | SW | S | SE | E | NE

    AXIS = N | W | S | E
    CORNERS = NW | SW | SE | NE

    NORTH = N | NW | NE
    WEAST = W | NW | SW
    EAST = E | NE | SE
    SOUTH = S | SW | SE

    def select_matrices(self, matrices: Sequence[T]) -> Sequence[T]:
        # In Python <3.11, composite flags are included in MagDirection's
        # collection and iteration interfaces.
        primary_flags = [
            flag for flag in MagDirection if flag != 0 and flag & (flag - 1) == 0
        ]
        assert len(matrices) == len(primary_flags) and self

        return [
            matrix for flag, matrix in zip(primary_flags, matrices) if self & flag
        ]


class BaseDetect:
    @staticmethod
    def from_param(
        cls: type[T],
        value: str | type[T] | T | None,
        exception_cls: type[CustomValueError],
        excluded: Sequence[type[T]] = [],
        func_except: FuncExceptT | None = None
    ) -> type[T]:
        if isinstance(value, str):
            all_edge_detects = get_subclasses(EdgeDetect, excluded)
            search_str = value.lower().strip()

            for edge_detect_cls in all_edge_detects:
                if edge_detect_cls.__name__.lower() == search_str:
                    return edge_detect_cls  # type: ignore

            raise exception_cls(func_except or cls.from_param, value)  # type: ignore

        if issubclass(value, cls):  # type: ignore
            return value  # type: ignore

        if isinstance(value, cls):
            return value.__class__

        return cls

    @staticmethod
    def ensure_obj(
        cls: type[T],
        value: str | type[T] | T | None,
        exception_cls: type[CustomValueError],
        excluded: Sequence[type[T]] = [],
        func_except: FuncExceptT | None = None
    ) -> T:
        new_edge_detect: T | None = None

        if not isinstance(value, cls):
            try:
                new_edge_detect = cls.from_param(value, func_except)()  # type: ignore
            except Exception:
                ...
        else:
            new_edge_detect = value

        if new_edge_detect is None:
            new_edge_detect = cls()

        if new_edge_detect.__class__ in excluded:
            raise exception_cls(
                func_except or cls.ensure_obj, new_edge_detect.__class__,  # type: ignore
                'This {cls_name} can\'t be instantiated to be used!',
                cls_name=new_edge_detect.__class__
            )

        return new_edge_detect


class EdgeDetect(ABC):
    """Abstract edge detection interface."""

    kwargs: KwargsT | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        self.kwargs = kwargs

    @classmethod
    def from_param(
        cls: type[EdgeDetect], edge_detect: EdgeDetectT | None = None, func_except: FuncExceptT | None = None
    ) -> type[EdgeDetect]:
        return BaseDetect.from_param(cls, edge_detect, UnknownEdgeDetectError, [], func_except)

    @classmethod
    def ensure_obj(
        cls: type[EdgeDetect], edge_detect: EdgeDetectT | None = None, func_except: FuncExceptT | None = None
    ) -> EdgeDetect:
        return BaseDetect.ensure_obj(cls, edge_detect, UnknownEdgeDetectError, [], func_except)

    @inject_self
    def edgemask(
        self, clip: vs.VideoNode, lthr: float = 0.0, hthr: float | None = None, multi: float = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        planes: PlanesT | tuple[PlanesT, bool] = None, **kwargs: Any
    ) -> vs.VideoNode:
        """
        Makes edge mask based on convolution kernel.
        The resulting mask can be thresholded with lthr, hthr and multiplied with multi.

        :param clip:            Source clip
        :param lthr:            Low threshold. Anything below lthr will be set to 0
        :param hthr:            High threshold. Anything above hthr will be set to the range max
        :param multi:           Multiply all pixels by this before thresholding
        :param clamp:           Clamp to TV or full range if True or specified range `(low, high)`

        :return:                Mask clip
        """
        return self._mask(clip, lthr, hthr, multi, clamp, _Feature.EDGE, planes, **kwargs)

    def _mask(
        self,
        clip: vs.VideoNode,
        lthr: float = 0.0, hthr: float | None = None,
        multi: float = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        feature: _Feature = _Feature.EDGE, planes: PlanesT | tuple[PlanesT, bool] = None, **kwargs: Any
    ) -> vs.VideoNode:
        assert check_variable(clip, self.__class__)

        if self.kwargs:
            kwargs = self.kwargs | kwargs

        peak = get_peak_value(clip)
        hthr = 1.0 if hthr is None else hthr

        lthr, hthr = scale_value(lthr, 32, clip), scale_value(hthr, 32, clip)

        discard_planes = False
        if isinstance(planes, tuple):
            planes, discard_planes = planes

        planes = normalize_planes(clip, planes)  # type: ignore

        wclip = plane(clip, planes[0]) if len(planes) == 1 else clip

        clip_p = self._preprocess(wclip)

        try:
            if feature == _Feature.EDGE:
                mask = self._compute_edge_mask(clip_p, **kwargs)
            elif feature == _Feature.RIDGE:
                if not isinstance(self, RidgeDetect):
                    raise RuntimeError
                mask = self._compute_ridge_mask(clip_p, **kwargs)
        except Exception as e:
            raise CustomRuntimeError('There was an error processing the mask! Are you using an abstract class?') from e

        mask = self._postprocess(mask, clip.format.bits_per_sample)

        if multi != 1:
            mask = ExprOp.MUL(mask, suffix=str(multi), planes=planes)

        if lthr == hthr:
            mask = norm_expr(mask, f'x {hthr} >= {ExprToken.RangeMax} 0 ?', planes)
        elif lthr > 0 and hthr < peak:
            mask = norm_expr(mask, f'x {hthr} > {ExprToken.RangeMax} x {lthr} < 0 x ? ?', planes)
        elif lthr > 0:
            mask = norm_expr(mask, f'x {lthr} < 0 x ?', planes)
        elif hthr < peak:
            mask = norm_expr(mask, f'x {hthr} > {ExprToken.RangeMax} x ?', planes)

        if clamp:
            if clamp is True:
                crange = ColorRange.from_video(clip)
                clamp = list(zip(get_lowest_values(mask, crange), get_peak_values(mask, crange)))

            if isinstance(clamp, list):
                mask = norm_expr(mask, [ExprOp.clamp(*c, c='x') for c in clamp], planes)
            elif isinstance(clamp, tuple):
                mask = ExprOp.clamp(*clamp, c='x')(mask, planes=planes)

        assert mask.format

        if mask.format.num_planes != clip.format.num_planes and not discard_planes:
            return join({None: clip.std.BlankClip(color=[0] * clip.format.num_planes, keep=True), planes[0]: mask})

        return mask

    @abstractmethod
    def _compute_edge_mask(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        raise NotImplementedError

    def _preprocess(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip

    def _postprocess(self, clip: vs.VideoNode, input_bits: int) -> vs.VideoNode:
        return clip


class MatrixEdgeDetect(EdgeDetect):
    matrices: ClassVar[Sequence[Sequence[float]]]
    divisors: ClassVar[Sequence[float] | None] = None
    mode_types: ClassVar[Sequence[str] | None] = None

    def _compute_edge_mask(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        return self._merge_edge([
            clip.std.Convolution(matrix=mat, divisor=div, saturate=False, mode=mode)
            for mat, div, mode in zip(self._get_matrices(), self._get_divisors(), self._get_mode_types())
        ])

    def _compute_ridge_mask(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        def _x(c: vs.VideoNode) -> vs.VideoNode:
            return c.std.Convolution(matrix=self._get_matrices()[0], divisor=self._get_divisors()[0])

        def _y(c: vs.VideoNode) -> vs.VideoNode:
            return c.std.Convolution(matrix=self._get_matrices()[1], divisor=self._get_divisors()[1])

        x = _x(clip)
        y = _y(clip)
        xx = _x(x)
        yy = _y(y)
        xy = _x(x)
        return self._merge_ridge([xx, yy, xy])

    @abstractmethod
    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        raise NotImplementedError

    @abstractmethod
    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError

    def _get_matrices(self) -> Sequence[Sequence[float]]:
        return self.matrices

    def _get_divisors(self) -> Sequence[float]:
        return self.divisors if self.divisors else [0.0] * len(self._get_matrices())

    def _get_mode_types(self) -> Sequence[str]:
        return self.mode_types if self.mode_types else ['s'] * len(self._get_matrices())

    def _postprocess(self, clip: vs.VideoNode, input_bits: int) -> vs.VideoNode:
        if len(self.matrices[0]) > 9 or (self.mode_types and self.mode_types[0] != 's'):
            clip = clip.std.Crop(
                right=clip.format.subsampling_w * 2 if clip.format and clip.format.subsampling_w != 0 else 2
            ).resize.Point(clip.width, src_width=clip.width)
        return clip


class MagnitudeMatrix(MatrixEdgeDetect):
    def __init__(self, mag_directions: MagDirection = MagDirection.ALL, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.mag_directions = mag_directions

    def _get_matrices(self) -> Sequence[Sequence[float]]:
        return self.mag_directions.select_matrices(self.matrices)


class RidgeDetect(MatrixEdgeDetect):
    @classmethod
    def from_param(  # type: ignore
        cls: type[RidgeDetect], edge_detect: RidgeDetectT | None = None, func_except: FuncExceptT | None = None
    ) -> type[RidgeDetect]:
        return BaseDetect.from_param(cls, edge_detect, UnknownRidgeDetectError, [], func_except)

    @classmethod
    def ensure_obj(  # type: ignore
        cls: type[RidgeDetect], edge_detect: RidgeDetectT | None = None, func_except: FuncExceptT | None = None
    ) -> RidgeDetect:
        return BaseDetect.ensure_obj(cls, edge_detect, UnknownRidgeDetectError, [], func_except)

    @inject_self
    def ridgemask(
        self, clip: vs.VideoNode, lthr: float = 0.0, hthr: float | None = None, multi: float = 1.0,
        clamp: bool | tuple[float, float] | list[tuple[float, float]] = False,
        planes: PlanesT | tuple[PlanesT, bool] = None, **kwargs: Any
    ) -> vs.VideoNode | NoReturn:
        """
        Makes ridge mask based on convolution kernel.
        The resulting mask can be thresholded with lthr, hthr and multiplied with multi.

        :param clip:            Source clip
        :param lthr:            Low threshold. Anything below lthr will be set to 0
        :param hthr:            High threshold. Anything above hthr will be set to the range max
        :param multi:           Multiply all pixels by this before thresholding
        :param clamp:           Clamp to TV or full range if True or specified range `(low, high)`

        :return:                Mask clip
        """
        return self._mask(clip, lthr, hthr, multi, clamp, _Feature.RIDGE, planes, **kwargs)

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return core.std.Expr(clips, 'x y * 2 * -1 * x dup * z dup * 4 * + y dup * + + sqrt x y + +')


class SingleMatrix(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return clips[0]

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError


class EuclideanDistance(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return core.std.Expr(clips, 'x x * y y * + sqrt')

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError


class Max(MatrixEdgeDetect, ABC):
    def _merge_edge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        return ExprOp.MAX.combine(*clips)

    def _merge_ridge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode | NoReturn:
        raise NotImplementedError


EdgeDetectT: TypeAlias = type[EdgeDetect] | EdgeDetect | str  # type: ignore
RidgeDetectT: TypeAlias = type[RidgeDetect] | RidgeDetect | str  # type: ignore


def get_all_edge_detects(
    clip: vs.VideoNode,
    lthr: float = 0.0, hthr: float | None = None,
    multi: float = 1.0,
    clamp: bool | tuple[float, float] | list[tuple[float, float]] = False
) -> list[vs.VideoNode]:
    """
    Returns all the EdgeDetect subclasses

    :param clip:        Source clip
    :param lthr:        See :py:func:`EdgeDetect.get_mask`
    :param hthr:        See :py:func:`EdgeDetect.get_mask`
    :param multi:       See :py:func:`EdgeDetect.get_mask`
    :param clamp:       Clamp to TV or full range if True or specified range `(low, high)`

    :return:            A list edge masks
    """
    def _all_subclasses(cls: type[EdgeDetect] = EdgeDetect) -> set[type[EdgeDetect]]:
        return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in _all_subclasses(c))

    all_subclasses = {
        s for s in _all_subclasses()
        if s.__name__ not in {
            'MatrixEdgeDetect', 'RidgeDetect', 'SingleMatrix', 'EuclideanDistance', 'Max',
            'Matrix1D', 'SavitzkyGolay', 'SavitzkyGolayNormalise',
            'Matrix2x2', 'Matrix3x3', 'Matrix5x5'
        }
    }
    return [
        edge_detect().edgemask(clip, lthr, hthr, multi, clamp).text.Text(edge_detect.__name__)
        for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__)
    ]


def get_all_ridge_detect(
    clip: vs.VideoNode, lthr: float = 0.0, hthr: float | None = None, multi: float = 1.0,
    clamp: bool | tuple[float, float] | list[tuple[float, float]] = False
) -> list[vs.VideoNode]:
    """
    Returns all the RidgeDetect subclasses

    :param clip:        Source clip
    :param lthr:        See :py:func:`EdgeDetect.get_mask`
    :param hthr:        See :py:func:`EdgeDetect.get_mask`
    :param multi:       See :py:func:`EdgeDetect.get_mask`
    :param clamp:       Clamp to TV or full range if True or specified range `(low, high)`

    :return:            A list edge masks
    """
    def _all_subclasses(cls: type[RidgeDetect] = RidgeDetect) -> set[type[RidgeDetect]]:
        return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in _all_subclasses(c))

    all_subclasses = {
        s for s in _all_subclasses()
        if s.__name__ not in {
            'MatrixEdgeDetect', 'RidgeDetect', 'SingleMatrix', 'EuclideanDistance', 'Max',
            'Matrix1D', 'SavitzkyGolay', 'SavitzkyGolayNormalise',
            'Matrix2x2', 'Matrix3x3', 'Matrix5x5'
        }
    }
    return [
        edge_detect().ridgemask(clip, lthr, hthr, multi, clamp).text.Text(edge_detect.__name__)
        for edge_detect in sorted(all_subclasses, key=lambda x: x.__name__)
    ]
