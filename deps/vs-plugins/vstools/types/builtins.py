from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING, Any, Callable, Iterable, ParamSpec, Protocol, Sequence, SupportsFloat, SupportsIndex, Tuple,
    TypeAlias, TypeVar, Union, overload, runtime_checkable
)

import vapoursynth as vs

__all__ = [
    'T', 'T0', 'T1', 'T2', 'T_contra',

    'F', 'F0', 'F1', 'F2', 'F_VD',

    'P', 'P0', 'P1', 'P2',
    'R', 'R0', 'R1', 'R2', 'R_contra',

    'Nb',

    'PlanesT', 'VideoNodeIterable',

    'FrameRange', 'FrameRangeN', 'FrameRangesN',

    'Self',

    'SingleOrArr', 'SingleOrArrOpt',
    'SingleOrSeq', 'SingleOrSeqOpt',

    'SimpleByteData', 'SimpleByteDataArray',
    'ByteData',

    'KwargsT',

    'SupportsTrunc',

    'SupportsString',

    'SupportsDunderLT', 'SupportsDunderGT',
    'SupportsDunderLE', 'SupportsDunderGE',

    'SupportsFloatOrIndex',

    'SupportsIndexing',
    'SupportsKeysAndGetItem',

    'SupportsAllComparisons',
    'SupportsRichComparison', 'SupportsRichComparisonT',
    'ComparatorFunc'
]

Nb = TypeVar('Nb', float, int)

T = TypeVar('T')
T0 = TypeVar('T0')
T1 = TypeVar('T1')
T2 = TypeVar('T2')

F = TypeVar('F', bound=Callable[..., Any])
F0 = TypeVar('F0', bound=Callable[..., Any])
F1 = TypeVar('F1', bound=Callable[..., Any])
F2 = TypeVar('F2', bound=Callable[..., Any])

P = ParamSpec('P')
P0 = ParamSpec('P0')
P1 = ParamSpec('P1')
P2 = ParamSpec('P2')

R = TypeVar('R')
R0 = TypeVar('R0')
R1 = TypeVar('R1')
R2 = TypeVar('R2')


_KT = TypeVar('_KT')
_VT_co = TypeVar('_VT_co', covariant=True)

F_VD = TypeVar('F_VD', bound=Callable[..., vs.VideoNode])

T_contra = TypeVar('T_contra', contravariant=True)
R_contra = TypeVar('R_contra', contravariant=True)

PlanesT: TypeAlias = int | Sequence[int] | None
FrameRange: TypeAlias = int | Tuple[int, int] | Sequence[int]
FrameRangeN: TypeAlias = int | Tuple[int | None, int | None] | None

if TYPE_CHECKING:
    FrameRangesN: TypeAlias = Sequence[FrameRangeN]
else:
    FrameRangesN: TypeAlias = list[FrameRangeN]


VideoNodeIterable: TypeAlias = vs.VideoNode | Iterable[vs.VideoNode | Iterable[vs.VideoNode]] | Iterable[
    vs.VideoNode | Iterable[vs.VideoNode | Iterable[vs.VideoNode]]
]

Self = TypeVar('Self')

SingleOrArr = Union[T, list[T]]
SingleOrSeq = Union[T, Sequence[T]]
SingleOrArrOpt = Union[SingleOrArr[T], None]
SingleOrSeqOpt = Union[SingleOrSeq[T], None]

SimpleByteData: TypeAlias = str | bytes | bytearray
SimpleByteDataArray = Union[SimpleByteData, Sequence[SimpleByteData]]

ByteData: TypeAlias = SupportsFloat | SupportsIndex | SimpleByteData | memoryview

SupportsFloatOrIndex: TypeAlias = SupportsFloat | SupportsIndex

KwargsT = dict[str, Any]


@runtime_checkable
class SupportsTrunc(Protocol):
    def __trunc__(self) -> int:
        ...


@runtime_checkable
class SupportsString(Protocol):
    @abstractmethod
    def __str__(self) -> str:
        ...


@runtime_checkable
class SupportsDunderLT(Protocol[T_contra]):
    def __lt__(self, __other: T_contra) -> bool:
        ...


@runtime_checkable
class SupportsDunderGT(Protocol[T_contra]):
    def __gt__(self, __other: T_contra) -> bool:
        ...


@runtime_checkable
class SupportsDunderLE(Protocol[T_contra]):
    def __le__(self, __other: T_contra) -> bool:
        ...


@runtime_checkable
class SupportsDunderGE(Protocol[T_contra]):
    def __ge__(self, __other: T_contra) -> bool:
        ...


@runtime_checkable
class SupportsAllComparisons(
    SupportsDunderLT[Any], SupportsDunderGT[Any], SupportsDunderLE[Any], SupportsDunderGE[Any], Protocol
):
    ...


SupportsRichComparison: TypeAlias = SupportsDunderLT[Any] | SupportsDunderGT[Any]
SupportsRichComparisonT = TypeVar('SupportsRichComparisonT', bound=SupportsRichComparison)


class ComparatorFunc(Protocol):
    @overload
    def __call__(
        self, __arg1: SupportsRichComparisonT, __arg2: SupportsRichComparisonT,
        *_args: SupportsRichComparisonT, key: None = ...
    ) -> SupportsRichComparisonT:
        ...

    @overload
    def __call__(self, __arg1: T0, __arg2: T0, *_args: T0, key: Callable[[T0], SupportsRichComparison]) -> T0:
        ...

    @overload
    def __call__(self, __iterable: Iterable[SupportsRichComparisonT], *, key: None = ...) -> SupportsRichComparisonT:
        ...

    @overload
    def __call__(self, __iterable: Iterable[T0], *, key: Callable[[T0], SupportsRichComparison]) -> T0:
        ...

    @overload
    def __call__(
        self, __iterable: Iterable[SupportsRichComparisonT], *, key: None = ..., default: T0
    ) -> SupportsRichComparisonT | T0:
        ...

    @overload
    def __call__(
        self, __iterable: Iterable[T1], *, key: Callable[[T1], SupportsRichComparison], default: T2
    ) -> T1 | T2:
        ...


class SupportsIndexing(Protocol[_VT_co]):
    def __getitem__(self, __k: int) -> _VT_co:
        ...


class SupportsKeysAndGetItem(Protocol[_KT, _VT_co]):
    def keys(self) -> Iterable[_KT]:
        ...

    def __getitem__(self, __k: _KT) -> _VT_co:
        ...
