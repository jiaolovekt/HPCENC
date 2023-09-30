from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from math import floor, sqrt
from typing import Any, Literal, Sequence, Tuple

from vsexprtools import ExprList, ExprOp, ExprToken, complexpr_available, norm_expr
from vsrgtools.util import wmean_matrix
from vstools import (
    ConvMode, CustomIndexError, FuncExceptT, PlanesT, StrList, check_variable, copy_signature, core, fallback,
    inject_self, interleave_arr, iterate, scale_value, to_arr, vs
)

from .types import Coordinates, MorphoFunc, XxpandMode

__all__ = [
    'Morpho', 'CoordsT',
    'grow_mask'
]

CoordsT = int | Tuple[int, ConvMode] | Sequence[int]


def _minmax_method(  # type: ignore
    self: Morpho, src: vs.VideoNode, thr: float | None = None,
    coords: CoordsT | None = [1] * 8,
    iterations: int = 1, multiply: float | None = None, planes: PlanesT = None,
    *, func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    ...


def _morpho_method(  # type: ignore
    self: Morpho, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
    coords: CoordsT = 5, multiply: float | None = None,
    *, func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    ...


def _morpho_method2(  # type: ignore
    self: Morpho, clip: vs.VideoNode, sw: int, sh: int | None = None, mode: XxpandMode = XxpandMode.RECTANGLE,
    thr: float | None = None, planes: PlanesT = None, *, func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    ...


@dataclass
class Morpho:
    planes: PlanesT = None
    func: FuncExceptT | None = None
    fast: bool | None = None

    def __post_init__(self) -> None:
        self._fast = fallback(self.fast, complexpr_available) and complexpr_available

    def _check_params(
        self, radius: int, thr: float | None, coords: CoordsT, planes: PlanesT, func: FuncExceptT
    ) -> tuple[FuncExceptT, PlanesT]:
        if radius < 1:
            raise CustomIndexError('radius has to be greater than 0!', func, radius)

        if isinstance(coords, (int, tuple)):
            size = coords if isinstance(coords, int) else coords[0]

            if size < 2:
                raise CustomIndexError('when int or tuple, coords has to be greater than 1!', func, coords)

            if not self._fast and size != 3:
                raise CustomIndexError(
                    'with fast=False or no akarin plugin, you must have coords=3!', func, coords
                )
        elif len(coords) != 8:
            raise CustomIndexError('when a list, coords must contain exactly 8 numbers!', func, coords)

        if thr is not None and thr < 0.0:
            raise CustomIndexError('thr must be a positive number!', func, coords)

        return self.func or func, self.planes if planes is None else planes

    @classmethod
    def _morpho_xx_imum(
        cls, src: vs.VideoNode, thr: float | None, op: Literal[ExprOp.MIN, ExprOp.MAX],
        coords: CoordsT, multiply: float | None = None, clamp: bool = False
    ) -> StrList:
        exclude = list[tuple[int, int]]()

        if isinstance(coords, (int, tuple)):
            if isinstance(coords, tuple):
                size, mode = coords
            else:
                size, mode = coords, ConvMode.SQUARE

            assert size > 1

            radius = size // 2

            if size % 2 == 0:
                exclude.extend((x, radius) for x in range(-radius, radius + 1))
                exclude.append((radius, radius - 1))
        else:
            coords = list(coords)
            coords.insert(len(coords) // 2, 1)
            radius, mode = floor(sqrt(len(coords)) / 2), ConvMode.SQUARE

        matrix = ExprOp.matrix('x', radius, mode, exclude)

        if not isinstance(coords, (int, tuple)):
            matrix = ExprList([x for x, coord in zip(matrix, coords) if coord])

        matrix = ExprList(interleave_arr(matrix, op * matrix.mlength, 2))

        if thr is not None:
            matrix.append('x', scale_value(thr, 32, src), ExprOp.SUB, ExprOp.MAX)

        if multiply is not None:
            matrix.append(multiply, ExprOp.MUL)

        if clamp:
            matrix.append(ExprOp.clamp())

        return matrix

    def _mm_func(
        self, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
        coords: CoordsT = 5, multiply: float | None = None, *, func: FuncExceptT,
        mm_func: MorphoFunc, op: Literal[ExprOp.MIN, ExprOp.MAX], **kwargs: Any
    ) -> vs.VideoNode:
        func, planes = self._check_params(radius, thr, coords, planes, func)

        if self._fast:
            mm_func = norm_expr  # type: ignore[assignment]
            kwargs.update(expr=self._morpho_xx_imum(src, thr, op, coords, multiply))
        elif isinstance(coords, (int, tuple)):
            if isinstance(coords, tuple):
                if coords[1] is not ConvMode.SQUARE:
                    raise CustomIndexError(
                        'with fast=False or no akarin plugin, you must have ConvMode.SQUARE!', func, coords
                    )

                coords = coords[0]

            if coords != 3:
                raise CustomIndexError(
                    'with fast=False or no akarin plugin, you must have coords=3!', func, coords
                )

            kwargs.update(coordinates=[1] * 8)

        if not self._fast:
            if thr is not None:
                kwargs.update(threshold=scale_value(thr, 32, src))

            if multiply is not None:
                orig_mm_func = mm_func

                @copy_signature(mm_func)
                def _mm_func(*args: Any, **kwargs: Any) -> Any:
                    return orig_mm_func(*args, **kwargs).std.Expr(f'x {multiply} *')

                mm_func = _mm_func

        return iterate(src, mm_func, radius, planes=planes, **kwargs)

    @inject_self
    def xxpand_transform(
        self, clip: vs.VideoNode, op: Literal[ExprOp.MIN, ExprOp.MAX], sw: int, sh: int | None = None,
        mode: XxpandMode = XxpandMode.RECTANGLE, thr: float | None = None,
        planes: PlanesT = None, *, func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        func, planes = self._check_params(1, thr, 3, planes, func or self.xxpand_transform)

        sh = fallback(sh, sw)

        if op not in {ExprOp.MIN, ExprOp.MAX}:
            raise NotImplementedError

        function = self.maximum if op is ExprOp.MAX else self.minimum

        for wi, hi in zip_longest(range(sw, -1, -1), range(sh, -1, -1), fillvalue=0):
            if wi > 0 and hi > 0:
                coords = Coordinates.from_xxpand_mode(mode, wi)
            elif wi > 0:
                coords = Coordinates.HORIZONTAL
            elif hi > 0:
                coords = Coordinates.VERTICAL
            else:
                break

            clip = function(clip, thr, coords, planes=planes, func=func)

        return clip

    def _xxflate(
        self: Morpho, inflate: bool, src: vs.VideoNode, radius: int, planes: PlanesT, thr: float | None,
        multiply: float | None, *, func: FuncExceptT
    ) -> vs.VideoNode:
        assert check_variable(src, func)

        expr = ExprOp.matrix('x', radius, exclude=[(0, 0)])

        conv_len = len(expr)

        expr.append(ExprOp.ADD * expr.mlength)

        if src.format.sample_type is vs.INTEGER:
            expr.append(radius * 4, ExprOp.ADD)

        expr.append(conv_len, ExprOp.DIV)
        expr.append('x', ExprOp.MAX if inflate else ExprOp.MIN)

        if thr is not None:
            thr = scale_value(thr, 32, src)
            limit = ['x', thr, ExprOp.ADD] if inflate else ['x', thr, ExprOp.SUB, ExprToken.RangeMin, ExprOp.MAX]

            expr.append(limit, ExprOp.MIN if inflate else ExprOp.MAX)

        if multiply is not None:
            expr.append(multiply, ExprOp.MUL)

        return norm_expr(src, expr, planes)

    @inject_self
    @copy_signature(_minmax_method)
    def maximum(
        self, src: vs.VideoNode, thr: float | None = None, coords: CoordsT | None = None,
        iterations: int = 1, multiply: float | None = None, planes: PlanesT = None,
        *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        return self.dilation(src, iterations, planes, thr, coords or ([1] * 8), multiply, func=func, **kwargs)

    @inject_self
    @copy_signature(_minmax_method)
    def minimum(
        self, src: vs.VideoNode, thr: float | None = None, coords: CoordsT | None = None,
        iterations: int = 1, multiply: float | None = None, planes: PlanesT = None,
        *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        return self.erosion(src, iterations, planes, thr, coords or ([1] * 8), multiply, func=func, **kwargs)

    @inject_self
    def inflate(
        self: Morpho, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
        iterations: int = 1, multiply: float | None = None, *, func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        for _ in range(iterations):
            src = self._xxflate(True, src, radius, planes, thr, multiply, func=func or self.inflate)
        return src

    @inject_self
    def deflate(
        self: Morpho, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
        iterations: int = 1, multiply: float | None = None, *, func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        for _ in range(iterations):
            src = self._xxflate(False, src, radius, planes, thr, multiply, func=func or self.deflate)
        return src

    @inject_self
    @copy_signature(_morpho_method)
    def dilation(self, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self._mm_func(*args, func=func or self.dilation, mm_func=core.std.Maximum, op=ExprOp.MAX, **kwargs)

    @inject_self
    @copy_signature(_morpho_method)
    def erosion(self, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self._mm_func(*args, func=func or self.erosion, mm_func=core.std.Minimum, op=ExprOp.MIN, **kwargs)

    @inject_self
    @copy_signature(_morpho_method2)
    def expand(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self.xxpand_transform(clip, ExprOp.MAX, *args, func=func, **kwargs)

    @inject_self
    @copy_signature(_morpho_method2)
    def inpand(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self.xxpand_transform(clip, ExprOp.MIN, *args, func=func, **kwargs)

    @inject_self
    @copy_signature(_morpho_method)
    def closing(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        func = func or self.closing

        dilated = self.dilation(src, *args, func=func, **kwargs)
        eroded = self.erosion(dilated, *args, func=func, **kwargs)

        return eroded

    @inject_self
    @copy_signature(_morpho_method)
    def opening(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        func = func or self.closing

        eroded = self.erosion(src, *args, func=func, **kwargs)
        dilated = self.dilation(eroded, *args, func=func, **kwargs)

        return dilated

    @inject_self
    def gradient(
        self, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
        coords: CoordsT = 5, multiply: float | None = None, *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        func, planes = self._check_params(radius, thr, coords, planes, func or self.gradient)

        if radius == 1 and self._fast:
            return norm_expr(
                src, '{dilated} {eroded} - {multiply}', planes,
                dilated=self._morpho_xx_imum(src, thr, ExprOp.MAX, coords, None, True),
                eroded=self._morpho_xx_imum(src, thr, ExprOp.MIN, coords, None, True),
                multiply='' if multiply is None else f'{multiply} *'
            )

        eroded = self.erosion(src, radius, planes, thr, coords, multiply, func=func, **kwargs)
        dilated = self.dilation(src, radius, planes, thr, coords, multiply, func=func, **kwargs)

        return norm_expr([dilated, eroded], 'x y -', planes)

    @inject_self
    @copy_signature(_morpho_method)
    def top_hat(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        opened = self.opening(src, *args, func=func or self.top_hat, **kwargs)

        return norm_expr([src, opened], 'x y -', kwargs.get('planes', args[1] if len(args) > 1 else None))

    @inject_self
    @copy_signature(_morpho_method)
    def black_hat(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        closed = self.closing(src, *args, func=func or self.black_hat, **kwargs)

        return norm_expr([closed, src], 'x y -', kwargs.get('planes', args[1] if len(args) > 1 else None))

    @inject_self
    def outer_hat(
        self, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
        coords: CoordsT = 5, multiply: float | None = None, *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        func, planes = self._check_params(radius, thr, coords, planes, func or self.outer_hat)

        if radius == 1 and self._fast:
            return norm_expr(
                src, '{dilated} {multiply} x -', planes,
                dilated=self._morpho_xx_imum(src, thr, ExprOp.MAX, coords, None, True),
                multiply='' if multiply is None else f'{multiply} *'
            )

        dilated = self.dilation(src, radius, planes, thr, coords, multiply, func=func, **kwargs)

        return norm_expr([dilated, src], 'x y -', planes)

    @inject_self
    def inner_hat(
        self, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: float | None = None,
        coords: CoordsT = 5, multiply: float | None = None, *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        func, planes = self._check_params(radius, thr, coords, planes, func or self.inner_hat)

        if radius == 1 and self._fast:
            return norm_expr(
                src, '{eroded} {multiply} x -', planes,
                eroded=self._morpho_xx_imum(src, thr, ExprOp.MIN, coords),
                multiply='' if multiply is None else f'{multiply} *'
            )

        eroded = self.erosion(src, radius, planes, thr, coords, multiply, func=func, **kwargs)

        return norm_expr([eroded, src], 'x y -', planes)

    @inject_self
    def binarize(
        self, src: vs.VideoNode, midthr: float | list[float] | None = None,
        lowval: float | list[float] | None = None, highval: float | list[float] | None = None,
        planes: PlanesT = None
    ) -> vs.VideoNode:
        midthr, lowval, highval = (
            thr and list(
                scale_value(t, 32, src, chroma=i != 0)
                for i, t in enumerate(to_arr(thr))
            ) for thr in (midthr, lowval, highval)
        )

        return src.std.Binarize(midthr, lowval, highval, planes)


def grow_mask(
    mask: vs.VideoNode, radius: int = 1, multiply: float = 1.0,
    planes: PlanesT = None, coords: CoordsT = 5, thr: float | None = None,
    *, func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    func = func or grow_mask

    assert check_variable(mask, func)

    morpho = Morpho(planes, func)

    kwargs.update(thr=thr, coords=coords)

    closed = morpho.closing(mask, **kwargs)
    dilated = morpho.dilation(closed, **kwargs)
    outer = morpho.outer_hat(dilated, radius, **kwargs)

    blurred = outer.std.Convolution(wmean_matrix, planes=planes)

    if multiply != 1.0:
        return blurred.std.Expr(f'x {multiply} *')

    return blurred
