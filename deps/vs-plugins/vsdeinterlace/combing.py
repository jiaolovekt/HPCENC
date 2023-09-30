from __future__ import annotations

from typing import cast

from vsexprtools import ExprVars, complexpr_available, norm_expr
from vsrgtools import sbr
from vstools import (
    ConvMode, CustomEnum, FieldBased, FieldBasedT, FuncExceptT, FunctionUtil, PlanesT, core, depth, expect_bits,
    get_neutral_values, scale_8bit, vs
)

__all__ = [
    'fix_telecined_fades',

    'vinverse'
]


def fix_telecined_fades(
    clip: vs.VideoNode, tff: bool | FieldBasedT | None = None, colors: float | list[float] = 0.0,
    planes: PlanesT = None, func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Give a mathematically perfect solution to fades made *after* telecining (which made perfect IVTC impossible).

    This is an improved version of the Fix-Telecined-Fades plugin.

    Make sure to run this *after* IVTC/deinterlacing!

    :param clip:                            Clip to process.
    :param tff:                             Top-field-first. `False` sets it to Bottom-Field-First.
                                            If `None`, get the field order from the _FieldBased prop.
    :param colors:                          Color offset for the plane average.

    :return:                                Clip with fades (and only fades) accurately deinterlaced.

    :raises UndefinedFieldBasedError:       No automatic ``tff`` can be determined.
    """
    func = func or fix_telecined_fades

    if not complexpr_available:
        raise ExprVars._get_akarin_err()(func=func)

    clip = FieldBased.ensure_presence(clip, tff, func)

    f = FunctionUtil(clip, func, planes, (vs.GRAY, vs.YUV), 32)

    fields = f.work_clip.std.Limiter().std.SeparateFields()

    for i in f.norm_planes:
        fields = fields.std.PlaneStats(None, i, f'P{i}')

    props_clip = core.akarin.PropExpr(
        [f.work_clip, fields[::2], fields[1::2]], lambda: {  # type: ignore[misc]
            f'f{t}Avg{i}': f'{c}.P{i}Average {color} -'  # type: ignore[has-type]
            for t, c in ['ty', 'bz']
            for i, color in zip(f.norm_planes, f.norm_seq(colors))
        }
    )

    fix = norm_expr(
        props_clip, 'Y 2 % BF! BF@ x.fbAvg{i} x.ftAvg{i} ? AVG! '
        'AVG@ 0 = x x {color} - AVG@ BF@ x.ftAvg{i} x.fbAvg{i} ? + 2 / AVG@ / * ? {color} +',
        planes, i=f.norm_planes, color=colors, force_akarin=func
    )

    return f.return_clip(fix)


class Vinverse(CustomEnum):
    V1 = object()
    V2 = object()
    MASKED = object()
    MASKEDV1 = object()
    MASKEDV2 = object()

    def __call__(
        self, clip: vs.VideoNode, sstr: float = 2.7, amount: int = 255, scale: float = 0.25,
        mode: ConvMode = ConvMode.VERTICAL, planes: PlanesT = None
    ) -> vs.VideoNode:
        if amount <= 0:
            return clip

        clip, bits = expect_bits(clip, 32)

        neutral = get_neutral_values(clip)

        expr = f'y z - {sstr} * D1! x y - D2! D1@ abs D1A! D2@ abs D2A! '
        expr += f'D1@ D2@ xor D1A@ D2A@ < D1@ D2@ ? {scale} * D1A@ D2A@ < D1@ D2@ ? ? y + '

        if self in {Vinverse.V1, Vinverse.MASKEDV1}:
            blur = clip.std.Convolution([50, 99, 50], mode=mode, planes=planes)
            blur2 = blur.std.Convolution([1, 4, 6, 4, 1], mode=mode, planes=planes)
        elif self in {Vinverse.V2, Vinverse.MASKEDV2}:
            blur = sbr(clip, mode=mode, planes=planes)
            blur2 = blur.std.Convolution([1, 2, 1], mode=mode, planes=planes)

        if self in {Vinverse.MASKED, Vinverse.MASKEDV1, Vinverse.MASKEDV2}:
            search_str = 'x[-1,0] x[1,0]' if mode == ConvMode.HORIZONTAL else 'x[0,-1] x[0,1]'
            mask_search_str = search_str.replace('x', 'y')

            if self is Vinverse.MASKED:
                find_combs = norm_expr(clip, f'x x 2 * {search_str} + + 4 / - {{n}} +', planes, n=neutral)
                decomb = norm_expr(
                    [find_combs, clip],
                    'x x 2 * {search_str} + + 4 / - B! y B@ x {n} - * 0 '
                    '< {n} B@ abs x {n} - abs < B@ {n} + x ? ? - {n} +', n=neutral, search_str=search_str
                )
            else:
                decomb = norm_expr(
                    [clip, blur, blur2], 'x x 2 * y + 4 / - {n} + FC! FC@ FC@ 2 * y z - {n} + + 4 / - B! '
                    'x B@ FC@ {n} - * 0 < {n} B@ abs FC@ {n} - abs < B@ {n} + FC@ ? ? - {n} +', n=neutral
                )

            return norm_expr(
                [clip, decomb], f'{scale_8bit(clip, amount)} a! y y y y 2 * {mask_search_str} + + 4 / - {sstr} '
                '* + y - {n} + D1! x y - {n} + D2! D1@ {n} - D2@ {n} - * 0 < D1@ {n} - abs D2@ {n} - abs < D1@ '
                'D2@ ? {n} - {scale} * {n} + D1@ {n} - abs D2@ {n} - abs < D1@ D2@ ? ? {n} - + merge! '
                'x a@ + merge@ < x a@ + x a@ - merge@ > x a@ - merge@ ? ?', n=neutral, scale=scale
            )

        if amount < 255:
            amn = scale_8bit(clip, amount)
            expr += f'LIM! x {amn} + LIM@ < x {amn} + x {amn} - LIM@ > x {amn} - LIM@ ? ?'

        return depth(norm_expr([clip, blur, blur2], expr, planes), bits)


vinverse = cast(Vinverse, Vinverse.V1)
