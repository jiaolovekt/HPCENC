from __future__ import annotations

from functools import lru_cache, partial
from itertools import count
from math import e, log, log2, sqrt
from typing import Any

from vsexprtools import ExprOp, ExprVars, complexpr_available, norm_expr
from vspyplugin import FilterMode, ProcessMode, PyPluginCuda
from vstools import (
    ConvMode, CustomNotImplementedError, CustomOverflowError, CustomValueError, FuncExceptT, FunctionUtil,
    NotFoundEnumValue, PlanesT, StrList, check_variable, core, depth, fallback, get_depth, get_neutral_value, join,
    normalize_planes, normalize_seq, split, to_arr, vs
)

from .enum import BlurMatrix, LimitFilterMode
from .limit import limit_filter
from .util import normalize_radius

__all__ = [
    'blur', 'box_blur', 'side_box_blur',
    'gauss_blur', 'gauss_fmtc_blur',
    'min_blur', 'sbr', 'median_blur',
    'bilateral'
]


def blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, blur, radius, planes, mode=mode)

    if mode == ConvMode.SQUARE:
        matrix2 = [1, 3, 4, 3, 1]
        matrix3 = [1, 4, 8, 10, 8, 4, 1]
    elif mode in {ConvMode.HORIZONTAL, ConvMode.VERTICAL}:
        matrix2 = [1, 6, 15, 20, 15, 6, 1]
        matrix3 = [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]
    else:
        raise NotFoundEnumValue('Invalid mode specified!', blur)

    if radius == 1:
        matrix = [1, 2, 1]
    elif radius == 2:
        matrix = matrix2
    elif radius == 3:
        matrix = matrix3
    else:
        raise CustomNotImplementedError('This radius isn\'t supported!', blur)

    return clip.std.Convolution(matrix, planes=planes, mode=mode)


def box_blur(clip: vs.VideoNode, radius: int | list[int] = 1, passes: int = 1, planes: PlanesT = None) -> vs.VideoNode:
    assert check_variable(clip, box_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, box_blur, radius, planes, passes=passes)

    if not radius:
        return clip

    fp16 = clip.format.sample_type == vs.FLOAT and clip.format.bits_per_sample == 16

    if radius > 12 and not fp16:
        blurred = clip.std.BoxBlur(planes, radius, passes, radius, passes)
    else:
        matrix_size = radius * 2 | 1

        if fp16:
            matrix_size **= 2

        blurred = clip
        for _ in range(passes):
            if fp16:
                blurred = norm_expr(blurred, [
                    ExprOp.matrix('x', radius), ExprOp.ADD * (matrix_size - 1), matrix_size, ExprOp.DIV
                ], planes)
            else:
                blurred = blurred.std.Convolution([1] * matrix_size, planes=planes, mode=ConvMode.SQUARE)

    return blurred


def side_box_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, planes: PlanesT = None,
    inverse: bool = False
) -> vs.VideoNode:
    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, side_box_blur, radius, planes, inverse=inverse)

    half_kernel = [(1 if i <= 0 else 0) for i in range(-radius, radius + 1)]

    conv_m1 = partial(core.std.Convolution, matrix=half_kernel, planes=planes)
    conv_m2 = partial(core.std.Convolution, matrix=half_kernel[::-1], planes=planes)
    blur_pt = partial(core.std.BoxBlur, planes=planes)

    vrt_filters, hrz_filters = [
        [
            partial(conv_m1, mode=mode), partial(conv_m2, mode=mode),
            partial(blur_pt, hradius=hr, vradius=vr, hpasses=h, vpasses=v)
        ] for h, hr, v, vr, mode in [
            (0, None, 1, radius, ConvMode.VERTICAL), (1, radius, 0, None, ConvMode.HORIZONTAL)
        ]
    ]

    vrt_intermediates = (vrt_flt(clip) for vrt_flt in vrt_filters)
    intermediates = list(
        hrz_flt(vrt_intermediate)
        for i, vrt_intermediate in enumerate(vrt_intermediates)
        for j, hrz_flt in enumerate(hrz_filters) if not i == j == 2
    )

    comp_blur = None if inverse else box_blur(clip, radius, 1, planes)

    if complexpr_available:
        template = '{cum} x - abs {new} x - abs < {cum} {new} ?'

        cum_expr, cumc = '', 'y'
        n_inter = len(intermediates)

        for i, newc, var in zip(count(), ExprVars[2:26], ExprVars[4:26]):
            if i == n_inter - 1:
                break

            cum_expr += template.format(cum=cumc, new=newc)

            if i != n_inter - 2:
                cumc = var.upper()
                cum_expr += f' {cumc}! '
                cumc = f'{cumc}@'

        if comp_blur:
            clips = [clip, *intermediates, comp_blur]
            cum_expr = f'x {cum_expr} - {ExprVars[n_inter + 1]} +'
        else:
            clips = [clip, *intermediates]

        cum = norm_expr(clips, cum_expr, planes, force_akarin='vsrgtools.side_box_blur')
    else:
        cum = intermediates[0]
        for new in intermediates[1:]:
            cum = limit_filter(clip, cum, new, LimitFilterMode.SIMPLE2_MIN, planes)

        if comp_blur:
            cum = clip.std.MakeDiff(cum).std.MergeDiff(comp_blur)

    if comp_blur:
        return box_blur(cum, 1, min(radius // 2, 1))

    return cum


def _norm_gauss_sigma(clip: vs.VideoNode, sigma: float | None, sharp: float | None, func: FuncExceptT) -> float:
    if sigma is None:
        if sharp is None:
            raise CustomValueError("Sigma and sharp can't be both None!", func)
        sigma = sqrt(1.0 / (2.0 * (sharp / 10.0) * log(2)))
    elif sharp is not None:
        raise CustomValueError("Sigma and sharp can't both be float!", func)

    if sigma >= min(clip.width, clip.height):
        raise CustomOverflowError("Sigma can't be bigger or equal than the smaller size of the clip!", func)

    return sigma


def gauss_blur(
    clip: vs.VideoNode,
    sigma: float | list[float] | None = 0.5, sharp: float | list[float] | None = None,
    mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, gauss_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(sigma, list):
        return normalize_radius(clip, gauss_blur, ('sigma', sigma), planes, mode=mode)
    elif isinstance(sharp, list):
        return normalize_radius(clip, gauss_blur, ('sharp', sharp), planes, mode=mode)

    sigma = _norm_gauss_sigma(clip, sigma, sharp, gauss_blur)

    if sigma <= 0.333:
        return clip

    if sigma <= 4.333:
        return BlurMatrix.gauss(sigma)(clip, planes, mode)

    return gauss_fmtc_blur(clip, sigma, sharp, True, mode, planes)


def gauss_fmtc_blur(
    clip: vs.VideoNode,
    sigma: float | list[float] | None = 0.5, sharp: float | list[float] | None = None,
    strict: bool = True, mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, gauss_fmtc_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(sigma, list):
        return normalize_radius(clip, gauss_blur, ('sigma', sigma), planes, strict=strict, mode=mode)
    elif isinstance(sharp, list):
        return normalize_radius(clip, gauss_blur, ('sharp', sharp), planes, strict=strict, mode=mode)

    if strict:
        sigma = _norm_gauss_sigma(clip, sigma, sharp, gauss_fmtc_blur)
        wdown, hdown = [
            round((max(round(size / sigma), 2) if char in mode else size) / 2) * 2
            for size, char in [(clip.width, 'h'), (clip.height, 'v')]
        ]

        def _fmtc_blur(clip: vs.VideoNode) -> vs.VideoNode:
            down = clip.resize.Bilinear(wdown, hdown)
            down = down.fmtc.resample(clip.width, clip.height, kernel='gauss', a1=9)

            return depth(down, clip)
    else:
        if sigma is None or sigma < 1.0 or sigma > 100.0:
            raise CustomValueError('Sigma has to be > 1 and < 100!', gauss_fmtc_blur, reason='strict=True')
        elif sharp is None:
            sharp = 100
        elif sharp < 1.0 or sharp > 100.0:
            raise CustomValueError('Sharp has to be > 1 and < 100!', gauss_fmtc_blur, reason='strict=True')

        def _fmtc_blur(clip: vs.VideoNode) -> vs.VideoNode:
            down = clip.fmtc.resample(
                clip.width * 2, clip.height * 2, kernel='gauss', a1=sharp
            )
            down = down.fmtc.resample(clip.width, clip.height, kernel='gauss', a1=sigma)

            return depth(down, clip)

    if not {*range(clip.format.num_planes)} - {*planes}:
        return _fmtc_blur(clip)

    return join([
        _fmtc_blur(p) if i in planes else p
        for i, p in enumerate(split(clip))
    ])


def min_blur(clip: vs.VideoNode, radius: int | list[int] = 1, planes: PlanesT = None) -> vs.VideoNode:
    """
    MinBlur by Didée (http://avisynth.nl/index.php/MinBlur)
    Nifty Gauss/Median combination
    """
    assert check_variable(clip, min_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, min_blur, radius, planes)

    if radius in {0, 1}:
        median = clip.std.Median(planes)
    else:
        median = median_blur(clip, radius, planes=planes)

    if radius:
        weighted = blur(clip, radius)
    else:
        weighted = sbr(clip, planes=planes)

    return limit_filter(weighted, clip, median, LimitFilterMode.DIFF_MIN, planes)


def sbr(
    clip: vs.VideoNode,
    radius: int | list[int] = 1, mode: ConvMode = ConvMode.SQUARE,
    planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, sbr)

    planes = normalize_planes(clip, planes)

    neutral = [get_neutral_value(clip), get_neutral_value(clip, True)]

    blur_func = partial(blur, radius=radius, mode=mode, planes=planes)

    weighted = blur_func(clip)

    diff = clip.std.MakeDiff(weighted, planes)

    diff_weighted = blur_func(diff)

    clips = [diff, diff_weighted]

    if complexpr_available:
        clips.append(clip)
        expr = 'x y - A! x {mid} - XD! z A@ XD@ * 0 < {mid} A@ abs XD@ abs < A@ {mid} + x ? ? {mid} - -'
    else:
        expr = 'x y - x {mid} - * 0 < {mid} x y - abs x {mid} - abs < x y - {mid} + x ? ?'

    diff = norm_expr(clips, expr, planes, mid=neutral)

    if complexpr_available:
        return diff

    return clip.std.MakeDiff(diff, planes)


def median_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    def _get_vals(radius: int) -> tuple[StrList, int, int, int]:
        matrix = ExprOp.matrix('x', radius, mode, [(0, 0)])
        rb = len(matrix) + 1
        st = rb - 1
        sp = rb // 2 - 1
        dp = st - 2

        return matrix, st, sp, dp

    return norm_expr(clip, (
        f"{matrix} sort{st} swap{sp} min! swap{sp} max! drop{dp} x min@ max@ clip"
        for matrix, st, sp, dp in map(_get_vals, to_arr(radius))
    ), planes, force_akarin=median_blur)


class BilateralFilter(PyPluginCuda[None]):
    cuda_kernel = 'bilateral'
    filter_mode = FilterMode.Parallel

    input_per_plane = True
    output_per_plane = True

    @PyPluginCuda.process(ProcessMode.SingleSrcIPP)
    def _(self, src: BilateralFilter.DT, dst: BilateralFilter.DT, f: vs.VideoFrame, plane: int, n: int) -> None:
        self.kernel.bilateral[plane](src, dst)

    @lru_cache
    def get_kernel_shared_mem(
        self, plane: int, func_name: str, blk_size_w: int, blk_size_h: int, dtype_size: int
    ) -> int:
        return (2 * self.bil_radius[plane] + blk_size_w) * (2 * self.bil_radius[plane] + blk_size_h) * dtype_size

    def __init__(
        self, clip: vs.VideoNode, sigmaS: float | list[float], sigmaR: float | list[float],
        radius: int | list[int] | None, **kwargs: Any
    ) -> None:
        sigmaS, sigmaR = normalize_seq(sigmaS), normalize_seq(sigmaR)

        sigmaS_scaled, sigmaR_scaled = [
            [(-0.5 / (val * val)) * log2(e) for val in vals]
            for vals in (sigmaS, sigmaR)
        ]

        if radius is None:
            radius = [max(1, round(s * 3)) for s in sigmaS]

        self.bil_radius = normalize_seq(radius)

        return super().__init__(
            clip, kernel_planes_kwargs=[
                dict(sigmaS=s, sigmaR=r, radius=rad)
                for s, r, rad in zip(sigmaS_scaled, sigmaR_scaled, self.bil_radius)
            ], **kwargs
        )


def bilateral(
    clip: vs.VideoNode, sigmaS: float | list[float] = 3.0, sigmaR: float | list[float] = 0.02,
    ref: vs.VideoNode | None = None, radius: int | list[int] | None = None,
    device_id: int = 0, num_streams: int | None = None, use_shared_memory: bool = True,
    block_x: int | None = None, block_y: int | None = None, planes: PlanesT = None,
    *, gpu: bool | None = None
) -> vs.VideoNode:
    func = FunctionUtil(clip, bilateral, planes)

    sigmaS, sigmaR = func.norm_seq(sigmaS), func.norm_seq(sigmaR)

    if not ref and gpu is not False:
        if min(sigmaS) < 4 and PyPluginCuda.backend.is_available:
            block_x = fallback(block_x, block_y, 16)
            block_y = fallback(block_y, block_x)

            return BilateralFilter(
                clip, sigmaS, sigmaR, radius,
                kernel_size=(block_x, block_y),
                use_shared_memory=use_shared_memory
            ).invoke()

        try:
            if hasattr(core, 'bilateralgpu_rtc'):
                return clip.bilateralgpu_rtc.Bilateral(
                    sigmaS, sigmaR, radius, device_id, num_streams, use_shared_memory, block_x, block_y
                )
            if hasattr(core, 'bilateralgpu'):
                try:
                    return clip.bilateralgpu.Bilateral(
                        sigmaS, sigmaR, radius, device_id, num_streams, use_shared_memory
                    )
                except vs.Error:
                    # Old versions
                    return clip.bilateralgpu.Bilateral(sigmaS, sigmaR, radius, device_id)
        except vs.Error as e:
            # has the plugin but no cuda GPU available
            if 'cudaGetDeviceCount' in str(e):
                pass

    if (bits := get_depth(clip)) > 16:
        clip = depth(clip, 16)

    if ref and clip.format != ref.format:
        ref = depth(ref, clip)

    return depth(clip.bilateral.Bilateral(ref, sigmaS, sigmaR), bits)
