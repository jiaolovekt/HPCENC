from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from vsexprtools import complexpr_available, norm_expr
from vskernels import Catrom, NoScale, Scaler, ScalerT, Spline144
from vsmasktools import EdgeDetect, EdgeDetectT, Prewitt, ScharrTCanny
from vsrgtools import RepairMode, box_blur, contrasharpening_median, median_clips, repair, unsharp_masked
from vstools import (
    MISSING, CustomOverflowError, CustomRuntimeError, FunctionUtil, MissingT, PlanesT, check_ref_clip, get_h,
    get_peak_value, get_w, get_y, join, normalize_planes, plane, scale_8bit, scale_value, split, vs
)

from .abstract import Antialiaser, SingleRater
from .antialiasers import Eedi3, Nnedi3
from .enums import AADirection
from .mask import resize_aa_mask

__all__ = [
    'pre_aa',
    'upscaled_sraa',
    'transpose_aa',
    'clamp_aa', 'masked_clamp_aa',
    'fine_aa',
    'based_aa'
]


def pre_aa(
    clip: vs.VideoNode, radius: int = 1, strength: int = 100,
    aa: type[Antialiaser] | Antialiaser = Nnedi3(0, pscrn=1),
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode:
    func = FunctionUtil(clip, pre_aa, planes)

    if isinstance(aa, Antialiaser):
        aa = aa.copy(field=3, **kwargs)  # type: ignore
    else:
        aa = aa(field=3, **kwargs)

    clip = func.work_clip

    for _ in range(2):
        bob = aa.interpolate(clip, False)
        sharp = unsharp_masked(clip, radius, strength)
        limit = median_clips(sharp, clip, bob[::2], bob[1::2])
        clip = limit.std.Transpose()

    return func.return_clip(clip)


def upscaled_sraa(
    clip: vs.VideoNode, rfactor: float = 1.5,
    width: int | None = None, height: int | None = None,
    ssfunc: ScalerT = Nnedi3(), aafunc: SingleRater = Eedi3(),
    direction: AADirection = AADirection.BOTH,
    downscaler: ScalerT = Catrom(), planes: PlanesT = 0
) -> vs.VideoNode:
    """
    Super-sampled single-rate AA for heavy aliasing and broken line-art.

    It works by super-sampling the clip, performing AA, and then downscaling again.
    Downscaling can be disabled by setting `downscaler` to `None`, returning the super-sampled luma clip.
    The dimensions of the downscaled clip can also be adjusted by setting `height` or `width`.
    Setting either `height`, `width` or 1,2 in planes will also scale the chroma accordingly.

    :param clip:            Clip to process.
    :param rfactor:         Image enlargement factor.
                            It is not recommended to go below 1.3
    :param width:           Target resolution width. If None, determined from `height`.
    :param height:          Target resolution height.
    :param ssfunc:          Super-sampler used for upscaling before AA.
    :param aafunc:          Function used to antialias after super-sampling.
    :param direction:       Direction in which antialiasing will be performed.
    :param downscaler:      Downscaler to use after super-sampling.
    :param planes:          Planes to do antialiasing on.

    :return:                Antialiased clip.

    :raises ValueError:     ``rfactor`` is not above 1.
    """
    assert clip.format

    planes = normalize_planes(clip, planes)

    if height is None:
        height = clip.height

    if width is None:
        if height == clip.height:
            width = clip.width
        else:
            width = get_w(height, clip)

    aa_chroma = 1 in planes or 2 in planes
    scale_chroma = ((clip.width, clip.height) != (width, height)) or aa_chroma

    work_clip, *chroma = [clip] if scale_chroma else split(clip)

    if rfactor <= 1:
        raise ValueError('upscaled_sraa: rfactor must be above 1!')

    ssh = get_h(work_clip.width * rfactor, work_clip, 2)
    ssw = get_w(ssh, work_clip, 2)

    ssfunc = Scaler.ensure_obj(ssfunc, upscaled_sraa)
    downscaler = Scaler.ensure_obj(downscaler, upscaled_sraa)

    up = ssfunc.scale(work_clip, ssw, ssh)
    up, *chroma = [up] if aa_chroma or scale_chroma else [up, *chroma]

    aa = aafunc.aa(up, *direction.to_yx())

    if downscaler:
        aa = downscaler.scale(aa, width, height)

    if not chroma:
        return aa

    if 0 not in planes:
        return join(clip, aa, clip.format.color_family)

    return join([aa, *chroma], clip.format.color_family)


def transpose_aa(clip: vs.VideoNode, aafunc: SingleRater, planes: PlanesT = 0) -> vs.VideoNode:
    """
    Perform transposed AA.

    :param clip:        Clip to process.
    :param aafun:       Antialiasing function.
    :return:            Antialiased clip.
    """
    func = FunctionUtil(clip, transpose_aa, planes)

    aafunc = aafunc.copy(transpose_first=True, drop_fields=False)  # type: ignore

    aa = aafunc.aa(func.work_clip)  # type: ignore

    return func.return_clip(aa)


def clamp_aa(
    src: vs.VideoNode, weak: vs.VideoNode, strong: vs.VideoNode,
    strength: float = 1, ref: vs.VideoNode | None = None, planes: PlanesT = 0
) -> vs.VideoNode:
    """
    Clamp stronger AAs to weaker AAs.
    Useful for clamping :py:func:`upscaled_sraa` or :py:func:`Eedi3`
    to :py:func:`Nnedi3` for a strong but more precise AA.

    :param src:         Non-AA'd source clip.
    :param weak:        Weakly-AA'd clip.
    :param strong:      Strongly-AA'd clip.
    :param strength:    Clamping strength.
    :param ref:         Reference clip for clamping.
    :param planes:      Planes to process.

    :return:            Clip with clamped anti-aliasing.
    """
    assert src.format

    planes = normalize_planes(src, planes)

    if src.format.sample_type == vs.INTEGER:
        thr = strength * get_peak_value(src)
    else:
        thr = strength / 219

    if thr == 0:
        return median_clips(src, weak, strong, planes=planes)

    if ref:
        if complexpr_available:
            expr = f'y z - ZD! y a - AD! ZD@ AD@ xor x ZD@ abs AD@ abs < a z {thr} + min z {thr} - max a ? ?'
        else:
            expr = f'y z - y a - xor x y z - abs y a - abs < a z {thr} + min z {thr} - max a ? ?'
    else:
        if complexpr_available:
            expr = f'x y - XYD! x z - XZD! XYD@ XZD@ xor x XYD@ abs XZD@ abs < z y {thr} + min y {thr} - max z ? ?'
        else:
            expr = f'x y - x z - xor x x y - abs x z - abs < z y {thr} + min y {thr} - max z ? ?'

    return norm_expr([src, ref, weak, strong] if ref else [src, weak, strong], expr, planes)


def masked_clamp_aa(
    clip: vs.VideoNode, strength: float = 1.0,
    mthr: float = 0.25, mask: vs.VideoNode | EdgeDetectT | None = None,
    weak_aa: vs.VideoNode | SingleRater = Nnedi3(),
    strong_aa: vs.VideoNode | SingleRater = Eedi3(),
    opencl: bool | None = False, ref: vs.VideoNode | None = None,
    planes: PlanesT = 0
) -> vs.VideoNode:
    """
    Clamp a strong aa to a weaker one for the purpose of reducing the stronger's artifacts.

    :param clip:                Clip to process.
    :param strength:            Set threshold strength for over/underflow value for clamping.
    :param mthr:                Binarize threshold for the mask, float.
    :param mask:                Clip to use for custom mask or an EdgeDetect to use custom masker.
    :param weak_aa:             SingleRater for the weaker aa.
    :param strong_aa:           SingleRater for the stronger aa.
    :param opencl:              Whether to force OpenCL acceleration, None to leave as is.
    :param ref:                 Reference clip for clamping.

    :return:                    Antialiased clip.
    """
    assert clip.format

    func = FunctionUtil(clip, masked_clamp_aa, planes)

    if not isinstance(mask, vs.VideoNode):
        bin_thr = scale_value(mthr, 32, clip)

        mask = ScharrTCanny.ensure_obj(mask).edgemask(func.work_clip)  # type: ignore
        mask = mask.std.Binarize(bin_thr)
        mask = mask.std.Maximum()
        mask = box_blur(mask)
        mask = mask.std.Minimum().std.Deflate()

    if isinstance(weak_aa, SingleRater):
        if opencl is not None and hasattr(weak_aa, 'opencl'):
            weak_aa.opencl = opencl

        weak_aa = transpose_aa(func.work_clip, weak_aa, func.norm_planes)
    elif func.luma_only:
        weak_aa = get_y(weak_aa)

    if isinstance(strong_aa, SingleRater):
        if opencl is not None and hasattr(strong_aa, 'opencl'):
            strong_aa.opencl = opencl

        strong_aa = transpose_aa(func.work_clip, strong_aa, func.norm_planes)
    elif func.luma_only:
        strong_aa = get_y(strong_aa)

    if ref and func.luma_only:
        ref = get_y(ref)

    clamped = clamp_aa(func.work_clip, weak_aa, strong_aa, strength, ref, func.norm_planes)

    merged = func.work_clip.std.MaskedMerge(clamped, mask, func.norm_planes)  # type: ignore

    return func.return_clip(merged)


def fine_aa(
    clip: vs.VideoNode, taa: bool = False,
    singlerater: SingleRater = Eedi3(),
    rep: int | RepairMode = RepairMode.LINE_CLIP_STRONG,
    planes: PlanesT = 0
) -> vs.VideoNode:
    """
    Taa and optionally repair clip that results in overall lighter anti-aliasing, downscaled with Spline kernel.

    :param clip:            Clip to process.
    :param taa:             Whether to perform a normal or transposed aa
    :param singlerater:     Singlerater used for aa.
    :param rep:             Repair mode.

    :return:                Antialiased clip.
    """
    assert clip.format

    singlerater = singlerater.copy(shifter=Spline144())  # type: ignore

    func = FunctionUtil(clip, fine_aa, planes)

    if taa:
        aa = transpose_aa(func.work_clip, singlerater)
    else:
        aa = singlerater.aa(func.work_clip)  # type: ignore

    contra = contrasharpening_median(func.work_clip, aa, planes=func.norm_planes)

    repaired = repair(contra, func.work_clip, func.norm_seq(rep))

    return func.return_clip(repaired)


if TYPE_CHECKING:
    from vsdenoise import Prefilter
    from vsscale import FSRCNNXShader, FSRCNNXShaderT, ShaderFile

    def based_aa(
        clip: vs.VideoNode, rfactor: float = 2.0,
        mask_thr: int = 60, lmask: vs.VideoNode | EdgeDetectT = Prewitt,
        downscaler: ScalerT = Catrom,
        supersampler: ScalerT | FSRCNNXShaderT | ShaderFile | Path | Literal[False] = FSRCNNXShader.x56,
        antialiaser: Antialiaser = Eedi3(0.125, 0.25, vthresh0=12, vthresh1=24, field=1, sclip_aa=None),
        prefilter: Prefilter | vs.VideoNode = Prefilter.NONE, show_mask: bool = False, planes: PlanesT = 0,
        **kwargs: Any
    ) -> vs.VideoNode:
        ...
else:
    def based_aa(
        clip: vs.VideoNode, rfactor: float = 2.0,
        mask_thr: int = 60, lmask: vs.VideoNode | EdgeDetectT = Prewitt,
        downscaler: ScalerT = Catrom,
        supersampler: ScalerT | FSRCNNXShaderT | ShaderFile | Path | Literal[False] | MissingT = MISSING,
        antialiaser: Antialiaser = Eedi3(0.125, 0.25, vthresh0=12, vthresh1=24, field=1, sclip_aa=None),
        prefilter: Prefilter | vs.VideoNode | MissingT = MISSING, show_mask: bool = False, planes: PlanesT = 0,
        **kwargs: Any
    ) -> vs.VideoNode:
        try:
            from vsscale import FSRCNNXShader, PlaceboShader  # noqa: F811
        except ModuleNotFoundError:
            raise CustomRuntimeError(
                'You\'re missing the "vsscale" package! You can install it with "pip install vsscale".', based_aa
            )

        try:
            from vsdenoise import Prefilter  # noqa: F811
        except ModuleNotFoundError:
            raise CustomRuntimeError(
                'You\'re missing the "vsdenoise" package! You can install it with "pip install vsdenoise".', based_aa
            )

        func = FunctionUtil(clip, based_aa, planes)

        if supersampler is False:
            supersampler = downscaler = NoScale
        else:
            if supersampler is MISSING:
                supersampler = FSRCNNXShader.x56

        if prefilter is MISSING:
            prefilter = Prefilter.NONE

        if isinstance(prefilter, vs.VideoNode):
            work_clip = plane(prefilter, 0) if func.luma_only else prefilter
            check_ref_clip(func.work_clip, work_clip, based_aa)
        else:
            work_clip = prefilter(func.work_clip)

        antialiaser = antialiaser.copy(**kwargs)

        if rfactor < 1.0:
            raise CustomOverflowError(
                f'"rfactor" must be bigger than 1.0! ({rfactor})', based_aa
            )

        if not isinstance(lmask, vs.VideoNode):
            lmask = EdgeDetect.ensure_obj(lmask, based_aa)

            if mask_thr > 255:
                raise CustomOverflowError(
                    f'"mask_thr" must be less or equal than 255! ({mask_thr})', based_aa
                )

            lmask = lmask.edgemask(plane(work_clip, 0)).std.Binarize(scale_8bit(work_clip, mask_thr))
            lmask = box_blur(lmask.std.Maximum()).std.Limiter()

            if not func.luma_only:
                lmask = Catrom.resample(join(lmask, lmask, lmask), work_clip)

        if show_mask:
            return lmask

        if isinstance(supersampler, (str, Path)):
            supersampler = PlaceboShader(supersampler)

        supersampler = Scaler.ensure_obj(supersampler, based_aa)
        downscaler = Scaler.ensure_obj(downscaler, based_aa)

        aaw, aah = [(round(r * rfactor) + 1) & ~1 for r in (clip.width, clip.height)]

        aa = supersampler.scale(func.work_clip.std.Transpose(), aah, aaw)

        mclip_up = resize_aa_mask(lmask.std.Transpose(), aa.width, aa.height)

        aa = antialiaser.interpolate(aa, False, sclip=aa, mclip=mclip_up).std.Transpose()
        aa = antialiaser.interpolate(aa, False, sclip=aa, mclip=mclip_up.std.Transpose())

        aa = downscaler.scale(aa, func.work_clip.width, func.work_clip.height)

        aa_merge = func.work_clip.std.MaskedMerge(aa, lmask)

        return func.return_clip(aa_merge)
