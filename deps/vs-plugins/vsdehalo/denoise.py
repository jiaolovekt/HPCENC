from __future__ import annotations

from math import ceil

from vsaa import Nnedi3
from vsdenoise import Prefilter
from vsexprtools import ExprOp, ExprToken, norm_expr
from vskernels import NoShift, Point, Scaler, ScalerT
from vsmasktools import Morpho, Prewitt
from vsrgtools import LimitFilterMode, contrasharpening, contrasharpening_dehalo, limit_filter, repair
from vstools import FunctionUtil, PlanesT, check_ref_clip, fallback, mod4, plane, vs

__all__ = [
    'smooth_dering'
]


def smooth_dering(
    clip: vs.VideoNode,
    smooth: vs.VideoNode | Prefilter = Prefilter.MINBLUR1,
    ringmask: vs.VideoNode | None = None,
    mrad: int = 1, msmooth: int = 1, minp: int = 1, mthr: float = 0.24, incedge: bool = False,
    thr: int = 12, elast: float = 2.0, darkthr: int | None = None,
    contra: int | float | bool = 1.2, drrep: int = 13, pre_ss: float = 1.0,
    pre_supersampler: ScalerT = Nnedi3(0, field=0, shifter=NoShift),
    pre_downscaler: ScalerT = Point, planes: PlanesT = 0, show_mask: bool = False
) -> vs.VideoNode:
    """
    :param clip:        Clip to process.
    :param smooth:      Already smoothed clip, or a Prefilter, tuple for [luma, chroma] prefilter.
    :param ringmask:    Custom ringing mask.
    :param mrad:        Expanding iterations of edge mask, higher value means more aggressive processing.
    :param msmooth:     Inflating iterations of edge mask, higher value means smoother edges of mask.
    :param minp:        Inpanding iterations of prewitt edge mask, higher value means more aggressive processing.
    :param mthr:        Threshold of prewitt edge mask, lower value means more aggressive processing
                        but for strong ringing, lower value will treat some ringing as edge,
                        which "protects" this ringing from being processed.
    :param incedge:     Whether to include edge in ring mask, by default ring mask only include area near edges.
    :param thr:         Threshold (8-bit scale) to limit filtering diff.
                        Smaller thr will result in more pixels being taken from processed clip.
                        Larger thr will result in less pixels being taken from input clip.
                            PDiff: pixel value diff between processed clip and input clip
                            ODiff: pixel value diff between output clip and input clip
                            PDiff, thr and elast is used to calculate ODiff:
                            ODiff = PDiff when [PDiff <= thr]
                            ODiff gradually smooths from thr to 0 when [thr <= PDiff <= thr * elast]
                            For elast>2.0, ODiff reaches maximum when [PDiff == thr * elast / 2]
                            ODiff = 0 when [PDiff >= thr * elast]
    :param elast:       Elasticity of the soft threshold.
                        Larger "elast" will result in more pixels being blended from.
    :param darkthr:     Threshold (8-bit scale) for darker area near edges, for filtering diff
                        that brightening the image by default equals to thr/4.
                        Set it lower if you think de-ringing destroys too much lines, etc.
                        When darkthr is not equal to ``thr``, ``thr`` limits darkening,
                        while ``darkthr`` limits brightening.
                        This is useful to limit the overshoot/undershoot/blurring introduced in deringing.
                        Examples:
                            ``thr=0``,   ``darkthr=0``  : no limiting
                            ``thr=255``, ``darkthr=255``: no limiting
                            ``thr=8``,   ``darkthr=2``  : limit darkening with 8, brightening is limited to 2
                            ``thr=8``,   ``darkthr=0``  : limit darkening with 8, brightening is limited to 0
                            ``thr=255``, ``darkthr=0``  : limit darkening with 255, brightening is limited to 0
                            For the last two examples, output will remain unchanged. (0/255: no limiting)
    :param contra:      Whether to use contra-sharpening to resharp deringed clip:
                            False: no contrasharpening
                            True: auto radius for contrasharpening
                            int 1-3: represents radius for contrasharpening
                            float: represents level for contrasharpening_dehalo
    :param drrep:       Use repair for details retention, recommended values are 13/12/1.
    :param planes:      Planes to be processed.
    :param show_mask:   Show the computed ringing mask.
    :param kwargs:      Kwargs to be passed to the prefilter function.

    :return:            Deringed clip.
    """
    func = FunctionUtil(clip, smooth_dering, planes, (vs.GRAY, vs.YUV))
    planes = func.norm_planes
    work_clip = func.work_clip

    pre_supersampler = Scaler.ensure_obj(pre_supersampler, smooth_dering)
    pre_downscaler = Scaler.ensure_obj(pre_downscaler, smooth_dering)

    if pre_ss > 1.0:
        work_clip = pre_supersampler.scale(  # type: ignore
            work_clip, mod4(work_clip.width * pre_ss), mod4(work_clip.height * pre_ss)
        )

    darkthr = fallback(darkthr, thr // 4)

    rep_dr = [drrep if i in planes else 0 for i in range(work_clip.format.num_planes)]

    if not isinstance(smooth, vs.VideoNode):
        smoothed = smooth(work_clip, planes)  # type: ignore
    else:
        check_ref_clip(clip, smooth)  # type: ignore

        smoothed = plane(smooth, 0) if func.luma_only else smooth  # type: ignore

        if pre_ss > 1.0:
            smoothed = pre_supersampler.scale(smoothed, work_clip.width, work_clip.height)  # type: ignore

    if contra:
        if isinstance(contra, int):
            smoothed = contrasharpening(smoothed, work_clip, contra, 13, planes)
        else:
            smoothed = contrasharpening_dehalo(smoothed, work_clip, contra, planes=planes)

    if set(rep_dr) != {0}:
        repclp = repair(work_clip, smoothed, drrep)
    else:
        repclp = work_clip

    limitclp = limit_filter(
        repclp, work_clip, None, LimitFilterMode.CLAMPING, planes, thr, elast, darkthr
    )

    if ringmask is None:
        prewittm = Prewitt.edgemask(work_clip, mthr)

        fmask = prewittm.std.Median(planes).misc.Hysteresis(prewittm, planes)

        omask = Morpho.expand(fmask, mrad, mrad, planes=planes) if mrad > 0 else fmask

        if msmooth > 0:
            omask = Morpho.inflate(omask, msmooth, planes)

        if incedge:
            ringmask = omask
        else:
            if minp <= 0:
                imask = fmask
            elif minp % 2 == 0:
                imask = Morpho.inpand(fmask, minp // 2, planes=planes)
            else:
                imask = Morpho.inpand(Morpho.inflate(fmask, 1, planes), ceil(minp / 2), planes=planes)

            ringmask = norm_expr(
                [omask, imask], [f'{ExprToken.RangeMax} {ExprToken.RangeMax} y - / x *', ExprOp.clamp()]
            )

    dering = work_clip.std.MaskedMerge(limitclp, ringmask, planes)

    if show_mask:
        return ringmask

    if (dering.width, dering.height) != (clip.width, clip.height):
        dering = pre_downscaler.scale(work_clip, clip.width, clip.height)

    return func.return_clip(dering)
