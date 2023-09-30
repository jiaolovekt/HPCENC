from __future__ import annotations

import warnings
from typing import Any

from vstools import DependencyNotFoundError, FieldBased, FieldBasedT, InvalidFramerateError, check_variable, core, vs

__all__ = [
    'pulldown_credits'
]


def pulldown_credits(
    clip: vs.VideoNode, frame_ref: int, tff: bool | FieldBasedT | None = None,
    interlaced: bool = True, dec: bool | None = None,
    bob_clip: vs.VideoNode | None = None, qtgmc_args: dict[str, Any] | None = None
) -> vs.VideoNode:
    """
    Deinterlacing function for interlaced credits (60i/30p) on top of telecined video (24p).

    This is a combination of havsfunc's dec_txt60mc, ivtc_txt30mc, and ivtc_txt60mc functions.
    The credits are interpolated and decimated to match the output clip.

    The function assumes you're passing a telecined clip (that's native 24p).
    If your clip is already fieldmatched, decimation will automatically be enabled unless set it to False.
    Likewise, if your credits are 30p (as opposed to 60i), you should set `interlaced` to False.

    The recommended way to use this filter is to trim out the area with interlaced credits,
    apply this function, and `vstools.insert_clip` the clip back into a properly IVTC'd clip.
    Alternatively, use `muvsfunc.VFRSplice` to splice the clip back in if you're dealing with a VFR clip.

    :param clip:                    Clip to process. Framerate must be 30000/1001.
    :param frame_ref:               First frame in the pattern. Expected pattern is ABBCD,
                                    except for when ``dec`` is enabled, in which case it's AABCD.
    :param tff:                     Top-field-first. `False` sets it to Bottom-Field-First.
    :param interlaced:              60i credits. Set to false for 30p credits.
    :param dec:                     Decimate input clip as opposed to IVTC.
                                    Automatically enabled if certain fieldmatching props are found.
                                    Can be forcibly disabled by setting it to `False`.
    :param bob_clip:                Custom bobbed clip. If `None`, uses a QTGMC clip.
                                    Framerate must be 60000/1001.
    :param qtgmc_args:              Arguments to pass on to QTGMC.
                                    Accepts any parameter except for FPSDivisor and TFF.

    :return:                        IVTC'd/decimated clip with credits pulled down to 24p.

    :raises ModuleNotFoundError:    Dependencies are missing.
    :raises ValueError:             Clip does not have a framerate of 30000/1001 (29.97).
    :raises TopFieldFirstError:     No automatic ``tff`` can be determined.
    :raises InvalidFramerateError:  Bobbed clip does not have a framerate of 60000/1001 (59.94)
    """

    try:
        from havsfunc import QTGMC  # type: ignore[import]
    except ModuleNotFoundError:
        raise DependencyNotFoundError(pulldown_credits, 'havsfunc')

    try:
        from vsdenoise import prefilter_to_full_range
    except ModuleNotFoundError:
        from havsfunc import DitherLumaRebuild as prefilter_to_full_range  # type: ignore
        warnings.warn("pulldown_credits: missing dependency `vsdenoise`!", ImportWarning)

    assert check_variable(clip, "pulldown_credits")

    InvalidFramerateError.check(pulldown_credits, clip, (30000, 1001))

    tff = FieldBased.from_param(tff, pulldown_credits) or FieldBased.from_video(clip, True)
    clip = FieldBased.ensure_presence(clip, tff)

    qtgmc_kwargs = dict[str, Any](
        SourceMatch=3, Lossless=2, TR0=2, TR1=2, TR2=3, Preset="Placebo"
    ) | (qtgmc_args or {}) | dict(FPSDivisor=1, TFF=tff.field)

    if dec is not False:  # Automatically enable dec unless set to False
        dec = any(x in clip.get_frame(0).props for x in {"VFMMatch", "TFMMatch"})

        if dec:
            warnings.warn("pulldown_credits: 'Fieldmatched clip passed to function! "
                          "dec is set to `True`. If you want to disable this, set `dec=False`!'")

    # motion vector and other values
    field_ref = frame_ref * 2
    frame_ref %= 5
    invpos = (5 - field_ref) % 5

    offset = [0, 0, -1, 1, 1][frame_ref]
    pattern = [0, 1, 0, 0, 1][frame_ref]
    direction = [-1, -1, 1, 1, 1][frame_ref]

    blksize = 16 if clip.width > 1024 or clip.height > 576 else 8
    overlap = blksize // 2

    ivtc_fps = dict(fpsnum=24000, fpsden=1001)
    ivtc_fps_div = dict(fpsnum=12000, fpsden=1001)

    # Bobbed clip
    bobbed = bob_clip or QTGMC(clip, **qtgmc_kwargs)

    InvalidFramerateError.check(pulldown_credits, bobbed, (60000, 1001))

    if interlaced:  # 60i credits. Start of ABBCD
        if dec:  # Decimate the clip instead of properly IVTC
            clean = bobbed.std.SelectEvery(5, [4 - invpos])

            if invpos > 2:
                jitter = core.std.AssumeFPS(
                    bobbed[0] * 2 + bobbed.std.SelectEvery(5, [6 - invpos, 7 - invpos]),
                    **ivtc_fps)  # type:ignore[arg-type]
            elif invpos > 1:
                jitter = core.std.AssumeFPS(
                    bobbed[0] + bobbed.std.SelectEvery(5, [2 - invpos, 6 - invpos]),
                    **ivtc_fps)  # type:ignore[arg-type]
            else:
                jitter = bobbed.std.SelectEvery(5, [1 - invpos, 2 - invpos])
        else:  # Properly IVTC
            if invpos > 1:
                clean = core.std.AssumeFPS(bobbed[0] + bobbed.std.SelectEvery(5, [6 - invpos]),
                                           **ivtc_fps_div)  # type:ignore[arg-type]
            else:
                clean = bobbed.std.SelectEvery(5, [1 - invpos])

            if invpos > 3:
                jitter = core.std.AssumeFPS(bobbed[0] + bobbed.std.SelectEvery(5, [4 - invpos, 8 - invpos]),
                                            **ivtc_fps)  # type:ignore[arg-type]
            else:
                jitter = bobbed.std.SelectEvery(5, [3 - invpos, 4 - invpos])

        jsup_pre = prefilter_to_full_range(jitter, 1.0).mv.Super(pel=2)
        jsup = jitter.mv.Super(pel=2, levels=1)
        vect_f = jsup_pre.mv.Analyse(blksize=blksize, isb=False, delta=1, overlap=overlap)
        vect_b = jsup_pre.mv.Analyse(blksize=blksize, isb=True, delta=1, overlap=overlap)
        comp = core.mv.FlowInter(jitter, jsup, vect_b, vect_f)
        out = core.std.Interleave([comp[::2], clean] if dec else [clean, comp[::2]])
        offs = 3 if dec else 2
        return out[invpos // offs:]
    else:  # 30i credits
        if pattern == 0:
            if offset == -1:
                c1 = core.std.AssumeFPS(bobbed[0] + bobbed.std.SelectEvery(
                    10, [2 + offset, 7 + offset, 5 + offset, 10 + offset]), **ivtc_fps)  # type:ignore[arg-type]
            else:
                c1 = bobbed.std.SelectEvery(10, [offset, 2 + offset, 7 + offset, 5 + offset])

            if offset == 1:
                c2 = core.std.Interleave([
                    bobbed.std.SelectEvery(10, [4]),
                    bobbed.std.SelectEvery(10, [5]),
                    bobbed[10:].std.SelectEvery(10, [0]),
                    bobbed.std.SelectEvery(10, [9])
                ])
            else:
                c2 = bobbed.std.SelectEvery(10, [3 + offset, 4 + offset, 9 + offset, 8 + offset])
        else:
            if offset == 1:
                c1 = core.std.Interleave([
                    bobbed.std.SelectEvery(10, [3]),
                    bobbed.std.SelectEvery(10, [5]),
                    bobbed[10:].std.SelectEvery(10, [0]),
                    bobbed.std.SelectEvery(10, [8])
                ])
            else:
                c1 = bobbed.std.SelectEvery(10, [2 + offset, 4 + offset, 9 + offset, 7 + offset])

            if offset == -1:
                c2 = core.std.AssumeFPS(bobbed[0] + bobbed.std.SelectEvery(
                    10, [1 + offset, 6 + offset, 5 + offset, 10 + offset]), **ivtc_fps)  # type:ignore[arg-type]
            else:
                c2 = bobbed.std.SelectEvery(10, [offset, 1 + offset, 6 + offset, 5 + offset])

        super1_pre = prefilter_to_full_range(c1, 1.0).mv.Super(pel=2)
        super1 = c1.mv.Super(pel=2, levels=1)
        vect_f1 = super1_pre.mv.Analyse(blksize=blksize, isb=False, delta=1, overlap=overlap)
        vect_b1 = super1_pre.mv.Analyse(blksize=blksize, isb=True, delta=1, overlap=overlap)
        fix1 = c1.mv.FlowInter(super1, vect_b1, vect_f1, time=50 + direction * 25).std.SelectEvery(4, [0, 2])

        super2_pre = prefilter_to_full_range(c2, 1.0).mv.Super(pel=2)
        super2 = c2.mv.Super(pel=2, levels=1)
        vect_f2 = super2_pre.mv.Analyse(blksize=blksize, isb=False, delta=1, overlap=overlap)
        vect_b2 = super2_pre.mv.Analyse(blksize=blksize, isb=True, delta=1, overlap=overlap)
        fix2 = c2.mv.FlowInter(super2, vect_b2, vect_f2).std.SelectEvery(4, [0, 2])

        return core.std.Interleave([fix1, fix2] if pattern == 0 else [fix2, fix1])
