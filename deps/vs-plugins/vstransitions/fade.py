from __future__ import annotations

from vstools import FrameRangeN, FrameRangesN, insert_clip, normalize_ranges, vs

from .easing import EasingT, Linear

__all__ = [
    'fade', 'fade_freeze',

    'fade_in', 'fade_out',
    'fade_in_freeze', 'fade_out_freeze',

    'crossfade',

    'fade_ranges'
]


def fade(
    clipa: vs.VideoNode, clipb: vs.VideoNode, invert: bool, start: int,
    end: int, function: EasingT = Linear
) -> vs.VideoNode:
    clipa_cut = clipa[start:end]
    clipb_cut = clipb[start:end]

    if invert:
        fade = crossfade(clipa_cut, clipb_cut, function)
    else:
        fade = crossfade(clipb_cut, clipa_cut, function)

    return insert_clip(clipa, fade, start)


def fade_freeze(
    clipa: vs.VideoNode, clipb: vs.VideoNode, invert: bool, start: int,
    end: int, function: EasingT = Linear
) -> vs.VideoNode:
    start_f, end_f = (start, end) if invert else (end, start)

    length = end - start + 1

    return fade(
        insert_clip(clipa, clipa[start_f] * length, start),
        insert_clip(clipb, clipb[end_f] * length, start),
        invert, start, end, function
    )


def fade_in(clip: vs.VideoNode, start: int, end: int, function: EasingT = Linear) -> vs.VideoNode:
    return fade(clip, clip.std.BlankClip(), False, start, end, function)


def fade_out(clip: vs.VideoNode, start: int, end: int, function: EasingT = Linear) -> vs.VideoNode:
    return fade(clip, clip.std.BlankClip(), True, start, end, function)


def fade_in_freeze(clip: vs.VideoNode, start: int, end: int, function: EasingT = Linear) -> vs.VideoNode:
    return fade_in(insert_clip(clip, clip[end] * (end - start + 1), start), start, end, function)


def fade_out_freeze(clip: vs.VideoNode, start: int, end: int, function: EasingT = Linear) -> vs.VideoNode:
    return fade_out(insert_clip(clip, clip[start] * (end - start + 1), start), start, end, function)


def crossfade(
    clipa: vs.VideoNode, clipb: vs.VideoNode, function: EasingT,
    debug: bool | int | tuple[int, int] = False
) -> vs.VideoNode:
    assert clipa.format and clipb.format

    if not clipa.height == clipb.height and clipa.width == clipb.width and clipa.format.id == clipb.format.id:
        raise ValueError('crossfade: Both clips must have the same length, dimensions and format.')

    ease_function = function(0, 1, clipa.num_frames)

    def _fading(n: int) -> vs.VideoNode:
        weight = ease_function.ease(n)
        merge = clipa.std.Merge(clipb, weight)
        return merge.text.Text(str(weight), 9, 2) if debug else merge

    return clipa.std.FrameEval(_fading)


def fade_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode, ranges: FrameRangeN | FrameRangesN,
    fade_length: int = 5, ease_func: EasingT = Linear
) -> vs.VideoNode:
    nranges = normalize_ranges(clip_b, ranges)
    nranges = [(s - fade_length, e + fade_length) for s, e in nranges]
    nranges = normalize_ranges(clip_b, nranges)

    franges = [range(s, e + 1) for s, e in nranges]

    ease_function = ease_func(0, 1, fade_length)

    def _fading(n: int) -> vs.VideoNode:
        frange: range | None = next((x for x in franges if n in x), None)

        if frange is None:
            return clip_a

        if frange.start + fade_length >= n >= frange.start:
            weight = ease_function.ease(n - frange.start)

            return clip_a.std.Merge(clip_b, weight)

        if frange.stop - fade_length <= n <= frange.stop:
            weight = ease_function.ease(frange.stop - n)

            return clip_b.std.Merge(clip_a, 1 - weight)

        return clip_b

    return clip_a.std.FrameEval(_fading)
