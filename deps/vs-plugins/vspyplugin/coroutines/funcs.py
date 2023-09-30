"""
This module and original idea is by cid-chan (Sarah <cid@cid-chan.moe>)
"""

from __future__ import annotations

from typing import Iterable

import vapoursynth as vs
from vstools import T0, T

from .coros import GatherRequests, SingleFrameRequest
from .types import AnyCoroutine

__all__ = [
    'get_frame', 'get_frames', 'get_frames_shifted',
    'gather', 'wait'
]


async def get_frame(clip: vs.VideoNode, frame_no: int) -> vs.VideoFrame:
    return await SingleFrameRequest(clip, frame_no)


async def get_frames(*clips: vs.VideoNode, frame_no: int) -> tuple[vs.VideoFrame, ...]:
    return await wait(get_frame(clip, frame_no) for clip in clips)


async def get_frames_shifted(
    clip: vs.VideoNode, frame_no: int, shifts: int | tuple[int, int] | Iterable[int] = (-1, 1)
) -> tuple[vs.VideoFrame, ...]:
    if isinstance(shifts, int):
        shifts = (-shifts, shifts)

    if isinstance(shifts, tuple):
        start, stop = shifts
        step = -1 if stop < start else 1
        shifts = range(start, stop + step, step)

    return await wait(get_frame(clip, frame_no + shift) for shift in shifts)


async def gather(*coroutines: AnyCoroutine[T0, T]) -> tuple[T, ...]:
    return await GatherRequests(coroutines)


async def wait(coroutines: Iterable[AnyCoroutine[T0, T]]) -> tuple[T, ...]:
    return await GatherRequests(tuple(coroutines))
