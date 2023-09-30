"""
This module and original idea is by cid-chan (Sarah <cid@cid-chan.moe>)
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine

import vapoursynth as vs
from vstools import T0, CustomNotImplementedError, T


class FrameRequest:
    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int, continuation: Callable[[Any], vs.VideoNode]
    ) -> vs.VideoNode:
        raise CustomNotImplementedError(func=self.__class__)


AnyCoroutine = Coroutine[FrameRequest, T0 | None, T]
