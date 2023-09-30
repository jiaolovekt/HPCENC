"""
This module and original idea is by cid-chan (Sarah <cid@cid-chan.moe>)
"""

from __future__ import annotations

from typing import Any, Callable, Generator, Generic

import vapoursynth as vs
from vstools import T0, CustomTypeError, CustomValueError, T

from .types import AnyCoroutine, FrameRequest

core = vs.core


__all__ = [
    'Atom',
    'SingleFrameRequest', 'GatherRequests',
    'coro2node'
]


UNWRAP_NAME = '__vspyplugin_unwrap'


class Atom(Generic[T]):
    value: T | None

    def __init__(self) -> None:
        self.value = None

    def set(self, value: T) -> None:
        self.value = value

    def unset(self) -> None:
        self.value = None


class SingleFrameRequest(FrameRequest):
    def __init__(self, clip: vs.VideoNode, frame_no: int) -> None:
        self.clip = clip
        self.frame_no = frame_no

    def __await__(self) -> Generator[SingleFrameRequest, None, vs.VideoFrame]:
        return (yield self)  # type: ignore

    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int,
        continuation: Callable[[Any], vs.VideoNode]
    ) -> vs.VideoNode:
        req_clip = self.clip[self.frame_no]

        return clip.std.FrameEval(
            lambda n, f: continuation(f), [req_clip]
        )


class GatherRequests(Generic[T], FrameRequest):
    def __init__(self, coroutines: tuple[AnyCoroutine[T0, T], ...]) -> None:
        if len(coroutines) <= 1:
            raise CustomValueError('You need to pass at least 2 coroutines!', self.__class__)

        self.coroutines = coroutines

    def __await__(self) -> Generator[GatherRequests[T], None, tuple[T, ...]]:
        return (yield self)  # type: ignore

    @staticmethod
    def _unwrap(frame: vs.VideoFrame, atom: Atom[T]) -> vs.VideoFrame | T | None:
        if frame.props.get(UNWRAP_NAME, False):
            return atom.value

        return frame

    def unwrap_coros(self, clip: vs.VideoNode, frame_no: int) -> tuple[list[vs.VideoNode], list[Atom[T]]]:
        return zip(*[  # type: ignore
            _coro2node_wrapped(clip, frame_no, coro) for coro in self.coroutines
        ])

    def wrap_frames(self, frames: list[vs.VideoFrame], atoms: list[Atom[T]]) -> tuple[vs.VideoFrame | T | None, ...]:
        return tuple(
            self._unwrap(frame, atom) for frame, atom in zip(frames, atoms)
        )

    def build_frame_eval(
        self, clip: vs.VideoNode, frame_no: int,
        continuation: Callable[[tuple[vs.VideoFrame | T | None, ...]], vs.VideoNode]
    ) -> vs.VideoNode:
        clips, atoms = self.unwrap_coros(clip, frame_no)

        def _apply(n: int, f: list[vs.VideoFrame]) -> vs.VideoNode:
            return continuation(self.wrap_frames(f, atoms))

        return clip.std.FrameEval(_apply, clips)


def _wrapped_modify_frame(blank_clip: vs.VideoNode) -> Callable[[vs.VideoFrame], vs.VideoNode]:
    def _wrap_frame(frame: vs.VideoFrame) -> vs.VideoNode:
        def _return_frame(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            return frame

        return blank_clip.std.ModifyFrame(blank_clip, _return_frame)

    return _wrap_frame


def _coro2node_wrapped(
    base_clip: vs.VideoNode, frameno: int, coro: AnyCoroutine[T0, T]
) -> tuple[vs.VideoNode, Atom[T]]:
    atom = Atom[T]()
    return coro2node(base_clip, frameno, coro, atom), atom


def coro2node(
    base_clip: vs.VideoNode, frameno: int, coro: AnyCoroutine[T0, T], wrap: Atom[T] | None = None
) -> vs.VideoNode:
    assert base_clip.format

    props_clip = base_clip.std.BlankClip()
    blank_clip = core.std.BlankClip(
        length=1, fpsnum=1, fpsden=1, keep=True,
        width=base_clip.width, height=base_clip.height,
        format=base_clip.format.id
    )

    _wrap_frame = _wrapped_modify_frame(blank_clip)

    def _continue(wrapped_value: T0 | None) -> vs.VideoNode:
        if wrap:
            wrap.unset()

        try:
            next_request = coro.send(wrapped_value)
        except StopIteration as e:
            value = e.value

            if isinstance(value, vs.VideoNode):
                return value  # type: ignore

            if isinstance(value, vs.VideoFrame):
                return _wrap_frame(value)

            if not wrap:
                raise CustomTypeError('You can only return a VideoFrame or VideoNode!', coro2node)

            wrap.set(value)

            return props_clip.std.SetFrameProp(UNWRAP_NAME, intval=True)
        except Exception as e:
            raise e

        return next_request.build_frame_eval(base_clip, frameno, _continue)

    return _continue(None)
