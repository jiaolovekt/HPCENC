from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, Generic, Type, cast, overload, ClassVar

import vapoursynth as vs
from vstools import CustomIndexError, CustomTypeError, CustomValueError, InvalidSubsamplingError, copy_signature

from .abstracts import PyPluginBackendBase
from .backends import PyBackend
from .coroutines import frame_eval_async, get_frame, get_frames
from .types import DT_T, FD_T, FilterMode, OutputFunc_T

__all__ = [
    'PyPluginBase', 'PyPlugin',
    'PyPluginOptions',
    'PyPluginUnavailableBackendBase',
    'PyPluginUnavailableBackend'
]


@dataclass
class PyPluginOptions:
    force_precision: int | None = None
    shift_chroma: bool = False

    @overload
    def norm_clip(self, clip: vs.VideoNode) -> vs.VideoNode:
        ...

    @overload
    def norm_clip(self, clip: None) -> None:
        ...

    def norm_clip(self, clip: vs.VideoNode | None) -> vs.VideoNode | None:
        if not clip:
            return clip

        assert (fmt := clip.format)

        if self.force_precision:
            if fmt.sample_type is not vs.FLOAT or fmt.bits_per_sample != self.force_precision:
                clip = clip.resize.Point(
                    format=fmt.replace(sample_type=vs.FLOAT, bits_per_sample=self.force_precision).id,
                    dither_type='none'
                )

        if self.shift_chroma:
            if fmt.sample_type is not vs.FLOAT and self.force_precision != 32:
                raise CustomValueError(
                    f'{self.__class__.__name__}: You need to have a clip with float sample type for shift_chroma=True!'
                )

            if fmt.num_planes == 3:
                clip = clip.std.Expr(['', 'x 0.5 +'])

        return clip

    def ensure_output(self, plugin: PyPlugin[FD_T], clip: vs.VideoNode) -> vs.VideoNode:
        assert plugin.ref_clip.format
        assert (fmt := clip.format)

        if plugin.out_format.id != plugin.ref_clip.format.id:
            clip = clip.resize.Bicubic(format=plugin.out_format.id, dither_type='none')

        if self.shift_chroma and fmt.num_planes == 3:
            clip = clip.std.Expr(['', 'x 0.5 -'])

        return clip


class PyPluginBase(Generic[FD_T, DT_T], PyPluginBackendBase[DT_T]):
    if not TYPE_CHECKING:
        __slots__ = (
            'filter_data', 'clips', 'ref_clip', 'fd', '_input_per_plane'
        )

    debug: bool = False
    backend: ClassVar[PyBackend] = PyBackend.NONE
    filter_data: Type[FD_T]
    filter_mode: FilterMode

    options: PyPluginOptions

    input_per_plane: bool | list[bool]
    output_per_plane: bool
    channels_last: bool

    min_clips: int
    max_clips: int

    clips: list[vs.VideoNode]
    ref_clip: vs.VideoNode
    out_format: vs.VideoFormat

    fd: FD_T

    def __class_getitem__(cls, fdata: Type[FD_T] | None = None) -> Type[PyPlugin[FD_T]]:
        class PyPluginInnerClass(cls):  # type: ignore
            filter_data = fdata

        return PyPluginInnerClass

    def __init__(
        self,
        ref_clip: vs.VideoNode,
        clips: list[vs.VideoNode] | None = None,
        *,
        filter_mode: FilterMode | None = None,
        options: PyPluginOptions | None = None,
        input_per_plane: bool | list[bool] | None = None,
        output_per_plane: bool | None = None,
        channels_last: bool | None = None,
        min_clips: int | None = None,
        max_clips: int | None = None,
        **kwargs: Any
    ) -> None:
        assert ref_clip.format

        arguments = [
            (filter_mode, 'filter_mode', FilterMode.Parallel),
            (options, 'options', PyPluginOptions()),
            (input_per_plane, 'input_per_plane', True),
            (output_per_plane, 'output_per_plane', True),
            (channels_last, 'channels_last', False),
            (min_clips, 'min_clips', 1),
            (max_clips, 'max_clips', -1)
        ]

        for value, name, default in arguments:
            if value is not None:
                setattr(self, name, value)
            elif not hasattr(self, name):
                setattr(self, name, default)

        self.out_format = ref_clip.format

        self.ref_clip = self.options.norm_clip(ref_clip)

        self.clips = [self.options.norm_clip(clip) for clip in clips] if clips else []

        self_annotations = self.__annotations__.keys()

        for name, value in list(kwargs.items()):
            if name in self_annotations:
                setattr(self, name, value)
                kwargs.pop(name)

        if callable(self.filter_data):
            self.fd = self.filter_data(**kwargs)  # type: ignore
        else:
            self.fd = None  # type: ignore

        n_clips = 1 + len(self.clips)

        inputs_per_plane = self.input_per_plane

        if not isinstance(inputs_per_plane, list):
            inputs_per_plane = [inputs_per_plane]

        for _ in range((1 + len(self.clips)) - len(inputs_per_plane)):
            inputs_per_plane.append(inputs_per_plane[-1])

        self._input_per_plane = inputs_per_plane

        self.is_single_plane = [
            bool(clip.format and clip.format.num_planes == 1)
            for clip in (self.ref_clip, *self.clips)
        ]

        if n_clips < self.min_clips or (self.max_clips > 0 and n_clips > self.max_clips):
            max_clips_str = 'inf' if self.max_clips == -1 else self.max_clips
            raise CustomIndexError(
                f'You must pass {self.min_clips} <= n clips <= {max_clips_str}!', self.__class__
            )

        if not self.output_per_plane and (ref_clip.format.subsampling_w or ref_clip.format.subsampling_h):
            raise CustomValueError(
                'You can\'t have output_per_plane=False with a subsampled clip!', self.__class__
            )

        for idx, clip, ipp in zip(count(-1), (self.ref_clip, *self.clips), self._input_per_plane):
            assert clip.format
            if not ipp and (clip.format.subsampling_w or clip.format.subsampling_h):
                raise InvalidSubsamplingError(
                    self.__class__,
                    'You can\'t have input_per_plane=False with a subsampled clip! ({clip_type})',
                    clip_type='Ref Clip' if idx == -1 else f'Clip Index: {idx}'
                )

    if TYPE_CHECKING:
        def _invoke_func(self) -> OutputFunc_T:
            ...

    @PyPluginBackendBase.ensure_output
    def invoke(self) -> vs.VideoNode:
        output_func = self._invoke_func()

        modify_frame_partial = partial(
            vs.core.std.ModifyFrame, self.ref_clip, (self.ref_clip, *self.clips), output_func
        )

        if self.filter_mode is FilterMode.Serial:
            output = modify_frame_partial()
        elif self.filter_mode is FilterMode.Parallel:
            output = self.ref_clip.std.FrameEval(lambda n: modify_frame_partial())
        else:
            if self.clips:
                output_func_multi = cast(Callable[[tuple[vs.VideoFrame, ...], int], vs.VideoFrame], output_func)

                @frame_eval_async(self.ref_clip)
                async def output(n: int) -> vs.VideoFrame:
                    return output_func_multi(await get_frames(self.ref_clip, *self.clips, frame_no=n), n)
            else:
                output_func_single = cast(Callable[[vs.VideoFrame, int], vs.VideoFrame], output_func)

                @frame_eval_async(self.ref_clip)
                async def output(n: int) -> vs.VideoFrame:
                    return output_func_single(await get_frame(self.ref_clip, n), n)

        return output

    def __call__(self, func: Callable[..., Any]) -> vs.VideoNode:
        this_args = {'self', 'f', 'src', 'dst', 'plane', 'n'}

        annotations = set(func.__annotations__.keys()) - {'return'}

        if not annotations:
            raise CustomTypeError(f'{self.__class__.__name__}: You must type hint the function!', self.__class__)

        if annotations - this_args:
            raise CustomTypeError(f'{self.__class__.__name__}: Unknown arguments specified!', self.__class__)

        miss_args = this_args - annotations

        if 'self' in annotations:
            func = partial(func, self)
            annotations.remove('self')

        if not miss_args:
            self.process_SingleSrcIPP = self.process_SingleSrcIPF = func
            self.process_MultiSrcIPP = self.process_MultiSrcIPF = func
        else:
            def _wrapper_ipf(src: Any, dst: Any, f: vs.VideoFrame, n: int) -> None:
                curr_locals = locals()
                func(**{name: curr_locals[name] for name in annotations})

            def _wrapper_ipp(src: Any, dst: Any, f: vs.VideoFrame, plane: int, n: int) -> None:
                curr_locals = locals()
                func(**{name: curr_locals[name] for name in annotations})

            self.process_SingleSrcIPF = self.process_MultiSrcIPF = _wrapper_ipf
            self.process_SingleSrcIPP = self.process_MultiSrcIPP = _wrapper_ipp

        return self.invoke()


class PyPlugin(PyPluginBase[FD_T, memoryview]):
    def _invoke_func(self) -> OutputFunc_T:
        assert self.ref_clip.format

        def _stack_frame(frame: vs.VideoFrame, idx: int) -> memoryview | list[memoryview]:
            return frame[0] if self.is_single_plane[idx] else [frame[p] for p in {0, 1, 2}]

        output_func: OutputFunc_T

        if self.output_per_plane:
            if self.clips:
                assert self.process_MultiSrcIPP
                func_MultiSrcIPP = self.process_MultiSrcIPP

                def output_func(f: tuple[vs.VideoFrame, ...], n: int) -> vs.VideoFrame:
                    fout = f[0].copy()

                    pre_stacked_clips = {
                        idx: _stack_frame(frame, idx)
                        for idx, frame in enumerate(f)
                        if not self._input_per_plane[idx]
                    }

                    for p in range(fout.format.num_planes):
                        inputs_data = [
                            frame[p] if self._input_per_plane[idx] else pre_stacked_clips[idx]
                            for idx, frame in enumerate(f)
                        ]

                        func_MultiSrcIPP(inputs_data, fout[p], fout, p, n)  # type: ignore

                    return fout
            else:
                assert self.process_SingleSrcIPP
                func_SingleSrcIPP = self.process_SingleSrcIPP

                if self._input_per_plane[0]:
                    def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                        fout = f.copy()

                        for p in range(fout.format.num_planes):
                            func_SingleSrcIPP(f[p], fout[p], fout, p, n)

                        return fout
                else:
                    def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                        fout = f.copy()

                        pre_stacked_clip = _stack_frame(f, 0)

                        for p in range(fout.format.num_planes):
                            func_SingleSrcIPP(pre_stacked_clip, fout[p], fout, p, n)  # type: ignore

                        return fout
        else:
            if self.clips:
                assert self.process_MultiSrcIPF
                func_MultiSrcIPF = self.process_MultiSrcIPF

                def output_func(f: tuple[vs.VideoFrame, ...], n: int) -> vs.VideoFrame:
                    fout = f[0].copy()

                    src_arrays = [_stack_frame(frame, idx) for idx, frame in enumerate(f)]

                    func_MultiSrcIPF(src_arrays, fout, fout, n)  # type: ignore

                    return fout
            else:
                if self.ref_clip.format.num_planes == 1:
                    if self.process_SingleSrcIPP:
                        func_SingleSrcIPP = self.process_SingleSrcIPP

                        def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                            fout = f.copy()

                            func_SingleSrcIPP(f[0], fout[0], fout, 0, n)

                            return fout
                    else:
                        assert self.process_SingleSrcIPF
                        func_SingleSrcIPF = self.process_SingleSrcIPF

                        def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                            fout = f.copy()

                            func_SingleSrcIPF(f[0], fout, fout, n)

                            return fout
                else:
                    assert self.process_SingleSrcIPF
                    func_SingleSrcIPF = self.process_SingleSrcIPF

                    def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                        fout = f.copy()

                        func_SingleSrcIPF(f, fout, fout, n)  # type: ignore

                        return fout

        return output_func


class PyPluginUnavailableBackendBase(PyPluginBase[FD_T, DT_T]):
    @copy_signature(PyPlugin.__init__)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from .exceptions import UnavailableBackend
        raise UnavailableBackend(self.backend, self.__class__)


class PyPluginUnavailableBackend(PyPluginUnavailableBackendBase[FD_T, Any]):
    ...
