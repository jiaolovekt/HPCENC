from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, cast

import vapoursynth as vs
from vstools import CustomValueError, copy_signature, get_resolutions

from .backends import PyBackend
from .base import PyPlugin, PyPluginUnavailableBackendBase
from .types import DT_T, FD_T, OutputFunc_T

__all__ = [
    'PyPluginCupyBase', 'PyPluginCupy'
]

this_backend = PyBackend.CUPY
this_backend.set_dependencies({'cupy': '11.0.0'}, PyBackend.NUMPY)


try:
    if PyBackend.is_cli:
        raise ModuleNotFoundError

    from cupy_backends.cuda.api import runtime

    import cupy as cp
    from cupy import cuda
    from numpy.typing import NDArray

    from .numpy import NDT_T, PyPluginNumpy, PyPluginNumpyBase

    if TYPE_CHECKING:
        concatenate: Callable[..., NDT_T]
    else:
        from cupy._core import concatenate_method as concatenate

    class PyPluginCupyBase(PyPluginNumpyBase[FD_T, NDT_T]):
        backend = this_backend

        cuda_num_streams: int = 0

        def _synchronize(self, events: list[cuda.Event]) -> None:
            for event in events:
                self.cuda_default_stream.wait_event(event)

            self.cuda_device.synchronize()

        def _memcpy_async(self, dst_ptr: int, src_ptr: int, amount: int, kind: int) -> list[cuda.Event]:
            offset = 0

            stop_events = []
            for stream in self.cuda_streams:
                with stream:
                    runtime.memcpyAsync(dst_ptr + offset, src_ptr + offset, amount, kind, stream.ptr)

                    stop_events.append(cuda.Event())

                    offset += amount

            return stop_events

        def to_device(self, f: vs.VideoFrame, idx: int, plane: int) -> NDT_T:
            self._memcpy_func(
                int(self.src_arrays[plane][idx].data),
                cast(int, f.get_read_ptr(plane).value),
                self.src_data_lengths[plane][idx],
                runtime.memcpyHostToDevice
            )

            return self.src_arrays[plane][idx]

        def from_device(self, dst: vs.VideoFrame) -> vs.VideoFrame:
            if self.cuda_num_streams:
                events = []
                for plane in range(dst.format.num_planes):
                    events.extend(
                        self._memcpy_async(
                            cast(int, dst.get_write_ptr(plane).value),
                            self._dst_pointers[plane],
                            self.out_data_lengths[plane],
                            runtime.memcpyDeviceToHost
                        )
                    )
                self._synchronize(events)
            else:
                for plane in range(dst.format.num_planes):
                    runtime.memcpy(
                        cast(int, dst.get_write_ptr(plane).value),
                        self._dst_pointers[plane],
                        self.out_data_lengths[plane],
                        runtime.memcpyDeviceToHost
                    )

            return dst

        @staticmethod
        def alloc_plane_arrays(clip: vs.VideoNode | vs.VideoFrame, fill: int | float | None = 0) -> list[NDT_T]:
            assert clip.format

            function = cp.empty if fill is None else cp.zeros if fill == 0 else partial(cp.full, fill_value=fill)

            return [
                function((height, width), dtype=PyPluginNumpy.get_dtype(clip), order='C')
                for _, width, height in get_resolutions(clip)
            ]

        def _get_data_len(self, arr: NDT_T) -> int:
            return round(super()._get_data_len(arr) / max(1, self.cuda_num_streams))

        @copy_signature(PyPlugin.__init__)
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

            if self.cuda_num_streams != 0 and self.cuda_num_streams < 2:
                raise CustomValueError('"cuda_num_streams" must be 0 or >= 2!', self.__class__)

            self.cuda_device = cuda.Device()
            self.cuda_memory_pool = cuda.MemoryPool()

            cuda.set_allocator(self.cuda_memory_pool.malloc)

            self.cuda_default_stream = cuda.Stream(non_blocking=True)
            self.cuda_streams = [cuda.Stream(non_blocking=True) for _ in range(self.cuda_num_streams)]

            self.cuda_is_101 = 10010 <= runtime.runtimeGetVersion()
            self._memcpy_func = self._memcpy_async if self.cuda_num_streams else runtime.memcpy

        def allocate_src_dst_memory(self) -> None:
            assert self.ref_clip.format

            src_arrays = [self.alloc_plane_arrays(clip) for clip in (self.ref_clip, *self.clips)]
            self.src_arrays = [
                [array[plane] for array in src_arrays] for plane in range(self.ref_clip.format.num_planes)
            ]
            self.src_data_lengths = [[self._get_data_len(a) for a in arr] for arr in self.src_arrays]

            self.out_arrays = self.alloc_plane_arrays(self.ref_clip)
            self.out_data_lengths = [self._get_data_len(arr) for arr in self.out_arrays]

            if self.output_per_plane:
                self.dst_stacked_planes = self.alloc_plane_arrays(self.ref_clip)
            else:
                shape = (self.ref_clip.height, self.ref_clip.width)

                shape_channels: tuple[int, ...]
                if self.is_single_plane[0]:
                    shape_channels = shape + (1, )
                elif self.channels_last:
                    shape_channels = shape + (3, )
                else:
                    shape_channels = (3, ) + shape

                planes_slices = self.get_planes_slices(self.ref_clip, self.channels_last)

                self.dst_stacked_arr = cp.zeros(shape_channels, self.get_dtype(self.ref_clip))
                self.dst_stacked_planes = [
                    self.dst_stacked_arr[planes_slices[plane]]
                    for plane in range(self.ref_clip.format.num_planes)
                ]

            self._dst_pointers = [int(source.data) for source in self.dst_stacked_planes]

    class PyPluginCupy(PyPluginCupyBase[FD_T, NDArray[Any]]):
        def _invoke_func(self) -> OutputFunc_T:
            assert self.ref_clip.format

            self.allocate_src_dst_memory()

            if self.ref_clip.format.num_planes == 1:
                def _stack_whole_frame(frame: vs.VideoFrame, idx: int) -> NDArray[Any]:
                    return self.to_device(frame, idx, 0)
            elif self.channels_last:
                stack_slice = (slice(None), slice(None), None)

                def _stack_whole_frame(frame: vs.VideoFrame, idx: int) -> NDArray[Any]:
                    return concatenate([
                        self.to_device(frame, idx, plane)[stack_slice] for plane in {0, 1, 2}
                    ], axis=2)
            else:
                def _stack_whole_frame(frame: vs.VideoFrame, idx: int) -> NDArray[Any]:
                    return concatenate([
                        self.to_device(frame, idx, plane) for plane in {0, 1, 2}
                    ], axis=0)

            def _stack_frame(frame: vs.VideoFrame, idx: int) -> NDArray[Any]:
                if self.is_single_plane[idx]:
                    return self.to_device(frame, idx, 0)

                return _stack_whole_frame(frame, idx)

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
                                self.to_device(frame, idx, p)
                                if self._input_per_plane[idx]
                                else pre_stacked_clips[idx]
                                for idx, frame in enumerate(f)
                            ]

                            func_MultiSrcIPP(inputs_data, self.dst_stacked_planes[p], fout, p, n)

                        return self.from_device(fout)
                else:
                    assert self.process_SingleSrcIPP
                    func_SingleSrcIPP = self.process_SingleSrcIPP

                    if self._input_per_plane[0]:
                        def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                            fout = f.copy()

                            for p in range(fout.format.num_planes):
                                func_SingleSrcIPP(self.to_device(f, 0, p), self.dst_stacked_planes[p], fout, p, n)

                            return self.from_device(fout)
                    else:
                        def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                            fout = f.copy()

                            pre_stacked_clip = _stack_frame(f, 0)

                            for p in range(fout.format.num_planes):
                                func_SingleSrcIPP(pre_stacked_clip, self.dst_stacked_planes[p], fout, p, n)

                            return self.from_device(fout)
            else:
                if self.clips:
                    assert self.process_MultiSrcIPF
                    func_MultiSrcIPF = self.process_MultiSrcIPF

                    def output_func(f: tuple[vs.VideoFrame, ...], n: int) -> vs.VideoFrame:
                        fout = f[0].copy()

                        func_MultiSrcIPF(
                            [_stack_frame(frame, idx) for idx, frame in enumerate(f)],
                            self.dst_stacked_arr, fout, n
                        )

                        return self.from_device(fout)
                else:
                    assert self.process_SingleSrcIPF
                    func_SingleSrcIPF = self.process_SingleSrcIPF

                    def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                        fout = f.copy()

                        func_SingleSrcIPF(_stack_whole_frame(f, 0), self.dst_stacked_arr, fout, n)

                        return self.from_device(fout)

            return output_func

    this_backend.set_available(True)
except ModuleNotFoundError as e:
    this_backend.set_available(False, e)

    class PyPluginCupyBase(PyPluginUnavailableBackendBase[FD_T, DT_T]):  # type: ignore
        backend = this_backend

    class PyPluginCupy(PyPluginCupyBase[FD_T]):  # type: ignore
        ...
