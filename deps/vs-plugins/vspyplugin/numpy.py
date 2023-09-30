from __future__ import annotations

from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import vapoursynth as vs
from vstools import copy_signature, get_resolutions

from .backends import PyBackend
from .base import PyPlugin, PyPluginBase, PyPluginUnavailableBackendBase
from .types import DT_T, FD_T, OutputFunc_T

__all__ = [
    'PyPluginNumpyBase', 'PyPluginNumpy',
    'NDT_T'
]

this_backend = PyBackend.NUMPY
this_backend.set_dependencies({'numpy': '1.23.5'}, PyBackend.NONE)

try:
    if PyBackend.is_cli:
        raise ModuleNotFoundError

    from ctypes import POINTER
    from ctypes import _cast as ctypes_cast  # type: ignore
    from ctypes import _Pointer as PointerType
    from ctypes import memmove

    import numpy as np
    from numpy import dtype
    from numpy.core._multiarray_umath import asarray
    from numpy.ctypeslib import _scalar_type_map  # type: ignore
    from numpy.typing import NDArray

    NDT_T = TypeVar('NDT_T', bound=NDArray[Any])

    if TYPE_CHECKING:
        concatenate: Callable[..., NDT_T]
    else:
        from numpy.core.numeric import concatenate

    _cache_dtypes = dict[int, dtype[Any]]()
    _cache_arr_dtypes = dict[int, type[PointerType]]()  # type: ignore

    class PyPluginNumpyBase(PyPluginBase[FD_T, NDT_T]):
        backend = this_backend

        @classmethod
        def to_host(cls, f: vs.VideoFrame, plane: int, strict: bool = False) -> NDT_T:
            ptr = f.get_read_ptr(plane)
            ctype = cls.get_arr_ctype_from_clip(f, plane, strict)

            return asarray(ctypes_cast(ptr, ptr, ctype).contents)  # type: ignore

        @classmethod
        def from_host(
            self, src: NDArray[Any], dst: vs.VideoFrame, planes_slices: tuple[slice, ...]
        ) -> None:
            for plane in range(dst.format.num_planes):
                src_ptr, length = src[planes_slices[plane]].__array_interface__['data']
                memmove(dst.get_write_ptr(plane), src_ptr, length)

        @classmethod
        def get_arr_ctype_from_clip(
            cls, clip: vs.VideoNode | vs.VideoFrame, plane: int, strict: bool = False
        ) -> type[PointerType]:  # type: ignore
            key = hash((clip.format.id, clip.width, clip.height, plane))  # type: ignore

            if key not in _cache_arr_dtypes:
                if plane == 0:
                    _cache_arr_dtypes[key] = cls.get_arr_ctype(
                        clip.width, clip.height, cls.get_dtype(clip, strict)
                    )
                else:
                    assert clip.format

                    _cache_arr_dtypes[key] = cls.get_arr_ctype(
                        clip.width >> clip.format.subsampling_h,
                        clip.height >> clip.format.subsampling_h,
                        cls.get_dtype(clip, strict)
                    )

            return _cache_arr_dtypes[key]

        @classmethod
        @lru_cache
        def get_arr_ctype(cls, width: int, height: int, data_type: dtype[Any]) -> type[PointerType]:  # type: ignore
            ctypes_type = _scalar_type_map[data_type.newbyteorder('=')]

            cast_type = POINTER(ctypes_type)

            element_type = height * (width * cast_type._type_)

            return POINTER(element_type)

        @staticmethod
        def get_dtype(clip: vs.VideoNode | vs.VideoFrame, strict: bool = False) -> dtype[Any]:
            fmt = cast(vs.VideoFormat, clip.format)

            if fmt.id not in _cache_dtypes:
                stype = 'float' if fmt.sample_type is vs.FLOAT else 'uint'
                bits = (fmt.bytes_per_sample * 8) if strict else fmt.bits_per_sample
                _cache_dtypes[fmt.id] = dtype(f'{stype}{bits}')

            return _cache_dtypes[fmt.id]

        @staticmethod
        def alloc_plane_arrays(clip: vs.VideoNode | vs.VideoFrame, fill: int | float | None = 0) -> list[NDT_T]:
            assert clip.format

            function = np.empty if fill is None else np.zeros if fill == 0 else partial(np.full, fill_value=fill)

            return [
                function((height, width), dtype=PyPluginNumpy.get_dtype(clip), order='C')  # type: ignore
                for _, width, height in get_resolutions(clip)
            ]

        def _get_data_len(self, arr: NDT_T) -> int:
            return arr.shape[0] * arr.shape[1] * arr.dtype.itemsize

        @copy_signature(PyPlugin.__init__)
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

        @classmethod
        def get_planes_slices(
            cls, clip: vs.VideoNode | vs.VideoFrame, channels_last: bool = False, n_channels: int = 3
        ) -> tuple[slice, ...]:
            assert clip.format

            no_slice = slice(None, None, None)

            return cast(
                tuple[slice, ...],
                tuple(
                    (
                        [plane, no_slice][channels_last],
                        *([no_slice] * (n_channels - 2)),
                        [no_slice, plane][channels_last]
                    ) for plane in range(clip.format.num_planes)
                )
            )

        @classmethod
        def get_stack_whole_frame_func(
            cls, channels_last: bool, n_channels: int = 3, strict: bool = False
        ) -> Callable[[vs.VideoFrame], NDT_T]:
            if channels_last:
                axis = n_channels - 1

                stack_slice = (slice(None), slice(None), None)

                def _stack_whole_frame(frame: vs.VideoFrame) -> NDT_T:
                    return concatenate([
                        cls.to_host(frame, plane, strict)[stack_slice] for plane in {0, 1, 2}
                    ], axis=axis)
            else:
                axis = n_channels - 3

                def _stack_whole_frame(frame: vs.VideoFrame) -> NDT_T:
                    return concatenate([
                        cls.to_host(frame, plane, strict) for plane in {0, 1, 2}
                    ], axis=axis)

            return _stack_whole_frame

        def _invoke_func(self) -> OutputFunc_T:
            assert self.ref_clip.format

            planes_idx = self.get_planes_slices(self.ref_clip, self.channels_last)

            stack_whole_frame = self.get_stack_whole_frame_func(self.channels_last)

            def _stack_frame(frame: vs.VideoFrame, idx: int) -> NDT_T:
                if self.is_single_plane[idx]:
                    return self.to_host(frame, 0)

                return stack_whole_frame(frame)

            if not self.output_per_plane:
                dst_stacked_arr = np.zeros(
                    (self.ref_clip.height, self.ref_clip.width, 3), self.get_dtype(self.ref_clip)
                )

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
                                self.to_host(frame, p)
                                if self._input_per_plane[idx]
                                else pre_stacked_clips[idx]
                                for idx, frame in enumerate(f)
                            ]

                            func_MultiSrcIPP(inputs_data, self.to_host(fout, p), fout, p, n)

                        return fout
                else:
                    assert self.process_SingleSrcIPP
                    func_SingleSrcIPP = self.process_SingleSrcIPP

                    if self._input_per_plane[0]:
                        def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                            fout = f.copy()

                            for p in range(fout.format.num_planes):
                                func_SingleSrcIPP(self.to_host(f, p), self.to_host(fout, p), fout, p, n)

                            return fout
                    else:
                        def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                            fout = f.copy()

                            pre_stacked_clip = _stack_frame(f, 0)

                            for p in range(fout.format.num_planes):
                                func_SingleSrcIPP(pre_stacked_clip, self.to_host(fout, p), fout, p, n)

                            return fout
            else:
                if self.clips:
                    assert self.process_MultiSrcIPF
                    func_MultiSrcIPF = self.process_MultiSrcIPF

                    def output_func(f: tuple[vs.VideoFrame, ...], n: int) -> vs.VideoFrame:
                        fout = f[0].copy()

                        func_MultiSrcIPF(
                            [_stack_frame(frame, idx) for idx, frame in enumerate(f)],
                            dst_stacked_arr, fout, n
                        )

                        self.from_host(dst_stacked_arr, fout, planes_idx)

                        return fout
                else:
                    if self.ref_clip.format.num_planes == 1:
                        if self.process_SingleSrcIPP:
                            func_SingleSrcIPP = self.process_SingleSrcIPP

                            def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                                fout = f.copy()

                                func_SingleSrcIPP(self.to_host(f, 0), self.to_host(fout, 0), fout, 0, n)

                                return fout
                        else:
                            assert self.process_SingleSrcIPF
                            func_SingleSrcIPF = self.process_SingleSrcIPF

                            def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                                fout = f.copy()

                                func_SingleSrcIPF(self.to_host(f, 0), [self.to_host(fout, 0)], fout, n)

                                return fout
                    else:
                        assert self.process_SingleSrcIPF
                        func_SingleSrcIPF = self.process_SingleSrcIPF

                        def output_func(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
                            fout = f.copy()

                            func_SingleSrcIPF(stack_whole_frame(f), dst_stacked_arr, fout, n)

                            self.from_host(dst_stacked_arr, fout, planes_idx)

                            return fout

            return output_func

    class PyPluginNumpy(PyPluginNumpyBase[FD_T, NDArray[Any]]):
        ...

    this_backend.set_available(True)
except ModuleNotFoundError as e:
    this_backend.set_available(False, e)

    class PyPluginNumpyBase(PyPluginUnavailableBackendBase[FD_T, DT_T]):  # type: ignore
        backend = this_backend

    class PyPluginNumpy(PyPluginNumpyBase[FD_T]):  # type: ignore
        ...

    if not TYPE_CHECKING:
        NDT_T = TypeVar('NDT_T', bound=Any)
