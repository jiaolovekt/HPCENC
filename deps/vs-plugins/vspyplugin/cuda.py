from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, Generic, Literal, Sequence, TypeVar, cast

from vstools import CustomRuntimeError, get_lowest_value, get_neutral_value, get_peak_value, get_resolutions, vs

from .backends import PyBackend
from .base import PyPluginOptions, PyPluginUnavailableBackendBase
from .types import DT_T, FD_T, FilterMode
from .utils import get_c_dtype_long

__all__ = [
    'PyPluginCudaBase', 'PyPluginCuda',
    'CudaCompileFlags', 'PyPluginCudaOptions'
]

this_backend = PyBackend.CUDA
this_backend.set_dependencies({}, PyBackend.CUPY)

T = TypeVar('T')


@dataclass
class CudaCompileFlags:
    std: Literal[3, 11, 14, 17, 20] = 17
    use_fast_math: bool = True
    extra_vectorization: bool = True
    options: tuple[str, ...] | None = None

    def to_tuple(self) -> tuple[str, ...]:
        options = [] if self.options is None else list(self.options)

        if self.use_fast_math:
            options.append('--use_fast_math')

        if self.std:
            options.append(f'--std=c++{self.std:02}')

        if self.extra_vectorization:
            options.append('--extra-device-vectorization')

        return tuple(set(options))


@dataclass
class PyPluginCudaOptions(PyPluginOptions):
    compile_flags: CudaCompileFlags = field(default_factory=lambda: CudaCompileFlags())
    backend: Literal['nvrtc', 'nvcc'] = 'nvrtc'
    translate_cucomplex: bool = False
    enable_cooperative_groups: bool = False
    jitify: bool = False


try:
    if PyBackend.is_cli:
        raise ModuleNotFoundError

    import cupy
    from numpy.typing import NDArray

    from .cupy import PyPluginCupy, PyPluginCupyBase
    from .numpy import NDT_T

    class CudaKernelFunction(Generic[NDT_T]):
        def __call__(
            self, src: NDT_T, dst: NDT_T, *args: Any,
            kernel_size: tuple[int, ...] = ..., block_size: tuple[int, ...] = ..., shared_mem: int = ...
        ) -> Any:
            ...

    class CudaKernelFunctionPlanes(CudaKernelFunction[NDT_T]):
        def __new__(
            cls, function: CudaKernelFunction[NDT_T], planes_functions: list[CudaKernelFunction[NDT_T]] | None = None
        ) -> CudaKernelFunctionPlanes[NDT_T]:
            class func_inner(dict[int | None, CudaKernelFunction[NDT_T]]):
                __call__ = lambda _, *args, **kwargs: function(*args, **kwargs)  # noqa: E731

            if planes_functions is None:
                planes_functions = [function]
            planes_functions += planes_functions[-1:] * (3 - len(planes_functions))

            return func_inner(  # type: ignore
                {idx: func for idx, func in enumerate(planes_functions)} | {None: function}
            )

        if TYPE_CHECKING:
            def __getitem__(self, plane: int | None, /) -> CudaKernelFunction[NDT_T]:
                ...

    class CudaKernelFunctions(Generic[NDT_T]):
        def __new__(cls, **kwargs: CudaKernelFunctionPlanes[NDT_T]) -> CudaKernelFunctions[NDT_T]:
            self = type('kernel_inner', (object, ), dict(__slots__=list(kwargs.keys())))

            for key, func in kwargs.items():
                setattr(self, key, func)

            return self  # type: ignore

        if TYPE_CHECKING:
            def __getattribute__(self, __name: str) -> CudaKernelFunctionPlanes[NDT_T]:
                ...

    class PyPluginCudaBase(PyPluginCupyBase[FD_T, NDT_T]):
        backend = this_backend

        cuda_kernel: str | tuple[str | Path, str | Sequence[str]]

        kernel_size: int | tuple[int, ...] = 16

        use_shared_memory: bool = False

        options: PyPluginCudaOptions = PyPluginCudaOptions()

        kernel: CudaKernelFunctions[NDT_T]

        @lru_cache
        def get_kernel_size(self, plane: int, func_name: str, width: int, height: int) -> tuple[int, ...]:
            if isinstance(self.kernel_size, tuple):
                return self.kernel_size

            return self.kernel_size, self.kernel_size

        @lru_cache
        def normalize_kernel_size(
            self, plane: int, func_name: str, blk_size_w: int, blk_size_h: int, width: int, height: int
        ) -> tuple[int, ...]:
            return ((width + blk_size_w - 1) // blk_size_w, (height + blk_size_h - 1) // blk_size_h)

        @lru_cache
        def get_kernel_shared_mem(
            self, plane: int, func_name: str, blk_size_w: int, blk_size_h: int, dtype_size: int
        ) -> int:
            return blk_size_w * blk_size_h * dtype_size

        def get_kernel_args(self, plane: int, func_name: str, width: int, height: int, **kwargs: Any) -> dict[str, Any]:
            assert self.ref_clip.format

            block_x, block_y, *block_xx = self.get_kernel_size(plane, func_name, width, height)

            kernel_args = dict[str, Any](
                use_shared_memory=self.use_shared_memory,
                block_x=block_x, block_y=block_y,
                data_type=get_c_dtype_long(self.ref_clip),
                is_float=self.ref_clip.format.sample_type is vs.FLOAT,
                lowest_value=float(get_lowest_value(self.ref_clip)),
                neutral_value=float(get_neutral_value(self.ref_clip)),
                peak_value=float(get_peak_value(self.ref_clip)),
            )

            if block_xx:
                kernel_args |= dict(block_z=block_xx[0])

            if self.fd:
                try:
                    kernel_args |= self.fd  # type: ignore
                except BaseException:
                    ...

            return kwargs | kernel_args | dict(width=width, height=height)

        def normalize_kernel_arg(self, value: Any) -> str:
            string = str(value)

            if isinstance(value, bool):
                return string.lower()

            return string

        def __init__(
            self,
            ref_clip: vs.VideoNode,
            clips: list[vs.VideoNode] | None = None,
            cuda_kernel: str | tuple[str | Path, str | Sequence[str]] | None = None,
            kernel_size: int | tuple[int, ...] | None = None,
            use_shared_memory: bool | None = None,
            *,
            kernel_kwargs: dict[str, Any] | None = None,
            kernel_planes_kwargs: list[dict[str, Any] | None] | None = None,
            filter_mode: FilterMode | None = None,
            options: PyPluginOptions | None = None,
            input_per_plane: bool | list[bool] | None = None,
            output_per_plane: bool | None = None,
            channels_last: bool | None = None,
            min_clips: int | None = None,
            max_clips: int | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(
                ref_clip, clips,
                filter_mode=filter_mode, options=options, channels_last=channels_last,
                input_per_plane=input_per_plane, output_per_plane=output_per_plane,
                min_clips=min_clips, max_clips=max_clips, **kwargs
            )

            arguments = [
                (cuda_kernel, 'cuda_kernel', None),
                (kernel_size, 'kernel_size', 16),
                (use_shared_memory, 'use_shared_memory', True)
            ]

            for value, name, default in arguments:
                if value is not None:
                    setattr(self, name, value)
                elif not hasattr(self, name) and default is not None:
                    setattr(self, name, default)

            assert self.ref_clip.format

            if kernel_kwargs is None:
                kernel_kwargs = {}

            if kernel_planes_kwargs:
                kernel_planes_kwargs += kernel_planes_kwargs[-1:] * (3 - len(kernel_planes_kwargs))

            if not hasattr(self, 'cuda_kernel'):
                raise CustomRuntimeError('You\'re missing cuda_kernel!', self.__class__)

            if isinstance(self.cuda_kernel, tuple):
                self_cuda_path, cuda_functions = self.cuda_kernel
            else:
                self_cuda_path, cuda_functions = self.cuda_kernel, Path(self.cuda_kernel).stem

            if isinstance(cuda_functions, str):
                cuda_functions = [cuda_functions]

            cuda_path = Path(self_cuda_path)

            if not cuda_path.suffix:
                cuda_path = cuda_path.with_suffix('.cu')

            paths = [
                cuda_path, cuda_path.absolute().resolve(),
                *(
                    Path(inspect.getfile(cls)).parent / cuda_path.name
                    for cls in self.__class__.mro()
                    if cls.__module__.strip('_') != 'builtins'
                )
            ]

            cuda_kernel_code: str | None = None

            for path in paths:
                if path.exists():
                    cuda_kernel_code = path.read_text()
                    break

            if cuda_kernel_code is None:
                if cuda_path.suffix == '.cu' or len(str(self_cuda_path)) < 24:
                    raise CustomRuntimeError('Cuda Kernel file not found!', self.__class__)
                elif isinstance(self_cuda_path, str):
                    cuda_kernel_code = self_cuda_path

            if cuda_kernel_code:
                cuda_kernel_code = cuda_kernel_code.strip()

            if not cuda_kernel_code:
                raise CustomRuntimeError('Cuda Kernel code not found!', self.__class__)

            def _wrap_kernel_function(
                def_kernel_size: tuple[int, ...],
                def_block_size: tuple[int, ...],
                def_shared_mem: int, function: Any
            ) -> CudaKernelFunction[NDT_T]:
                def _inner_function(
                    *args: Any,
                    kernel_size: tuple[int, ...] = def_kernel_size,
                    block_size: tuple[int, ...] = def_block_size,
                    shared_mem: int = def_shared_mem
                ) -> Any:
                    return function(kernel_size, block_size, args, shared_mem=shared_mem)

                return cast(CudaKernelFunction[NDT_T], _inner_function)

            raw_kernel_kwargs = dict(
                options=('-Xptxas', '-O3', *self.options.compile_flags.to_tuple()),
                backend=self.options.backend,
                translate_cucomplex=self.options.translate_cucomplex,
                enable_cooperative_groups=self.options.enable_cooperative_groups,
                jitify=self.options.jitify, name_expressions=None,
                log_stream=sys.stdout if self.debug else None
            )

            _cache_modules_funcs = dict[int, Any]()
            _cache_kernel_funcs = dict[tuple[int, str], CudaKernelFunction[NDT_T]]()

            def _get_kernel_func(name: str, plane: int, width: int, height: int) -> CudaKernelFunction[NDT_T]:
                assert self.ref_clip.format and cuda_kernel_code and kernel_kwargs is not None

                kernel_args = self.get_kernel_args(plane, name, width, height, **kernel_kwargs)
                block_sizes = self.get_kernel_size(plane, name, width, height)[:2]

                if kernel_planes_kwargs and (p_kwargs := kernel_planes_kwargs[plane]):
                    kernel_args |= p_kwargs

                kernel_args = {
                    name: self.normalize_kernel_arg(value)
                    for name, value in kernel_args.items()
                }

                def_kernel_size = self.normalize_kernel_size(
                    plane, name, *block_sizes, self.ref_clip.width, self.ref_clip.height
                )

                def_shared_mem = self.get_kernel_shared_mem(
                    plane, name, *block_sizes, self.ref_clip.format.bytes_per_sample
                ) if self.use_shared_memory else 0

                sub_kernel_code = Template(cuda_kernel_code).substitute(kernel_args)

                module_key = hash(sub_kernel_code)
                kernel_key = module_key, name

                if module_key not in _cache_modules_funcs:
                    _cache_modules_funcs[module_key] = cupy._cupy._core.raw._get_raw_module(
                        sub_kernel_code, None, **raw_kernel_kwargs
                    )

                if kernel_key not in _cache_kernel_funcs:
                    kernel = _cache_modules_funcs[module_key].get_function(name)

                    _cache_kernel_funcs[kernel_key] = _wrap_kernel_function(
                        def_kernel_size, block_sizes, def_shared_mem, kernel
                    )

                return _cache_kernel_funcs[kernel_key]

            resolutions = get_resolutions(self.ref_clip)

            kernel_functions = {
                name: [
                    _get_kernel_func(name, plane, width, height)
                    for plane, width, height in resolutions
                ] for name in cuda_functions
            }

            self.kernel = CudaKernelFunctions(**{
                name: CudaKernelFunctionPlanes(funcs[0], funcs)
                for name, funcs in kernel_functions.items()
            })

    class PyPluginCuda(PyPluginCupy[FD_T], PyPluginCudaBase[FD_T, NDArray[Any]]):
        ...

    this_backend.set_available(True)
except ModuleNotFoundError as e:
    this_backend.set_available(False, e)

    class PyPluginCudaBase(PyPluginUnavailableBackendBase[FD_T, DT_T]):  # type: ignore
        backend = this_backend

    class PyPluginCuda(PyPluginCudaBase[FD_T]):  # type: ignore
        ...
