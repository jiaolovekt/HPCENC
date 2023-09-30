import multiprocessing

__all__ = ['cupy', 'cupyx', 'cufft', 'is_cuda_101', 'cuda_stream', 'fftw_cpu_kwargs']

try:
    import cupy  # type: ignore
    import cupyx  # type: ignore
    from cupy.cuda import cufft  # type: ignore

    cuda_error = None
    cuda_available = True
except ImportError as e:
    cuda_error = e
    cuda_available = False

if cuda_available:
    is_cuda_101 = 10010 <= cupy.cuda.runtime.runtimeGetVersion()

    cuda_stream = cupy.cuda.Stream(True, True, True)
else:
    cupy = cupyx = cufft = is_cuda_101 = cuda_stream = None  # noqa

fftw_cpu_kwargs = {
    'axes': (0, 1),
    'flags': ['FFTW_MEASURE'],
    'threads': multiprocessing.cpu_count()
}
