from __future__ import annotations

from functools import partial
from math import floor
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import vapoursynth as vs
from pyfftw import FFTW, empty_aligned  # type: ignore

from .utils import cuda_available, cuda_error, cuda_stream, cufft, cupy, cupyx, fftw_cpu_kwargs


core = vs.core

__all__ = ['FFTSpectrum']


def _fast_roll(fdst: Any, fsrc: Any, yh: int, xh: int) -> None:
    fdst[:-yh, :-xh] = fsrc[yh:, xh:]
    fdst[:-yh, -xh:] = fsrc[yh:, :xh]
    fdst[-yh:, :-xh] = fsrc[:yh, xh:]
    fdst[-yh:, -xh:] = fsrc[:yh, :xh]


def _fftspectrum_cpu_modifyframe(
    f: List[vs.VideoFrame], n: int,
    rollfunc: Any, fft_out_arr: np.typing.NDArray[np.complex64]
) -> vs.VideoFrame:
    fdst = f[1].copy()

    farr: np.typing.NDArray[np.complex64] = np.asarray(f[0][0]).astype(np.complex64)

    FFTW(farr, fft_out_arr, **fftw_cpu_kwargs).execute()

    fdst_arr: np.typing.NDArray[np.float32] = np.asarray(fdst[0])

    rollfunc(fdst_arr, fft_out_arr.real)

    return fdst


if cuda_available:
    is_cuda_101 = 10010 <= cupy.cuda.runtime.runtimeGetVersion()

    def _fftspectrum_gpu_modifyframe(
        f: List[vs.VideoFrame], n: int, fft_thr: float, fft_scale: float,
        rollfunc: Any, fft_out_arr: cupy.typing.NDArray[cupy.complex64],
        cuda_plan: Any, cuda_fout: cupy.typing.NDArray[cupy.uint8]
    ) -> vs.VideoFrame:
        fdst = f[1].copy()

        farr = cupy.asarray(f[0][0])

        farr = cupy.ascontiguousarray(farr)

        farr = farr.astype(cupy.complex64)

        if (is_cuda_101):
            farr = farr.copy()

        cuda_plan.fft(farr, cuda_fout, cufft.CUFFT_FORWARD)

        fft_norm = cupy.abs(cupy.log(cupy.sqrt(cupy.abs(cuda_fout.real))))

        fft_norm = cupy.where(fft_norm > fft_thr, fft_norm * fft_scale, 0)

        fft_norm = fft_norm.astype(cupy.uint8)

        rollfunc(fft_out_arr, fft_norm)

        fft_out_arr.get(cuda_stream, 'C', np.asarray(fdst[0]))

        return fdst

_fft_modifyframe_cache: Dict[Tuple[Tuple[int, int], bool], Callable[[vs.VideoFrame, int], vs.VideoFrame]] = {}


def FFTSpectrum(
    clip: vs.VideoNode, threshold: float = 2.25, target_size: Tuple[int, int] | None = None, cuda: bool | None = None
) -> vs.VideoNode:
    assert clip.format

    cuda = (cuda_available if cuda is None else bool(cuda)) and ((clip.width + 31) & ~31) == clip.width

    if cuda and not cuda_available:
        raise ValueError(
            f"FFTSpectrum: Cuda acceleration isn't available!\nError: `{cuda_error}`"
        )

    if clip.format.bits_per_sample != 8 and clip.format.sample_type != vs.INTEGER:
        clip = clip.resize.Bicubic(
            format=clip.format.replace(
                sample_type=vs.INTEGER, bits_per_sample=8
            ).id, dither_type='error_diffusion'
        )

    shape = (clip.height, clip.width)

    cache_key = (shape, cuda)

    if cache_key not in _fft_modifyframe_cache:
        if cuda:
            fft_aligned_zeros = cupy.empty(shape, cupy.uint8, 'C')
            fft_cuplan_zeros = cupy.empty(shape, cupy.complex64, 'C')

            cuda_plan = cupyx.scipy.fftpack.get_fft_plan(
                fft_cuplan_zeros, shape, fftw_cpu_kwargs['axes']
            )
        else:
            fft_aligned_zeros = empty_aligned(shape, np.complex64)

        rollfunc = partial(_fast_roll, xh=clip.width // 2, yh=clip.height // 2,)

        _modify_frame_args = dict(rollfunc=rollfunc, fft_out_arr=fft_aligned_zeros)

        if cuda:
            _fft_modifyframe_cache[cache_key] = partial(
                _fftspectrum_gpu_modifyframe, **_modify_frame_args,
                cuda_plan=cuda_plan, cuda_fout=fft_cuplan_zeros
            )
        else:
            _fft_modifyframe_cache[cache_key] = partial(
                _fftspectrum_cpu_modifyframe, **_modify_frame_args
            )

    truemax = 9.846

    fft_thr = truemax / threshold

    fft_scale = truemax * threshold * 1.1681139254640247

    _modify_frame_func = _fft_modifyframe_cache[cache_key]

    if cuda:
        _modify_frame_func = partial(
            _modify_frame_func, fft_thr=fft_thr, fft_scale=fft_scale
        )

    blankclip = clip.std.BlankClip(
        format=vs.GRAY8 if cuda else vs.GRAYS, color=0, keep=True
    )

    fftclip = blankclip.std.ModifyFrame([clip, blankclip], _modify_frame_func)

    if target_size:
        max_width, max_height = target_size

        if clip.width != max_width or clip.height != max_height:
            w_diff, h_diff = max_width - clip.width, max_height - clip.height
            w_pad, w_mod = (floor(w_diff / 2), w_diff % 2) if w_diff > 0 else (0, 0)
            h_pad, h_mod = (floor(h_diff / 2), h_diff % 2) if h_diff > 0 else (0, 0)

            fftclip = fftclip.std.AddBorders(w_pad, w_pad + w_mod, h_pad, h_pad + h_mod)

            if w_mod or h_mod:
                fftclip = fftclip.resize.Bicubic(src_top=h_mod / 2, src_left=w_mod / 2)

    if not cuda:
        fftclip = fftclip.akarin.Expr(
            f'x abs sqrt log abs S! S@ {fft_thr} > S@ {fft_scale} * 0 ?', vs.GRAY8
        )

    return fftclip
