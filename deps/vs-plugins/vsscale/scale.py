from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from math import ceil, floor, log2
from typing import Any, Literal

from vsexprtools import complexpr_available, expr_func, norm_expr
from vskernels import ScalerT, SetsuCubic, ZewiaCubic
from vsrgtools import box_blur, gauss_blur
from vstools import (
    DependencyNotFoundError, KwargsT, Matrix, MatrixT, PlanesT, Transfer, VSFunction, check_ref_clip, check_variable,
    core, depth, fallback, get_depth, get_nvidia_version, inject_self, padder, vs
)

from .gamma import gamma2linear, linear2gamma
from .helpers import GenericScaler

__all__ = [
    'DPID',
    'SSIM',
    'DLISR',
    'Waifu2x'
]


@dataclass
class DPID(GenericScaler):
    """Rapid, Detail-Preserving Image Downscaler for VapourSynth"""

    sigma: float = 0.1
    """
    The power factor of range kernel. It can be used to tune the amplification of the weights of pixels
    that represent detail—from a box filter over an emphasis of distinct pixels towards a selection
    of only the most distinct pixels.
    """

    ref: vs.VideoNode | ScalerT = ZewiaCubic
    """VideoNode or Scaler to obtain the downscaled reference for DPID."""

    planes: PlanesT = None
    """Sets which planes will be processed. Any unprocessed planes will be simply copied from ref."""

    @inject_self
    def scale(  # type: ignore[override]
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        ref = clip

        if isinstance(self.ref, vs.VideoNode):
            check_ref_clip(clip, self.ref)  # type: ignore
            ref = self.ref  # type: ignore
            scaler = self.ensure_scaler(self.scaler)
        else:
            scaler = self.ensure_scaler(self.ref)  # type: ignore

        if (ref.width, ref.height) != (width, height):
            ref = scaler.scale(ref, width, height)

        kwargs |= {
            'lambda_': self.sigma, 'planes': self.planes,
            'src_left': shift[1], 'src_top': shift[0]
        } | kwargs | {'read_chromaloc': True}

        return core.dpid.DpidRaw(clip, ref, **kwargs)  # type: ignore


@dataclass
class SSIM(GenericScaler):
    """
    SSIM downsampler is an image downscaling technique that aims to optimize
    for the perceptual quality of the downscaled results.

    Image downscaling is considered as an optimization problem
    where the difference between the input and output images is measured
    using famous Structural SIMilarity (SSIM) index.

    The solution is derived in closed-form, which leads to the simple, efficient implementation.
    The downscaled images retain perceptually important features and details,
    resulting in an accurate and spatio-temporally consistent representation of the high resolution input.
    """

    smooth: int | float | VSFunction | None = None
    """
    Image smoothening method.
    If you pass an int, it specifies the "radius" of the internally-used boxfilter,
    i.e. the window has a size of (2*smooth+1)x(2*smooth+1).
    If you pass a float, it specifies the "sigma" of gauss_blur,
    i.e. the standard deviation of gaussian blur.
    If you pass a function, it acts as a general smoother.
    Default uses a gaussian blur.
    """

    curve: Transfer | bool | None = None
    """
    Perform a gamma conversion prior to scaling and after scaling. This must be set for `sigmoid` to function.
    If True it will auto-determine the curve based on the input props or resolution.
    Can be specified with for example `curve=TransferCurve.BT709`.
    """

    sigmoid: bool | float | tuple[float, float] | None = None
    """When True, applies a sigmoidal curve after the power-like curve
    (or before when converting from linear to gamma-corrected).
    This helps reduce the dark halo artefacts found around sharp edges
    caused by resizing in linear luminance.
    It can be a float or a tuple of two float values.
    When float or first value, it specifies the sigmoid slope. (cont)
    The second value specifies the sigmoid center for the curve. (thr)
    """

    epsilon: float = 1e-6
    """Variable used for math operations."""

    @inject_self
    def scale(  # type: ignore[override]
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0),
        smooth: int | float | VSFunction = ((3 ** 2 - 1) / 12) ** 0.5,
        curve: Transfer | bool = False, sigmoid: bool | float | tuple[float, float] = False, **kwargs: Any
    ) -> vs.VideoNode:
        assert check_variable(clip, self.scale)

        smooth = fallback(self.smooth, smooth)  # type: ignore
        curve = fallback(self.curve, curve)  # type: ignore
        sigmoid = fallback(self.sigmoid, sigmoid)

        if callable(smooth):
            filter_func = smooth
        elif isinstance(smooth, int):
            filter_func = partial(box_blur, radius=smooth)
        elif isinstance(smooth, float):
            filter_func = partial(gauss_blur, sigma=smooth)

        sigmoid_kwargs = KwargsT(epsilon=self.epsilon)

        if isinstance(sigmoid, bool):
            sigmoid_kwargs.update(sigmoid=bool(sigmoid))
        else:
            if isinstance(sigmoid, tuple):
                sig_cont, sig_thr = sigmoid
            else:
                sig_cont, sig_thr = sigmoid, 0.5

            sigmoid_kwargs.update(sigmoid=True, cont=sig_cont, thr=sig_thr)

        if curve is True:
            try:
                curve = Transfer.from_video(clip, True)
            except ValueError:
                curve = Transfer.from_matrix(Matrix.from_video(clip))

        bits, clip = get_depth(clip), depth(clip, 32)

        convert_csp = None

        if curve:
            if clip.format and clip.format.color_family is not vs.RGB:
                convert_csp = (Matrix.from_transfer(curve), clip.format)
                clip = self._kernel.resample(clip, vs.RGBS, None, convert_csp[0])
            clip = gamma2linear(clip, curve, **sigmoid_kwargs)

        l1 = self._scaler.scale(clip, width, height, shift, **kwargs)

        l1_sq, c_sq = [expr_func(x, 'x dup *') for x in (l1, clip)]

        l2 = self._scaler.scale(c_sq, width, height, shift, **kwargs)

        m, sl_m_square, sh_m_square = [filter_func(x) for x in (l1, l1_sq, l2)]

        if complexpr_available:
            merge_expr = f'z dup * SQ! x SQ@ - SQD! SQD@ {self.epsilon} < 0 y SQ@ - SQD@ / sqrt ?'
        else:
            merge_expr = f'x z dup * - {self.epsilon} < 0 y z dup * - x z dup * - / sqrt ?'

        r = expr_func([sl_m_square, sh_m_square, m], merge_expr)
        t = expr_func([r, m], 'x y *')
        d = expr_func([filter_func(m), filter_func(r), l1, filter_func(t)], 'x y z * + a -')

        if curve:
            d = linear2gamma(d, curve, **sigmoid_kwargs)

        if convert_csp is not None:
            d = self._kernel.resample(d, convert_csp[1], convert_csp[0])

        return depth(d, bits)


@dataclass
class DLISR(GenericScaler):
    """Use Nvidia NGX Technology DLISR DNN to scale up nodes. https://developer.nvidia.com/rtx/ngx"""

    scaler: ScalerT = field(default_factory=lambda: DPID(0.5, SetsuCubic))
    """Scaler to use to downscale clip to desired resolution, if necessary."""

    matrix: MatrixT | None = None
    """Input clip's matrix. Set only if necessary."""

    device_id: int | None = None
    """Which cuda device to run this filter on."""

    @inject_self
    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0),
        *, matrix: MatrixT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        output = clip

        assert check_variable(clip, self.__class__)

        if width > clip.width or height > clip.width:
            if not matrix:
                matrix = Matrix.from_param(matrix or self.matrix, self.__class__) or Matrix.from_video(clip, False)

            output = self._kernel.resample(output, vs.RGBS, Matrix.RGB, matrix)
            output = output.std.Limiter()

            max_scale = max(ceil(width / clip.width), ceil(height / clip.height))

            output = output.akarin.DLISR(max_scale, self.device_id)

        return self._finish_scale(output, clip, width, height, shift, matrix)


@dataclass
class Waifu2x(GenericScaler):
    """Use Waifu2x neural network to scale clips."""

    cuda: bool | Literal['trt'] | None = None
    """Whether to run this on cpu, gpu, or use trt technology. None will pick the fastest automatically."""

    num_streams: int | None = None
    """Number of gpu streams for the model."""

    fp16: bool = True
    """Whether to use float16 precision if available."""

    device_id: int = 0
    """Id of the cuda device to use."""

    matrix: MatrixT | None = None
    """Input clip's matrix. Set only if necessary."""

    tiles: int | tuple[int, int] | None = None
    """Process in separate tiles instead of the whole frame. Use if [V]RAM limited."""

    tilesize: int | tuple[int, int] | None = None
    """Manually specify the size of a single tile."""

    overlap: int | tuple[int, int] | None = None
    """Overlap for reducing blocking artifacts between tile borders."""

    backend_kwargs: KwargsT | None = None
    """Kwargs passed to create the backend instance."""

    @classmethod
    def mod_padding(cls, clip: vs.VideoNode, mod: int = 4, min: int = 4) -> tuple[int, int, int, int]:
        ph, pv = (mod - (((x + min * 2) - 1) % mod + 1) for x in (clip.width, clip.height))
        left, top = floor(ph / 2), floor(pv / 2)
        return tuple(x + min for x in (left, ph - left, top, pv - top))  # type: ignore

    def __post_init__(self) -> None:
        try:
            from vsmlrt import Backend  # type: ignore
        except ModuleNotFoundError as e:
            raise DependencyNotFoundError(self.__class__, e)

        cuda = self.cuda

        if cuda is True:
            self.fp16 = False
        elif self.fp16:
            self.fp16 = complexpr_available.fp16

        bkwargs = (self.backend_kwargs or KwargsT()) | KwargsT(fp16=self.fp16, device_id=self.device_id)

        # All this will eventually be in vs-nn
        if cuda is None:
            try:
                data: KwargsT = core.trt.DeviceProperties(self.device_id)  # type: ignore
                memory = data.get('total_global_memory', 0)
                def_num_streams = data.get('async_engine_count', 1)

                cuda = 'trt'

                bkwargs = KwargsT(
                    workspace=memory / (1 << 22) if memory else None,
                    use_cuda_graph=True, use_cublas=True, use_cudnn=True,
                    use_edge_mask_convolutions=True, use_jit_convolutions=True,
                    static_shape=True, heuristic=True, output_format=int(self.fp16),
                    tf32=not self.fp16, force_fp16=self.fp16, num_streams=def_num_streams
                ) | bkwargs

                streams_info = 'OK' if bkwargs['num_streams'] == def_num_streams else 'MISMATCH'

                core.log_message(
                    vs.MESSAGE_TYPE_DEBUG,
                    f'Selected [{data.get("name", b"<unknown>").decode("utf8")}] '
                    f'with {f"{(memory / (1 << 30))}GiB" if memory else "<unknown>"} of VRAM, '
                    f'num_streams={def_num_streams} ({streams_info})'
                )
            except Exception:
                self.fp16 = False
                bkwargs['fp16'] = False
                cuda = get_nvidia_version() is not None

        if bkwargs.get('num_streams', None) is None:
            bkwargs.update(num_streams=fallback(self.num_streams, 1))

        if cuda is True:
            if hasattr(core, 'ort'):
                self.backend = Backend.ORT_CUDA(**bkwargs)
            else:
                self.backend = Backend.OV_GPU(**bkwargs)
        elif cuda is False:
            if hasattr(core, 'ncnn'):
                self.backend = Backend.NCNN_VK(**bkwargs)
            else:
                bkwargs.pop('device_id')

                if hasattr(core, 'ort'):
                    self.backend = Backend.ORT_CPU(**bkwargs)
                else:
                    self.backend = Backend.OV_CPU(**bkwargs)
        else:
            self.backend = Backend.TRT(**bkwargs)

        super().__post_init__()

    @inject_self.init_kwargs.clean
    def scale(  # type:ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        wclip = clip

        assert check_variable(clip, self.scale)

        matrix = self.matrix
        is_gray = clip.format.color_family is vs.GRAY
        planes = 0 if is_gray else None
        kwargs.update(tiles=self.tiles, tilesize=self.tilesize, overlap=self.overlap)

        if (is_upscale := width > clip.width or height > clip.width):
            from vsmlrt import Waifu2x, Waifu2xModel

            if clip.format.color_family is vs.YUV:
                if not matrix:
                    matrix = Matrix.from_param(matrix or self.matrix, self.__class__) or Matrix.from_video(clip, False)
                wclip = self._kernel.resample(wclip, vs.RGBH if self.fp16 else vs.RGBS, Matrix.RGB, matrix)
            else:
                wclip = depth(wclip, 16 if self.fp16 else 32, vs.FLOAT)

                if is_gray:
                    wclip = wclip.std.ShufflePlanes(0, vs.RGB)

            try:
                wclip = wclip.std.Limiter(planes=planes)
            except vs.Error:
                wclip = norm_expr(wclip, 'x 0 1 clamp', planes=planes)

            mult = max(int(log2(ceil(size))) for size in (width / wclip.width, height / wclip.height))

            for _ in range(mult):
                padding = self.mod_padding(wclip)

                padded = padder(wclip, *padding)

                upscaled = Waifu2x(
                    padded, noise=-1, scale=2, model=Waifu2xModel.cunet, backend=self.backend, **kwargs
                )

                cropped = upscaled.std.Crop(*(p * 2 for p in padding))

                try:
                    wclip = norm_expr(cropped, 'x 0.5 255 / + 0 1 clamp', planes=planes)
                except RuntimeError:
                    wclip = norm_expr(depth(cropped, 32), 'x 0.5 255 / + 0 max 1 min', planes=planes)

            if is_gray:
                wclip = wclip.std.ShufflePlanes(0, vs.GRAY)

        return self._finish_scale(wclip, clip, width, height, shift, matrix, is_upscale)
