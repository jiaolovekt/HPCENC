from __future__ import annotations

from functools import partial
from itertools import groupby
from math import log2
from typing import Callable, Iterable, Literal, Sequence, cast, overload

from vsaa import Nnedi3
from vskernels import Catrom, Kernel, KernelT, Scaler, ScalerT, Spline144
from vsmasktools import GenericMaskT, normalize_mask
from vstools import (
    CustomValueError, FieldBased, FieldBasedT, FuncExceptT, core, depth, expect_bits, get_h, get_prop, get_w, get_y,
    join, normalize_seq, split, to_arr, vs
)

from .helpers import scale_var_clip
from .mask import descale_detail_mask
from .types import DescaleAttempt, DescaleMode, DescaleModeWithInfo, DescaleResult, PlaneStatsKind

__all__ = [
    'get_select_descale', 'descale',

    'descale_fields'
]


def get_select_descale(
    clip: vs.VideoNode, descale_attempts: list[DescaleAttempt], mode: DescaleModeWithInfo
) -> tuple[Callable[[list[vs.VideoFrame], int], vs.VideoNode], list[vs.VideoNode]]:
    """Get callables for FrameEval/ModifyFrame and prop clips for the specified params."""
    curr_clip = clip
    main_kernel = descale_attempts[0].kernel.__class__.__name__
    attempts_by_idx = list(dict.fromkeys(descale_attempts).keys())

    threshold = mode.thr
    res_operator = mode.mode.res_op
    diff_operator = mode.mode.diff_op

    res_props_key = DescaleMode.PlaneDiff.prop_value(PlaneStatsKind.AVG)
    diff_prop_key = DescaleMode.KernelDiff.prop_value(PlaneStatsKind.DIFF)

    diff_clips = [attempt.diff for attempt in attempts_by_idx]

    def _get_descale_score(diff_vals: list[float], i: int) -> float:
        height_log = log2(curr_clip.height - attempts_by_idx[i].resolution.height)
        pstats_avg = round(1 / max(diff_vals[i], 1e-12))

        return height_log * pstats_avg ** 0.2  # type: ignore

    def _parse_attemps(f: list[vs.VideoFrame], indices: list[int]) -> tuple[vs.VideoNode, list[float], int]:
        diff_vals = [get_prop(frame, res_props_key, float) for frame in f]

        best_res = res_operator(indices, key=partial(_get_descale_score, diff_vals))

        best_attempt = attempts_by_idx[best_res]

        return best_attempt.descaled, diff_vals, best_res

    if mode.mode.is_average:
        clips_indices = list(range(len(diff_clips)))

        if threshold <= 0.0:
            def _select_descale(f: list[vs.VideoFrame], n: int) -> vs.VideoNode:
                return _parse_attemps(f, clips_indices)[0]
        else:
            def _select_descale(f: list[vs.VideoFrame], n: int) -> vs.VideoNode:
                best_attempt, plane_averages, best_res = _parse_attemps(f, clips_indices)

                if plane_averages[best_res] > threshold:
                    return curr_clip

                return best_attempt
    elif mode.mode.is_kernel_diff:
        group_by_kernel = {
            key: list(grouped) for key, grouped in groupby(
                enumerate(attempts_by_idx), lambda x: x[1].kernel.__class__.__name__
            )
        }

        if len(group_by_kernel) < 2:
            raise CustomValueError(
                'With KernelDiff mode you need to specify at least two kernels!\n'
                '(First will be the main kernel, others will be compared to it)',
                get_select_descale
            )

        kernel_indices = {
            name: [x[0] for x in attempts] for name, attempts in group_by_kernel.items()
        }
        other_kernels = {
            key: val for key, val in kernel_indices.items() if key != main_kernel
        }

        main_kernel_indices = list(kernel_indices[main_kernel])
        other_kernel_indices = list(other_kernels.values())
        other_kernel_enum_indices = list(enumerate(other_kernels.values()))

        main_clips_indices = list(range(len(main_kernel_indices)))
        other_clips_indices = [
            list(range(len(kernel_indices)))
            for kernel_indices in other_kernel_indices
        ]
        comp_other_indices = list(range(len(other_clips_indices)))

        def _select_descale(f: list[vs.VideoFrame], n: int) -> vs.VideoNode:
            main_clip, _, main_best_val_idx = _parse_attemps(
                [f[i] for i in main_kernel_indices], main_clips_indices
            )

            other_diffs_parsed = [
                _parse_attemps([f[i] for i in indices], other_clips_indices[j])
                for j, indices in other_kernel_enum_indices
            ]

            other_best_idx = diff_operator(
                comp_other_indices, key=lambda i: other_diffs_parsed[i][1][other_diffs_parsed[i][2]]
            )
            other_clip, _, other_best_val_idx = other_diffs_parsed[other_best_idx]

            main_value = get_prop(f[main_kernel_indices[main_best_val_idx]], diff_prop_key, float)
            other_value = get_prop(f[other_kernel_indices[other_best_idx][other_best_val_idx]], diff_prop_key, float)

            if other_value - threshold > main_value:
                return main_clip

            return other_clip

    else:
        raise CustomValueError('Incorrect descale mode specified!', get_select_descale)

    return _select_descale, diff_clips


@overload
def descale(  # type: ignore
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None, height: int | Iterable[int] = 720,
    kernels: KernelT | Sequence[KernelT] = Catrom, upscaler: ScalerT | None = Nnedi3,
    mask: GenericMaskT | Literal[False] | tuple[
        GenericMaskT | Literal[False] | None, GenericMaskT
    ] = descale_detail_mask,
    mode: DescaleMode | DescaleModeWithInfo = DescaleMode.PlaneDiff(0.0),
    dst_width: int | None = None, dst_height: int | None = None,
    shift: tuple[float, float] = (0, 0), scaler: ScalerT = Spline144,
    result: Literal[False] = False
) -> vs.VideoNode:
    ...


@overload
def descale(
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None, height: int | Iterable[int] = 720,
    kernels: KernelT | Sequence[KernelT] = Catrom, upscaler: ScalerT | None = Nnedi3,
    mask: GenericMaskT | Literal[False] | tuple[
        GenericMaskT | Literal[False] | None, GenericMaskT
    ] = descale_detail_mask,
    mode: DescaleMode | DescaleModeWithInfo = DescaleMode.PlaneDiff(0.0),
    dst_width: int | None = None, dst_height: int | None = None,
    shift: tuple[float, float] = (0, 0), scaler: ScalerT = Spline144,
    result: Literal[True] = True
) -> DescaleResult:
    ...


def descale(
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None, height: int | Iterable[int] = 720,
    kernels: KernelT | Sequence[KernelT] = Catrom, upscaler: ScalerT | None = Nnedi3,
    mask: GenericMaskT | Literal[False] | tuple[
        GenericMaskT | Literal[False] | None, GenericMaskT
    ] = descale_detail_mask,
    mode: DescaleMode | DescaleModeWithInfo = DescaleMode.PlaneDiff(0.0),
    dst_width: int | None = None, dst_height: int | None = None,
    shift: tuple[float, float] = (0, 0), scaler: ScalerT = Spline144,
    result: bool = False
) -> vs.VideoNode | DescaleResult:
    """
    A unified and expansive descaling function, includes support for handling
    fractional resolutions, multi-res, FHD detail masking, and conditional descaling.

    .. warning::

        Only descale if you are absolutely certain you know the correct native resolution(s) and kernel(s)!
        Wrong rescales will be more destructive than simply AAing/dehaloing instead!

    ``width`` and ``height`` both accept multiple values, allowing you to descale to multiple resolutions
    in case your source is upscaled from different native resolutions. If ``width`` is not set,
    it will be auto-calculated using the clip's aspect ratio and given height(s) as reference.

    You can pass a list of :py:class:`vskernels.Kernel` objects, allowing you to handle mixed sources.
    Common examples include shows that have been upscaled using either SharpBicubic or Spline36 throughout its runtime.
    If you find yourself working on a SharpBicubic, Spline36, or Lanczos source,
    consider double-checking other scenes and episodes for any of the other kernels mentioned.
    **Especially** if you've got either SharpCubic or Spline36!

    For fractional resolution descaling, you can pass the ``src_width`` and ``src_height`` to the kernel(s)
    passed to the function.

    Conditional descaling can be achieved by making use of ``mode``. Setting a specific mode determines
    how the condition is applied, and a threshold can be set. If the difference between the descaled
    and re-upscaled clip (using the same kernel) exceeds the threshold, it will not descale the frame.
    For more information, please refer to :py:class:`DescaleMode`'s docstring.

    Custom upscalers can be passed as well. If you'd like to write a custom upscaler, make sure
    you create a class with a ``scale`` method that adheres to the following API:

    ```py
    class.scale(clip, width, height)
    ```

    All the results can be returned using ``result``. This will return a :py:class:`DescaleResult` object
    containing the following attributes:

    * ``descaled`` (descaled clip, can be a variable-resolution clip)
    * ``rescaled`` (rescaled clip, can be a variable-resolution clip)
    * ``upscaled`` (upscaled clip)
    * ``mask`` (Descale error mask)
    * ``attempts`` (List of :py:class:`DescaleAttempt`s)
    * ``out`` (Final rescaled, masked, and chroma-merged clip)

    This has multiple applications, like for example needing the mask for post-rescaling masking of credits
    (like for post-AA, for example), or if you want to do additional filtering while it's descaled.

    :param clip:            Clip to process.
    :param width:           Width to descale to. Pass a list if your source has multiple native resolutions.
                            If None, auto-calculates based on the clip's AR and given height(s).
                            For fractional resolutions, set ``src_width`` in the :py:class:`vskernels.Kernel`.
                            Make sure you're confident about the native resolution(s) before descaling!
                            Default: None.
    :param height:          Height to descale to. Pass a list if your source has multiple native resolutions.
                            For fractional resolutions, set ``src_height`` in the :py:class:`vskernels.Kernel`.
                            Make sure you're confident about the native resolution(s) before descaling!
                            Default: 720.
    :param kernels:         :py:class:`vskernels.Kernel`s to descale to. Pass a list to enable multi-kernel descaling.
                            When passing multiple kernels, make sure you use a :py:class:`DescaleMode` that supports it.
                            Default: :py:class:`vskernels.Catrom`.
    :param upscaler:        Scaler used for rescaling. A passing scaler can be passed, but it must be a class with
                            a ``scale`` method that accepts a `clip, width, height`.
                            Setting this to False will disable upscaling entirely.
                            Default: :py:class:`vsaa.Nnedi3`.
    :param mask:            Internal masking function. This can be either a custom mask or a masking function.
                            Setting this to False will disable masking entirely.
                            Default: :py:func:`vsscale.mask.descale_detail_mask`.
    :param mode:            :py:class:`DescaleMode` used to determine how conditional descaling is handled.
                            For more information, please refer to :py:class:`DescaleMode`'s docstring.
                            Default: :py:func:`DescaleMode.PlaneDiff(0.0)`.
    :param dst_width:       Destination width. Set the clip's output width.
                            If None and a ``dst_height`` is passed, auto-calculate based on
                            the clip's AR and given ``dst_height``. Else, use the input clip's width.
                            Default: None.
    :param dst_height:      Destination height. Set the clip's output height.
                            If None, use the input clip's height.
                            Default: None.
    :param shift:           Shift the clip. Useful for cross-conversions, or if the image was shifted in the upscale.
                            Default: (0, 0), no shifting.
    :param scaler:          Scaler used for operations where regular scaling is required (for example,
                            ``upscaler=None``, for scaling chroma and descale detail mask).
                            Default: :py:class:`vskernels.Spline144`
    :param result:          Return the :py:class:`DescaleResult` object.
                            If False, return the regularly rescaled output.
                            Default: False.

    :raises ValueError:     Number of given heights and width don't match.
    :raises ValueError:     No kernel is specified.

    :returns:               Either a rescaled clip (mask applied, chroma re-added),
                            or a :py:class:`DescaleResult` object containing the results from ``descale``.
    """
    assert clip.format

    if isinstance(height, int):
        heights = [height]
    else:
        heights = list(height)

    if width is None:
        widths = [get_w(h, clip) for h in heights]
    elif isinstance(width, int):
        widths = [width]
    else:
        widths = list(width)

    if dst_width is None and dst_height:
        dest_width, dest_height = get_w(dst_height, clip), dst_height
    elif dst_height is None and dst_width:
        dest_width, dest_height = dst_width, get_h(dst_width, clip)
    else:
        dest_width, dest_height = clip.width, clip.height

    norm_resolutions = list(zip(widths, heights))
    norm_kernels = [Kernel.ensure_obj(kernel, descale) for kernel in to_arr(kernels)]

    scaler = Scaler.ensure_obj(scaler, descale)

    if len(widths) != len(heights):
        raise CustomValueError('Number of heights and widths specified mismatch!', descale)

    if not norm_kernels:
        raise CustomValueError('You must specify at least one kernel!', descale)

    work_clip, *chroma = split(clip)

    clip_y, bit_depth = expect_bits(work_clip, 32)

    max_kres_len = max(len(norm_kernels), len(norm_resolutions))

    kernel_combinations = list[tuple[Kernel, tuple[int, int]]](zip(*(
        normalize_seq(x, max_kres_len) for x in (norm_kernels, norm_resolutions)  # type: ignore
    )))

    mode = mode if isinstance(mode, DescaleModeWithInfo) else mode()

    descale_attempts = [
        DescaleAttempt.from_args(
            clip_y, width, height, shift, kernel, mode,
            descale_attempt_idx=i, descale_height=height, descale_kernel=kernel.__class__.__name__
        )
        for i, (kernel, (width, height)) in enumerate(kernel_combinations)
    ]

    if len(descale_attempts) > 1:
        norm_set_res = set(norm_resolutions)
        if len(norm_set_res) > 1:
            ref_clip = core.std.Splice([
                clip_y.std.BlankClip(length=len(clip_y) - 1, keep=True),
                clip_y.std.BlankClip(length=1, width=clip_y.width + 1, keep=True)
            ], mismatch=True)
        else:
            target_width, target_height = norm_set_res.pop()
            ref_clip = clip_y.std.BlankClip(target_width, target_height, keep=True)

        select_partial, prop_clips = get_select_descale(clip_y, descale_attempts, mode)

        descaled = ref_clip.std.FrameEval(select_partial, prop_clips)
    else:
        descaled = descale_attempts[0].descaled

    if len(kernel_combinations) == 1:
        rescaled = descale_attempts[0].rescaled
    else:
        rescaled = clip_y.std.FrameEval(
            lambda f, n: descale_attempts[cast(int, f.props.descale_attempt_idx)].rescaled, descaled
        )

    clip_y = depth(clip_y, bit_depth)
    descaled = depth(descaled, bit_depth)
    rescaled = depth(rescaled, bit_depth)

    upscaled = None
    if upscaler:
        upscaled = scale_var_clip(
            descaled, dest_width, dest_height, scaler=Scaler.ensure_obj(upscaler, descale)
        )
        upscaled = depth(upscaled, bit_depth)

    if isinstance(mask, tuple):
        error_mask, pproc_mask = mask
    else:
        error_mask, pproc_mask = mask, None

    if upscaled:
        clip_y = scaler.scale(clip_y, dest_width, dest_height)

        if error_mask:
            error_mask = normalize_mask(error_mask, clip_y, rescaled)
            error_mask = scaler.scale(error_mask, dest_width, dest_height)

            upscaled = upscaled.std.MaskedMerge(clip_y, error_mask)

        if pproc_mask:
            pproc_mask = normalize_mask(pproc_mask, clip_y, upscaled)
            pproc_mask = scaler.scale(pproc_mask, dest_width, dest_height)

            upscaled = clip_y.std.MaskedMerge(upscaled, pproc_mask)
    else:
        error_mask = pproc_mask = None

    if upscaled:
        out = upscaled
    elif upscaler:
        out = rescaled
    else:
        out = descaled

    if chroma and upscaled and (clip.width, clip.height) == (dest_width, dest_height):
        out = join([upscaled, *chroma], clip.format.color_family)

    out = out.std.CopyFrameProps(clip)

    if result:
        return DescaleResult(
            descaled, rescaled, upscaled, error_mask, pproc_mask, descale_attempts, out  # type: ignore
        )

    return out


def descale_fields(
    clip: vs.VideoNode, width: int | None = None, height: int = 720,
    tff: bool | FieldBasedT = True, kernel: KernelT = Catrom,
    src_top: float | tuple[float, float] = 0.0,
    src_left: float | tuple[float, float] = 0.0,
    debug: bool = False, func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Descale interwoven upscaled fields, also known as a cross conversion.

    ``src_top``, ``src_left`` allow you to to shift the clip prior to descaling.
    This may be useful, as sometimes clips are shifted before or after the original upscaling.

    :param clip:        Clip to process.
    :param width:       Native width. Will be automatically determined if set to `None`.
    :param height:      Native height. Will be divided by two internally.
    :param tff:         Top-field-first. `False` sets it to Bottom-Field-First.
    :param kernel:      py:class:`vskernels.Kernel` used for the descaling.
    :param src_top:     Shifts the clip vertically during the descaling.
                        Can be a tuple, defining the shift per-field.
    :param src_left:    Shifts the clip horizontally during the descaling.
                        Can be a tuple, defining the shift per-field.
    :param debug:       Set a frameprop with the kernel that was used.

    :return:            Descaled GRAY clip.
    """

    func = func or descale_fields

    height_field = int(height / 2)
    width = width or get_w(height, clip)

    kernel = Kernel.ensure_obj(kernel, func)

    clip = FieldBased.ensure_presence(clip, tff, func)

    y = get_y(clip).std.SeparateFields()

    if isinstance(src_top, tuple):
        ff_top, sf_top = src_top
    else:
        ff_top = sf_top = src_top

    if isinstance(src_left, tuple):
        ff_left, sf_left = src_left
    else:
        ff_left = sf_left = src_left

    if (ff_top, ff_left) == (sf_top, sf_left):
        descaled = kernel.descale(y, width, height_field, (ff_top, ff_left))
    else:
        descaled = core.std.Interleave([
            kernel.descale(y[::2], width, height_field, (ff_top, ff_left)),
            kernel.descale(y[1::2], width, height_field, (sf_top, sf_left))
        ])

    weave_y = descaled.std.DoubleWeave()

    if debug:
        weave_y = weave_y.std.SetFrameProp('scaler', data=f'{kernel.__class__.__name__} (Fields)')

    return weave_y.std.SetFieldBased(0)[::2]


# TODO: Write a function that checks every possible combination of B and C in bicubic
#       and returns a list of the results.
#       Possibly return all the frames in order of smallest difference to biggest.
#       Not reliable, but maybe useful as starting point.


# TODO: Write "multi_descale", a function that allows you to descale a frame twice,
#       like for example when the CGI in a show is handled in a different resolution
#       than the drawn animation.
