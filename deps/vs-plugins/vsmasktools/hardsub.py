from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vsexprtools import ExprOp, ExprToken, expr_func, norm_expr
from vskernels import Catrom
from vsrgtools.util import mean_matrix
from vstools import (
    CustomOverflowError, FileNotExistsError, FormatsRefClipMismatchError, FrameRangesN, LengthMismatchError, VSFunction,
    check_variable, core, depth, fallback, get_neutral_value, get_neutral_values, get_y, insert_clip, iterate,
    normalize_ranges, replace_ranges, scale_8bit, scale_value, vs, vs_object
)

from .abstract import DeferredMask, GeneralMask
from .edge import Sobel
from .morpho import Morpho
from .types import GenericMaskT
from .utils import normalize_mask

__all__ = [
    'HardsubManual',

    'HardsubMask',
    'HardsubSignFades',
    'HardsubSign',
    'HardsubLine',
    'HardsubLineFade',
    'HardsubASS',

    'bounded_dehardsub',
    'diff_hardsub_mask',

    'get_all_sign_masks',

    'custom_mask_clip'
]


@dataclass
class HardsubManual(GeneralMask, vs_object):
    path: str | Path
    processing: VSFunction = core.lazy.std.Binarize

    def __post_init__(self) -> None:
        if not (path := Path(self.path)).is_dir():
            raise FileNotExistsError('"path" must be an existing path directory!', self.get_mask)

        files = [file.stem for file in path.glob('*')]

        self.clips = [
            core.imwri.Read(file) for file in files
        ]

        self.ranges = [
            (other[-1] if other else end, end)
            for (*other, end) in (map(int, name.split('_')) for name in files)
        ]

    def get_mask(self, clip: vs.VideoNode, *args: Any) -> vs.VideoNode:  # type: ignore[override]
        assert check_variable(clip, self.get_mask)

        mask = clip.std.BlankClip(
            format=clip.format.replace(color_family=vs.GRAY, subsampling_h=0, subsampling_w=0).id,
            keep=True, color=0
        )

        for maskclip, (start_frame, end_frame) in zip(self.clips, self.ranges):
            maskclip = maskclip.std.AssumeFPS(clip).resize.Point(format=mask.format.id)  # type: ignore
            maskclip = self.processing(maskclip).std.Loop(end_frame - start_frame + 1)

            mask = insert_clip(mask, maskclip, start_frame)

        return mask

    def __vs_del__(self, core_id: int) -> None:
        super().__vs_del__(core_id)

        self.clips.clear()


class HardsubMask(DeferredMask):
    bin_thr: float = 0.75

    def get_progressive_dehardsub(
        self, hardsub: vs.VideoNode, ref: vs.VideoNode, partials: list[vs.VideoNode]
    ) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]:
        """
        Dehardsub using multiple superior hardsubbed sources and one inferior non-subbed source.

        :param hardsub:  Hardsub master source (eg Wakanim RU dub).
        :param ref:      Non-subbed reference source (eg CR, Funi, Amazon).
        :param partials: Sources to use for partial dehardsubbing (eg Waka DE, FR, SC).

        :return:         Dehardsub stages and masks used for progressive dehardsub.
        """

        masks = [self.get_mask(hardsub, ref)]
        partials_dehardsubbed = [hardsub]
        dehardsub_masks = []
        partials = partials + [ref]

        assert masks[-1].format is not None

        thr = scale_value(self.bin_thr, 32, masks[-1])

        for p in partials:
            masks.append(
                ExprOp.SUB.combine(masks[-1], self.get_mask(p, ref))
            )
            dehardsub_masks.append(
                iterate(expr_func([masks[-1]], f"x {thr} < 0 x ?"), core.std.Maximum, 4).std.Inflate()
            )
            partials_dehardsubbed.append(
                partials_dehardsubbed[-1].std.MaskedMerge(p, dehardsub_masks[-1])
            )

            masks[-1] = masks[-1].std.MaskedMerge(masks[-1].std.Invert(), masks[-2])

        return partials_dehardsubbed, dehardsub_masks

    def apply_dehardsub(
        self, hardsub: vs.VideoNode, ref: vs.VideoNode, partials: list[vs.VideoNode] | None = None
    ) -> vs.VideoNode:
        if partials:
            partials_dehardsubbed, _ = self.get_progressive_dehardsub(hardsub, ref, partials)
            dehardsub = partials_dehardsubbed[-1]
        else:
            dehardsub = hardsub.std.MaskedMerge(ref, self.get_mask(hardsub, ref))

        return replace_ranges(hardsub, dehardsub, self.ranges)


class HardsubSignFades(HardsubMask):
    highpass: float
    expand: int
    edgemask: GenericMaskT

    def __init__(
        self, *args: Any, highpass: float = 0.0763, expand: int = 8, edgemask: GenericMaskT = Sobel, **kwargs: Any
    ) -> None:
        self.highpass = highpass
        self.expand = expand
        self.edgemask = edgemask

        super().__init__(*args, **kwargs)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        clipedge, refedge = (
            normalize_mask(self.edgemask, x, **kwargs).std.Convolution(mean_matrix)
            for x in (clip, ref)
        )

        highpass = scale_value(self.highpass, 32, clip)

        mask = norm_expr(
            [clipedge, refedge], f'x y - {highpass} < 0 {ExprToken.RangeMax} ?'
        ).std.Median()

        return Morpho.inflate(Morpho.maximum(mask, iterations=self.expand), iterations=4)


class HardsubSign(HardsubMask):
    """
    Hardsub scenefiltering helper using `Zastin <https://github.com/kgrabs>`_'s hardsub mask.

    :param thr:     Binarization threshold, [0, 1] (Default: 0.06).
    :param expand:  std.Maximum iterations (Default: 8).
    :param inflate: std.Inflate iterations (Default: 7).
    """

    thr: float
    minimum: int
    expand: int
    inflate: int

    def __init__(
        self, *args: Any, thr: float = 0.06, minimum: int = 1, expand: int = 8, inflate: int = 7, **kwargs: Any
    ) -> None:
        self.thr = thr
        self.minimum = minimum
        self.expand = expand
        self.inflate = inflate
        super().__init__(*args, **kwargs)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        hsmf = norm_expr([clip, ref], 'x y - abs')
        hsmf = hsmf.resize.Point(format=clip.format.replace(subsampling_w=0, subsampling_h=0).id)  # type: ignore

        hsmf = ExprOp.MAX(hsmf, split_planes=True)

        hsmf = Morpho.binarize(hsmf, self.thr)
        hsmf = Morpho.minimum(hsmf, iterations=self.minimum)
        hsmf = Morpho.maximum(hsmf, iterations=self.expand)
        hsmf = Morpho.inflate(hsmf, iterations=self.inflate)

        return hsmf.std.Limiter()


class HardsubLine(HardsubMask):
    expand: int | None

    def __init__(self, *args: Any, expand: int | None = None, **kwargs: Any) -> None:
        self.expand = expand

        super().__init__(*args, **kwargs)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        assert clip.format

        expand_n = fallback(self.expand, clip.width // 200)

        y_range = scale_8bit(clip, 219) if clip.format.sample_type == vs.INTEGER else 1
        uv_range = scale_8bit(clip, 224) if clip.format.sample_type == vs.INTEGER else 1
        offset = scale_8bit(clip, 16) if clip.format.sample_type == vs.INTEGER else 0

        uv_abs = ' abs ' if clip.format.sample_type == vs.FLOAT else f' {get_neutral_value(clip)} - abs '
        yexpr = f'x y - abs {y_range * 0.7} > 255 0 ?'
        uv_thr = uv_range * 0.8
        uvexpr = f'x {uv_abs} {uv_thr} < y {uv_abs} {uv_thr} < and 255 0 ?'

        upper = y_range * 0.8 + offset
        lower = y_range * 0.2 + offset
        mindiff = y_range * 0.1

        difexpr = f'x {upper} > x {lower} < or x y - abs {mindiff} > and 255 0 ?'

        right = core.resize.Point(clip, src_left=4)

        subedge = norm_expr(
            [clip, right], (yexpr, uvexpr), format=clip.format.replace(sample_type=vs.INTEGER, bits_per_sample=8)
        )

        subedge = ExprOp.MIN(Catrom.resample(subedge, vs.YUV444P8), split_planes=True)

        clip_y, ref_y = get_y(clip), depth(get_y(ref), clip)

        clips = [clip_y.std.Convolution(mean_matrix), ref_y.std.Convolution(mean_matrix)]
        diff = core.std.Expr(clips, difexpr, vs.GRAY8).std.Maximum().std.Maximum()

        mask: vs.VideoNode = core.misc.Hysteresis(subedge, diff)
        mask = iterate(mask, core.std.Maximum, expand_n)
        mask = mask.std.Inflate().std.Inflate().std.Convolution(mean_matrix)

        return depth(mask, clip, range_in=1, range_out=1)


class HardsubLineFade(HardsubLine):
    def __init__(self, *args: Any, refframe: float = 0.5, **kwargs: Any) -> None:
        if refframe < 0 or refframe > 1:
            raise CustomOverflowError('"refframe" must be between 0 and 1!', self.__class__)

        self.ref_float = refframe

        super().__init__(*args, refframes=None, **kwargs)

    def get_mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:  # type: ignore
        self.refframes = [
            r[0] + round((r[1] - r[0]) * self.ref_float)
            for r in normalize_ranges(ref, self.ranges)
        ]

        return super().get_mask(clip, ref)


class HardsubASS(HardsubMask):
    filename: str
    fontdir: str | None
    shift: int | None

    def __init__(
        self, filename: str, *args: Any, fontdir: str | None = None, shift: int | None = None, **kwargs: Any
    ) -> None:
        self.filename = filename
        self.fontdir = fontdir
        self.shift = shift
        super().__init__(*args, **kwargs)

    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        ref = ref[0] * self.shift + ref if self.shift else ref
        mask: vs.VideoNode = ref.sub.TextFile(  # type: ignore[attr-defined]
            self.filename, fontdir=self.fontdir, blend=False
        )[1]
        mask = mask[self.shift:] if self.shift else mask
        mask = mask.std.Binarize(1)
        mask = iterate(mask, core.std.Maximum, 3)
        mask = iterate(mask, core.std.Inflate, 3)
        return mask.std.Limiter()


def bounded_dehardsub(
    hrdsb: vs.VideoNode, ref: vs.VideoNode, signs: list[HardsubMask], partials: list[vs.VideoNode] | None = None
) -> vs.VideoNode:
    for sign in signs:
        hrdsb = sign.apply_dehardsub(hrdsb, ref, partials)

    return hrdsb


def diff_hardsub_mask(a: vs.VideoNode, b: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
    assert check_variable(a, diff_hardsub_mask)
    assert check_variable(b, diff_hardsub_mask)

    return a.std.BlankClip(color=get_neutral_values(a), keep=True).std.MaskedMerge(
        a.std.MakeDiff(b), HardsubLine(**kwargs).get_mask(a, b)
    )


def get_all_sign_masks(hrdsb: vs.VideoNode, ref: vs.VideoNode, signs: list[HardsubMask]) -> vs.VideoNode:
    assert check_variable(hrdsb, get_all_sign_masks)
    assert check_variable(ref, get_all_sign_masks)

    mask = ref.std.BlankClip(
        format=ref.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id, keep=True
    )

    for sign in signs:
        mask = replace_ranges(mask, ExprOp.ADD.combine(mask, sign.get_mask(hrdsb, ref)), sign.ranges)

    return mask.std.Limiter()


def custom_mask_clip(
    clip: vs.VideoNode, ref: vs.VideoNode | None = None,
    imgs: list[str | Path] = [], ranges: FrameRangesN = [],
    show_mask: bool = True
) -> vs.VideoNode:
    if ref:
        FormatsRefClipMismatchError.check(custom_mask_clip, clip, ref)

    LengthMismatchError.check(
        custom_mask_clip, len(imgs), len(ranges),
        message="`imgs` and `ranges` must be of the same length!"
    )

    ref = ref or clip
    blank = ref.std.BlankClip(keep=True)

    masks = [core.imwri.Read(str(x)) * ref.num_frames for x in imgs]

    for mask, frange in zip(masks, ranges):
        blank = replace_ranges(blank, mask, frange)

    if show_mask:
        return blank

    return core.std.MaskedMerge(clip, ref, blank)
