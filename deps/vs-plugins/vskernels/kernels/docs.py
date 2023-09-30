from __future__ import annotations

from typing import Any, overload

from vstools import HoldsVideoFormatT, MatrixT, VideoFormatT, core, get_video_format, inject_self, vs

from .abstract import Kernel

__all__ = [
    'Example'
]


class Example(Kernel):
    """Example Kernel class for documentation purposes."""

    def __init__(self, b: float = 0, c: float = 1 / 2, **kwargs: Any) -> None:
        self.b = b
        self.c = c
        super().__init__(**kwargs)

    @inject_self.cached
    def scale(  # type: ignore[override]
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        """
        Perform a regular scaling operation.

        :param clip:        Input clip
        :param width:       Output width
        :param height:      Output height
        :param shift:       Shift clip during the operation.
                            Expects a tuple of (src_top, src_left).

        :rtype:             ``VideoNode``
        """
        return core.resize.Bicubic(
            clip, width, height, src_top=shift[0], src_left=shift[1],
            filter_param_a=self.b, filter_param_b=self.c, **self.kwargs, **kwargs
        )

    @inject_self.cached
    def descale(  # type: ignore[override]
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        """
        Perform a regular descaling operation.

        :param clip:        Input clip
        :param width:       Output width
        :param height:      Output height
        :param shift:       Shift clip during the operation.
                            Expects a tuple of (src_top, src_left).

        :rtype:             ``VideoNode``
        """
        return core.descale.Debicubic(
            clip, width, height, b=self.b, c=self.c, src_top=shift[0], src_left=shift[1], **kwargs
        )

    @inject_self.cached
    def resample(  # type: ignore[override]
        self, clip: vs.VideoNode, format: int | VideoFormatT | HoldsVideoFormatT,
        matrix: MatrixT | None = None, matrix_in: MatrixT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        """
        Perform a regular resampling operation.

        :param clip:        Input clip
        :param format:      Output format
        :param matrix:      Output matrix. If `None`, will take the matrix from the input clip's frameprops.
        :param matrix_in:   Input matrix. If `None`, will take the matrix from the input clip's frameprops.

        :rtype:             ``VideoNode``
        """
        return core.resize.Bicubic(
            clip, format=get_video_format(format).id,
            filter_param_a=self.b, filter_param_b=self.c,
            matrix=matrix, matrix_in=matrix_in, **self.kwargs, **kwargs
        )

    @overload  # type: ignore
    def shift(self, clip: vs.VideoNode, shift: tuple[float, float] = (0, 0), **kwargs: Any) -> vs.VideoNode:
        ...

    def shift(  # type: ignore
        self, clip: vs.VideoNode,
        shift_top: float | list[float] = 0.0, shift_left: float | list[float] = 0.0, **kwargs: Any
    ) -> vs.VideoNode:
        """
        Perform a regular shifting operation.

        :param clip:        Input clip
        :param shift:       Shift clip during the operation.\n
                            Expects a tuple of (src_top, src_left)\n
                            or two top, left arrays for shifting planes individually.

        :rtype:             ``VideoNode``
        """
        return core.resize.Bicubic(
            clip, src_top=shift_top, src_left=shift_left,  # type: ignore
            filter_param_a=self.b, filter_param_b=self.c,
            **self.kwargs, **kwargs
        )
