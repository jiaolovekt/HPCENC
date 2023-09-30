from __future__ import annotations

from typing import Any

from vstools import inject_self, vs

from .kernels import Bicubic, FmtConv, Impulse, Kernel, KernelT, Placebo, Scaler
from .kernels.docs import Example

__all__ = [
    'excluded_kernels',
    'NoShift', 'NoScale'
]


class NoShiftBase(Kernel):
    def get_scale_args(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().get_scale_args(clip, (0, 0), *(args and args[1:]), **kwargs)

    def get_descale_args(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().get_descale_args(clip, (0, 0), *(args and args[1:]), **kwargs)


class NoShift(Bicubic, NoShiftBase):  # type: ignore
    """
    Class util used to always pass shift=(0, 0)\n
    By default it inherits from :py:class:`vskernels.Bicubic`,
    this behaviour can be changed with :py:attr:`Noshift.from_kernel`\n

    Use case, for example vsaa's ZNedi3:
    ```
    test = ...  # some clip, 480x480
    doubled_no_shift = Znedi3(field=0, nsize=4, nns=3, shifter=NoShift).scale(test, 960, 960)
    down = Point.scale(double, 480, 480)
    ```
    """

    def __class_getitem__(cls, kernel: KernelT) -> type[Kernel]:
        return cls.from_kernel(kernel)

    @staticmethod
    def from_kernel(kernel: KernelT) -> type[Kernel]:
        """
        Function or decorator for making a kernel not shift.

        As example, in vsaa:
        ```
        doubled_no_shift = Znedi3(..., shifter=NoShift.from_kernel('lanczos')).scale(...)

        # which in *this case* can also be written as this
        doubled_no_shift = Znedi3(..., shifter=NoShift, scaler=Lanczos).scale(...)
        ```

        Or for some other code:
        ```
        @NoShift.from_kernel
        class CustomCatromWithoutShift(Catrom):
            # some cool code
            ...
        ```
        """

        kernel_t = Kernel.from_param(kernel)

        class inner_no_shift(NoShiftBase, kernel_t):  # type: ignore
            ...

        return inner_no_shift


class NoScaleBase(Scaler):
    @inject_self.cached
    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        try:
            return super().scale(clip, clip.width, clip.height, shift, **kwargs)  # type: ignore
        except Exception:
            return clip


class NoScale(NoScaleBase, Bicubic):  # type: ignore
    def __class_getitem__(cls, kernel: KernelT) -> type[Kernel]:
        return cls.from_kernel(kernel)

    @staticmethod
    def from_kernel(kernel: KernelT) -> type[Kernel]:
        kernel_t = Kernel.from_param(kernel)

        class inner_no_scale(kernel_t, NoScaleBase):  # type: ignore
            ...

        return inner_no_scale


excluded_kernels = [Kernel, FmtConv, Example, Impulse, Placebo, NoShiftBase, NoScaleBase]
