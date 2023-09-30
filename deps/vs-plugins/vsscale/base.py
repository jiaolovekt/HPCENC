from __future__ import annotations

from typing import TYPE_CHECKING

from vstools import CustomStrEnum

__all__ = [
    'ShaderFileBase',
    'ShaderFileCustom'
]

if TYPE_CHECKING:
    from .shaders import ShaderFile

    class ShaderFileCustomBase:
        CUSTOM: ShaderFileCustom

    class ShaderFileBase(ShaderFileCustomBase, CustomStrEnum):
        value: str

    class ShaderFileCustom(ShaderFile):  # type: ignore
        ...
else:
    ShaderFileBase = CustomStrEnum
    ShaderFileCustom = CustomStrEnum
