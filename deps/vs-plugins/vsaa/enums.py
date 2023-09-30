from __future__ import annotations

from vstools import CustomIntEnum

__all__ = [
    'AADirection'
]


class AADirection(CustomIntEnum):
    VERTICAL = 1
    HORIZONTAL = 2
    BOTH = 3

    def to_yx(self) -> tuple[bool, bool]:
        return (bool(self & AADirection.VERTICAL), bool(self & AADirection.HORIZONTAL))
