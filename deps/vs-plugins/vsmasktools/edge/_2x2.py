"""2D matrices 3x3"""

from __future__ import annotations

from abc import ABC

from ._abstract import EdgeDetect, EuclideanDistance, RidgeDetect

__all__ = [
    'Matrix2x2',
    'Roberts'
]


class Matrix2x2(EdgeDetect, ABC):
    ...


class Roberts(RidgeDetect, EuclideanDistance, Matrix2x2):
    """Lawrence Roberts operator. 2x2 matrices computed in 3x3 matrices."""

    matrices = [
        [0, 0, 0, 0, 1, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 1, 0, -1, 0]
    ]
