"""
    Kernels are a collection of wrappers pertaining to (de)scaling, format conversion,
    and other related operations, all while providing a consistent and clean interface.
    This allows for easy expansion and ease of use for any other maintainers
    who wishes to use them in their own functions.

    If you spot any issues, please do not hesitate to send in a Pull Request
    or reach out to me on Discord (LightArrowsEXE#0476)!

    For further support, drop by `#vs-kernels` in the `IEW Discord server <https://discord.gg/qxTxVJGtst>`_.
"""

# flake8: noqa

from . import exceptions, kernels, util
from .exceptions import *
from .kernels import *
from .util import *
