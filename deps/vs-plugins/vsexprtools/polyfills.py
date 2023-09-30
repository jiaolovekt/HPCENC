import builtins
from typing import Any

from vstools import copy_func

from .operators import ExprOperators
from .variables import ExprVar

__all__ = [
    'global_builtins', 'global_builtins_expr'
]

global_builtins = _originals = {
    'min': copy_func(builtins.min),
    'max': copy_func(builtins.max)
}


def _expr_min(*args: Any, **kwargs: Any) -> Any:
    if not any(isinstance(x, ExprVar) for x in args):
        return _originals['min'](*args, **kwargs)

    var = args[0]
    for arg in args[1:]:
        var = ExprOperators.MIN(var, arg)

    return var


def _expr_max(*args: Any, **kwargs: Any) -> Any:
    if not any(isinstance(x, ExprVar) for x in args):
        return _originals['max'](*args, **kwargs)

    var = args[0]
    for arg in args[1:]:
        var = ExprOperators.MAX(var, arg)

    return var


global_builtins_expr = {
    'min': _expr_min,
    'max': _expr_max
}
