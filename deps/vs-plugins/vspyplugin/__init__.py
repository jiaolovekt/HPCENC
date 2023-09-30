from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    import sys
    from pathlib import Path

    if sys.orig_argv:
        path = Path(sys.orig_argv[max(0, min(len(sys.orig_argv) - 1, 1))])

        if Path(sys.executable).parent == path.parent.parent and path.name in [
            'vspyplugin-script.py', 'vspyplugin.exe', 'vspyplugin'
        ]:
            import os
            os.environ['vspyplugin_is_cli'] = 'True'

from .abstracts import *  # noqa: F401, F403
from .backends import *  # noqa: F401, F403
from .base import *  # noqa: F401, F403
from .coroutines import *  # noqa: F401, F403
from .cuda import *  # noqa: F401, F403
from .cupy import *  # noqa: F401, F403
from .cython import *  # noqa: F401, F403
from .exceptions import *  # noqa: F401, F403
from .numpy import *  # noqa: F401, F403
from .types import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
