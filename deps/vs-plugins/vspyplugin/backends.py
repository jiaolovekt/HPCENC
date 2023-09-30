from __future__ import annotations

import os

from vstools import CustomIntEnum, classproperty

__all__ = [
    'PyBackend'
]


class PyBackend(CustomIntEnum):
    NONE = -1
    NUMPY = 0
    CUPY = 1
    CUDA = 2
    CYTHON = 3

    def set_available(self, is_available: bool, e: ModuleNotFoundError | None = None) -> None:
        if not is_available:
            _unavailable_backends.add((self, e))
        elif not self.is_available:
            unav_backs = _unavailable_backends.copy()
            _unavailable_backends.clear()
            _unavailable_backends.update({
                (backend, error) for backend, error in unav_backs if backend is not self
            })

    @property
    def is_available(self) -> bool:
        return self not in {backend for backend, _ in _unavailable_backends}

    @property
    def import_error(self) -> ModuleNotFoundError | None:
        return next((e for backend, e in _unavailable_backends if backend is self), None)

    @property
    def dependencies(self) -> dict[str, str]:
        deps = _dependecies_backends.get(self, {})

        for back in _dependecies_back_back.get(self, ()):
            deps |= back.dependencies

        return deps

    def set_dependencies(self, deps: dict[str, str], *backend_deps: PyBackend) -> None:
        _dependecies_backends[self] = {**deps}
        _dependecies_back_back[self] = backend_deps

    @classproperty
    def is_cli(cls) -> bool:
        try:
            return os.environ['vspyplugin_is_cli'] == 'True'
        except KeyError:
            return False


_unavailable_backends = set[tuple[PyBackend, ModuleNotFoundError | None]]()
_dependecies_backends = dict[PyBackend, dict[str, str]]()
_dependecies_back_back = dict[PyBackend, tuple[PyBackend, ...]]()
