from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from vstools import CustomNotImplementedError, CustomRuntimeError, copy_signature

from .backends import PyBackend
from .base import PyPlugin, PyPluginBase, PyPluginUnavailableBackendBase
from .types import DT_T, FD_T

__all__ = [
    'PyPluginCythonBase', 'PyPluginCython'
]

this_backend = PyBackend.CYTHON
this_backend.set_dependencies({'Cython': '0.29.32'}, PyBackend.NONE)

uniqey = 'cython'


try:
    if PyBackend.is_cli:
        raise ModuleNotFoundError

    from Cython.Build import cythonize  # noqa: F401

    class CythonKernelFunction:
        def __init__(self, function: Any) -> None:
            self.function = function

        if TYPE_CHECKING:
            def __call__(
                self, src: memoryview | list[memoryview], dst: memoryview | list[memoryview], *args: Any
            ) -> Any:
                ...
        else:
            def __call__(self, *args: Any) -> Any:
                return self.function(*args)

    class CythonKernelFunctions:
        def __init__(self, **kwargs: CythonKernelFunction) -> None:
            for key, func in kwargs.items():
                setattr(self, key, func)

        if TYPE_CHECKING:
            def __getattribute__(self, __name: str) -> CythonKernelFunction:
                ...

    class PyPluginCythonBase(PyPluginBase[FD_T, DT_T]):
        backend = this_backend

        cython_kernel: str | tuple[str | Path, str | Sequence[str]]

        kernel_kwargs: dict[str, Any]

        kernel: CythonKernelFunctions

        @copy_signature(PyPlugin.__init__)
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            import random
            import string
            import subprocess
            from hashlib import md5
            from importlib.util import module_from_spec, spec_from_file_location
            from os import listdir, makedirs, remove
            from shutil import copyfile, move

            super().__init__(*args, **kwargs)

            assert self.ref_clip.format

            if not hasattr(self, 'cython_kernel'):
                raise CustomRuntimeError('You\'re missing cython_kernel!', self.__class__)

            if isinstance(self.cython_kernel, tuple):
                cython_path, cython_functions = self.cython_kernel
            else:
                cython_path, cython_functions = self.cython_kernel, Path(self.cython_kernel).stem

            if isinstance(cython_functions, str):
                cython_functions = [cython_functions]

            if not isinstance(cython_path, Path):
                cython_path = Path(cython_path)

            if not cython_path.suffix:
                cython_path = cython_path.with_suffix('.pyx')

            paths = [
                cython_path, cython_path.absolute().resolve(),
                *(
                    Path(inspect.getfile(cls)).parent / cython_path.name
                    for cls in self.__class__.mro()
                    if cls.__module__.strip('_') != 'builtins'
                )
            ]

            for cython_path in paths:
                if cython_path.exists():
                    break
            else:
                raise CustomRuntimeError('Cython code not found!', self.__class__)

            if not cython_path.is_file():
                raise CustomNotImplementedError('Directories not yet supported!', self.__class__)

            if cython_path.suffix != '.pyx':
                raise CustomRuntimeError('Cython code must be a .pyx file!', self.__class__)

            cython_build_dir = cython_path.parent / '.vspyplugin'
            curr_md5 = str(md5(cython_path.read_bytes()).digest())

            if not cython_build_dir.exists():
                makedirs(cython_build_dir, 0o777, True)

            module_path: Path | None = None
            curr_files = listdir(cython_build_dir)

            md5filename = f'{uniqey}_{cython_path.stem}.md5'
            old_builds = list[str]()

            for file in curr_files:
                if file == md5filename:
                    md5file = cython_build_dir / file

                    module_hash, module_md5 = md5file.read_text().splitlines()

                    module_folder = cython_build_dir / module_hash

                    if module_folder.exists():
                        if curr_md5 == module_md5:
                            module_path = module_folder
                            break

                        old_builds.append(module_hash)

            if module_path is None:
                cython_new_path = cython_build_dir / cython_path.name

                copyfile(cython_path, cython_new_path)

                subprocess.run([
                    'cythonize', '--3str', '-i', '-q', '--lenient', '-k', cython_path.name
                ], cwd=cython_new_path.parent)

                new_files = {*listdir(cython_build_dir)} - {*curr_files}

                rand_str = ''.join(random.choice(string.ascii_lowercase) for _ in range(12))
                module_path = cython_build_dir / f'{uniqey}_{rand_str}'

                makedirs(module_path)

                for file in new_files:
                    if file.endswith('.pyd'):
                        curr_file, new_file = cython_build_dir / file, module_path / file
                        move(curr_file, new_file)
                        md5file = cython_build_dir / md5filename
                        md5file.open('w').write(f'{uniqey}_{rand_str}\n{curr_md5}')
                    elif not file.endswith('.md5'):
                        remove(cython_build_dir / file)

            if module_path is None:
                raise CustomRuntimeError('There was an error compiling the cython module!', self.__class__)

            for old_build in old_builds:
                for file in listdir(cython_build_dir / old_build):
                    try:
                        remove(cython_build_dir / old_build / file)
                    except BaseException:
                        ...

            pyd_file = next((file for file in listdir(module_path)), None)

            if pyd_file is None:
                raise CustomRuntimeError('There was an error locating the cython module!', self.__class__)

            spec = spec_from_file_location(cython_path.stem, module_path / pyd_file)

            if spec is None:
                raise CustomRuntimeError('There was an error loading the cython module!', self.__class__)

            module = module_from_spec(spec)

            if spec.loader is None:
                raise CustomRuntimeError('The cython module is missing the package loader!', self.__class__)

            spec.loader.exec_module(module)

            self.kernel = CythonKernelFunctions(**{
                name: CythonKernelFunction(
                    object.__getattribute__(module, name)
                ) for name in cython_functions
            })

    class PyPluginCython(PyPlugin[FD_T], PyPluginCythonBase[FD_T, memoryview]):
        ...

    this_backend.set_available(True)
except ModuleNotFoundError as e:
    this_backend.set_available(False, e)

    class PyPluginCythonBase(PyPluginUnavailableBackendBase[FD_T, DT_T]):  # type: ignore
        backend = this_backend

    class PyPluginCython(PyPluginCythonBase[FD_T]):  # type: ignore
        ...
