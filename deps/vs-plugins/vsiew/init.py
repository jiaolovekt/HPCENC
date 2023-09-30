# Shrimply

from typing import Iterator

base_org = 'Irrational-Encoding-Wizardry'


def update(action_: list[str] | None) -> None:
    import sys

    action = 'update'

    if action_:
        if action_[-1] == 'latest':
            action = 'update-git'
        elif 'Scripts' not in action_[-1] or '.py' not in action_[-1]:
            action = action_[-1].strip()

    def _get_install_call(package: str, do_git: bool) -> int:
        from subprocess import check_call

        args = list[str]()

        if do_git:
            package = f'git+https://github.com/{base_org}/{package}.git'
            args.extend(['--force', '--no-deps'])

        try:
            return check_call([
                sys.executable, '-m', 'pip', 'install',
                package, '-U', '--no-cache-dir', *args
            ])
        except Exception:
            return 1

    def _get_uninstall_call(package: str) -> int:
        from subprocess import check_call

        try:
            return check_call([
                sys.executable, '-m', 'pip', 'uninstall', package, '-y'
            ])
        except Exception:
            return 1

    def _get_iew_packages() -> Iterator[tuple[str, str]]:
        from http.client import HTTPSConnection

        conn = HTTPSConnection('raw.githubusercontent.com', 443)
        conn.request(
            'GET', f'https://raw.githubusercontent.com/{base_org}'
            '/vs-iew/master/requirements.txt'
        )

        res = conn.getresponse()

        for line in res.readlines():
            if b'#' in line:
                line_s = line.decode('utf-8').strip()

                *left, pypi_package = line_s.split('# ')
                package = left[0].split('=')[0]

                yield (package, pypi_package)

    err = color = 0
    message = default_message = 'No error message specified'

    def _set_message(
        message_succ: str = default_message, message_err: str = default_message
    ) -> None:
        nonlocal color, message, err
        color = 31 if err else 32
        message = (message_err if err else message_succ).format(err=err)

    if action == 'update':
        packages = list(_get_iew_packages())
        for name, _ in reversed(packages):
            _get_uninstall_call(name)

        for name, _ in packages:
            if _get_install_call(name, False):
                err += 1

        _set_message(
            'Successfully updated IEW packages!',
            'There was an error updating IEW packages!'
        )
    elif action == 'update-git':
        packages = list(_get_iew_packages())
        for name, _ in reversed(packages):
            _get_uninstall_call(name)

        for _, repo_name in packages:
            if _get_install_call(repo_name, True):
                err += 1

        _set_message(
            'Successfully updated all IEW packages to latest git!',
            'There was an error updating ({err}) IEW packages to latest git!'
        )
    elif action == 'uninstall':
        for name, _ in reversed(list(_get_iew_packages())):
            if _get_uninstall_call(name):
                err += 1

        _set_message(
            'Successfully uninstalled all IEW packages!',
            'There was an error uninstalling ({err}) IEW packages!'
        )
    else:
        err = 1
        _set_message(message_err=f'There\'s no action called "{action}"!')

    if sys.stdout and sys.stdout.isatty():
        message = f'\033[0;{color};1m{message}\033[0m'

    print(f'\n\t{message}\n')
