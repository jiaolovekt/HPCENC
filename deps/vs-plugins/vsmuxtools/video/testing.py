from pathlib import Path
import re
from vstools import vs
from shlex import split
from numpy import arange

from .encoders import SupportsQP, VideoEncoder, x265
from ..utils.src import SRC_FILE, src
from muxtools import make_output

__all__ = ["SettingsTester"]


def settings_to_dict(settings: str | list[str]) -> dict[str, str | None]:
    args = split(settings) if isinstance(settings, str) else settings
    args_dict = dict[str, str | None]()

    for i, arg in enumerate(args):
        if not arg.startswith("-") or arg[1:].isnumeric():
            continue
        nextarg = None if len(args) < i + 2 else args[i + 1]
        if not nextarg or (nextarg.startswith("-") and not nextarg[1:].isnumeric()):
            args_dict.update({arg: None})
        else:
            args_dict.update({arg: nextarg})

    return args_dict


def resolve_var(var: str) -> list[str]:
    has_equals = False
    if "=" in var:
        var = var.split("=")[1]
        has_equals = True
    var = var[1:-1]

    if "/" in var:
        start = float(var.split("/")[0].strip())
        end = float(var.split("/")[1].strip())
        step = var.split("/")[2].strip()
        decimals = len(step.split(".")[1]) if "." in step else None
        step = float(step)
        return [("=" if has_equals else " ") + str(round(val, decimals)) for val in arange(start, end + step, step) if round(val, decimals) <= end]
    else:
        return [f"{'=' if has_equals else ' '}{val.strip()}" for val in var.split(",")]


def generate_settings(settings: str) -> list[tuple[str, str]]:
    yes = ["y", "yes", "true"]
    no = ["n", "no", "false"]
    pattern = re.compile(r"\[(.*?(?:\/|\,).*?)\]")
    variable_args = dict[str, str]({k: v for (k, v) in settings_to_dict(settings).items() if (v and pattern.search(v)) or pattern.search(k)})
    args_dict = dict[str, str | None]({k: v for (k, v) in settings_to_dict(settings).items() if k not in variable_args.keys()})

    settings_list = list[tuple[str, str]]()

    for k, v in variable_args.items():
        sett = settings_list.copy() if settings_list else [("", " ".join([f"{k} {v if v else ''}".strip() for (k, v) in args_dict.items()]))]
        settings_list.clear()
        resolved = resolve_var(v if v else k)
        if "=" in k:
            k = k.split("=")[0]
        for s in sett:
            for res in resolved:
                if res.strip().lower() in yes + no:
                    if res.strip().lower() in yes:
                        key = k if "--no-" not in k else f"--{k[5:]}"
                    else:
                        key = k if "--no-" in k else f"--no-{k[2:]}"
                    key_name = key[2:] if key.startswith("--") else key[1:]
                    settings_list.append((s[0] + f" {key_name}", f"{s[1]} {key}"))
                else:
                    key_name = k[2:] if k.startswith("--") else k[1:]
                    name = s[0] + f" {key_name}{'=' if '=' not in res else ''}{res.strip()}"
                    settings_list.append((name.strip(), f"{s[1]} {k}{res}"))

    return settings_list


class SettingsTester:
    """
    A utility class for making test encodes.
    This can automatically parse strings like

    `--preset veryfast --crf [14/15/0.5]`
    and will then run encodes with CRF 14, 14.5 and 15.


    The same works with non stepping options like

    `--preset [fast,veryfast,slow,slower]`
    or
    `--sao [true,false]` (can also use yes/no and y/n for these)


    Keep in mind that this will create an encode for EVERY combination and not some order.

    `--preset [fast,veryfast,ultrafast] --crf [14/15/0.5]`
    For example will end up with 9 encodes.
    """

    encoder: VideoEncoder
    encodes = list[tuple[str | None, str]]()
    qp_file: str | None = None

    def __init__(self, settings: str | list[str], encoder: VideoEncoder | None = None, qp_clip: SRC_FILE | vs.VideoNode | None = None) -> None:
        if not encoder:
            self.encoder = x265("--kek")
        else:
            self.encoder = encoder

        if isinstance(self.encoder, SupportsQP) and qp_clip:
            self.encoder.qp_clip = qp_clip
            self.qp_file = self.encoder._get_qpfile()

        if isinstance(settings, str):
            self.encodes = generate_settings(settings)
        else:
            self.encodes = [(None, s) for s in settings]

    def run(self, clip: vs.VideoNode, output_clips: bool = True) -> None:
        """
        Runs all encodes with the settings specified/generated.

        :param output_clips:        Will index and output clips with proper naming if vspreview is installed.
                                    This might obviously end up using quite a lot of ram.
        """
        for encode in self.encodes:
            encoder = self.encoder.__class__(settings=encode[1])
            encoder.resumable = False
            if isinstance(self.encoder, SupportsQP) and self.qp_file:
                encoder.qp_file = self.qp_file

            out = make_output("encode", "test", suffix="" if not encode[0] else f"[{encode[0]}]")

            f = encoder.encode(clip, out)
            if output_clips:
                done = src(f.file, True)
                try:
                    from vspreview import set_output

                    set_output(done, name=encode[0], cache=False)
                except:
                    done.set_output(len(vs.get_outputs()))
