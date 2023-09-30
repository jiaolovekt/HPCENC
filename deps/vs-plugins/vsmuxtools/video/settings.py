from genericpath import isfile
import re
import os
from tracemalloc import start
from muxtools import error, PathLike, ensure_path, warn
from vstools import vs, Matrix, Primaries, Transfer, ColorRange, ChromaLocation, get_prop, Colorspace
from ..utils.types import Zone

__all__ = ["settings_builder_x265", "settings_builder_x264", "sb", "sb265", "sb264"]


def is_full_zone(zone: Zone) -> bool:
    if isinstance(zone[2], str):
        if len(zone) < 4:
            raise error(f"Zone '{zone}' is invalid.")
        return True
    else:
        return False


def shift_zones(zones: Zone | list[Zone] | None, start_frame: int = 0) -> list[Zone] | None:
    if not zones:
        return None
    if not isinstance(zones, list):
        zones = [zones]

    newzones = list[Zone]()

    for zone in zones:
        zone = list(zone)
        zone[0] = zone[0] - start_frame
        zone[1] = zone[1] - start_frame
        if zone[1] < 0:
            continue
        if zone[0] < 0:
            zone[0] = 0
        zone = tuple(zone)
        newzones.append(zone)

    return newzones


def zones_to_args(zones: Zone | list[Zone] | None, x265: bool) -> list[str]:
    args: list[str] = []
    if not zones:
        return args
    if not isinstance(zones, list):
        zones = [zones]
    zones_settings: str = ""
    for i, zone in enumerate(zones):
        if is_full_zone(zone):
            if x265 and zone[2].lower() not in ["q", "b"]:
                raise error(f"Zone '{zone}' is invalid for x265. Please only use b or q.")
            zones_settings += f"{zone[0]},{zone[1]},{zone[2]}={zone[3]}"
        else:
            zones_settings += f"{zone[0]},{zone[1]},b={zone[2]}"
        if i != len(zones) - 1:
            zones_settings += "/"
    args.extend(["--zones", zones_settings])
    return args


def settings_builder_x265(
    preset: str | int = "slower",
    crf: float = 14.0,
    qcomp: float = 0.75,
    psy_rd: float = 2.0,
    psy_rdoq: float = 2.0,
    aq_strength: float = 0.75,
    aq_mode: int = 3,
    rd: int = 4,
    rect: bool = True,
    amp: bool = False,
    chroma_qpoffsets: int = -2,
    tu_intra_depth: int = 2,
    tu_inter_depth: int = 2,
    rskip: bool | int = 0,
    tskip: bool = False,
    ref: int = 4,
    bframes: int = 16,
    cutree: bool = False,
    rc_lookahead: int = 60,
    subme: int = 5,
    me: int = 3,
    b_intra: bool = True,
    weightb: bool = True,
    deblock: list[int] | str = [-2, -2],
    append: str = "",
    **kwargs,
) -> str:
    # Simple insert values
    settings = f" --preset {preset} --crf {crf} --bframes {bframes} --ref {ref} --rc-lookahead {rc_lookahead} --subme {subme} --me {me}"
    settings += f" --aq-mode {aq_mode} --aq-strength {aq_strength} --qcomp {qcomp} --cbqpoffs {chroma_qpoffsets} --crqpoffs {chroma_qpoffsets}"
    settings += f" --rd {rd} --psy-rd {psy_rd} --psy-rdoq {psy_rdoq} --tu-intra-depth {tu_intra_depth} --tu-inter-depth {tu_inter_depth}"

    # Less simple
    settings += f" --{'rect' if rect else 'no-rect'} --{'amp' if amp else 'no-amp'} --{'tskip' if tskip else 'no-tskip'}"
    settings += f" --{'b-intra' if b_intra else 'no-b-intra'} --{'weightb' if weightb else 'no-weightb'} --{'cutree' if cutree else 'no-cutree'}"
    settings += f" --rskip {int(rskip) if isinstance(rskip, bool) else rskip}"

    if isinstance(deblock, list):
        deblock = f"{str(deblock[0])}:{str(deblock[1])}"
    settings += f" --deblock={deblock}"

    # Don't need to change these lol
    settings += " --no-sao --no-sao-non-deblock --no-strong-intra-smoothing --no-open-gop"

    for k, v in kwargs.items():
        prefix = "--"
        if k.startswith("_"):
            prefix = "-"
            k = k[1:]
        settings += f" {prefix}{k.replace('_', '-')} {v}"

    settings += (" " + append.strip()) if append.strip() else ""
    return settings


def settings_builder_x264(
    preset: str = "placebo",
    crf: float = 13,
    qcomp: float = 0.7,
    psy_rd: float = 1.0,
    psy_trellis: float = 0.0,
    trellis: int = 0,
    aq_strength: float = 0.8,
    aq_mode: int = 3,
    ref: int = 16,
    bframes: int = 16,
    mbtree: bool = False,
    rc_lookahead: int = 250,
    me: str = "umh",
    subme: int = 11,
    threads: int = 6,
    merange: int = 32,
    deblock: list[int] | str = [-1, -1],
    dct_decimate: bool = False,
    append: str = "",
    **kwargs,
) -> str:
    # Simple insert values
    settings = f" --preset {preset} --crf {crf} --bframes {bframes} --ref {ref} --rc-lookahead {rc_lookahead} --me {me} --merange {merange}"
    settings += f" --aq-mode {aq_mode} --aq-strength {aq_strength} --qcomp {qcomp}"
    settings += f" --psy-rd {psy_rd}:{psy_trellis} --trellis {trellis} --subme {subme} --threads {threads}"

    # Less simple
    settings += f" {'--no-mbtree' if not mbtree else ''} {'--no-dct-decimate' if not dct_decimate else ''}"

    if isinstance(deblock, list):
        deblock = f"{str(deblock[0])}:{str(deblock[1])}"
    settings += f" --deblock={deblock}"

    for k, v in kwargs.items():
        prefix = "--"
        if k.startswith("_"):
            prefix = "-"
            k = k[1:]
        settings += f" {prefix}{k.replace('_', '-')} {v}"

    settings += (" " + append.strip()) if append.strip() else ""
    return settings


sb = settings_builder_x265
sb265 = sb
sb264 = settings_builder_x264


def file_or_default(file: PathLike, default: str, no_warn: bool = False) -> tuple[str | list[str], bool]:
    if isinstance(file, list):
        return file, False
    if os.path.isfile(file):
        file = ensure_path(file, None)
        if file.exists():
            with open(file, "r") as r:
                settings = str(r.read())
                settings = settings.replace("\n", " ")
                settings = re.sub(r"(?:-o|--output) {clip.+?}", "", settings, flags=re.I).strip()
                return settings, True

    if not no_warn:
        warn("Settings file wasn't found. Using default.", None, 3)
    return default, False


def get_props(clip: vs.VideoNode, x265: bool) -> dict[str, str]:
    crange = ColorRange.from_video(clip)
    is_limited = crange.is_limited
    c_range = crange.string if x265 else ("tv" if is_limited else "pc")
    bits = clip.format.bits_per_sample
    props = clip.get_frame(0).props
    return {
        "depth": str(bits),
        "chromaloc": str(int(ChromaLocation.from_video(clip))),
        "range": c_range,
        "transfer": Transfer.from_video(clip).string,
        "colormatrix": Matrix.from_video(clip).string,
        "primaries": Primaries.from_video(clip).string,
        "sarnum": str(props.get("_SARNum", 1)),
        "sarden": str(props.get("_SARDen", 1)),
        "keyint": str(round(clip.fps) * 10),
        "min_keyint": str(round(clip.fps)),
        "frames": str(clip.num_frames),
        "fps_num": str(clip.fps_num),
        "fps_den": str(clip.fps_den),
        "min_luma": str(16 << (bits - 8) if is_limited else 0),
        "max_luma": str(235 << (bits - 8) if is_limited else (1 << bits) - 1),
        "lookahead": str(min(clip.fps_num * 5, 250)),
    }


def fill_props(settings: str, clip: vs.VideoNode, x265: bool, sar: str | None = None) -> str:
    props = get_props(clip, x265)
    if sar is not None:
        if not isinstance(sar, str):
            sar = str(sar)
        sarnum = sar if ":" not in sar else sar.split(":")[0]
        sarden = sar if ":" not in sar else sar.split(":")[1]
    else:
        sarnum = props.get("sarnum")
        sarden = props.get("sarden")
        if sarnum != "1" or sarden != "1":
            warn(f"Are you sure your SAR ({sarnum}:{sarden}) is correct?\nAre you perhaps working on an anamorphic source?", None, 2)
    settings = re.sub(r"{chromaloc(?::.)?}", props.get("chromaloc"), settings)
    settings = re.sub(r"{primaries(?::.)?}", props.get("primaries"), settings)
    settings = re.sub(r"{bits(?::.)?}", props.get("depth"), settings)
    settings = re.sub(r"{matrix(?::.)?}", props.get("colormatrix"), settings)
    settings = re.sub(r"{range(?::.)?}", props.get("range"), settings)
    settings = re.sub(r"{transfer(?::.)?}", props.get("transfer"), settings)
    settings = re.sub(r"{frames(?::.)?}", props.get("frames"), settings)
    settings = re.sub(r"{fps_num(?::.)?}", props.get("fps_num"), settings)
    settings = re.sub(r"{fps_den(?::.)?}", props.get("fps_den"), settings)
    settings = re.sub(r"{min_keyint(?::.)?}", props.get("min_keyint"), settings)
    settings = re.sub(r"{keyint(?::.)?}", props.get("keyint"), settings)
    settings = re.sub(r"{sarnum(?::.)?}", sarnum, settings)
    settings = re.sub(r"{sarden(?::.)?}", sarden, settings)
    settings = re.sub(r"{min_luma(?::.)?}", props.get("min_luma"), settings)
    settings = re.sub(r"{max_luma(?::.)?}", props.get("max_luma"), settings)
    settings = re.sub(r"{lookahead(?::.)?}", props.get("lookahead"), settings)
    return settings


def props_args(clip: vs.VideoNode, x265: bool, sar: str | None = None) -> list[str]:
    args: list[str] = []
    props = get_props(clip, x265)
    if sar is not None:
        if not isinstance(sar, str):
            sar = str(sar)
        sarnum = sar if ":" not in sar else sar.split(":")[0]
        sarden = sar if ":" not in sar else sar.split(":")[1]
    else:
        sarnum = props.get("sarnum")
        sarden = props.get("sarden")
        if sarnum != "1" or sarden != "1":
            warn(f"Are you sure your SAR ({sarnum}:{sarden}) is correct?\nAre you perhaps working on an anamorphic source?", None, 2)

    # fmt: off
    args.extend([
        "--input-depth", props.get("depth"),
        "--output-depth", props.get("depth"),
        "--transfer", props.get("transfer"),
        "--chromaloc", props.get("chromaloc"),
        "--colormatrix", props.get("colormatrix"),
        "--range", props.get("range"),
        "--colorprim", props.get("primaries"),
        "--sar", f"{sarnum}:{sarden}"
    ])
    if x265:
        args.extend([
            "--min-luma", props.get("min_luma"),
            "--max-luma", props.get("max_luma")
        ])
    return args
    # fmt: on
