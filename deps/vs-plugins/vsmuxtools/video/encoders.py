import shlex
import subprocess
from vstools import finalize_clip, initialize_clip, vs
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from muxtools import get_executable, ensure_path_exists, VideoFile, PathLike, make_output, info, warn, get_setup_attr, ensure_path

from ..utils.types import LosslessPreset, Zone
from ..utils.src import generate_qp_file, src_file
from .resumable import merge_parts, parse_keyframes
from .settings import file_or_default, fill_props, props_args, sb264, sb265, shift_zones, zones_to_args

__all__ = ["x264", "x265", "LosslessX264", "FFV1"]


@dataclass
class VideoEncoder(ABC):
    resumable = False

    @abstractmethod
    def encode(self, clip: vs.VideoNode, outfile: PathLike | None = None) -> VideoFile:
        """
        To actually run the encode.

        :param clip:            Your videonode
        :param outfile:         Can be a custom output file or directory.
                                The correct extension will automatically be appended.

        Returns a VideoFile object.
        If you're only interested in the path you can just do `VideoFile.file`.
        """
        ...

    def _update_progress(self, current_frame, total_frames):
        print(f"\rVapoursynth: {current_frame} / {total_frames} " f"({100 * current_frame // total_frames}%) || Encoder: ", end="")


@dataclass
class SupportsQP(VideoEncoder):
    settings: str | PathLike | None = None
    zones: Zone | list[Zone] | None = None
    qp_file: PathLike | bool | None = None
    qp_clip: src_file | vs.VideoNode | None = None
    add_props: bool | None = None
    sar: str | None = None
    quiet_merging: bool = True
    x265 = True

    def _get_qpfile(self, start_frame: int = 0) -> str:
        if not self.qp_file and not self.qp_clip:
            return ""

        if not isinstance(self.qp_file, bool) and self.qp_file is not None:
            return str(ensure_path_exists(self.qp_file, self).resolve())

        if self.qp_clip:
            if isinstance(self.qp_clip, src_file):
                self.qp_clip = self.qp_clip.src_cut
            return generate_qp_file(self.qp_clip, start_frame)

    def _init_settings(self, x265: bool):
        if not self.settings:
            s, p = file_or_default(f"{'x265' if x265 else 'x264'}_settings", sb265() if x265 else sb264())
            self.was_file = p
            self.settings = s
        else:
            s, p = file_or_default(self.settings, self.settings, True)
            self.was_file = p
            self.settings = s

        if self.add_props is None:
            self.add_props = not getattr(self, "was_file", False)

    def _update_settings(self, clip: vs.VideoNode, x265: bool):
        if self.was_file:
            self.settings = fill_props(self.settings, clip, x265, self.sar)

        self.settings = self.settings if isinstance(self.settings, list) else shlex.split(self.settings)

        if self.add_props:
            self.settings.extend(props_args(clip, x265, self.sar))

    @abstractmethod
    def _encode_clip(self, clip: vs.VideoNode, out: Path) -> Path:
        ...

    def encode(self, clip: vs.VideoNode, outfile: PathLike | None = None) -> VideoFile:
        if clip.format.bits_per_sample > (12 if self.x265 else 10):
            warn(f"This encoder does not support a bit depth over {(12 if self.x265 else 10)}.\nClip will be dithered to 10 bit.", self, 2)
            clip = finalize_clip(clip, 10)
        self._update_settings(clip, self.x265)
        out = make_output(
            Path(self.qp_clip.file).stem if isinstance(self.qp_clip, src_file) else "encoded",
            "265" if self.x265 else "264",
            "encoded" if isinstance(self.qp_clip, src_file) else "",
            outfile,
        )
        if not self.resumable:
            return VideoFile(self._encode_clip(clip, out, self._get_qpfile()))

        pattern = out.with_stem(out.stem + "_part_???")
        parts = sorted(pattern.parent.glob(pattern.name))
        info(f"Found {len(parts)} part{'s' if len(parts) != 1 else ''} for this encode")

        keyframes = list[int]()
        for i, p in enumerate(parts):
            try:
                info(f"Parsing keyframes for part {i}...")
                kf = parse_keyframes(p)[-1]
                if kf == 0:
                    del parts[-1]
                else:
                    keyframes.append(kf)
            except:
                del parts[-1]
        fout = out.with_stem(out.stem + f"_part_{len(parts):03.0f}")
        start_frame = sum(keyframes)
        info(f"Starting encode at frame {start_frame}")

        # TODO: Adjust existing zones to the new start frame

        clip = clip[start_frame:]
        self._encode_clip(clip, fout, self._get_qpfile(start_frame), start_frame)

        info("Remuxing and merging parts...")
        merge_parts(fout, out, keyframes, parts, self.quiet_merging)
        return VideoFile(out, source=self.qp_clip.file if isinstance(self.qp_clip, src_file) else None)


@dataclass
class x264(SupportsQP):
    """
    Encodes your clip to an avc/h264 file using x264.

    :param settings:            This will by default try to look for an `x264_settings` file in your cwd.
                                If it doesn't find one it will warn you and resort to the default settings_builder preset.
                                You can either pass settings as usual or a filepath here.
                                If the filepath doesn't exist it will assume you passed actual settings and pass those to the encoder.

    :param zones:               With this you can tweak settings of specific regions of the video.
                                In x264 this includes but is not limited to CRF.
                                For example (100, 300, "crf", 12) or [(100, 300, "crf", 12), (500, 750, 1.3)]
                                If the third part is not a string it will assume a bitrate multiplier (or "b")

    :param qp_file:             Here you can pass a bool to en/disable or an existing filepath for one.
    :param qp_clip:             Can either be a straight up VideoNode or a SRC_FILE/FileInfo from this package.
                                If neither a clip or a file are given it will simply skip.
                                If only a clip is given it will generate one.

    :param add_props:           This will explicitly add all props taken from the clip to the command line.
                                This will be disabled by default if you are using a file and otherwise enabled.
                                Files can have their own tokens like in vs-encode/vardautomation that will be filled in.

    :param sar:                 Here you can pass your Pixel / Sample Aspect Ratio. This will overwrite whatever is in the clip if passed.
    :param resumable:           Enable or disable resumable encodes. Very useful for people that have scripts that crash their PC (skill issue tbh)
    """

    resumable: bool = True
    x265 = False

    def __post_init__(self):
        self.executable = get_executable("x264")
        self._init_settings(self.x265)

    def _encode_clip(self, clip: vs.VideoNode, out: Path, qpfile: str, start_frame: int = 0) -> Path:
        args = [self.executable, "-o", str(out.resolve())]
        if qpfile:
            args.extend(["--qpfile", qpfile])
        if self.settings:
            args.extend(self.settings if isinstance(self.settings, list) else shlex.split(self.settings))
        if self.zones:
            if start_frame:
                self.zones = shift_zones(self.zones, start_frame)
            args.extend(zones_to_args(self.zones, False))
        args.extend(["--demuxer", "y4m", "-"])

        process = subprocess.Popen(args, stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda x, y: self._update_progress(x, y))
        process.communicate()
        return out


@dataclass
class x265(SupportsQP):
    """
    Encodes your clip to an hevc/h265 file using x265.

    :param settings:            This will by default try to look for an `x265_settings` file in your cwd.
                                If it doesn't find one it will warn you and resort to the default settings_builder preset.
                                You can either pass settings as usual or a filepath here.
                                If the filepath doesn't exist it will assume you passed actual settings and pass those to the encoder.

    :param zones:               With this you can tweak settings of specific regions of the video.
                                In x265 you're basically limited to a flat bitrate multiplier or force QP ("q")
                                For example (100, 300, "b", 1.2) or [(100, 300, "q", 12), (500, 750, 1.3)]
                                If the third part is not a string it will assume a bitrate multiplier (or "b")

    :param qp_file:             Here you can pass a bool to en/disable or an existing filepath for one.
    :param qp_clip:             Can either be a straight up VideoNode or a SRC_FILE/FileInfo from this package.
                                If neither a clip or a file are given it will simply skip.
                                If only a clip is given it will generate one.

    :param add_props:           This will explicitly add all props taken from the clip to the command line.
                                This will be disabled by default if you are using a file and otherwise enabled.
                                Files can have their own tokens like in vs-encode/vardautomation that will be filled in.

    :param sar:                 Here you can pass your Pixel / Sample Aspect Ratio. This will overwrite whatever is in the clip if passed.
    :param resumable:           Enable or disable resumable encodes. Very useful for people that have scripts that crash their PC (skill issue tbh)
    :param csv:                 Either a bool to enable or disable csv logging or a Filepath for said csv.
    """

    resumable: bool = True
    csv: bool | PathLike = True
    x265 = True

    def __post_init__(self):
        self.executable = get_executable("x265")
        self._init_settings(self.x265)

    def _encode_clip(self, clip: vs.VideoNode, out: Path, qpfile: str, start_frame: int = 0) -> Path:
        args = [self.executable, "-o", str(out.resolve())]
        if self.csv:
            if isinstance(self.csv, bool):
                show_name = get_setup_attr("show_name", "")
                csv_file = Path(show_name + f"{'_' if show_name else ''}log_x265.csv").resolve()
            else:
                csv_file = ensure_path(csv_file)
            args.extend(["--csv", str(csv_file)])
        if qpfile:
            args.extend(["--qpfile", qpfile])
        if self.settings:
            args.extend(self.settings if isinstance(self.settings, list) else shlex.split(self.settings))
        if self.zones:
            if start_frame:
                self.zones = shift_zones(self.zones, start_frame)
            args.extend(zones_to_args(self.zones, True))
        args.extend(["--y4m", "-"])

        process = subprocess.Popen(args, stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda x, y: self._update_progress(x, y))
        process.communicate()
        return out


@dataclass
class LosslessX264(VideoEncoder):
    """
    Uses x264 to encode clip to a lossless avc stream.

    :param preset:          Can either be a string of some x264 preset or any of the 3 predefined presets.
    :param settings:        Any other settings you might want to pass. Entirely optional.
    :param add_props:       This will explicitly add all props taken from the clip to the command line.
    """

    preset: str | LosslessPreset = LosslessPreset.MIDDLEGROUND
    settings: str | None = None
    add_props: bool = True

    def encode(self, clip: vs.VideoNode, outfile: PathLike | None = None) -> VideoFile:
        out = make_output("lossless", "264", user_passed=outfile)
        match self.preset:
            case LosslessPreset.SPEED:
                preset = "ultrafast"
            case LosslessPreset.COMPRESSION:
                preset = "veryslow"
            case LosslessPreset.MIDDLEGROUND:
                preset = "medium"
            case _:
                preset = self.preset
        settings = ["--preset", preset, "--qp", "0"]
        if clip.format.bits_per_sample > 10:
            warn(f"This encoder does not support a bit depth over 10.\nClip will be dithered to 10 bit.", self, 2)
            clip = finalize_clip(clip, 10)

        if self.settings:
            settings.extend(shlex.split(self.settings))
        avc = x264(settings, add_props=self.add_props, resumable=False)
        avc._update_settings(clip, False)
        avc._encode_clip(clip, out, None, 0)
        return VideoFile(out)


@dataclass
class FFV1(VideoEncoder):
    """
    Uses ffmpeg to encode clip to a lossless ffv1 stream.

    :param settings:        Can either be a string of your own settings or any of the 3 presets.
    :param ensure_props:    Calls initialize_clip on the clip to have at the very least guessed props
    """

    settings: str | LosslessPreset = LosslessPreset.MIDDLEGROUND
    ensure_props: bool = True

    def __post_init__(self):
        self.executable = get_executable("ffmpeg")

    def encode(self, clip: vs.VideoNode, outfile: PathLike | None = None) -> VideoFile:
        bits = clip.format.bits_per_sample
        if bits > 10:
            warn(f"You are encoding FFV1 with {bits} bits. It will be massive.", self, 1)
        if self.ensure_props:
            clip = initialize_clip(clip, bits)
            clip = finalize_clip(clip, bits)
        _base = "-coder 1 -context 0 -g 1 -level 3 -threads 0"
        match self.settings:
            case LosslessPreset.SPEED:
                self.settings = _base + " -slices 30 -slicecrc 0"
            case LosslessPreset.COMPRESSION:
                self.settings = _base + " -slices 16 -slicecrc 1"
            case LosslessPreset.MIDDLEGROUND:
                self.settings = _base + " -slices 24 -slicecrc 1"
            case _:
                self.settings = self.settings

        out = make_output("encoded_ffv1", "mkv", user_passed=outfile)

        args = [self.executable, "-f", "yuv4mpegpipe", "-i", "-", "-c:v", "ffv1"]
        if self.settings:
            args.extend(shlex.split(self.settings))
        args.append(str(out))

        process = subprocess.Popen(args, stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda x, y: self._update_progress(x, y))
        process.communicate()
        return VideoFile(out)
