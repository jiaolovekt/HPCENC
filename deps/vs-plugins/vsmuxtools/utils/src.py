import shutil as sh
from pathlib import Path
from typing import Callable
from fractions import Fraction
from vstools import vs, core, initialize_clip, copy_signature
from muxtools import (
    Trim,
    PathLike,
    parse_m2ts_path,
    ensure_path_exists,
    warn,
    info,
    get_workdir,
    get_temp_workdir,
    clean_temp_files,
    get_absolute_track,
    TrackType,
)


__all__ = ["src_file", "SRC_FILE", "FileInfo", "src", "frames_to_samples", "f2s"]


class src_file:
    file: PathLike
    force_lsmas: bool = False
    trim: Trim = None
    idx: Callable[[str], vs.VideoNode] | None = None
    idx_args = {}

    def __init__(self, file: PathLike, force_lsmas: bool = False, trim: Trim = None, idx: Callable[[str], vs.VideoNode] | None = None, **kwargs):
        """
        Custom `FileInfo` kind of thing for convenience

        :param file:            Either a string based filepath or a Path object
        :param force_lsmas:     Forces the use of lsmas inside of `vodesfunc.src`
        :param trim:            Can be a single trim or a sequence of trims.
        :param idx:             Indexer for the input file. Pass a function that takes a string in and returns a vs.VideoNode.
                                Defaults to `vodesfunc.src`
        """
        self.file = ensure_path_exists(file, self)
        self.force_lsmas = force_lsmas
        self.trim = trim
        self.idx = idx
        self.idx_args = kwargs

    def __index_clip(self):
        indexed = self.idx(str(self.file.resolve())) if self.idx else src(str(self.file.resolve()), self.force_lsmas, **self.idx_args)
        cut = indexed
        if self.trim:
            self.trim = list(self.trim)
            if self.trim[0] is None:
                self.trim[0] = 0
            if self.trim[1] is None or self.trim[1] == 0:
                cut = indexed[self.trim[0] :]
            else:
                cut = indexed[self.trim[0] : self.trim[1]]
            self.trim = tuple(self.trim)

        if self.file.suffix.lower() == ".dgi":
            if self.file.with_suffix(".m2ts").exists():
                self.file = self.file.with_suffix(".m2ts")
            else:
                self.file = parse_m2ts_path(self.file)

        setattr(self, "clip", indexed)
        setattr(self, "clip_cut", cut)

    @property
    def src(self) -> vs.VideoNode:
        if not hasattr(self, "clip"):
            self.__index_clip()
        return self.clip

    @property
    def src_cut(self) -> vs.VideoNode:
        if not hasattr(self, "clip_cut"):
            self.__index_clip()
        return self.clip_cut

    @copy_signature(initialize_clip)
    def init(self, *args, **kwargs) -> vs.VideoNode:
        """
        Getter that calls `vstools.initialize_clip` on the src clip for convenience

        :param kwargs:      Any other args passed to initialize_clip
        """
        return initialize_clip(self.src, *args, **kwargs)

    @copy_signature(initialize_clip)
    def init_cut(self, *args, **kwargs) -> vs.VideoNode:
        """
        Getter that calls `vstools.initialize_clip` on the src_cut clip for convenience

        :param kwargs:      Any other args passed to initialize_clip
        """
        return initialize_clip(self.src_cut, *args, **kwargs)

    def get_audio(self, track: int = 0, **kwargs) -> vs.AudioNode:
        """
        Indexes the specified audio track from the input file.
        """
        args = dict(exact=True)
        args.update(**kwargs)

        absolute = get_absolute_track(self.file, track, TrackType.AUDIO)

        return core.bs.AudioSource(str(self.file.resolve()), absolute.track_id, **args)

    def get_audio_trimmed(self, track: int = 0, **kwargs) -> vs.AudioNode:
        """
        Gets the indexed audio track with the trim specified in the src_file.
        """
        node = self.get_audio(track, **kwargs)
        if self.trim:
            if self.trim[1] is None or self.trim[1] == 0:
                node = node[f2s(self.trim[0], node, self.src) :]
            else:
                node = node[f2s(self.trim[0], node, self.src) : f2s(self.trim[1], node, self.src)]
        return node


SRC_FILE = src_file
FileInfo = src_file


def src(filePath: PathLike, force_lsmas: bool = False, delete_dgi_log: bool = True, **kwargs) -> vs.VideoNode:
    """
    Uses dgindex as Source and requires dgindexnv in path
    to generate files if they don't exist.

    :param filepath:        Path to video or dgi file
    :param force_lsmas:     Skip dgsource entirely and use lsmas
    :param delete_dgi_log:  Delete the .log files dgindexnv creates
    :return:                Video Node
    """
    filePath = ensure_path_exists(filePath, src)
    filePath = str(filePath)
    if filePath.lower().endswith(".dgi"):
        return core.lazy.dgdecodenv.DGSource(filePath, **kwargs)

    forceFallBack = sh.which("dgindexnv") is None or not hasattr(core, "dgdecodenv")

    if not force_lsmas:
        import pymediainfo as pym

        parsed = pym.MediaInfo.parse(filePath, parse_speed=0.25)
        trackmeta = parsed.video_tracks[0].to_data()
        format = trackmeta.get("format")
        bitdepth = trackmeta.get("bit_depth")
        if format is not None and bitdepth is not None:
            if str(format).strip().lower() == "avc" and int(bitdepth) > 8:
                forceFallBack = True
                warn(f"Falling back to lsmas for Hi10 ({Path(filePath).name})", src)
            elif str(format).strip().lower() == "ffv1":
                forceFallBack = True
                warn(f"Falling back to lsmas for FFV1 ({Path(filePath).name})", src)

    if force_lsmas or forceFallBack:
        return core.lsmas.LWLibavSource(filePath, **kwargs)

    path = Path(filePath)
    dgiFile = path.with_suffix(".dgi")

    if dgiFile.exists():
        return core.dgdecodenv.DGSource(dgiFile.resolve(True), **kwargs)
    else:
        info("Generating dgi file...", src)
        import os
        import subprocess as sub

        sub.Popen(f'dgindexnv -i "{path.name}" -h -o "{dgiFile.name}" -e', shell=True, stdout=sub.DEVNULL, cwd=path.parent.resolve(True)).wait()
        if path.with_suffix(".log").exists() and delete_dgi_log:
            os.remove(path.with_suffix(".log").resolve(True))
        return core.dgdecodenv.DGSource(dgiFile.resolve(True), **kwargs)


def frames_to_samples(frame: int, sample_rate: vs.AudioNode | int = 48000, fps: vs.VideoNode | Fraction = Fraction(24000, 1001)) -> int:
    """
    Converts a frame number to a sample number

    :param frame:           The frame number
    :param sample_rate:     Can be a flat number like 48000 (=48 kHz) or an AudioNode to get the sample rate from
    :param fps:             Can be a Fraction or a VideoNode to get the fps from

    :return:                The sample number
    """
    if frame == 0:
        return 0
    sample_rate = sample_rate.sample_rate if isinstance(sample_rate, vs.AudioNode) else sample_rate
    fps = Fraction(fps.fps_num, fps.fps_den) if isinstance(fps, vs.VideoNode) else fps
    return int(sample_rate * (fps.denominator / fps.numerator) * frame)


f2s = frames_to_samples


def generate_qp_file(clip: vs.VideoNode, start_frame: int = 0) -> str:
    filepath = Path(get_workdir(), f"qpfile_{start_frame}.txt")
    temp = Path(get_temp_workdir(), "qpfile.txt")
    if filepath.exists():
        info("Reusing existing QP File.")
        return str(filepath.resolve())
    info("Generating QP File...")

    clip = clip.resize.Bicubic(640, 360, format=vs.YUV410P8)
    clip = clip.wwxd.WWXD()
    if start_frame:
        clip = clip[start_frame:]
    out = ""
    for i in range(1, clip.num_frames):
        if clip.get_frame(i).props.Scenechange == 1:
            out += f"{i} I -1\n"

    with open(temp, "w") as file:
        file.write(out)

    temp.rename(filepath)
    clean_temp_files()

    return str(filepath.resolve())
