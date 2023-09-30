import subprocess
from vstools import vs
from fractions import Fraction
from dataclasses import dataclass

from muxtools import (
    PathLike,
    ensure_path_exists,
    debug,
    error,
    get_absolute_track,
    get_executable,
    TrackType,
    frame_to_timedelta,
    VideoFile,
    VideoTrack,
    MkvTrack,
)
from muxtools import SubFile as MTSubFile
from muxtools.subtitle.sub import SubFileSelf

__all__ = ["SubFile"]


@dataclass
class SubFile(MTSubFile):
    """
    Utility class representing a subtitle file with various functions to run on.

    :param file:            Can be a string, Path object or GlobSearch.
                            If the GlobSearch returns multiple results or if a list was passed it will merge them.

    :param container_delay: Set a container delay used in the muxing process later.
    :param source:          The file this sub originates from, will be set by the constructor.
    :param encoding:        Encoding used for reading and writing the subtitle files.
    """

    def truncate_by_video(
        self: SubFileSelf, source: PathLike | VideoTrack | MkvTrack | VideoFile | vs.VideoNode, fps: Fraction | None = None
    ) -> SubFileSelf:
        """
        Removes lines that start after the video ends and trims lines that extend past it.

        :param source:      Can be any video file or a VideoNode
        :param fps:         FPS Fraction; Will be parsed from the video by default
        """
        if isinstance(source, vs.VideoNode):
            frames = source.num_frames
            if not fps:
                fps = Fraction(source.fps_num, source.fps_den)
        else:
            if isinstance(source, VideoTrack) or isinstance(source, MkvTrack) or isinstance(source, VideoFile):
                file = ensure_path_exists(source.file, self)
            else:
                file = ensure_path_exists(source, self)
            # Unused variable, just used to have a simple validation
            track = get_absolute_track(file, 0, TrackType.VIDEO, self)
            ffprobe = get_executable("ffprobe")
            args = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate : stream_tags", str(file)]
            out = subprocess.run(args, capture_output=True, text=True)
            frames = 0

            for line in out.stdout.splitlines():
                if "=" not in line:
                    continue
                line = line.strip()
                if "r_frame_rate" in line and not fps:
                    fps = Fraction(line.split("=")[1])
                    debug(f"Parsed FPS from file: {fps}", self)
                elif "NUMBER_OF_FRAMES" in line:
                    line = line.split("=")[1]
                    try:
                        frames = int(line)
                        debug(f"Parsed frames from file: {frames}", self)
                    except:
                        continue

            if not fps or not frames:
                raise error(f"Could not parse frames or fps from file '{file.stem}'!", self)

        cutoff = frame_to_timedelta(frames + 1, fps)
        removed = 0
        trimmed = 0
        doc = self._read_doc()
        lines: list = doc.events
        for line in lines:
            if line.start > cutoff:
                lines.remove(line)
                removed += 1
                continue
            if line.end > cutoff:
                line.end = frame_to_timedelta(frames, fps)
                trimmed += 1

        if removed or trimmed:
            if removed:
                debug(f"Removed {removed} line{'s' if removed != 1 else ''} that started past the video", self)
            if trimmed:
                debug(f"Trimmed {trimmed} line{'s' if trimmed != 1 else ''} that extended past the video", self)
            doc.events = lines
            self.__update_doc(doc)

        return self
