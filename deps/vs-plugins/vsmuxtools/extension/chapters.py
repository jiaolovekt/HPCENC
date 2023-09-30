from fractions import Fraction
from muxtools import parse_chapters_bdmv, PathLike, GlobSearch, Chapter, Chapters as Ch

from ..utils.src import src_file

__all__ = ["Chapters"]


class Chapters(Ch):
    def __init__(
        self, chapter_source: src_file | PathLike | GlobSearch | Chapter | list[Chapter], fps: Fraction | None = None, _print: bool = True
    ) -> None:
        """
        Convenience class for chapters

        :param chapter_source:      Input either src_file/FileInfo, txt with ogm chapters, xml or (a list of) self defined chapters.
        :param fps:                 Needed for timestamp convertion. Gets the fps from the clip if src_file and otherwise assumes 24000/1001.
        :param _print:              Prints chapters after parsing and after trimming.
        """
        if isinstance(chapter_source, src_file):
            clip_fps = Fraction(chapter_source.src.fps_num, chapter_source.src.fps_den)
            self.fps = fps if fps else clip_fps
            self.chapters = parse_chapters_bdmv(chapter_source.file, self.fps, chapter_source.src_cut.num_frames, _print)
            if self.chapters and chapter_source.trim:
                self.trim(chapter_source.trim[0], chapter_source.trim[1], chapter_source.src_cut.num_frames)
                if _print:
                    print("After trim:")
                    self.print()
        else:
            super().__init__(chapter_source, fps if fps else Fraction(24000, 1001), _print)
