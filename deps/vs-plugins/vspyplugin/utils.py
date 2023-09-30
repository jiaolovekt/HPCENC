from __future__ import annotations

import vapoursynth as vs
from vstools import CustomRuntimeError, HoldsVideoFormatT, get_video_format

__all__ = [
    'get_c_dtype_short',
    'get_c_dtype_long'
]


def get_c_dtype_short(clip: HoldsVideoFormatT) -> str:
    fmt = get_video_format(clip)

    if fmt.sample_type is vs.FLOAT:
        return get_c_dtype_long(clip)

    if fmt.bytes_per_sample == 1:
        return 'uchar'
    elif fmt.bytes_per_sample == 2:
        return 'ushort'
    elif fmt.bytes_per_sample == 4:
        return 'uint'

    raise CustomRuntimeError(func=get_c_dtype_short)


def get_c_dtype_long(clip: HoldsVideoFormatT) -> str:
    fmt = get_video_format(clip)

    if fmt.sample_type is vs.FLOAT:
        if fmt.bytes_per_sample == 2:
            return 'half'
        return 'float'

    if fmt.bytes_per_sample == 1:
        return 'unsigned char'
    elif fmt.bytes_per_sample == 2:
        return 'unsigned short'
    elif fmt.bytes_per_sample == 4:
        return 'unsigned int'

    raise CustomRuntimeError(func=get_c_dtype_long)
