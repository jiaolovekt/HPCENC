#!/bin/bash

export ENCROOT="$PWD" && module use "$ENCROOT/deps/modfiles"

module load avsp3.7.2  cython  DevIL  ffmpeg-5-pre  ffms2  fribidi  gcc-12.3  harfbuzz-7.3.0  hpcenc  libass-0.17.1  libfdk_aac  lsmash-2.14.5  opencl  python-3.11.3  vs-r62  x264-b-broadwell  x265-p-broadwell  zimg-3.0.4
