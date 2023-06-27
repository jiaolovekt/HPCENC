#!/bin/bash

if [ -z "$ENCROOT" ] ; then
	export ENCROOT="$PWD" 
fi
module use "$ENCROOT/deps/modfiles"

module load avsp3.7.2  cython  DevIL  ffmpeg5-gcc12  ffms2  fribidi  gcc-12.3  harfbuzz-7.3.0  hpcenc  libass-0.17.1  libfdk_aac  lsmash-2.14.5  opencl  python-3.11.3 x264-gcc12-broadwell x265-gcc12-broadwell vs-r62    zimg-3.0.4
