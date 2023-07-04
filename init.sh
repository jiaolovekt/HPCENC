#!/bin/bash

if [ -z "$ENCROOT" ] ; then
	export ENCROOT="$PWD" 
fi
module use "$ENCROOT/deps/modfiles"

module load avsp3.7.2  cython  DevIL  ffmpeg-5.1.3  ffms2  gcc-12.3  hpcenc  lsmash-2.14.5  opencl  python-3.11.3 x264-gcc12-broadwell x265-icc vs-r62    zimg-3.0.4 mkvtoolnix-78
