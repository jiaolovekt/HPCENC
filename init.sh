#!/bin/bash

if ! module > /dev/null ; then
	echo "environment modules not found or not configured on this cluster at the current shell."
	echo "try apt install environment-modules or yum install environment-modules.x86_64 if you are root"
	echo "for details, check https://github.com/cea-hpc/modules"
	return 1
fi

if [ -z "$ENCROOT" ] ; then
	export ENCROOT="$PWD" 
fi
module use "$ENCROOT/deps/modfiles"
module load avsp3.7.2  cython  DevIL  ffmpeg-5.1.3  ffms2  gcc-12.3  hpcenc  lsmash-2.14.5  opencl  python-3.11.3 x264-gcc12-broadwell x265-gcc12-broadwell vs-r62 zimg-3.0.4 mkvtoolnix-78

#CHOOSE ONE IF YOU NEED VSFM SUPPORT
#module load wine-8.0.1
#module load wine64-8.0.1
module load wine-8.0.1


. config/func
