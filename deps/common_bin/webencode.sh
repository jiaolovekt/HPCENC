#!/bin/bash
set -e
DEBUG=1 #Verbose logging
DRYRUN=0 #dryrun

#Env Part
WORKINGDIR="$ENCROOT/template"
SRCDIR="$WORKINGDIR/src"
ASSDIR="$WORKINGDIR/ass"
FONTDIR="$WORKINGDIR/font"
TMPDIR="$WORKINGDIR/tmp" # or /tmp
OUTDIR="$WORKINGDIR/out"
ASSFILE=
AVSFILE=
CONFIGDIR="${ENCROOT}/config"
. "$CONFIGDIR"/func
#Env Part End

usage()
{
	echo "Usage: encode [ -p profile ] -n projectname -i index -r [1080/720] -l [GB/B5] -c [264/265] [ -m AVS/VS ]" 1>&2
	exit 0
}

#load default config
if [ -r "$CONFIGDIR"/defaultprofile ] ; then
	logg "load default config" debug
	. "$CONFIGDIR"/defaultprofile
else
	logg "default config not exist" err
	exit 2
fi
#check namemap
if ! [ -r "$CONFIGDIR"/namemap ] ; then
	logg "namemap not exist" err
	exit 5
fi

#checkopt
if [ "$#" -lt "5" ] ; then
	usage
fi

#getopt
while getopts p:n:i:r:l:m:c:h OP ; do
	case "$OP" in
	p)
		PROFILE="$OPTARG"
	;;
	n)
		PROJECT="$OPTARG"
	;;
	i)
		INDEX="$OPTARG"
	;;
	r)
		RESO="$OPTARG"
	;;
	l)
		LANGG="$OPTARG"
	;;
	m)
		EMODE="$OPTARG"
	;;
	c)
		#TODO: Rework full struct
		[[ "$OPTARG" =~ 264|avc|AVC ]] && X264_EN=1 && CODEC=264
		[[ "$OPTARG" =~ 265|hevc|HEVC ]] && X265_EN=1 && CODEC=265
		[ -z "$X264_EN" ] && [ -z "$X265_EN" ] && logg "wrong codec $OPTARG specified" err && exit 9
	;;
	*)
		usage
	;;
	esac
done
#project spec profile
if [ -n "$PROFILE" ] ; then
	if [ -r "$PROFILE" ] ; then
		. "$PROFILE"
		logg "load profile $PROFILE" info
	elif [ -r "$CONFIGDIR"/"$PROFILE" ] ; then
		PROFILE="$CONFIGDIR"/"$PROFILE"
		. "$PROFILE"
		logg "load profile $PROFILE" info
	else 
		logg "cannot find profile $PROFILE" err
		exit 3
	fi
elif [ -r "$CONFIGDIR"/"$PROJECT"_profile ] ; then
	. "$CONFIGDIR"/"$PROJECT"_profile
	logg "autoload profile ${CONFIGDIR}/${PROJECT}_profile" info
else
	logg "continue with default profile" debug
fi
[ -n "$EMODE" ] && MODE="$EMODE"
#check path
for D in '$WORKINGDIR' '$SRCDIR' '$ASSDIR' '$SCDIR' '$FONTDIR' '$TMPDIR' '$OUTDIR' ; do 
	if [ -d "$(eval echo $D)" ] && [ -w "$(eval echo $D)" ] ; then
		logg "$(eval echo "$D") exists and writeable" debug
	else
		logg "$D not exist or not writeable" err
		exit 4
	fi
done
#Namemap
logg "Encode $PROJECT Eps $INDEX $RESO $LANGG Codec $CODEC input $MODE" info
[[ "$RESO" =~ 1080 ]] && RESO=1080p
[[ "$RESO" =~ 720 ]] && RESO=720p
[ "$LANGG" = GB ] && LANGG="$GBflag"	#GB by default
[ "$LANGG" = B5 ] || [ "$LANGG" = "BIG5" ] && LANGG="$B5flag"	#BIG5 by default
if [[ "$MODE" =~ avs|AVS ]] ; then
	 MODE=avs 
elif [[ "$MODE" =~ vs|VS|vpy|VPY ]] ; then
	 MODE=vpy
else
	logg "wrong mode specifiled, should be avs/vs" err
	exit 10
fi
logg "Encode $PROJECT Eps $INDEX $RESO $LANGG Codec $CODEC input $MODE" info
nmmap=$(grep "$PROJECT" "$CONFIGDIR"/namemap | grep -v ^# )
if [ -z "$nmmap" ] ; then 
	logg "No project namemap defined" err
	exit 6
else
	eval "$nmmap"
fi
logg "Output name $OUTNAME"

#Encode part
#TODO maybe a full rework to mix altogether
getwebsrc
getwebsrcinfo

getwebscript "$CODEC" "$MODE"

if [ "$MODE" = vpy ] ; then
	getvpyinfo
fi

#logs
if [ "$Separatelogfile" = "1" ] ; then
	X26x_logpara="--log-level info --log-file-level full --log-file ${TMPDIR}/${PROJECT}_${INDEX}_${LANGG}_${RESO}_${CODEC}.log"
fi

X26x_TMP="${TMPDIR}/${PROJECT}_${INDEX}_${LANGG}_${RESO}.${CODEC}"
#X264 part
if [ "$X264_EN" = "1" ] ; then
	#check previous tmp
	if [ -f "$X264_TMP" ] && [ "$SkipEVideo" != "0" ] ; then
		 logg "Warning: Skip video encode, using existing $X264_TMP" warn 
	else
		#check avs exists
		#check bins
		for f in "$X264_exec" "$FFMPEG_exec" "$MEDIAINFO_exec" "vspipe" ;do
			if ! [ -x "$(which "$f")" ] ; then
				logg "$f not found or no exec perm" err
				exit 8
			fi
		done
		#video part
	if [ $MODE = vpy ] ; then
		O_ISCRIPT="$ISCRIPT"
		X264_exec="vspipe -c y4m \"$ISCRIPT\" - | $X264_exec"
		ISCRIPT="--demuxer y4m -"
	fi
		cmdline="$X264_exec --level 5.1 --crf $X264_crf --tune $X264_tune --keyint $X264_keyint --min-keyint 1 --threads $X264_threads --bframes $X264_bframes --qpmin 0 --qpmax $X264_qpmax  --b-adapt $X264_badapt --ref $X264_ref --chroma-qp-offset -2 --vbv-bufsize $X264_vbvbuf --vbv-maxrate $X264_vbvmaxrate --qcomp 0.7 --rc-lookahead $X264_lookahead --aq-strength 0.9 --deblock 1:1  --direct auto  --merange $X264_merange --me $X264_me --subme $X264_subme --trellis 2 --psy-rd 0.6:0.10 --no-fast-pskip --stylish --aq-mode $X264_aqmode --fgo 4 --partitions all --opts 0  --fade-compensate 0.10 ${X264_custom} ${X26x_logpara} -o \"$X26x_TMP\" $ISCRIPT"
		if [ "$DRYRUN" = 0 ] ; then
			logg "starting X264 video encode" info
			logg "$cmdline" debug
			eval $cmdline && logg "X264 encode done" info
		else
			logg "$cmdline" info
		fi
	fi
	[ -n "$O_ISCRIPT" ] && ISCRIPT="$O_ISCRIPT"
fi
#X265 part
if [ "$X265_EN" = "1" ] ; then
	#check previous tmp
	if [ -f "$X265_TMP" ] && [ "$SkipEVideo" != "0" ] ; then
		 logg "Warning: Skip video encode, using existing $X265_TMP" warn 
	else
		#check avs exists
		#check bins
		for f in "$X265_exec" "$FFMPEG_exec" "$MEDIAINFO_exec" "vspipe" ;do
			if ! [ -x "$(which "$f")" ] ; then
				logg "$f not found or no exec perm" err
				exit 8
			fi
		done
		#video part
		if [ $MODE = vpy ] ; then
			O_ISCRIPT="$ISCRIPT"
			X265_exec="vspipe -c y4m \"$ISCRIPT\" - | $X265_exec"
			ISCRIPT="--y4m -"
		fi
		logg "starting X265 video encode" info
		cmdline="$X265_exec --preset $X265_preset --no-open-gop --profile $X265_profile --crf $X265_crf --deblock -1:-1 --ref $X265_ref --keyint $X265_keyint --min-keyint 1 --rd $X265_rd --ctu $X265_ctu --max-tu-size $X265_maxtu --no-amp --rdoq-level $X265_rdoq  --rdpenalty 1 --me $X265_me --subme $X265_subme --merange $X265_merange --temporal-mvp --weightp --weightb --b-adapt $X265_badapt --psy-rd 4.0  --psy-rdoq $X265_psyrdoq --aq-mode $X265_aqmode --aq-strength 1.0 --qg-size $X265_qgsize --cutree --qcomp 0.7 --colormatrix $X265_colormatx --allow-non-conformance --rc-lookahead $X265_lookahead --scenecut 40 --dither --no-sao $X265_custom ${X26x_logpara} --output \"$X26x_TMP\" $ISCRIPT"
                if [ "$DRYRUN" = 0 ] ; then
			logg "$cmdline" debug
			eval "$cmdline"
			logg "X265 encode done" info
                else
			logg "$cmdline" info
                fi
	fi
	[ -n "$O_ISCRIPT" ] && ISCRIPT="$O_ISCRIPT"
fi
#New AAC part
if [ "$X264_AUD" = "AAC" ] || [ "$X265_AUD" = "AAC" ] ; then
	X26x_AUD_TMP="${TMPDIR}/${PROJECT}_${INDEX}_${LANGG}_${RESO}_${CODEC}.aac"
	if [ -f "$X26x_AUD_TMP" ] && [ "$SkipEAudio" != "0" ] ; then
                logg "Warning Skip Audio encode, using existing $X26x_AUD_TMP" warn
	elif [ "$AudDirectmux" != "0" ] && [ "$Audiorate" -le "$AudDirectmuxthreshold" ] && [[ "$AudioFmt" =~ AAC|aac ]]; then
		#acopy
                logg "Audio bit rate $Audiorate lower than $AudDirectmuxthreshold, direct mux" info
                cmdline="$FFMPEG_exec -nostdin -i \"${SRCFILE}\" -vn -c:a copy \"$X26x_AUD_TMP\" -y"
                if [ "$DRYRUN" = 0 ] ; then
			logg "$cmdline" debug
			eval "$cmdline"
                else
			logg "$cmdline" info
                fi
	elif [ "$VPY_AFallback" = "1" ] ; then
		#vpy has no audio, fallback to src
		logg "vpy audio fallback, using $SRCFILE" warn
                cmdline="$FFMPEG_exec -nostdin -i \"${SRCFILE}\" -vn -c:a copy \"$X26x_AUD_TMP\" -y"
                if [ "$DRYRUN" = 0 ] ; then
			logg "$cmdline" debug
			eval "$cmdline"
                else
			logg "$cmdline" info
                fi
	else
		#encode
		logg "Start audio encode" info
		case "$MODE" in
		avs)
		cmdline="$FFMPEG_exec -nostdin -i \"$ISCRIPT\" -vn -c:a aac -q $AAC_Q \"$X26x_AUD_TMP\" -y"
		if [ "$DRYRUN" = 0 ] ; then
			logg "$cmdline" debug
			eval "$cmdline"
		else
			logg "$cmdline" info
		fi
		;;
		vpy)
		cmdline="vspipe -c wav \"$ISCRIPT\" -o 1 - | $FFMPEG_exec -nostdin -i - -vn -c:a aac -q $AAC_Q \"$X26x_AUD_TMP\" -y"
		if [ "$DRYRUN" = 0 ] ; then
			logg "$cmdline" debug
			eval "$cmdline"
		else
			logg "$cmdline" info
		fi
		;;
		*)
		;;
		esac
	fi
fi

#AAC part - deprecated
oldaac()
{
if [ "$X264_EN" = "1" ] && [ "$X264_AUD" = "AAC" ] ;then
	X264_AUD_TMP="${TMPDIR}/${PROJECT}_${INDEX}_${LANGG}_${RESO}_264.aac"
	# check previous tmp
	if [ -f "$X264_AUD_TMP" ] && [ "$SkipEAudio" != "0" ] ; then
		logg "Warning Skip Audio encode, using existing $X264_AUD_TMP" warn
	elif [ "$AudDirectmux" != "0" ] && [ "$Audiorate" -le "$AudDirectmuxthreshold" ] && [[ "$AudioFmt" =~ AAC|aac ]]; then
		#acopy
		logg "Audio bit rate $Audiorate lower than $AudDirectmuxthreshold, direct mux" info
		if [ "$DRYRUN" = 0 ] ; then
		"$FFMPEG_exec" -nostdin -i "${SRCFILE}" -vn -c:a copy "$X264_AUD_TMP" -y
		else
		logg "$FFMPEG_exec -i ${SRCFILE} -vn -c:a copy $X264_AUD_TMP -y" info
		fi
	else
		#encode
		if [ "$DRYRUN" = 0 ] ; then
		"$FFMPEG_exec" -nostdin -i "$ISCRIPT" -vn -c:a aac -q "$AAC_Q" "$X264_AUD_TMP" -y
		else
		logg "$FFMPEG_exec -i $ISCRIPT -vn -c:a aac -q $AAC_Q $X264_AUD_TMP -y" info
		fi
	fi
fi
if [ "$X265_EN" = "1" ] && [ "$X265_AUD" = "AAC" ] ;then
	X265_AUD_TMP="${TMPDIR}/${PROJECT}_${INDEX}_${LANGG}_${RESO}_265.aac"
	# check previous tmp
	if [ -f "$X265_AUD_TMP" ] && [ "$SkipEAudio" != "0" ] ; then
		logg "Warning Skip Audio encode, using existing $X265_AUD_TMP" warn
	elif [ "$AudDirectmux" != "0" ] && [ "$Audiorate" -le "$AudDirectmuxthreshold" ] && [[ "$AudioFmt" =~ AAC|aac ]]; then
		#acopy
		logg "Audio bit rate $Audiorate lower than $AudDirectmuxthreshold, direct mux" info
		if [ "$DRYRUN" = 0 ] ; then
		"$FFMPEG_exec" -nostdin -i "${SRCFILE}" -vn -c:a copy "$X265_AUD_TMP" -y
		else
		logg "$FFMPEG_exec -i ${SRCFILE} -vn -c:a copy $X265_AUD_TMP -y" info
		fi
	else
		#encode
		if [ "$DRYRUN" = 0 ] ; then
		"$FFMPEG_exec" -nostdin -i "$ISCRIPT" -vn -c:a aac -q "$AAC_Q" "$X265_AUD_TMP" -y
		else
		logg "$FFMPEG_exec -i $ISCRIPT -vn -c:a aac -q $AAC_Q $X265_AUD_TMP -y" info
		fi
	fi
fi
}
#FLAC part - X265 only
#currently not available. Use FLAC for WEBrip??

#New mux part
logg "mux $X26x_TMP $X26x_AUD_TMP" info
case "$CODEC" in 
	264)
	MUX="$X264_MUX"
	;;
	265)
	MUX="$X265_MUX"
	;;
esac

case "$MUX" in
	mp4)
	[ "$MCOMMENTCPU" = 1 ] && commentcpu
	[ "$MCOMMENTCPU" = 1 ] && MCOMMENT="$MCOMMENT on $CPUCNT x $CPUID"
	cmdline="$FFMPEG_exec -nostdin -i \"$X26x_TMP\" -i \"$X26x_AUD_TMP\" -c:v copy -c:a copy -metadata comment=$MCOMMENT -map 0:v -map 1:a \"${OUTDIR}/${OUTNAME}.${MUX}\" -y"
	;;
	mkv)
	cmdline="mkvmerge -o \"${OUTDIR}/${OUTNAME}.${MUX}\" -v \"$X26x_TMP\" --language 0:jpn \"$X26x_AUD_TMP\" --global-tags \"$MKVCOMMENT\""
	;;
	*)
	;;
esac
if [ "$DRYRUN" = 0 ] ; then
	logg "$cmdline" debug
	eval "$cmdline"
else
	logg "$cmdline" info
fi

#mux part - deprecated
oldmux()
{
if [ "$X264_EN" = "1" ] ; then
	if [ "$DRYRUN" = 0 ] ; then
	"$FFMPEG_exec" -nostdin -i "$X264_TMP" -i "$X26x_AUD_TMP" -c:v copy -c:a copy -metadata comment="$MCOMMENT" -map 0:v -map 1:a "${OUTDIR}/${OUTNAME}.${X264_MUX}" -y
	else
	logg "$FFMPEG_exec -i $X264_TMP -i $X26x_AUD_TMP -c:v copy -c:a copy -metadata comment=$MCOMMENT -map 0:v -map 1:a ${OUTDIR}/${OUTNAME}.${X264_MUX} -y" info
	fi
fi
if [ "$X265_EN" = "1" ] ; then
	if [ "$DRYRUN" = 0 ] ; then
	"$FFMPEG_exec" -nostdin -i "$X265_TMP" -i "$X26x_AUD_TMP" -c:v copy -c:a copy -metadata comment="$MCOMMENT" -map 0:v -map 1:a "${OUTDIR}/${OUTNAME}.${X265_MUX}" -y
	else
	logg "$FFMPEG_exec -i $X265_TMP -i $X26x_AUD_TMP -c:v copy -c:a copy -metadata comment=$MCOMMENT -map 0:v -map 1:a ${OUTDIR}/${OUTNAME}.${X265_MUX} -y" info
	fi
fi
}
#Postprocess
cleanuptmp
#may call sth
