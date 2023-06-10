#!/bin/bash

DEBUG=0
DRYRUN=1
CONFIGDIR="$PWD/config"
. "$CONFIGDIR"/func

#Env Part
WORKINGDIR="$PWD/template"
SRCDIR="$WORKINGDIR/src"
ASSDIR="$WORKINGDIR/ass"
FONTDIR="$WORKINGDIR/font"
TMPDIR="$WORKINGDIR/tmp" # or /tmp
OUTDIR="$WORKINGDIR/out"

#Env Part End

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

#getopt
while getopts p:n:i:r:l:h OP ; do
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
	*)
		echo "Usage: encode [ -p profile ] -n projectname -i index -r [1080/720] -l [GB/B5]" 1>&2
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
#check path
for D in '$WORKINGDIR' '$SRCDIR' '$ASSDIR' '$SCDIR' '$FONTDIR' '$TMPDIR' '$OUTDIR' ; do 
	if [ -d "$(eval echo $D)" ] && [ -w "$(eval echo $D)" ] ; then
		logg "$D exists and writeable" debug
	else
		logg "$D not exist or not writeable" err
		exit 4
	fi
done
#Namemap
logg "Encode $PROJECT Eps $INDEX $RESO $LANGG"
[[ "$RESO" =~ 1080 ]] && RESO=1080p
[[ "$RESO" =~ 720 ]] && RESO=720p
[ "$LANGG" = GB ] && LANGG="$GBflag"	#GB by default
[ "$LANGG" = B5 ] || [ "$LANGG" = "BIG5" ] && LANGG="$B5flag"	#BIG5 by default

nmmap=$(grep "$PROJECT" "$CONFIGDIR"/namemap | grep -v ^# )
if [ -z "$nmmap" ] ; then 
	logg "No project namemap defined" err
	exit 6
else
	eval "$nmmap"
fi
logg "Output name $OUTNAME"

#Encode part
#TODO: seperate logs?
#X264 part

if [[ $X264_EN =~ (1|Y|y|True|true|TRUE) ]] ; then
	X264_TMP="${TMPDIR}/${PROJECT}_${INDEX}_${LANGG}_${RESO}.264"
	#check previous tmp
	if [ -f "$X264_TMP" ] && [ "$SkipEVideo" != "0" ] ; then
		 logg "Warning: Skip video encode, using existing $X264_TMP" warn 
	else
		#check avs exists
		getwebavs 264
		#check bins
		for f in "$X264_exec" "$FFMPEG_exec" "$MEDIAINFO_exec" ;do
			if ! [ -x "$(which $f)" ] ; then
				logg "$f not found or no exec perm" err
				exit 8
			fi
		done
		#check src
		getwebsrc
		getwebsrcinfo
		#video part
		if [ "$DRYRUN" = 0 ] ; then
		logg "starting X264 video encode" info
		"$X264_exec" --level 5.1 --crf "$X264_crf" --tune "$X264_tune" --keyint "$X264_keyint" --min-keyint 1 --threads "$X264_threads" --bframes "$X264_bframes" --qpmin 0 --qpmax "$X264_qpmax"  --b-adapt "$X264_badapt" --ref "$X264_ref" --chroma-qp-offset -2 --vbv-bufsize "$X264_vbvbuf" --vbv-maxrate "$X264_vbvmaxrate" --qcomp 0.7 --rc-lookahead "$X264_lookahead" --aq-strength 0.9 --deblock 1:1  --direct auto  --merange "$X264_merange" --me "$X264_me" --subme "$X264_subme" --trellis 2 --psy-rd 0.6:0.10 --no-fast-pskip --stylish --aq-mode "$X264_aqmode" --fgo 4 --partitions all --opts 0  --fade-compensate 0.10 "$X264_custom" -o "$X264_TMP" "$AVSFILE"
		else
		logg "$X264_exec --level 5.1 --crf $X264_crf --tune $X264_tune --keyint $X264_keyint --min-keyint 1 --threads $X264_threads --bframes $X264_bframes --qpmin 0 --qpmax $X264_qpmax  --b-adapt $X264_badapt --ref $X264_ref --chroma-qp-offset -2 --vbv-bufsize $X264_vbvbuf --vbv-maxrate $X264_vbvmaxrate --qcomp 0.7 --rc-lookahead $X264_lookahead --aq-strength 0.9 --deblock 1:1  --direct auto  --merange $X264_merange --me $X264_me --subme $X264_subme --trellis 2 --psy-rd 0.6:0.10 --no-fast-pskip --stylish --aq-mode $X264_aqmode --fgo 4 --partitions all --opts 0  --fade-compensate 0.10 $X264_custom -o $X264_TMP $AVSFILE" info
		fi
	fi
fi
#X265 part
if [[ $X265_EN =~ (1|Y|y|True|true|TRUE) ]] ; then
	X265_TMP="${TMPDIR}/${PROJECT}_${INDEX}_${LANGG}_${RESO}.265"
	#check previous tmp
	if [ -f "$X265_TMP" ] && [ "$SkipEVideo" != "0" ] ; then
		 logg "Warning: Skip video encode, using existing $X265_TMP" warn 
	else
		#check avs exists
		getwebavs 265
		#check bins
		for f in "$X265_exec" "$FFMPEG_exec" "$MEDIAINFO_exec" ;do
			if ! [ -x "$(which $f)" ] ; then
				logg "$f not found or no exec perm" err
				exit 8
			fi
		done
		#check src
		getwebsrc
		getwebsrcinfo
		#video part
		if [ "$DRYRUN" = 0 ] ; then
		logg "starting X265 video encode" info
		"$X265_exec" --input-depth 10 --preset "$X265_preset" --no-open-gop --profile "$X265_profile" --crf "$X265_crf" --deblock -1:-1 --ref "$X265_ref" --keyint "$X265_keyint" --min-keyint 1 --rd "$X265_rd" --ctu "$X265_ctu" --max-tu-size "$X265_maxtu" --no-amp --rdoq-level "$X265_rdoq"  --rdpenalty 1 --me "$X265_me" --subme "$X265_subme" --merange "$X265_merange" --temporal-mvp --weightp --weightb --b-adapt "$X265_badapt" --psy-rd 4.0  --psy-rdoq "$X265_psyrdoq" --aq-mode "$X265_aqmode" --aq-strength 1.0 --qg-size "$X265_qgsize" --cutree --qcomp 0.7 --colormatrix "$X265_colormatx" --allow-non-conformance --rc-lookahead "$X265_lookahead" --scenecut 40 --dither --no-sao "$X265_custom" --output "$X265_TMP" "$AVSFILE"
		else
		logg "$X265_exec --input-depth 10 --preset $X265_preset --no-open-gop --profile $X265_profile --crf $X265_crf --deblock -1:-1 --ref $X265_ref --keyint $X265_keyint --min-keyint 1 --rd $X265_rd --ctu $X265_ctu --max-tu-size $X265_maxtu --no-amp --rdoq-level $X265_rdoq  --rdpenalty 1 --me $X265_me --subme $X265_subme --merange $X265_merange --temporal-mvp --weightp --weightb --b-adapt $X265_badapt --psy-rd 4.0  --psy-rdoq $X265_psyrdoq --aq-mode $X265_aqmode --aq-strength 1.0 --qg-size $X265_qgsize --cutree --qcomp 0.7 --colormatrix $X265_colormatx --allow-non-conformance --rc-lookahead $X265_lookahead --scenecut 40 --dither --no-sao $X265_custom --output $X265_TMP $AVSFILE" info
		fi
	fi
fi
#AAC part 
if [ "$X264_EN" != "0" ] && [ "$X264_AUD" = "AAC" ] ;then
	X264_AUD_TMP="${TMPDIR}/${PROJECT}_${INDEX}_${LANGG}_${RESO}_264.aac"
	# check previous tmp
	if [ -f "$X264_AUD_TMP" ] && [ "$SkipEAudio" != "0" ] ; then
		logg "Warning Skip Audio encode, using existing $X264_AUD_TMP" warn
	elif [ "$AudDirectmux" != "0" ] && [ "$Audiorate" -le "$AudDirectmuxthreshold" ] && [[ "$AudioFmt" =~ AAC|aac ]]; then
		#acopy
		logg "Audio bit rate $Audiorate lower than $AudDirectmuxthreshold, direct mux" info
		if [ "$DRYRUN" = 0 ] ; then
		"$FFMPEG_exec" -i "${SRCDIR}/${SRCFILE}" -vn -c:a copy "$X264_AUD_TMP"
		else
		logg "$FFMPEG_exec -i ${SRCDIR}/${SRCFILE} -vn -c:a copy $X264_AUD_TMP" info
		fi
	else
		#encode
		if [ "$DRYRUN" = 0 ] ; then
		"$FFMPEG_exec" -i "$AVSFILE" -vn -c:a aac -q "$AAC_Q" "$X264_AUD_TMP"
		else
		logg "$FFMPEG_exec -i $AVSFILE -vn -c:a aac -q $AAC_Q $X264_AUD_TMP" info
		fi
	fi
fi
#dup! may improve later
if [ "$X265_EN" != "0" ] && [ "$X265_AUD" = "AAC" ] ;then
	X265_AUD_TMP="${TMPDIR}/${PROJECT}_${INDEX}_${LANGG}_${RESO}_265.aac"
	# check previous tmp
	if [ -f "$X265_AUD_TMP" ] && [ "$SkipEAudio" != "0" ] ; then
		logg "Warning Skip Audio encode, using existing $X265_AUD_TMP" warn
	elif [ "$AudDirectmux" != "0" ] && [ "$Audiorate" -le "$AudDirectmuxthreshold" ] && [[ "$AudioFmt" =~ AAC|aac ]]; then
		#acopy
		logg "Audio bit rate $Audiorate lower than $AudDirectmuxthreshold, direct mux" info
		if [ "$DRYRUN" = 0 ] ; then
		"$FFMPEG_exec" -i "${SRCDIR}/${SRCFILE}" -vn -c:a copy "$X265_AUD_TMP"
		else
		logg "$FFMPEG_exec -i ${SRCDIR}/${SRCFILE} -vn -c:a copy $X265_AUD_TMP" info
		fi
	else
		#encode
		if [ "$DRYRUN" = 0 ] ; then
		"$FFMPEG_exec" -i "$AVSFILE" -vn -c:a aac -q "$AAC_Q" "$X265_AUD_TMP"
		else
		logg "$FFMPEG_exec -i $AVSFILE -vn -c:a aac -q $AAC_Q $X265_AUD_TMP" info
		fi
	fi
fi

#FLAC part - X265 only
#currently not available. Use FLAC for WEBrip??

#mux part
if [ "$X264_EN" != "0" ] ; then
	if [ "$DRYRUN" = 0 ] ; then
	"$FFMPEG_exec" -i "$X264_TMP" -i "$X264_AUD_TMP" -c:v copy -c:a copy -metadata comment="$MCOMMENT" -map 0:v -map 1:a "{$OUTDIR}/${OUTPUT}.${X264_MUX}"
	else
	logg "$FFMPEG_exec -i $X264_TMP -i $X264_AUD_TMP -c:v copy -c:a copy -metadata comment=$MCOMMENT -map 0:v -map 1:a ${OUTDIR}/${OUTPUT}.${X264_MUX}" info
	fi
fi
if [ "$X265_EN" != "0" ] ; then
	if [ "$DRYRUN" = 0 ] ; then
	"$FFMPEG_exec" -i "$X265_TMP" -i "$X265_AUD_TMP" -c:v copy -c:a copy -metadata comment="$MCOMMENT" -map 0:v -map 1:a "{$OUTDIR}/${OUTPUT}.${X265_MUX}"
	else
	logg "$FFMPEG_exec -i $X265_TMP -i $X265_AUD_TMP -c:v copy -c:a copy -metadata comment=$MCOMMENT -map 0:v -map 1:a ${OUTDIR}/${OUTPUT}.${X265_MUX}" info
	fi
fi

#Postprocess
cleanuptmp
#may call sth
