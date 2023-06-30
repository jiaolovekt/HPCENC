#!/bin/bash

usage()
{
	echo "Usage: webbatch [ -P partition -p profile ] -n projectname -i index -r [1080/720] -l [GB/B5] -c [264/265] [ -t job_template ] [ -m avs/vs ]" 1>&2
	exit 0
}
#checkopt
if [ "$#" -lt "5" ] ; then
	usage
fi

SLURM_TEMPLATE=

#getopt
while getopts P:p:n:i:r:l:m:c:t:h OP ; do
        case "$OP" in
	P)
		PART="$OPTARG"
	;;
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
		MODE="$OPTARG"
	;;
        c)
                [[ "$OPTARG" =~ 264|avc|AVC ]] && X264_EN=1 && CODEC=264
                [[ "$OPTARG" =~ 265|hevc|HEVC ]] && X265_EN=1 && CODEC=265
                [ -z "$X264_EN" ] && [ -z "$X265_EN" ] && echo "wrong codec $OPTARG specified" 1>&2 && exit 9
        ;;
	t)
		SLURM_TEMPLATE="$OPTARG"
	;;
        *)
		usage
        ;;
        esac
done
[ -n "$PART" ] && PART="-p $PART"
[ -z "$SLURM_TEMPLATE" ] && SLURM_TEMPLATE="${ENCROOT}/config/sub_web"

JNAME="${PROJECT}_${INDEX}_${LANGG}_${RESO}_${CODEC}.${MODE}"

sbatch $PART -o "$JNAME".log -e "$JNAME".log -J "$JNAME" --export=ALL,PROFILE="$PROFILE",PROJECT="$PROJECT",INDEX="$INDEX",RESO="$RESO",LANGG="$LANGG",CODEC="$CODEC",MODE="$MODE" "$SLURM_TEMPLATE"
