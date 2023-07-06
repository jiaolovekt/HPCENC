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
while getopts P:N:p:n:i:r:l:m:c:t:h OP ; do
        case "$OP" in
	P)
		PART="$OPTARG"
	;;
	N)
		NTHREADS="$OPTARG"
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
#limit 264 sub within 1 socket
#State=UP TotalCPUs=16000 TotalNodes=400 SelectTypeParameters=NONE
if [ -z "$NTHREADS" ] && [ "$CODEC" = 264 ]  ; then
	NTHREADS=0
	local State=
	local TotalCPUs=
	local TotalNodes=
	local SelectTypeParameters=
	if [ -n "$PART" ] && ! echo "$PART" | grep -Fq ',' ; then
	#only process single partition passwd
		eval $(scontrol show partition "$PART" | grep TotalCPUS)
	else
	# process default partition
		eval $(scontrol show partition | grep -F 'Default=YES' -A6 | grep TotalCPUs)
	fi
	if [ -n "$TotalCPUs" ] && [ -n "$TotalNodes" ] ; then
	#calc single socket, assume DP not MP
		NTHREADS=$((TotalCPUS/TotalNodes/2))
	fi
	# apply only when 16 <= threads <=32
	if [ "$NTHREADS" -ge 16 ] && [ "$NTHREADS" -le 32 ] ; then
		echo "use $NTHREADS threads for $CODEC job"
		THREADS="-n $NTHREADS"
	fi
elif [ -n "$NTHREADS" ] ; then
	echo "use $NTHREADS threads for $CODEC job"
	THREADS="-n $NTHREADS"
fi
				
		
[ -n "$PART" ] && PART="-p $PART"
[ -z "$SLURM_TEMPLATE" ] && SLURM_TEMPLATE="${ENCROOT}/config/sub_web"
[ -n "$MODE" ] && MODE=".${MODE}"
JNAME="${PROJECT}_${INDEX}_${LANGG}_${RESO}_${CODEC}${MODE}"


sbatch $PART $THREADS -o "$JNAME".log -e "$JNAME".log -J "$JNAME" --export=ALL,PROFILE="$PROFILE",PROJECT="$PROJECT",INDEX="$INDEX",RESO="$RESO",LANGG="$LANGG",CODEC="$CODEC",MODE="$MODE" "$SLURM_TEMPLATE"
