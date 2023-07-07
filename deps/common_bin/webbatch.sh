#!/bin/bash

usage()
{
	echo "Usage: webbatch [ -P partition -N nthreads -p profile ] -n projectname -i index -r [1080/720] -l [GB/B5] -c [264/265] [ -t job_template ] [ -m avs/vs ]" 1>&2
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
#State=UP TotalCPUs=16000 TotalNodes=400 SelectTypeParameters=NONE
#Adaptive thread control
if [ -z "$NTHREADS" ] ; then
	if ! [ -x $(which scontrol) ] ; then
		echo "scontrol not found, need specify -N manually"
	fi
	NTHREADS=0
	State=
	TotalCPUs=
	TotalNodes=
	SelectTypeParameters=
	if [ -n "$PART" ] && ! echo "$PART" | grep -Fq ',' ; then
	#only process single partition passwd
		eval $(scontrol show partition "$PART" | grep -F 'TotalCPUs')
		if [ -n "$TotalCPUs" ] && [ -n "$TotalNodes" ] ; then
		#calc single socket, assume DP not MP
			NTHREADS=$((TotalCPUs/TotalNodes/2))
		fi
	elif [ -z "$PART" ] ; then
	# process default partition
		eval $(scontrol show partition | grep -F 'Default=YES' -A6 | grep -F 'TotalCPUs')
		if [ -n "$TotalCPUs" ] && [ -n "$TotalNodes" ] ; then
			NTHREADS=$((TotalCPUs/TotalNodes/2))
		fi
	else
	# process all partition, use minimal core per socket
	eval $(scontrol show partition | grep -F 'TotalCPUs' | ( while read ln ; do
		eval "$ln"
		if [ -n "$TotalCPUs" ] && [ -n "$TotalNodes" ] ; then
			CUTHREADS=$((TotalCPUs/TotalNodes/2))
			[ "$NTHREADS" = 0 ] && NTHREADS="$CUTHREADS"
			[ "$NTHREADS" -gt "$CUTHREADS" ] && NTHREADS="$CUTHREADS" 
		fi
		done
		echo "NTHREADS=$NTHREADS" ))
	fi
	# assume 2 x 12 core if cannot get info
	[ "$NTHREADS" = 0 ] && NTHREADS=12 && echo "no partition info, use $NTHREADS threads per socket"
	#limit 264 sub within 1 socket if appropriate cores
	if [ "$CODEC" = 264 ]  ; then
		# apply only when 16 <= threads <=32
		if [ "$NTHREADS" -ge 16 ] && [ "$NTHREADS" -le 32 ] ; then
			#THREADS="-n $NTHREADS --sockets-per-node 1"
			THREADS="-n $NTHREADS"
		elif [ "$NTHREADS" -lt 16 ] ; then
			THREADS="-n 16"
		else
			THREADS="-n 32"
		fi
	else
		#full node batch
		THREADS="-n $((NTHREADS*2))"
	fi
fi
		
echo "use $NTHREADS threads for $CODEC job"
[ -n "$PART" ] && PART="-p $PART"
[ -z "$SLURM_TEMPLATE" ] && SLURM_TEMPLATE="${ENCROOT}/config/sub_web"
[ -n "$MODE" ] && SMODE=".${MODE}"
JNAME="${PROJECT}_${INDEX}_${LANGG}_${RESO}_${CODEC}${SMODE}"


sbatch $PART $THREADS --distribution=block:block -N 1 -o "$JNAME".log -e "$JNAME".log -J "$JNAME" --export=ALL,PROFILE="$PROFILE",PROJECT="$PROJECT",INDEX="$INDEX",RESO="$RESO",LANGG="$LANGG",CODEC="$CODEC",MODE="$MODE" "$SLURM_TEMPLATE"
