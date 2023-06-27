# HPCENC
A pack of various prebuilt tools(?) used for video encoding(?) on (x86) [HPC clusters](https://en.wikipedia.org/wiki/Supercomputer). 

## Introduction
  The motivation of this project is to find a faster way of encoding videos with avisynth/vapoursynth with x264/265.\
  My former solution is a series of semi-automatic windows cmd scripts, which distributes encoding jobs to a few windows slave machines.\
  However sometimes it is buggy because of various reasons(maybe Windows is the most significant one).
  So I decided to migrate everything to linux and, since there's some HPC resources available, submit encode jobs to clusters will [save a lot of time](https://en.wikipedia.org/wiki/Parallel_computing#Amdahl's_law_and_Gustafson's_law).\
  You can submit a TV series of 12 episodes * 2 (1080p and 720p) * 2 (GB and BIG5) = 48 jobs to the cluster and, if there's enough nodes, your jobs will complete within the time of encoding just 1 episode.

## Content
 - Avisynth+ 3.7.2
 - avs plugins 
 - ffmpeg
 - libass
 - lsmash
 - VS-r62
 - x264
 - x265
## Requirements
  The package almost include every binary files needed for encoding, even with a gcc.\
  So the only requirement is [environment modules](https://github.com/cea-hpc/modules)(to manage environment variables), and,
  this is a very fundamental part of supercomputers so every HPC cluster should have this.
## Scheduler support
  __Slurm Only__\
  Can be ported to pbs/toqrue by modifying batch scripts.
## Usage
Just clone this project to your compute cluster.
Initiate the environment by\
`. ./init.sh`
### Create project
 - Add a name map rule to config/namemap (Details below)
 - By default we use parameters and paths in config/defaultprofile, you can create a alternate one for each project(e.g. KAKT_profile) under the config dir, Just copy the default one to the new one and modify.
```sh
cp config/defaultprofile config/KAKT_profile
vi config/KAKT_profile
```
 - Prepeare project dirs, example(using project name KAKT):
```sh
mkdir KAKT
cd KAKT
mkdir ass font out script src tmp
```
### Prepare files
Copy sources, fonts, ass, and scripts(optional) into respective files.
|DIR name|Usage|
|--------|-----|
|ass|ass files here|
|font|fonts here|
|out|final output dir|
|scripts|copy avs scripts here(or automatically created here)|
|tmp|intermediate files here(.264 .265 .aac)|

A symlink can be used for "font" dir
### Submit encode jobs
For detailed information about cluster scheduler and job submitting, refer to https://slurm.schedmd.com/quickstart.html
#### Direct run
webrip:
```sh
srun -N 1 --ntasks-per-node 1 webencode.sh [ -p profile ] -n projectname -i index -r [1080/720] -l [GB/B5] -c [264/265] [ -m avs/vs ]
```
#### Submit job
```sh
webbatch [ -p profile ] -n projectname -i index -r [1080/720] -l [GB/B5] -c [264/265] [ -t job_template ] [ -m avs/vs ]
#e.g.
[root@f hpcenc]# webbatch.sh -n KAKT -i 01 -r 720 -l GB -c 264
Submitted batch job 60796
[root@f hpcenc]# webbatch.sh -n KAKT -i 01 -r 720 -l BIG5 -c 264
Submitted batch job 60797
[root@f hpcenc]# squeue
 JOBID PARTITION                 NAME         USER ST       TIME NODES NODELIST(REASON)
 60797      oooo KAKT_01_BIG5_720_264         root  R       4:21     1 y10
 60796      oooo   KAKT_01_GB_720_264         root  R       4:27     1 y7
 60795      oooo                 test         root  R       8:38     1 y2
```
#### And get your output file int the "out" dir

### File name map
if a project's name is KAKT, the file's naming rules will be following:

namemap:
```
OUTNAME=[KTXP][Ayakashi_Triangle][$INDEX][$LANGG][$RESO]; SRC="Ayakashi Triangle" # KAKT
```

Defines the output name, the SRC name is also used to search source file and ass file. \
The source file should contain "Ayakashi Triangle", like [SubsPlease] Ayakashi Triangle - 01 (1080p).mkv.\
The ass file like Ayakashi Triangle - 01_GB.ass or KAKT_01_GB.ass will be searched.
The name template pattern can be changed as your wish but do not change the $varaible name.

### Custom AVS/VS template
You can create your own avs template according to config/base_template.avs or config/base_template.vpy, using the following varaible for substitution.\
#### AVS/VS template Variables
|String|Usage|
|-----------|----------|
|\_\_ENCROOT\_\_|The HPCENC root dir|
|\_\_SRCFILE\_\_|The source file name, without path|
|\_\_ASSFILE\_\_|The ass file name without path|
|\_\_FONTDIR\_\_|The relative font dir(in the project's dir)|

#### AVS template rules
 - If using automatic avs creation and 720p output, the template's last line should be __8bit__ final clip (e.g. Return V / Return Audiodub(V,A)) 


