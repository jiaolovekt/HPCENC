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
 - vs plugins
 - ffmpeg
 - libass
 - lsmash
 - VS r62
 - x264
 - x265
 - Python 3.11
 - mkvtoolnix-r78
 - mktorrent v1.0
 - Wine64-8.0.1
 - Avisynth+ 3.7.2 Windows i386
 - Avisynth+ 3.7.2 Windows x64
 - VSFilterMod i386
 - VSFilterMod x64
## Requirements and limitations
  The package almost include every binary files needed for encoding, even with a gcc.\
  So the only requirement is [environment modules](https://github.com/cea-hpc/modules)(to manage environment variables), and,
  this is a very fundamental part of supercomputers so every HPC cluster should have this.
### Scheduler support
  __Slurm Only__\
  Can be ported to pbs/toqrue/LSF by modifying batch scripts.
  default batch template is config/sub_web.\
  Can be replaced with `webbatch -t`.
### Heterogeneous support
  Since it's difficult for some cluster to allocate CPUs and GPUs together, currently there'll be NO GPU support for plugins or encoders.
### Softsub support
  Currently not scheduled.
### Architecture support
  Currently Broadwell(E5v4) and later Intel CPUs. May add EPYC2 support if I get a cluster to test.
### Numa support
  Will try to bind x264 process to 1 socket on DP/MP platforms if there's more than 12 threads per socket available.
### AVS/VS Plugins support (Linux)
  Check AVS plugins at deps/avsp3.7.2/lib/avisynth\
  Check VS plugins at deps/vs-plugins
### AVS Plugins support (Win64)
  Currently only Lsmashsource and VSFilterMod\
  Put x64 plugin dlls in deps/wine/avs64/plugins
### AVS Plugins support (Win32)
  Currently only Lsmashsource and VSFilterMod\
  Put i386 plugin dlls in deps/wine/avs32/plugins
### VSFilterMod support
  Supported by wine, i386 and x64 versions available
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
|src|source files here|
|out|final output dir|
|scripts|copy avs scripts here(or automatically created here)|
|tmp|intermediate files here(.264 .265 .aac)|

A symlink can be used for "font" dir across different projects, or modify the project's profile.
### Submit encode jobs
For detailed information about cluster scheduler and job submitting, refer to https://slurm.schedmd.com/quickstart.html
#### Direct run
webrip:
```sh
srun -N 1 --ntasks-per-node 1 webencode.sh [ -p profile ] -n projectname -i index -r [1080/720] -l [GB/B5] -c [264/265] [ -m avs/vs ]
```
#### Submit job
```sh
webbatch [ -P partition -p profile ] -n projectname -i index -r [1080/720] -l [GB/B5] -c [264/265] [ -t job_template ] [ -m avs/vs ]
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
```
[root@f hpcenc]# squeue |grep root
 62783      oooo KKSC_07_B5_1080_264.         root  R       6:34     1 y8
 62782      oooo KKSC_08_B5_1080_264.         root  R       6:37     1 y13
 62781      oooo KKSC_09_B5_1080_264.         root  R       6:40     1 y1
 62789     ooood KKSC_10_B5_1080_264.         root  R       6:10     1 b26
 62788     ooood KKSC_06_B5_1080_264.         root  R       6:13     1 b25
 62787     ooood KKSC_05_B5_1080_264.         root  R       6:16     1 b24
 62791     oooof KKSC_12_B5_1080_264.         root  R       5:52     1 w3
 62790     oooof KKSC_11_B5_1080_264.         root  R       5:55     1 w3
 62786     oooof KKSC_04_B5_1080_264.         root  R       6:28     1 w2
 62784     oooof KKSC_02_B5_1080_264.         root  R       6:31     1 w1
 62785     oooof KKSC_03_B5_1080_264.         root  R       6:31     1 w2
 62772     oooof KKSC_01_B5_1080_264.         root  R       8:42     1 w1
```

#### And get your output file int the "out" dir

### File name map
if a project's name is KAKT, the file's naming rules will be following:

namemap:
```
OUTNAME=[KTXP][Ayakashi_Triangle][$INDEX][$LANGG][$RESO]; SRC="Ayakashi Triangle"; ASS=KAKT # KAKT
```

Defines the output name, the SRC name is also used to search source file and ass file. \
The source file should contain "Ayakashi Triangle", like [SubsPlease] Ayakashi Triangle - 01 (1080p).mkv.\
The ass file like Ayakashi Triangle - 01_GB.ass or KAKT_01_GB.ass will be searched.
The name template pattern can be changed as your wish but do not change the $varaible name.

### Custom AVS/VS template
You can create your own avs template according to config/base_template.avs or config/base_template.vpy, using the following varaible for substitution.
#### AVS/VS template Variables
|String|Usage|
|-----------|----------|
|\_\_ENCROOT\_\_|The HPCENC root dir|
|\_\_SRCFILE\_\_|The source file name, without path|
|\_\_ASSFILE\_\_|The ass file name without path|
|\_\_FONTDIR\_\_|The relative font dir(in the project's dir)|

#### AVS template rules
 - If using automatic avs creation and 720p output, the template's last line should be __8bit__ final clip (e.g. Return V / Return Audiodub(V,A)) 
#### VS template rules
 - Video stream should be output stream 0, and audio stream 1. 

## Current status
 - [x] Webrip batch with avs
 - [x] Webrip batch with vs (testing)
 - [ ] BDMV preprocess
 - [ ] BDMV batch with avs
 - [ ] BDMV batch with vs
 - [x] VSFM support
 - [x] mktorrent
 - [x] update tracker list
 - [ ] Automatic job submission as daemon
 - [ ] Automatic mktorrent
 - [ ] Automatic publsh
