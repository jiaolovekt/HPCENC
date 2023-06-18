# HPCENC
A pack of various prebuilt tools(?) used for video encoding(?). 

## Introduction
  The motivation of this project is to find a faster way of encoding videos with avisynth/vapoursynth with x264/265. 
  My former solution is a series of semi-automatic windows cmd scripts, which distributes encoding jobs to a few windows slave machines. Because of various reasons(maybe Windows is the most significant one), sometimes it is buggy.
  So I decided to migrate everything to linux and
## Content
  Avisynth+ 3.7.2
  avs plugins 
  ffmpeg
  libass
  lsmash
  VS-r62
  x264
  x265
## Requirements
  The package almost include every binary files needed for encoding, even with a gcc.
  So the only requirement is [environment modules](https://github.com/cea-hpc/modules)(to manage environment variables), and,
  this is a very fundamental part of supercomputers so every compute cluster should have this.
## Usage

### File name map
if a project's name is KAKT, the file's naming rules will be following:

namemap:
OUTNAME=[KTXP][Ayakashi_Triangle][$INDEX][$LANGG][$RESO]; SRC="Ayakashi Triangle" # KAKT

Defines the output name, the SRC name is also used to search source file and ass file. The source file should contain "Ayakashi Triangle", like [SubsPlease] Ayakashi Triangle - 01 (1080p).mkv. The ass file like Ayakashi Triangle - 01_GB.ass or KAKT_01_GB.ass will be searched.

### AVS template variable
You can create your own avs template according to config/base_template.avs, using the following varaible for substitution.
|----------|----------|
|__ENCROOT__|The HPCENC root dir|
|__SRCFILE__|The source file name, without path|
|__ASSFILE__|The ass file name without path|
|__FONTDIR__|The relative font dir(in the project's dir)|

A symlink can be used for FONTDIR

### Submit encode jobs
`encode [ -p profile ] -n projectname -i index -r [1080/720] -l [GB/B5] -c [264/265]`
