#!/bin/bash

# Author: Jerry Peng 2018
# In this script, we check the format of wav files inside a given directory

ipath2dir="/lan/ibdata/SPEECH_DATABASE/RTHK_raw_data/Sound_Archives/2017"


echo ">> Start format checking ============================================"

# list audios with different formats
audio_postfix="*.wav *.WAV *.mp3 *.MP3"

numfiles_total=0
for i in ${audio_postfix}; do
  numfiles=`find $ipath2dir -type f -name "$i" | wc -l`
  numfiles_total=$((numfiles_total+numfiles))
  printf "There are %8d audio files with postfix: %s\n" $numfiles $i
done
echo "In total, $numfiles_total audio files under dir $ipath2dir"

echo "Now, we only take *.MP3 into consideration."
fpaths="find $ipath2dir -type f -name "*.MP3" -print0"

# check sampling frequency
echo "1) check sampling frequency"
echo "The first column is #files, second column is sampling frequency"
$fpaths | xargs -0 soxi -r 2> /dev/null | sort | uniq -c

# the result shows two files are broken. And they should be get rid of.
# the rest files are all 48kHz.

echo "2) check number of channels"
echo "The first column is #files, second column is #channels"
$fpaths | xargs -0 soxi -c 2> /dev/null | sort | uniq -c
# the rest files are all 2-channel(stereo)

echo "3) check precision"
echo "The first column is #files, second column is #bits/sample"
$fpaths | xargs -0 soxi 2> /dev/null | grep "Precision" | sort | uniq -c
# the rest files are all 16-bit

echo "Now, we generate the valid audio file list and wav.scp"
$fpaths | xargs -0 soxi 2> /dev/null | grep "Input File" | cut -d\' -f2 > wavlist
numvalidfiles=`wc -l wavlist`
echo "After check, $numvalidfiles are written into wavlist"

perl -CSAD gen_wavscp.pl wavlist > wav.scp
numvalidformat_file=`wc -l wav.scp`
echo "And finally, ${numvalidformat_file} are written into wav.scp"



echo ">> Start format conversion ============================================"

# stereo to mono channel
# 44.1k to 16k
# mp3 to wav
perl -CSAD cvt_wavformat.pl wav.scp > wav_cvt.scp
