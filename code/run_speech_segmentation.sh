#!/bin/bash

# Author: Jerry Peng 2018
# In this script, 
#  1) we do VAD to select speech cuts
#  2) draw the VAD curve and speech spectromgram


. ./cmd.sh
. ./path.sh

stage=
exp=exp/speechseg
set -e

. ./utils/parse_options.sh

mkdir -p $exp