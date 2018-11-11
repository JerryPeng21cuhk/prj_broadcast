#!/bin/bash

# Author: Jerry Peng 2018
# In this script, 
#  1) we do VAD to select speech cuts
#  2) draw the VAD curve and speech spectromgram


. ./cmd.sh
. ./path.sh

set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
stage=


. ./utils/parse_options.sh

# mkdir -p exp

# data preparation
if [ $stage -le 0 ]; then
  # local/make_musan.sh /home/jerry/Downloads/musan data
  # 42.6h music, 6.2h noise, 60.4h speech


  local/check_and_format_data.sh /lan/ibdata/SPEECH_DATABASE/RTHK_raw_data/Sound_Archives/2017 data/hkbn_2017
  # 1144h

  paste -d' ' <(cut -d' ' -f1 data/hkbn_2017/wav.scp) \
    <(cut -d' ' -f1 data/hkbn_2017/wav.scp) \
    > data/hkbn_2017/spk2utt
  utils/spk2utt_to_utt2spk.pl data/hkbn_2017/spk2utt > data/hkbn_2017/utt2spk

  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 30 --cmd "$train_cmd" \
  #   data/musan_speech exp/make_mfcc $mfccdir

  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 30 --cmd "$train_cmd" \
  #   data/musan_music exp/make_mfcc $mfccdir

  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 5 --cmd "$train_cmd" \
  #   data/musan_noise exp/make_mfcc $mfccdir

  # steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 30 --cmd "$train_cmd" \
  #   data/hkbn_2017 exp/make_mfcc $mfccdir

  utils/fix_data_dir.sh data/musan_speech
  utils/fix_data_dir.sh data/musan_music
  utils/fix_data_dir.sh data/musan_noise
  utils/fix_data_dir.sh data/hkbn_2017

  
  sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
    data/musan_speech exp/make_vad $vaddir
  sid/compute_vad_decision.sh --nj 5 --cmd "$train_cmd" \
    data/musan_noise exp/make_vad $vaddir
  sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
    data/musan_music exp/make_vad $vaddir
  sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
    data/hkbn_2017 exp/make_vad $vaddir

fi


if [ $stage -le 1 ]; then
  sid/train_diag_ubm.sh --nj 10 --cmd "$train_cmd" --delta-window 2 \
    data/musan_noise 32 exp/diag_ubm_noise
  sid/train_diag_ubm.sh --nj 20 --cmd "$train_cmd" --delta-window 2 \
    data/musan_speech 32 exp/diag_ubm_speech
  sid/train_diag_ubm.sh --nj 20 --cmd "$train_cmd" --delta-window 2 \
    data/musan_music 32  exp/diag_ubm_music

  sid/train_full_ubm.sh --nj 20 --cmd "$train_cmd" \
    --remove-low-count-gaussians false data/musan_noise \
    exp/diag_ubm_noise exp/full_ubm_noise
  sid/train_full_ubm.sh --nj 20 --cmd "$train_cmd" \
    --remove-low-count-gaussians false data/musan_speech \
    exp/diag_ubm_speech exp/full_ubm_speech
  sid/train_full_ubm.sh --nj 20 --cmd "$train_cmd" \
    --remove-low-count-gaussians false data/musan_music \
    exp/diag_ubm_music exp/full_ubm_music


  # first we select a small subset from hkbn to check if this approach works
  subset_data_dir.sh data/hkbn_2017 10 data/hkbn_2017part10  

  local/compute_vad_decision_gmm.sh --nj 10 --cmd "$train_cmd" \
    --merge-map-config conf/merge_vad_map.txt --use-energy-vad true \
    --cleanup false \
    data/hkbn_2017part10 exp/full_ubm_noise exp/full_ubm_speech/ \
    exp/full_ubm_music/ exp/vad_gmm_hkbn2017part10 exp/vad_gmm_hkbn2017part10

fi

# show results
if [ $stage -le 2 ]; then
  python3 local/view_prob_curve.py


fi