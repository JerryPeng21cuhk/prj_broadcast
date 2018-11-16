# encoding: utf-8
# Author: Jerry Peng 2018 

# This script implements some visualization functions:
  # 1) given a wavid and a segment with (start-time, end-time)
  #      plot the spectrogram and decoding prob curves derived from several GMM
  # 2) given a wavid and a segment with (start-time, end-time)
  #      save prob curves into wavesfurer format file. It can be visualized in wavesfurer.


import math
import numpy as np
import sys
import os
import wave # use to read .wav file
import librosa #use to extract MFCC feature
import librosa.display
import librosa.core
import argparse
import re
import kaldi_io
import scipy
import soundfile as sf
import io
import pdb
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# # global io

# wavid="20170217-noon"
ipath2wav_scp = "/home/jerry/project_broadcast/data/hkbn_2017/wav.scp"

ipath2musicprob_scp = "/home/jerry/project_broadcast/exp/vad_gmm_hkbn2017part10/full_ubm_music_logprob.scp"
ipath2speechprob_scp = "/home/jerry/project_broadcast/exp/vad_gmm_hkbn2017part10/full_ubm_speech_logprob.scp"
ipath2noiseprob_scp = "/home/jerry/project_broadcast/exp/vad_gmm_hkbn2017part10/full_ubm_noise_logprob.scp"

# fs = 16000
# win_len = 0.025
# hop_len = 0.010
# seg = (10, 50)

# opath2probcurves_txt = "/home/jerry/project_broadcast/20170217-noon.txt"
# opath2probcurves_fig = "/home/jerry/project_broadcast/20170217-noon-10-50.txt"


def main():

  def regex_parse_seg(s, pat=re.compile(r"(\d*\.\d+|\d+)-(\d*\.\d+|\d+|inf)")):
    digits = pat.match(s)
    if not digits:
        raise argparse.ArgumentTypeError
    else:
      return [float(num) for num in digits.groups()]


  parser = argparse.ArgumentParser(description="Plot the prob curves given the timeslot of a segment",
                                 epilog="E.g. " + sys.argv[0] + " --seg 37.0-42.0",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("wavid")
  parser.add_argument("--seg", type=regex_parse_seg, default="0.0-inf",
                      help="timestamp in second of a segment. E.g. 37.0-42.0")
  parser.add_argument("--win-len", dest="win_len", type=float, default=0.025,
                      help="window length in second")
  parser.add_argument("--hop-len", dest="hop_len", type=float, default=0.010,
                      help="window hop in second")
  parser.add_argument("--fs", type=int, default=16000,
                      help="sampling frequency in Hz")
  parser.add_argument("--to-txt", dest="opath2probcurves_txt", default="",
                      help="save likelihood curves to hardisk. It can be further used in wavesurfer.")
  parser.add_argument("--to-fig", dest="opath2probcurves_fig", default="",
                      help="plot likelihood curves and spectrogram by matplotlib.pyplot and save it to hardisk.")

  args = parser.parse_args()

  assert args.opath2probcurves_txt or args.opath2probcurves_fig, "Both --to-txt and --to-fig are empty. Do nothing ?"

  if (args.opath2probcurves_txt):
    save_probcurves2wavesurf(args.wavid, 
                            args.opath2probcurves_txt,
                            seg=args.seg, 
                            fs=args.fs, 
                            win_len=args.win_len, 
                            hop_len=args.hop_len)

  if (args.opath2probcurves_fig):
    plt.clf()
    plt.figure(figsize=(args.seg[1] -args.seg[0] + 10, 10))
    plt.subplot(2, 1, 1)
    plot_spectrogram(args.wavid, 
                    ipath2wav_scp, 
                    seg=args.seg, 
                    fs=args.fs, 
                    win_len=args.win_len, 
                    hop_len=args.hop_len)
    plt.subplot(2, 1, 2)
    plot_probcurves(args.wavid, 
                    seg=args.seg, 
                    fs=args.fs, 
                    win_len=args.win_len, 
                    hop_len=args.hop_len)
    plt.savefig(args.opath2probcurves_fig)



# func list:

# plot_spectrogram
  # fetch_audio_segment
    # # kaldi_read_wav

# save_probcurves2wavesurf
  # fetch_probcurves
    # # fetch_llkprob_segment

# plot_probcurves
  # fetch_probcurves
    # # fetch_llkprob_segment


def kaldi_read_wav(file_or_fd):
  """
    given a audio file, return audio samples and sampling frequency

    args: file_or_fd -- <cmd pipeline string> or a normal filename string

    return: y -- audio samples with numpy format
            fs -- sampling frequency
  """
  if file_or_fd[-1] == '|':

    import subprocess, io, threading
    # cleanup function for subprocesses,
    def cleanup(proc, cmd):
      ret = proc.wait()
      if ret > 0:
        raise SubprocessFailed('cmd %s returned %d !' % (cmd,ret))
      return
    cmd = file_or_fd[:-1].encode('utf-8')
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    # return proc.stdout
    fd = proc.stdout
  else:
    # a normal file
    fd = open(file_or_fd, 'rb')
  # fd is a fhandle
  tmp = io.BytesIO(fd.read())
  y, fs = sf.read(tmp)
  return y, fs


def fetch_audio_segment(wavid, ipath2wav_scp, seg=(0.0, math.inf), fs=16000):
  """
    given wavid, return an audio segment from ipath2wav_scp

    args:  wavid       -- string, id of a audio file
           ipath2wav_scp -- the path to wav.scp 
                            (wav.scp has the same format as kaldi's:
                              each row of the file is <wavid> <wavpath>)
           seg          -- a tuple of float, (start_time, end_time)
           fs           -- sampling frequency

    return: y -- audio samples with numpy format
            fs -- sampling frequency
  """
  fd = kaldi_io.open_or_fd(ipath2wav_scp)
  for line in fd:
    (wid, path) = line.decode("utf-8").rstrip().split(' ', 1)
    if wavid == wid:
      y, fs = kaldi_read_wav(path)
      start_t, end_t = seg
      end_t = min(end_t, y.shape[0]/fs) # the second term is float by default
      assert start_t < end_t and start_t >= 0.0, "InputArg: seg {0} invalid".format(str(seg))
      # return the segment and fs
      return y[int(start_t*fs):int(end_t*fs)], fs
  #  wavid not found
  raise Exception("wavid: {0} not found in file {1}".format(wavid, ipath2wav_scp))


def fetch_llkprob_segment(wavid, ipath2prob_scp, seg=(0.0, math.inf), win_len=0.025, hop_len=0.010):
  """
    given wavid, return an loglikehood probability segment from ipath2prob_scp

    args: wavid       -- string, id of a audio file
          ipath2prob_scp -- the path to llk_prob.scp
                            each wavid corresponds to a float vector of llk_prob
                            llk_prob: the prob of a specific GMM generating a frame
          seg            -- a tuple of (start_time, end_time)
          win_len -- window length in second
          hop_len -- window shift in second

    return: vec        -- llk_prob curve with numpy format
  """
  fd = kaldi_io.open_or_fd(ipath2prob_scp)
  for line in fd:
    (wid, path) = line.decode("utf-8").rstrip().split(' ', 1)
    if wavid == wid:
      vec = kaldi_io.read_vec_flt(path) # np.array
      start_t, end_t = seg
      end_t = min(end_t, vec.shape[0]*hop_len) # the second term is float by default
      assert start_t < end_t and start_t >= 0.0, "InputArg: seg {0} invalid".format(str(seg))
      start_f = int(start_t / hop_len)
      end_f = int(end_t /hop_len)
      return vec[start_f:end_f]
      

def fetch_probcurves(wavid, seg=(0.0, math.inf), win_len=0.025, hop_len=0.010):
  """
    given wavid, return the normalized(softmax) llkprob curves
    caution: this func uses global variables: musiclogprob, speechlogprob, noiselogprob

    args: wavid       -- string, id of an audio file
          seg            -- a tuple of (start_time, end_time)
          win_len -- window length in second
          hop_len -- window shift in second
    
    return: likelihood_probs --  a tuple of 3 numpy array items
  """
  musiclogprob = fetch_llkprob_segment(wavid, ipath2musicprob_scp, seg=seg, win_len=win_len, hop_len=hop_len)
  speechlogprob = fetch_llkprob_segment(wavid, ipath2speechprob_scp, seg=seg, win_len=win_len, hop_len=hop_len)
  noiselogprob = fetch_llkprob_segment(wavid, ipath2noiseprob_scp, seg=seg, win_len=win_len, hop_len=hop_len)

  def softmax(logprobs):
    logprob_mat = np.vstack(logprobs)
    logprob_max = np.max(logprob_mat, axis=0, keepdims=True)
    numerator = np.exp(logprob_mat - logprob_max)
    denominator = np.sum(numerator, axis=0, keepdims=True)
    prob_mat = numerator/denominator
    return (row for row in prob_mat)

  return softmax( (musiclogprob, speechlogprob, noiselogprob) )


def save_probcurves2wavesurf(wavid, opath2probcurves_txt, seg=(0.0, math.inf), fs=16000, win_len=0.025, hop_len=0.010):
  """
    save curves with wavesurf format into file: opath2probcurves_txt.
    It is more convienent to use wavesurf to evaluate the results as we can listen on-the-fly.
    
    args: wavid       -- string, id of an audio file
          opath2probcurves_txt -- string, path
          seg            -- a tuple of (start_time, end_time)
          fs             -- sampling frequency
          win_len -- window length in second
          hop_len -- window shift in second
  """
  musicprob, speechprob, noiseprob = fetch_probcurves(wavid, seg=seg, win_len=win_len, hop_len=hop_len)
  with open(opath2probcurves_txt, 'w') as f:
    for i in range(musicprob.shape[0]):
      f.write("{0:.2f} {1:.2f} {2:.2f}\n".format(musicprob[i], speechprob[i], noiseprob[i]))


def plot_spectrogram(wavid, ipath2wav_scp, seg=(0.0, math.inf), fs=16000, win_len=0.025, hop_len=0.010):
  """
    plot spectrogram given an wavid in ipath2wav_scp.
    The fig is saved into .png under current dir

    args: wavid       -- string, id of an audio file
          ipath2wav_scp -- the path to wav.scp (The file has format as kaldi's.)
          seg            -- a tuple of (start_time, end_time)
          fs             -- sampling frequency
          win_len        -- window length (in second)
          hop_len        -- window shift (in second)

    Usage of this func:

      plt.clf()
      plt.figure(figsize=(seg[1] -seg[0] + 10, 5)
      plot_spectrogram(wavid, ipath2wav_scp, seg=(0.0, math.inf), fs=16000, win_len=0.025, hop_len=0.010)
      plt.savefig('spectro_{0}.png'.format(wavid))

  """
  y, fs = fetch_audio_segment(wavid, ipath2wav_scp, seg=seg, fs=fs)

  win_length = int(win_len * fs)
  hop_length = int(hop_len * fs)

  window = scipy.signal.hamming(win_length, sym=False)
  eps = np.spacing(1)
  D = librosa.amplitude_to_db(np.abs(librosa.stft(y + eps,
                                             n_fft=2048,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             center=True,
                                             window=window)
                          ))
  x_coords=librosa.core.frames_to_time(np.arange(D.shape[1]+1), sr=fs, hop_length=hop_length) + seg[0]
  librosa.display.specshow(D, x_axis='time', x_coords=x_coords, y_axis='linear')
  # plt.colorbar(format='%+2.0f dB')
  plt.ylabel('Log-frequency power spectrogram')


def plot_probcurves(wavid, seg=(0.0, math.inf), fs=16000, win_len=0.025, hop_len=0.010):
  """
    plot speech/noise/music loglikehood probility curve
    The fig is saved into .png under current dir

    arg: wavid       -- string, id of an audio file
         seg            -- a tuple of (start_time, end_time)
         fs             -- sampling frequency
         win_len  -- window length (in second)
         hop_len -- window shift (in second)

    Usage of this func:

      plt.clf()
      plt.figure(figsize=(seg[1] -seg[0] + 10, 5)
      plot_spectrogram(wavid, seg=(0.0, math.inf), fs=16000, win_len=0.025, hop_len=0.010)
      plt.savefig('spectro_{0}.png'.format(wavid))
  """
  
  musicprob, speechprob, noiseprob = fetch_probcurves(wavid, seg=seg, win_len=win_len, hop_len=hop_len)

  hop_length = int(hop_len * fs)

  # note that only seg[0] is reliable to be further use.
  x = librosa.core.frames_to_time(np.arange(musicprob.shape[0]), sr=fs, hop_length=hop_length) + seg[0]
  plt.plot(x, musicprob, 'r', label='music')
  plt.plot(x, speechprob, 'g', label='speech')
  plt.plot(x, noiseprob, 'b', label='noise')
  plt.xlim(x.min(), x.max())
  plt.legend()
  plt.ylabel('likehood curve')



if __name__ == '__main__':
  main()
