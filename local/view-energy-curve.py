# encoding: utf-8
import math
import numpy as np
import sys
import wave # use to read .wav file
import librosa #use to extract MFCC feature
import librosa.display
import kaldi_io
import scipy
import soundfile as sf
import io
# import pdb
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# from src.functions import * #use the functions defined by myself

wavid="20170217-noon"
ipath2wav_scp = "/home/jerry/project_broadcast/data/hkbn_2017/wav.scp"
ipath2musicprob_scp = "/home/jerry/project_broadcast/exp/vad_gmm_hkbn2017part10/full_ubm_music_logprob.scp"
ipath2speechprob_scp = "/home/jerry/project_broadcast/exp/vad_gmm_hkbn2017part10/full_ubm_speech_logprob.scp"
ipath2noiseprob_scp = "/home/jerry/project_broadcast/exp/vad_gmm_hkbn2017part10/full_ubm_noise_logprob.scp"


def kaldi_read_wav(file_or_fd):
  # fd = kaldi_io.open_or_fd(file_or_fd)
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

    args: ipath2wav_scp -- the path to wav.scp (The file has format as kaldi's.)
           seg          -- a tuple of (start_time, end_time)

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

      return y[int(start_t*fs):int(end_t*fs)], fs
  #  wavid not found
  raise Exception("wavid: {0} not found in file {1}".format(wavid, ipath2wav_scp))


def fetch_llkprob_segment(wavid, ipath2prob_scp, seg=(0.0, math.inf), win_len=0.025, hop_len=0.010):
  """
    given wavid, return an loglikehood probability segment from ipath2prob_scp

    args: ipath2prob_scp -- the path to llk_prob.scp
                            each wavid corresponds to a float vector of llk_prob
                            llk_prob: the prob of a specific GMM generating a frame
          seg            -- a tuple of (start_time, end_time)

    return: probs        -- llk_probs with numpy format
  """
  fd = kaldi_io.open_or_fd(ipath2prob_scp)
  for line in fd:
    (wid, path) = line.decode("utf-8").rstrip().split(' ', 1)
    if wavid == wid:
      vec = read_vec_flt(path) # np.array
      start_t, end_t = seg
      end_t = min(end_t, vec.shape[0]*hop_len) # the second term is float by default
      assert start_t < end_t and start_t >= 0.0, "InputArg: seg {0} invalid".format(str(seg))
      start_f = int(start_t / hop_len)
      end_f = int(end_t /hop_len)
      return vec[start_f:end_f]
      

def plot_spectrogram(y, fs=16000, win_len=0.025, hop_len=0.010):
  """
    plot spectrogram given an wavid in ipath2wav_scp.
    The fig is saved into .png under current dir

    args: y       -- audio samples with numpy format
          fs      -- sampling frequency
          win_len -- window length (in second)
          hop_len -- window shift (in second)

  """
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
  plt.clf()
  plt.figure(figsize=(50,5)) 
  librosa.display.specshow(D, y_axis='linear')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Log-frequency power spectrogram')
  plt.savefig('{0}.png'.format(wavid))


def plot_llkcurves(musicprob, speechprob, noiseprob, win_len=0.025, hop_len=0.010):
  """
    plot speech/noise/music loglikehood probility curve
    The fig is saved into .png under current dir

    arg: win_len  -- window length (in second)
         hop_len -- window shift (in second)

    TODO: this func will finally merged into plot_spectrogram and rename it
  """
  plt.clf()
  plt.figure(figsize=(50,5))

  t = 
  plt.plot(musicprob, '')
  plt.plot(speechprob)
  plt.plot(noiseprob)

  pass
  


def main():
  y, fs = fetch_audio_segment(wavid, ipath2wav_scp, seg=(10.0, 50.0), fs=16000)
  musicprob = fetch_llkprob_segment(wavid, ipath2musicprob_scp, seg=(10.0, 50.0), win_len=0.025, hop_len=0.010)
  speechprob = fetch_llkprob_segment(wavid, ipath2speechprob_scp, seg=(10.0, 50.0), win_len=0.025, hop_len=0.010)
  noiseprob = fetch_llkprob_segment(wavid, ipath2noiseprob_scp, seg=(10.0, 50.0), win_len=0.025, hop_len=0.010)

  plot_spectrogram(y, fs=fs, win_len=0.025, hop_len=0.010)


if __name__ == '__main__':
  main()


# energy_curve = np.mean(power_spectrogram[1:,:], axis=0)

# seq_isLargerThanThreshold = energy_curve > 3.0

# plt.plot(seq_isLargerThanThreshold)

