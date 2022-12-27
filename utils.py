# Audio processing
import librosa
# Data cleaning
import numpy as np
from numpy import mean, var


# Functions we will use
def transform_audio(audio, FRAMESIZE, HOPLENGTH, MELS, SPLITFREQ):
    audio_array, sr = librosa.load(audio)
    rms_audio = librosa.feature.rms(audio_array, frame_length=FRAMESIZE, hop_length=HOPLENGTH)[0]
    mel_audio = librosa.feature.melspectrogram(audio_array, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH, n_mels=MELS)
    log_mel_audio = librosa.power_to_db(mel_audio)[0]
    sc_audio = librosa.feature.spectral_centroid(y=audio_array, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH)[0]
    return np.hstack((mean(log_mel_audio), var(log_mel_audio), mean(rms_audio), var(rms_audio), mean(sc_audio)))


def fancy_amplitude_envelope(signal, framesize, hoplength):
    return np.array([max(signal[i:i + framesize]) for i in range(0, len(signal), hoplength)])

