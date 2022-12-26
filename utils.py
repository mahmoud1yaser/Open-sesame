# Audio processing
import librosa
# Data cleaning
import numpy as np
from numpy import mean, var

# Functions we will use
def transform_audio(audio, FRAMESIZE, HOPLENGTH, MELS, SPLITFREQ):
    audio_array, sr = librosa.load(audio)
    ae_audio = fancy_amplitude_envelope(audio_array, FRAMESIZE, HOPLENGTH)
    mel_audio = librosa.feature.melspectrogram(audio_array, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH, n_mels=MELS)
    log_mel_audio = librosa.power_to_db(mel_audio)[0]
    mfccs_audio = librosa.feature.mfcc(y=audio_array, n_mfcc=MELS, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH)[0]
    sc_audio = librosa.feature.spectral_centroid(y=audio_array, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH)[0]
    chromagram_audio = librosa.feature.chroma_stft(audio_array, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH)[0]
    tone_audio = librosa.feature.tonnetz(y=audio_array, sr=sr)[0]

    #     return np.hstack((log_mel_audio, mfccs_audio, sc_audio, rms_audio, ae_audio, chromagram_audio, tone_audio))
    return np.hstack((mean(log_mel_audio), var(log_mel_audio), mean(mfccs_audio), var(mfccs_audio), mean(sc_audio),
                      var(sc_audio), mean(ae_audio), var(ae_audio), mean(chromagram_audio), var(chromagram_audio),
                      mean(tone_audio), var(tone_audio)))


def fancy_amplitude_envelope(signal, framesize, hoplength):
    return np.array([max(signal[i:i + framesize]) for i in range(0, len(signal), hoplength)])