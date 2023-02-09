# Audio processing
import librosa
import noisereduce
# Data cleaning
import numpy as np
from numpy import mean, var
import speech_recognition as sr
import matplotlib.pyplot as plt


# Functions we will use
def transform_audio(audio, FRAMESIZE, HOPLENGTH, MELS):
    # audio_noised, sr = librosa.load(audio, duration=2)
    # audio_array = noisereduce.reduce_noise(y=audio_noised, sr=sr)
    audio_array, sr = librosa.load(audio, duration=2)

    log_mel_audio_list_mean = []
    log_mel_audio_list_var = []
    mfccs_audio_list_mean = []
    mfccs_audio_list_var = []
    cqt_audio_list_mean = []
    cqt_audio_list_var = []
    chromagram_audio_list_mean = []
    chromagram_audio_list_var = []
    tone_audio_list_mean = []
    tone_audio_list_var = []

    log_mel_audio = librosa.power_to_db(
        librosa.feature.melspectrogram(audio_array, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH, n_mels=MELS))
    mfccs_audio = librosa.feature.mfcc(y=audio_array, n_mfcc=MELS, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH)

    cqt_audio = np.abs(librosa.cqt(y=audio_array, sr=sr, hop_length=HOPLENGTH))
    chromagram_audio = librosa.feature.chroma_stft(audio_array, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH)
    tone_audio = librosa.feature.tonnetz(y=audio_array, sr=sr)

    for i in range(len(log_mel_audio)):
        log_mel_audio_list_mean.append(log_mel_audio[i].mean())
        log_mel_audio_list_var.append(log_mel_audio[i].var())

    for i in range(len(mfccs_audio)):
        mfccs_audio_list_mean.append(mfccs_audio[i].mean())
        mfccs_audio_list_var.append(mfccs_audio[i].var())

    for i in range(len(cqt_audio)):
        cqt_audio_list_mean.append(cqt_audio[i].mean())
        cqt_audio_list_var.append(cqt_audio[i].var())

    for i in range(len(chromagram_audio)):
        chromagram_audio_list_mean.append(chromagram_audio[i].mean())
        chromagram_audio_list_var.append(chromagram_audio[i].var())

    for i in range(len(tone_audio)):
        tone_audio_list_mean.append(tone_audio[i].mean())
        tone_audio_list_var.append(tone_audio[i].var())

    sb_audio = librosa.feature.spectral_bandwidth(y=audio_array, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH)

    ae_audio = fancy_amplitude_envelope(audio_array, FRAMESIZE, HOPLENGTH)
    rms_audio = librosa.feature.rms(audio_array, frame_length=FRAMESIZE, hop_length=HOPLENGTH)

    return np.hstack((mean(ae_audio), var(ae_audio), mean(rms_audio), var(rms_audio), mean(sb_audio), var(sb_audio),
                      chromagram_audio_list_mean, chromagram_audio_list_var, tone_audio_list_mean, tone_audio_list_var,
                      cqt_audio_list_mean, cqt_audio_list_var, mfccs_audio_list_mean, mfccs_audio_list_var,
                      log_mel_audio_list_mean, log_mel_audio_list_var))


def fancy_amplitude_envelope(signal, framesize, hoplength):
    return np.array([max(signal[i:i + framesize]) for i in range(0, len(signal), hoplength)])


def get_audio_features():
    x_ver = transform_audio('audio.wav', 1024, 512, 13)
    return x_ver


def check_word(wav):
    # initialize recognizer class (for recognizing the speech)
    r = sr.Recognizer()

    # Reading Microphone as source
    # listening the speech and store in audio_text variable
    with sr.AudioFile(wav) as source:
        audio_text = r.record(source)

    # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    try:
        # using google speech recognition
        transcribed_text = r.recognize_google(audio_text)
        if "open" in transcribed_text:
            return 1
        else:
            return 0
    except:
        return 0


def plot_feature_importance_bar(X, user_input, classifier, features_number=10):
    # Make a prediction on the user's input data
    user_prediction = classifier.predict(user_input)

    # Get the confidence score of the prediction
    confidence_score = classifier.predict_proba(user_input)[0][user_prediction[0]]

    # Get the feature importances from the classifier
    importances = classifier.feature_importances_

    # Get the indices of the top 10 most important features
    indices = np.argsort(importances)[::-1][:features_number]

    # Plot the feature importances of the top 10 features based on the confidence score
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(features_number), importances[indices], align='center', color='blue')
    ax.set_yticks(range(features_number))
    ax.set_yticklabels(X.columns[indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance Plot Based on Confidence Score')
    ax.text(0.95, 0.97, 'Input Confidence Score: {:.2f}'.format(confidence_score), ha='right', va='bottom',
            transform=ax.transAxes, fontsize=12)
    plt.tight_layout()
    fig.savefig('static\\assets\\dynamic_plot.png')


def plot_feature_importance_scatter(X, user_input, classifier, features_number=10):
    user_prediction = classifier.predict(user_input)
    confidence_score = classifier.predict_proba(user_input)[0][user_prediction[0]]

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:features_number]

    fig, ax = plt.subplots(figsize=(7, 5))
    c = ax.scatter(importances[indices], range(features_number), c=[confidence_score] * features_number, cmap='Blues')
    ax.set_yticks(range(features_number))
    ax.set_yticklabels(X.columns[indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance Plot Based on Confidence Score')
    ax.text(0.95, 0.95, 'Input Confidence Score: {:.2f}'.format(confidence_score), ha='right', va='bottom',
            transform=ax.transAxes, fontsize=12)
    plt.colorbar(c, ax=ax).set_label('Confidence Score')
    plt.tight_layout()
    fig.savefig('static\\assets\\dynamic_plot.png')
