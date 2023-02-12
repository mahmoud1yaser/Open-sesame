import librosa
import noisereduce
import pandas as pd
import numpy as np
from numpy import mean, var
import speech_recognition
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle


# Read pickle function
# This function reads the classifier from a pickle file and returns it
def read_pickle(classifier_path):
    with open(classifier_path, 'rb') as c:
        # Load classifier from file
        classifier = pickle.load(c)
    # Return the classifier
    return classifier


# Split and standardize function
# This function splits the data into training and testing sets and standardizes it using the given scaler
def split_standardize(x_path, y_path, sc):
    # Read feature data into a numpy array
    x_features = np.array(pd.read_csv(x_path))
    # Read label data into a pandas dataframe
    y_features = pd.read_csv(y_path)
    # Split the data into training and testing sets
    x_train, x_test, _, _ = train_test_split(x_features, y_features, test_size=0.3)
    # Standardize the training data
    sc.fit_transform(x_train)
    # Standardize the testing data
    sc.transform(x_test)
    # Return the feature data and the scaler
    return x_features, sc


# This function takes an audio file and returns audio features using different techniques like Melspectrogram, MFCC,
# CQT, Chroma STFT, Tonnetz and others.
def transform_audio(audio, FRAMESIZE, HOPLENGTH, MELS):
    # Load the audio file and apply noise reduction
    audio_noised, sr = librosa.load(audio, duration=2)
    audio_array = noisereduce.reduce_noise(y=audio_noised, sr=sr)

    # Initialize lists to store mean and variance of audio features
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

    # Calculate Melspectrogram of audio and convert it to log scale
    log_mel_audio = librosa.power_to_db(
        librosa.feature.melspectrogram(audio_array, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH, n_mels=MELS))

    # Calculate MFCC of audio
    mfccs_audio = librosa.feature.mfcc(y=audio_array, n_mfcc=MELS, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH)

    # Calculate CQT of audio
    cqt_audio = np.abs(librosa.cqt(y=audio_array, sr=sr, hop_length=HOPLENGTH))

    # Calculate Chromagram of audio
    chromagram_audio = librosa.feature.chroma_stft(audio_array, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH)

    # Calculate Tonnetz of audio
    tone_audio = librosa.feature.tonnetz(y=audio_array, sr=sr)

    # Calculate mean and variance of each frame of log mel spectrogram
    for i in range(len(log_mel_audio)):
        log_mel_audio_list_mean.append(log_mel_audio[i].mean())
        log_mel_audio_list_var.append(log_mel_audio[i].var())

    # Calculate mean and variance of each frame of MFCC
    for i in range(len(mfccs_audio)):
        mfccs_audio_list_mean.append(mfccs_audio[i].mean())
        mfccs_audio_list_var.append(mfccs_audio[i].var())

    # Calculate mean and variance of each frame of CQT
    for i in range(len(cqt_audio)):
        cqt_audio_list_mean.append(cqt_audio[i].mean())
        cqt_audio_list_var.append(cqt_audio[i].var())

    # Calculate mean and variance of each frame of chromagram
    for i in range(len(chromagram_audio)):
        chromagram_audio_list_mean.append(chromagram_audio[i].mean())
        chromagram_audio_list_var.append(chromagram_audio[i].var())

    # Calculate mean and variance of each frame of tone
    for i in range(len(tone_audio)):
        tone_audio_list_mean.append(tone_audio[i].mean())
        tone_audio_list_var.append(tone_audio[i].var())

    # Calculate spectral bandwidth
    sb_audio = librosa.feature.spectral_bandwidth(y=audio_array, sr=sr, n_fft=FRAMESIZE, hop_length=HOPLENGTH)

    # Calculate amplitude envelope and root mean square (time-domain features)
    ae_audio = fancy_amplitude_envelope(audio_array, FRAMESIZE, HOPLENGTH)
    rms_audio = librosa.feature.rms(audio_array, frame_length=FRAMESIZE, hop_length=HOPLENGTH)

    return np.hstack((mean(ae_audio), var(ae_audio), mean(rms_audio), var(rms_audio), mean(sb_audio), var(sb_audio),
                      chromagram_audio_list_mean, chromagram_audio_list_var, tone_audio_list_mean, tone_audio_list_var,
                      cqt_audio_list_mean, cqt_audio_list_var, mfccs_audio_list_mean, mfccs_audio_list_var,
                      log_mel_audio_list_mean, log_mel_audio_list_var))


def fancy_amplitude_envelope(signal, framesize, hoplength):
    """
    Calculate the fancy amplitude envelope of a signal.

    Parameters:
        signal (ndarray): the input signal
        framesize (int): size of the frames for the envelope calculation
        hoplength (int): the hop length between frames for the envelope calculation

    Returns:
        ndarray: the fancy amplitude envelope of the input signal
    """
    return np.array([max(signal[i:i + framesize]) for i in range(0, len(signal), hoplength)])


def get_audio_features():
    """
    Get the audio features of a wav file.

    Returns:
        ndarray: the audio features of the wav file.
    """
    x_ver = transform_audio('audio.wav', 1024, 512, 13)
    return x_ver


def check_password(wav):
    """
    Check if the audio file matches a password.

    Parameters:
        wav (str): the path to the audio file

    Returns:
        int: 1 if the password is correct, 0 otherwise
    """
    # initialize recognizer class (for recognizing the speech)
    r = speech_recognition.Recognizer()

    # Reading Microphone as source
    # listening the speech and store in audio_text variable
    with speech_recognition.AudioFile(wav) as source:
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
    """
        Plot the feature importance of a classifier as a bar graph.

        Parameters:
            X (DataFrame): the feature data
            user_input (ndarray): the user's input data
            classifier (object): the trained classifier
            features_number (int): the number of features to plot (default=10)

        Returns:
            None
        """
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
    ax.barh(range(features_number), importances[indices], align='center', color='#0062d6')
    ax.set_yticks(range(features_number))
    ax.set_yticklabels(X.columns[indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance Plot Based on Confidence Score')
    ax.text(0.97, 0.94, 'Score: {:.2f}'.format(confidence_score), ha='right', va='bottom',
            transform=ax.transAxes, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig('static\\assets\\dynamic_plot.png')


def plot_feature_importance_scatter(X, user_input, classifier, features_number=10):
    """
        Plot the feature importance of a classifier as a scatter graph.

        Parameters:
            X (DataFrame): the feature data
            user_input (ndarray): the user's input data
            classifier (object): the trained classifier
            features_number (int): the number of features to plot (default=10)

        Returns:
            None
        """
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

