import utils as u
# Data cleaning
import pandas as pd
import numpy as np
# Data visualization
# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Advanced options
import warnings
warnings.filterwarnings("ignore")

FRAMESIZE = 1024
HOPLENGTH = 512
# 13 mels for fingerprint and more than 20 mels for word detection
MELS = 13
SPLITFREQ = 2000
PCA_N = 2

df_big_chunk_s = pd.read_csv('data\\big_chunk_speaker.csv')

xb_features_s = np.array(df_big_chunk_s.iloc[:,:])

y_features_s = pd.read_csv('data\\target_data_speaker.csv')
y_features_s = np.array(y_features_s.iloc[:,:])

# ------------------------------------------------------------------

df_big_chunk_p = pd.read_csv('data\\big_chunk_word.csv')

xb_features_p = np.array(df_big_chunk_p.iloc[:,:])

y_features_p = pd.read_csv('data\\target_data_word.csv')
y_features_p = np.array(y_features_p.iloc[:,:])


xb_train_s, xb_test_s, y_train_s, y_test_s = train_test_split(xb_features_s, y_features_s, test_size=0.3)
xb_train_p, xb_test_p, y_train_p, y_test_p = train_test_split(xb_features_p, y_features_p, test_size=0.3)
# performing preprocessing part
sc = StandardScaler()
# pca = PCA(n_components=PCA_N)
# -----------------------------------------------
xb_train_s = sc.fit_transform(xb_train_s)
xb_test_s = sc.transform(xb_test_s)

xb_train_p = sc.fit_transform(xb_train_p)
xb_test_p = sc.transform(xb_test_p)
# xb_train_pca = pca.fit_transform(xb_train)
# xb_test_pca = pca.transform(xb_test)


def get_audio_features():
    x_ver = u.transform_audio('audio.wav', FRAMESIZE * 4, HOPLENGTH * 4, MELS, SPLITFREQ)  # Big-chunk
    x_ver = sc.transform(x_ver.reshape(1, -1))
    return x_ver


def get_plot_data():
    x_ver = get_audio_features()  # Big-chunk
    # x_ver_pca = pca.transform(x_ver)[0]
    return xb_train_s, y_train_s, xb_train_p, y_train_p


