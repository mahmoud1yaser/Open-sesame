import utils as u
# Data cleaning
import os
import pandas as pd
import numpy as np
# Data visualization
import matplotlib.pyplot as plt
# Machine learning
import sklearn
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Advanced options
import warnings
warnings.filterwarnings("ignore")
import pickle

FRAMESIZE = 1024
HOPLENGTH = 512
# 13 mels for fingerprint and more than 20 mels for word detection
MELS = 13
SPLITFREQ = 2000
PCA_N = 2

classifier_path = 'models\\rf_speaker.pkl'

df_big_chunk = pd.read_csv('data\\big_chunk_speaker.csv')

xb_features = np.array(df_big_chunk.iloc[:,:])

y_features = pd.read_csv('data\\target_data_speaker.csv')

best_features_big = list(np.loadtxt(open('data\\speaker_features_big.csv'), delimiter=",", dtype='str'))

xb_train, xb_test, y_train, y_test = train_test_split(xb_features, y_features, test_size=0.3)

sc = StandardScaler()
pca = PCA(n_components=PCA_N)

xb_train = sc.fit_transform(xb_train)
xb_train_pca = pca.fit_transform(xb_train)

classifier = pickle.load(open(classifier_path, 'rb'))
# classifier = XGBClassifier()
# classifier.load_model(classifier_path)

x_ver_features = []
for x in best_features_big:
    x_ver_features.append(int(x.split(' ')[1]))


def predict_speaker(audio):
    x_ver = u.transform_audio(audio, FRAMESIZE * 4, HOPLENGTH * 4, MELS, SPLITFREQ)  # Big-chunk
    x_ver = x_ver[x_ver_features]
    x_ver = sc.transform(x_ver.reshape(1, -1))
    x_ver = pca.transform(x_ver)[0]

    return classifier.predict(np.array([x_ver[0].ravel(), x_ver[1].ravel()]).T).reshape(x_ver[0].shape)
