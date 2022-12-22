import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import librosa
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from lazypredict.Supervised import LazyClassifier



df = pd.read_csv('data2.csv')
df.drop(df.columns[0], inplace=True, axis=1)
y = df['result']
X = df.loc[:, df.columns != 'result']

cols = X.columns
scaler = preprocessing.MinMaxScaler()
np_scaled = scaler.fit_transform(X)

X = pd.DataFrame(np_scaled, columns = cols)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model=SVC()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('Accuracy',':', round(accuracy_score(y_test, preds), 5), '\n')

#test
def speaker(audio):
    y, s = librosa.load(audio)
    audio, _ = librosa.effects.trim(y,top_db=30)
    

    mfcc = librosa.feature.mfcc(y=audio, sr=s,n_fft=1024)
    data=[]
    #features
    chromagram = librosa.feature.chroma_stft(audio, sr=s,n_fft=1024)
    data.append(chromagram.mean())
    mel = librosa.feature.melspectrogram(y=audio, sr=s,n_fft=1024)
    data.append(mel.mean())
    tone =librosa.feature.tonnetz (y=audio, sr=s)
    data.append(tone.mean())



    for i in range(20):
           data.append(mfcc[i].mean()) 
    input_data_as_numpy_array  = np.asarray(data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.transform(input_data_reshaped)
    speaker=model.predict(std_data)
    if speaker ==1:
        return ("Adham")
    elif speaker ==2:
        return ("Ahmed")
    elif speaker ==3:
        return ("Maha")
    elif speaker ==4:
        return("Mohmoud")
    elif speaker ==5:
        return("others")