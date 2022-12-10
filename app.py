from flask import Flask
from flask import request
from flask import render_template
import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import librosa
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from lazypredict.Supervised import LazyClassifier


app = Flask(__name__)

df = pd.read_csv('data.csv')
df.drop(df.columns[0], inplace=True, axis=1)
y = df['result']
X = df.loc[:, df.columns != 'result']

cols = X.columns
scaler = preprocessing.MinMaxScaler()
np_scaled = scaler.fit_transform(X)

X = pd.DataFrame(np_scaled, columns = cols)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



model=RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
# model=SVC()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('Accuracy',':', round(accuracy_score(y_test, preds), 5), '\n')


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        f = request.files['audio_data']
        # if f:
        #      f.save(os.path.join('static/assets/audio.wav'))
        
        with open('audio.wav','wb') as audio:
            f.save(audio)
        print('file uploaded successfully')
        
        
        y, s = librosa.load("audio.wav")
        audio, _ = librosa.effects.trim(y)
        
        mfcc = librosa.feature.mfcc(y=audio, sr=s)
    
        
        data=[]
        for i in range(20):
            data.append(mfcc[i].mean()) 
            data.append(mfcc[i].var())
        input_data_as_numpy_array  = np.asarray(data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        std_data = scaler.transform(input_data_reshaped)
        word=model.predict(std_data)
        if word ==0:
           return  "Not allowed"
        elif word ==1:
            return "welcome"
        
    else:
        return render_template("index.html")






if __name__ == "__main__":
    app.run(debug=True)
