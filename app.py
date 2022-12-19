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
import speaker_model
import text


app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        f = request.files['audio_data']
        # if f:
        #      f.save(os.path.join('static/assets/audio.wav'))
        
        with open('audio.wav','wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        text_resl=text.text('audio.wav')
        speaker_resl=speaker_model.speaker('audio.wav')
        return(str(text_resl)+", "+str(speaker_resl))        
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
