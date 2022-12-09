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
from pip import LazyClassifier


df = pd.DataFrame()
#load audio
directory= "data/"
models = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        data=f
        
    y, s = librosa.load(data)
    audio, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=audio, sr=s)
    mfccMeans=[]
    mfccVar=[]
    
    for i in range(20):
       mfccMeans.append(mfcc[i].mean()) 
       mfccVar.append(mfcc[i].var())
    if "close"  in filename:
        result=0
    elif "other"  in filename:
        result=0
    elif "open"  in filename:
        result=1
    elif "otd"  in filename:
        result=1    
    elif "Otd"  in filename:
        result=1    
    data={
        'mfcc1_mean':mfccMeans[0],
        'mfcc1_var' :mfccVar[0],
        'mfcc2_mean':mfccMeans[1],
        'mfcc2_var' :mfccVar[1],
        'mfcc3_mean':mfccMeans[2],
        'mfcc3_var' :mfccVar[2],
        'mfcc4_mean':mfccMeans[3],
        'mfcc4_var' :mfccVar[3],
        'mfcc5_mean':mfccMeans[4],
        'mfcc5_var' :mfccVar[4],
        'mfcc6_mean':mfccMeans[5],
        'mfcc6_var' :mfccVar[5],
        'mfcc7_mean':mfccMeans[6],
        'mfcc7_var' :mfccVar[6],
        'mfcc8_mean':mfccMeans[7],
        'mfcc8_var' :mfccVar[7],
        'mfcc9_mean':mfccMeans[8],
        'mfcc9_var' :mfccVar[8],
        'mfcc10_mean':mfccMeans[9],
        'mfcc10_var' :mfccVar[9],
        'mfcc11_mean':mfccMeans[10],
        'mfcc11_var' :mfccVar[10],
        'mfcc12_mean':mfccMeans[11],
        'mfcc12_var' :mfccVar[11],
        'mfcc13_mean':mfccMeans[12],
        'mfcc13_var' :mfccVar[12],
        'mfcc14_mean':mfccMeans[13],
        'mfcc14_var' :mfccVar[13],
        'mfcc15_mean':mfccMeans[14],
        'mfcc15_var' :mfccVar[14],
        'mfcc16_mean':mfccMeans[15],
        'mfcc16_var' :mfccVar[15],
        'mfcc17_mean':mfccMeans[16],
        'mfcc17_var' :mfccVar[16],
        'mfcc18_mean':mfccMeans[17],
        'mfcc18_var' :mfccVar[17],
        'mfcc19_mean':mfccMeans[18],
        'mfcc19_var' :mfccVar[18],
        'mfcc20_mean':mfccMeans[19],
        'mfcc20_var' :mfccVar[19],
        'result':[result]
    }
    dfTemp=pd.DataFrame(data)
    # df.append(dfTemp,ignore_index=True)
    df=pd.concat([df, dfTemp], ignore_index=True)

print("Convert to csv file? (T/N)")
convert=input()
if convert=="t":
    df.to_csv('data.csv')
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
    
    #test
    y, s = librosa.load("mohmoud other 11.wav")
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
        print("Not allowed")
    elif word ==1:
        print("open the door")
    