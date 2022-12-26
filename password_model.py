import utils as u
# Data cleaning
import os
import pandas as pd
import numpy as np
# Data visualization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
MELS = 20
SPLITFREQ = 2000
PCA_N = 2

classifier_path = 'models\\svm_word.pkl'

df_big_chunk = pd.read_csv('data\\big_chunk_word.csv')

xb_features = np.array(df_big_chunk.iloc[:,:])

y_features = pd.read_csv('data\\target_data_word.csv')

best_features_big = list(np.loadtxt(open('data\\word_features_big.csv'), delimiter=",", dtype='str'))

xb_train, xb_test, y_train, y_test = train_test_split(xb_features, y_features, test_size=0.1)

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


def predict_password(audio):
    x_ver = u.transform_audio(audio, FRAMESIZE * 4, HOPLENGTH * 4, MELS, SPLITFREQ)  # Big-chunk
    x_ver = x_ver[x_ver_features]
    x_ver = sc.transform(x_ver.reshape(1, -1))
    x_ver = pca.transform(x_ver)[0]

    # Predicting the training set
    # result through scatter plot
    fig = plt.figure()

    plt.scatter(x_ver[0].reshape(1,-1), x_ver[1].reshape(1,-1), color='red', label='input')
    print(x_ver)
    print(np.array(y_train.iloc[:,0]))
    X_set, y_set = xb_train_pca, np.array(y_train.iloc[:,0])
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                   stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1,
                                   stop=X_set[:, 1].max() + 1, step=0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                      X2.ravel()]).T).reshape(X1.shape), alpha=0.3,
                 cmap=ListedColormap(('white', 'aquamarine')))

    #         plt.xlim(X1.min(), X1.max())
    #         plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('green', 'blue'))(i), label=j)

    plt.title('Model (Training set)')
    plt.xlabel('PC1')  # for Xlabel
    plt.ylabel('PC2')  # for Ylabel
    plt.legend()  # to show legend
    # show scatter plot
    # plt.show()
    imageName='static/assets/Feature_visuals.png'
    fig.savefig(imageName)

    min_ver_0 = 1.2*xb_train_pca[:,0].min()
    max_ver_0 = 1.2*xb_train_pca[:,0].max()
    min_ver_1 = 1.2*xb_train_pca[:,1].min()
    max_ver_1 = 1.2*xb_train_pca[:,1].max()

    x_ver_0_condition = min_ver_0 < x_ver[0] < max_ver_0
    x_ver_1_condition = min_ver_1 < x_ver[1] < max_ver_1

    if classifier.predict(x_ver.reshape(1,-1)) == 0 or ~x_ver_0_condition or ~x_ver_1_condition:
        return 'Invalid Password'
    else:
        return 'Welcome'
