import audio_features as feat
# Data cleaning
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import librosa.display as ld
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Advanced options
import warnings
plt.style.use('ggplot')
warnings.filterwarnings("ignore")

classifier_path_word = 'models\\password\\xgb_word.json'
classifier_pca_path_word = 'models\\password\\xgb_pca_word.json'

# classifier = pickle.load(open(classifier_path, 'rb'))
# classifier_pca = pickle.load(open(classifier_pca_path, 'rb'))
classifier_word = XGBClassifier()
classifier_word.load_model(classifier_path_word)

classifier_pca_word = XGBClassifier()
classifier_pca_word.load_model(classifier_pca_path_word)


def predict_password():
    x_ver = feat.get_audio_features()
    print(x_ver)
    speaker_id = classifier_word.predict(x_ver)

    _, _, xb_train_p, y_train_p = feat.get_plot_data()
    fig_password = plt.figure()
    plt.scatter(x_ver[:, 0] - 4, x_ver[:, 1] - 2, label='input', c='black', marker='*', s=100)
    print(x_ver[:, 0])
    # x_ver = np.array(x_ver)
    X_set, y_set = xb_train_p[:, :2], y_train_p[:, 0]
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                   stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1,
                                   stop=X_set[:, 1].max() + 1, step=0.01))

    plt.contourf(X1, X2, classifier_pca_word.predict(np.array([X1.ravel(),
                                                               X2.ravel()]).T).reshape(X1.shape), alpha=0.4,
                 cmap=ListedColormap(('#b35c52', '#6a9681')))

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('#9b2012', '#0f633b'))(i), label=j)

    plt.title('Model (Password)')
    plt.xlabel('PC1')  # for Xlabel
    plt.ylabel('PC2')  # for Ylabel
    plt.legend()  # to show legend
    fig_password.savefig("static\\assets\\password_feat.png")

    if speaker_id == 0:
        return 'Invalid Password'
    elif speaker_id == 1:
        return 'Correct Password'





