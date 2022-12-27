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

classifier_path = 'models\\speaker\\xgb_speaker.json'
classifier_pca_path = 'models\\speaker\\xgb_pca_speaker.json'


classifier = XGBClassifier()
classifier.load_model(classifier_path)

classifier_pca = XGBClassifier()
classifier_pca.load_model(classifier_pca_path)


def predict_speaker():
    x_ver = feat.get_audio_features()
    print(x_ver)
    # x_ver = np.array(x_ver)
    speaker_id = classifier.predict(x_ver)

    fig_speaker = plt.figure()
    plt.scatter(x_ver[:, 0]-4, x_ver[:, 1] - 2, label='input', c='black', marker='*', s=100)
    print(x_ver[:, 0])

    xb_train, y_train, _, _ = feat.get_plot_data()
    X_set, y_set = xb_train[:, :2], y_train[:, 0]
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                   stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1,
                                   stop=X_set[:, 1].max() + 1, step=0.01))

    plt.contourf(X1, X2, classifier_pca.predict(np.array([X1.ravel(),
                                                          X2.ravel()]).T).reshape(X1.shape), alpha=0.4,
                 cmap=ListedColormap(('#b35c52', '#6a9681', '#6b7ca8', '#9483a0', '#adaa7b')))

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('#9b2012', '#0f633b', '#234499', '#602d90', '#88842a'))(i), label=j)

    plt.title('Model (Speaker)')
    plt.xlabel('PC1')  # for Xlabel
    plt.ylabel('PC2')  # for Ylabel
    plt.legend()  # to show legend
    fig_speaker.savefig("static\\assets\\speaker_feat.png")

    # fig_spect = plt.figure()
    # ld.specshow(x_spect,
    #             x_axis="time",
    #             y_axis="mfccs",
    #             sr=sr)
    # plt.colorbar(format="%+2.f")
    # fig_spect.savefig("static\\assets\\Feature_visuals.png")

    if speaker_id == 4:
        return 'Maha'
    elif speaker_id == 1:
        return 'Adham'
    elif speaker_id == 2:
        return 'Mahmoud'
    elif speaker_id == 3:
        return 'Ahmed'
    else:
        return 'User'




# def plot_speaker():
#     fig = plt.figure()
#     x_ver = feat.get_audio_features()
#     plt.scatter(x_ver[:, 0], x_ver[:, 1]-5, label='input', c='black', marker='*', s=100)
#     print(x_ver[:, 0])
#     X_set, y_set = xb_train[:, :2], y_train[:, 0]
#     X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
#                                    stop=X_set[:, 0].max() + 1, step=0.01),
#                          np.arange(start=X_set[:, 1].min() - 1,
#                                    stop=X_set[:, 1].max() + 1, step=0.01))
#
#     plt.contourf(X1, X2, classifier_pca.predict(np.array([X1.ravel(),
#                                                           X2.ravel()]).T).reshape(X1.shape), alpha=0.4,
#                  cmap=ListedColormap(('#b35c52', '#6a9681', '#6b7ca8', '#9483a0', '#adaa7b')))
#
#     for i, j in enumerate(np.unique(y_set)):
#         plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                     color=ListedColormap(('#9b2012', '#0f633b', '#234499', '#602d90', '#88842a'))(i), label=j)
#
#     plt.title('Model (Speaker)')
#     plt.xlabel('PC1')  # for Xlabel
#     plt.ylabel('PC2')  # for Ylabel
#     plt.legend()  # to show legend
#     imageName = "static\\assets\\speaker_feat.png"
#     # plt.show()
#     fig.savefig(imageName)

