import utils
import pickle

# Data cleaning
import pandas as pd
import numpy as np
# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

classifier_path = 'models\\rf_speaker_mod.pkl'
with open(classifier_path, 'rb') as c:
    classifier = pickle.load(c)

x_features = np.array(pd.read_csv('data\\speaker_data.csv'))
y_features = pd.read_csv('data\\speaker_target.csv')
x_train, x_test, _, _ = train_test_split(x_features, y_features, test_size=0.3)

# performing preprocessing part
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


def predict_speaker():
    x_ver = utils.get_audio_features()
    x_ver = sc.transform(x_ver.reshape(1, -1))
    utils.plot_feature_importance_bar(pd.DataFrame(x_features), x_ver, classifier)
    speaker_id = classifier.predict(x_ver)

    if speaker_id == 1:
        return 'Adham'
    elif speaker_id == 2:
        return 'Mahmoud'
    elif speaker_id == 3:
        return 'Ahmed'
    elif speaker_id == 4:
        return 'Maha'
    else:
        return 'User'
