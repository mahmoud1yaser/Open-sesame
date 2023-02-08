import utils
import pickle

# Data cleaning
import pandas as pd
import numpy as np
# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

classifier_path = 'models\\rf_password_mod.pkl'
with open(classifier_path, 'rb') as c:
    classifier = pickle.load(c)

x_features = np.array(pd.read_csv('data\\password_data.csv'))
y_features = pd.read_csv('data\\password_target.csv')
x_train, x_test, _, _ = train_test_split(x_features, y_features, test_size=0.3)

# performing preprocessing part
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


def predict_password():
    x_ver = utils.get_audio_features()
    x_ver = sc.transform(x_ver.reshape(1, -1))
    password_id = classifier.predict(x_ver)
    return password_id


