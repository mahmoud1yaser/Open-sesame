# Import the required utilities module
import utils

# Import necessary libraries for data cleaning and machine learning
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved classifier models for speaker identification and password prediction
classifier_fingerprint = utils.read_pickle('models\\rf_speaker_mod.pkl')
classifier_password = utils.read_pickle('models\\rf_password_mod.pkl')

# Initialize the StandardScaler objects for the speaker identification and password prediction
sc_fingerprint = StandardScaler()
sc_password = StandardScaler()

# Preprocess the speaker identification and password prediction data by splitting and standardizing the data
x_fingerprint_features, sc_fingerprint = utils.split_standardize('data\\speaker_data.csv', 'data\\speaker_target.csv'
                                                                 , sc_fingerprint)
x_password_features, sc_password = utils.split_standardize('data\\password_data.csv', 'data\\password_target.csv'
                                                           , sc_password)


# Function to predict the speaker identification based on the input audio features
def predict_speaker(x_input):
    # Standardize the input features
    x_input = sc_fingerprint.transform(x_input.reshape(1, -1))
    # Plot the feature importance bar plot
    utils.plot_feature_importance_bar(pd.DataFrame(x_fingerprint_features), x_input, classifier_fingerprint)
    # Predict the speaker identification based on the input features
    speaker_id = classifier_fingerprint.predict(x_input)

    # Map the speaker id to the speaker name
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


# Function to predict the password based on the input audio features
def predict_password(x_input):
    # Standardize the input features
    x_input = sc_password.transform(x_input.reshape(1, -1))
    # Predict the password based on the input features
    password_id = classifier_password.predict(x_input)
    return password_id
