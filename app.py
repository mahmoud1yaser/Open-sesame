# Importing necessary libraries and modules
from flask import Flask, request, render_template
import predict # Custom module for audio fingerprint prediction
import utils # Custom module for utility functions
import warnings

# Ignoring warnings for a cleaner output
warnings.filterwarnings("ignore")

# Creating Flask application object
app = Flask(__name__)


# Defining route for the index page
@app.route("/", methods=['POST', 'GET'])
def index():
    # Check if the request is a POST request
    if request.method == "POST":
        # Get the audio data from the POST request
        f = request.files['audio_data']
        # Save the audio data to a local file
        with open('audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        # Get the audio features from the saved audio file
        user_input = utils.get_audio_features()
        # Predict the speaker using the audio features
        speaker = predict.predict_speaker(user_input)
        # Check the password for the speaker
        # password = predict.predict_password(user_input)
        password = utils.check_password('audio.wav')
        # Return different responses based on the results of speaker and password predictions
        if speaker == "User":
            return "You are not" + ", " + "Registered"
        if password == 0:
            return "Welcome "+str(speaker) + ", " + "Invalid Password"
        if password == 1 and not speaker == "User":
            return "Welcome " + str(speaker) + ", " + "Correct Password"
    else:
        # Return the HTML template for GET requests
        return render_template("index.html")


# Running the application
if __name__ == "__main__":
    app.run(debug=True)
