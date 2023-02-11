from flask import Flask
from flask import request
from flask import render_template
import password_model
import speaker_model
import utils
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        f = request.files['audio_data']
        with open('audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        speaker = speaker_model.predict_speaker()
        # password = password_model.predict_password()
        password = utils.check_password('audio.wav')
        if speaker == "User":
            return "You are not" + ", " + "Registered"
        if password == 0:
            return "Welcome "+str(speaker) + ", " + "Invalid Password"
        if password == 1 and not speaker == "User":
            return "Welcome " + str(speaker) + ", " + "Correct Password"
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
