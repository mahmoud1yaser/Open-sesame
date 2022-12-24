from flask import Flask
from flask import request
from flask import render_template
# import speaker_model
# import text
import fingerprint_model as fp
import password_model as pm

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        f = request.files['audio_data']

        with open('input.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        password = pm.predict_password('input.wav')
        speaker = fp.predict_speaker('input.wav')
        if speaker == 1:
            user = 'Adham'
        elif speaker == 2:
            user = 'Mahmoud'
        elif speaker == 3:
            user = 'Ahmed'
        elif speaker == 4:
            user = 'Maha'
        else:
            user = 'User'

        if speaker != 0 and password == 1:
            access = 'Access Granted'
        elif speaker != 0 and password == 0:
            access = 'Invalid Password'
        else:
            access = 'Unauthorized Access'

        # if speaker == 'Speaker is not recognized':
        return (str(user) + ", "+str(access))
        # return (str(password) + ", " + str(speaker))
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
