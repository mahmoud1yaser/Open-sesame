from flask import Flask
from flask import request
from flask import render_template
import password_model as pm
import speaker_model
import text



app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        f = request.files['audio_data']
        # if f:
        #      f.save(os.path.join('static/assets/audio.wav'))
        
        with open('audio.wav','wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        # text_resl=text.text('audio.wav')
        text_resl = pm.predict_password('audio.wav')
        speaker_resl=speaker_model.speaker('audio.wav')
        return(str(text_resl)+", "+str(speaker_resl))        
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
