from django.shortcuts import render


# Create your views here.
def index(request):
    context = {"page_title": "Record audio"}

    if request.method == "POST":
        f = request.files['audio_data']
        with open('audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        return render(request,"index.html",context)
    else:

        return render(request,"index.html",context)
