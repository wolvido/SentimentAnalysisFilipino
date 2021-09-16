from flask import Flask, request, render_template 

#ML model driver code
from fastai.text.all import *
import pandas as pd 

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from google_trans_new import google_translator

def sentiment(word):
    #translate 
    translator = google_translator() 
    translate_text = translator.translate(word , lang_src='fil', lang_tgt='en') 
    #load model
    model = load_learner('sentiment') 

    if "pos" in model.predict(translate_text):
        return "positive"
    else:
        return "negative"
############

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        text = request.form.get("text")
        return render_template("index.html", 
            sentiment = sentiment(text))

if __name__ == "__main__":
    app.run(debug=True)
