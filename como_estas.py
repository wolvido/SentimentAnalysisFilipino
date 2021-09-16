from fastai.text.all import *
import pandas as pd 

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from google_trans_new import google_translator

def sentiment(word):
    translator = google_translator() 
    translate_text = translator.translate(word,lang_src='fil', lang_tgt='en') 

    model = load_learner('sentiment')
    return model.predict(translate_text), translate_text

print(sentiment("Youre a fucking piece of shit motherfucker bitch and I hate you I will kill you"))