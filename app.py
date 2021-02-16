import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import string

def remove_punctuation(text):
    text = str(text).lower()
    table1 = text.maketrans('1234567890','          ',string.punctuation)
    text = text.replace('not ','not')
    text = text.replace('dont ','dont')
    text = text.replace('very ','very')
    text = text.translate(table1)
    return text


app = Flask(__name__)

vect = pickle.load(open("vectorizer.pkl", "rb"))
vect_model = pickle.load(open("vect_model.pkl", "rb"))

tfidf = pickle.load(open("tfidf.pkl", "rb"))
tfidf_model = pickle.load(open("tfidf_model.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    review = str(request.form["review"])
    review_without_punctuations = remove_punctuation(review)
    review_totest1 = vect.transform([review_without_punctuations])
    review_totest2 = tfidf.transform([review_without_punctuations])
    
    
    prob1 = vect_model.predict_proba(review_totest1)
    prob2 = tfidf_model.predict_proba(review_totest2)
    
    finalpred = np.mean([prob1[0][1],prob2[0][1]])
    

    if finalpred >= 0.6:
        res = "Hurrah! That is " + str(round(finalpred,2)*100) +"%"+ "  POSITIVE🤩"
        src = "postive.jpg"
    elif finalpred <= 0.4:
        res = "Oops! That is " + str(100 - round(finalpred,2)*100)+ "%"+ "  NEGATIVE😥"
        src = "negative.jpg"
    else:
        res = "Can't say  That seems NEUTRAL🤔"
        src = "neutral.jpg"
    
    return render_template('index.html',prediction_text='{}'.format(res), alt= "nothing")
                           
   


if __name__ == "__main__":
    app.run(debug=True)
                           
                           
                           
                           
                           
                           
                           
                           
                           
                           
                           
                           
                           