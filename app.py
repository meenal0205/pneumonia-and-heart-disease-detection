from sklearn.preprocessing import MaxAbsScaler
from sklearn import preprocessing
import os
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import requests
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pandas_profiling as pp
import math
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


app = Flask(__name__)

data = pd.read_csv('heart disease/heart.csv')
data['total_risk_factor'] = 0
data.loc[data['trestbps'] > 140, 'total_risk_factor'] += 1
data.loc[data['chol'] > 240, 'total_risk_factor'] += 1
data.loc[data['fbs'] > 120, 'total_risk_factor'] += 1
data.loc[data['ca'] >= 1, 'total_risk_factor'] += 1
data.loc[data['thal'] == 3, 'total_risk_factor'] += 1
x = data.drop('target', axis=1)
y = data['target']
scaler = MaxAbsScaler()
x_scaled = scaler.fit_transform(x)
x_scaled_data = pd.DataFrame(x_scaled, columns=x.columns)


x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42)
model_path = 'C:\EDAI_SEM_2\model_2.h5'
model = load_model(model_path)
model.make_predict_function()


model5 = GradientBoostingClassifier()
model5.fit(x_train, y_train)
y_pred5 = model5.predict(x_test)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def upload():
    imagefile = request.files['imagefile']
    imagepath = "./images/"+imagefile.filename
    imagefile.save(imagepath)
    result = model_predict(imagepath, model)
    s = ""
    if(result == 1):
        s = "Pneumonic"
    else:
        s = "Normal"
    return render_template("result.html", result=s)


def model_predict(img_path, model):
    img = load_img(img_path, target_size=(150, 150, 3))
    img = img_to_array(img)
    img = img/255
    img = img.reshape(-1, 150, 150, 1)
    predictions = model.predict(img)
    predictions = np.round(predictions)
    predictions = predictions.reshape(1, -1)[0]
    return predictions[0]


@app.route("/pneumonia", methods=["GET", "POST"])
def pneumonia():
    return render_template("Pneumonia.html")


@app.route("/heart_disease", methods=["POST", "GET"])
def heart_disease():
    if request.method == "POST":
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        chestpain = int(request.form['chestpain'])
        restingbp = int(request.form['restingbp'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        rer = int(request.form['rer'])
        maxheartrate = int(request.form['maxheartrate'])
        diffbreath = int(request.form['diffbreath'])
        stdi = float(request.form['stdi'])
        shiftheart = int(request.form['shiftheart'])
        thal = int(request.form['thal'])
        ca = int(random.randint(0, 2))

        total_risk = 0
        if(restingbp > 140):
            total_risk += 1
        if(chol > 240):
            total_risk += 1
        if(fbs == 1):
            total_risk += 1
        if(ca >= 1):
            total_risk += 1
        if(thal == 3):
            total_risk += 1
        arr = np.array(
            [[age, sex, chestpain, restingbp, chol, fbs, rer, maxheartrate, diffbreath, stdi, shiftheart, ca, thal, total_risk]])
        result = model5.predict(arr)
        msg = ""
        if(result[0] == 1):
            msg = "You have high risk of getting a heart disease"
        else:
            msg = "You have low risk of getting a heart disease"
        return render_template("result.html", result=msg)

    return render_template("Heart_disease.html")


if __name__ == '__main__':
    app.run(debug=True)
