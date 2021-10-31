# -*- coding: utf-8 -*-

from flask import Flask,render_template,request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("student marks predict.pkl")

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST"])
def predict():
    global df
    
    input_feature = [int(x) for x in request.form.values()]
    feature_value = np.array(input_feature)
    
    if(input_feature[0] < 0 or input_feature[0] > 24):
        return render_template('index.html',prediction_text="please enter valid hours if you live on earth")
    
    output = model.predict([feature_value])[0][0].round(2)
  
    df = pd.concat([df,pd.DataFrame({'Study_hours': input_feature,'predicted_output' :[output]})],ignore_index = True)
    print(df)
    df.to_csv("C:\\Users\\hp\\Desktop\\student_marks.csv")
    return render_template('index.html',prediction_text = "you will get [{}%] marks when you study [{}] hours".format(output,int(input_feature[0])))

    
if(__name__=='__main__'):
    app.run()