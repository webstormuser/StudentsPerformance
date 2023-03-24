from flask import Flask,request,render_template
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

#route for home page 
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediction',methods=['GET','POST'])
def predict_datapoints():
    if request.method=='GET':
        return render_template("home.html")
    else:
        data=CustomData(
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('race_ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'), 
        test_preparation_course=request.form.get('test_preparation_course'), 
        writing_score=float(request.form.get('writing_score')), 
        reading_score=float(request.form.get('reading_score'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)
        return render_template('home.html',result=round(result[0],2))

if __name__=="__main__":
    app.run(debug=True,port=5001)
