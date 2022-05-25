import pandas as pd 
import joblib
import json
import numpy as np
model = joblib.load("rf_model.sav")
from flask import Flask, request, jsonify
app = Flask(__name__)


@app.route("/" ,methods=['GET','POST'])
def home():
    df=pd.read_csv("test_Data.csv")
    X = df.drop('Status', axis=1)
    y = df.Status
    prediction = model.predict(X)
    for i in range(0,len(prediction)):
        if prediction[i] ==1:
            print(i)
    return {"preds":str(prediction)}
    if prediction == 0:
        print("NOT")
        return {"prediction":"Not Recovered"}
    else:
        print("Rec")
        return 'Recovered'



@app.route("/api", methods=['GET','POST'])
def api():
    
    r = request.json
    msg = {

        "Occurrence_Year":r['occurrence_year'],
        "Occurrence_Month":r['occurrence_month'],
        "Occurrence_DayOfWeek":r['occurrence_dayofweek'],
        "Occurrence_DayOfMonth":r['occurrence_dayofmonth'],
        "Occurrence_DayOfYear":r['Occurrence_DayOfYear'],
        "Occurrence_Hour":r['occurrence_hour'],
        "Premises_Type":r['premises_type'],
        "Bike_Type":r['bike_type'],
        "Bike_Speed":r['bike_speed'],
        "Cost_of_Bike":r['bike_cost'],
        "Longitude":r['longitude'],
        "Latitude":r['latitude'],
         "ObjectId2":r['ObjectId2']

    }
    row = pd.DataFrame(msg, index=[0])
    print(row.head())
    prediction = model.predict(row)    
    if prediction == 0:
        return "Not Recovered"
    else:
        return 'Recovered'

if __name__ == '__main__':
    app.run()
