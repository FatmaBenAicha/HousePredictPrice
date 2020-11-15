from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
with open('models/BaggingRegressor.pkl', 'rb') as f:
    model = pickle.load(f)
with open('dataPreparation/pipline.pkl', 'rb') as p:
    pipline = pickle.load(p)
app = Flask('__name__')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/',methods=["POST"])
def predict():
    longitude = float(request.form['longitude'])    
    latitude = float(request.form['latitude'])
    housing_median_age = float(request.form['housing_median_age'])
    total_rooms = float(request.form['total_rooms'])
    population = float(request.form['population'])
    households = float(request.form['households'])
    median_income =float( request.form['median_income'])
    #calcul
    total_bedrooms = float(request.form['total_bedrooms'])
    rooms_per_household = float(total_rooms)/ float(households)
    bedrooms_per_room = float(total_bedrooms)/ float(total_rooms)
    population_per_household = float(population)/ float(households)
    
    if ( (request.form['ocean_proximity']) == 1 ):
        ocean_proximity1 =  1.0
        ocean_proximity2 =  0.0
        ocean_proximity3 =  0.0
        ocean_proximity4 =  0.0
        ocean_proximity5 =  0.0
    elif ((request.form['ocean_proximity']) == 2):
        ocean_proximity1 =  0.0
        ocean_proximity2 =  1.0
        ocean_proximity3 =  0.0
        ocean_proximity4 =  0.0
        ocean_proximity5 =  0.0
    elif ((request.form['ocean_proximity']) == 3):
        ocean_proximity1 =  0.0
        ocean_proximity2 =  0.0
        ocean_proximity3 =  1.0
        ocean_proximity4 =  0.0
        ocean_proximity5 =  0.0
    elif ((request.form['ocean_proximity']) == 4):
        ocean_proximity1 =  0.0
        ocean_proximity2 =  0.0
        ocean_proximity3 =  0.0
        ocean_proximity4 =  1.0
        ocean_proximity5 =  0.0
    else:
        ocean_proximity1 =  0.0
        ocean_proximity2 =  0.0
        ocean_proximity3 =  0.0
        ocean_proximity4 =  0.0
        ocean_proximity5 =  1.0
    
    arr = np.array([[longitude,latitude,housing_median_age,total_rooms,population,
    households,median_income,rooms_per_household ,bedrooms_per_room,population_per_household,
    ocean_proximity1,ocean_proximity2,ocean_proximity3,ocean_proximity4,ocean_proximity5]])
    
    new_features = pipline.transform(arr)
    prediction = model.predict(new_features)

    return render_template('index.html', prediction_text='{}$'.format(prediction))
if __name__=='__main__':
    app.run(debug=True)

