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
    longitude = request.form.get('longitude')
    latitude = request.form.get('latitude')
    housing_median_age = request.form.get('housing_median_age')
    total_rooms = request.form.get('total_rooms')
    total_bedrooms = request.form.get('total_bedrooms')
    population = request.form.get('population')
    households = request.form.get('households')
    median_income = request.form.get('median_income')
    
    rooms_per_household = float(total_rooms)/ float(households)
    bedrooms_per_room = float(total_bedrooms)/ float(total_rooms)
    population_per_household = float(population)/ float(households)
    
    
    if ( (request.form.get('ocean_proximity')) == 1 ):
        ocean_proximity1 =  1.0
        ocean_proximity2 =  0.0
        ocean_proximity3 =  0.0
        ocean_proximity4 =  0.0
        ocean_proximity5 =  0.0
    elif ((request.form.get('ocean_proximity')) == 2):
        ocean_proximity1 =  0.0
        ocean_proximity2 =  1.0
        ocean_proximity3 =  0.0
        ocean_proximity4 =  0.0
        ocean_proximity5 =  0.0
    elif ((request.form.get('ocean_proximity')) == 3):
        ocean_proximity1 =  0.0
        ocean_proximity2 =  0.0
        ocean_proximity3 =  1.0
        ocean_proximity4 =  0.0
        ocean_proximity5 =  0.0
    elif ((request.form.get('ocean_proximity')) == 4):
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
    arr = np.array([ longitude, latitude, housing_median_age, total_rooms, population, households,
                    median_income, rooms_per_household,bedrooms_per_room, population_per_household,
                    ocean_proximity1, ocean_proximity2, ocean_proximity3, ocean_proximity4, ocean_proximity5])
    
    dataF=pd.DataFrame(data=[arr], columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'population', 'households',
                                            'median_income','rooms_per_household', 'bedrooms_per_room', 
                                            'population_per_household', 'ocean_1', 'ocean_2', 'ocean_3', 'ocean_4', 'ocean_5'])
    new_features = pipline.transform(dataF)
    prediction = model.predict(new_features)
    return render_template('index.html', prediction_text='{}'.format(prediction))
if __name__=='__main__':
    app.run(debug=True)

