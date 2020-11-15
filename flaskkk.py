from flask import Flask, render_template, request
import pickle
import numpy as np

with open('models/LinearRegression.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def home():
    longitude = float(request.form['longitude'])
    latitude =  float(request.form['latitude'])
    housing_median_age =  float(request.form['housing_median_age'])
    total_room =  float(request.form['total_rooms'])
    total_bedrooms =  float(request.form['total_bedrooms'])
    population =  float(request.form['population'])
    households =  float(request.form['households'])
    median_income =  float(request.form['median_income'])
    bedrooms_per_room = (total_bedrooms)/ (total_rooms)
    population_per_household = (population)/ (households)
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
    
        
        
    
    arr = np.array([[longitude,latitude,housing_median_age,total_room,total_bedrooms,population,households,median_income,bedrooms_per_room,population_per_household,ocean_proximity1,ocean_proximity2,ocean_proximity3,ocean_proximity4,ocean_proximity5]])
    pred = model.predict(arr)

    data = round(pred[0], 2)
    return render_template('index.html', prediction_text='$ {}'.format(data))
if __name__ == "__main__":
    app.run()