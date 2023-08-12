
from flask import Flask, request, render_template
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))
model.check_input = False

print("Loaded model:", model)


@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the form and convert to the required data types
    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    # Create a feature array
    features = [N, P, K, temperature, humidity, ph, rainfall]
    
    print("Input features:", features)
    
    features = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)
    prediction = model.predict(features)
    
    
    print("Prediction:", prediction)

    # Crop recommendations
    crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
    }

    # if prediction[0] in crop_dict.values():
    #     crop = [key for key, value in crop_dict.items() if value == predict][0]
    #     result = "{} is the best crop to be cultivated in your farm".format(crop)
    # else:
    #     "Sorry!!! We are unable to recommend a crop for this environment"
    # return render_template("index.html", result=result)

        
    predict = prediction[0]
    if predict in crop_dict.values():
        crop = [key for key, value in crop_dict.items() if value == predict][0]
        result = "{} is the best crop to be cultivated in your farm".format(crop)
    else:
        result = "Sorry!!! We are unable to recommend a crop for this environment"
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
