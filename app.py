from flask import Flask, request, render_template
import ast
import numpy as np
import pandas 
import sklearn
import pickle


model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standardscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    # Create feature list
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    
    # Reshaping the features to match model input shape
    single_pred = np.array(feature_list).reshape(1,-1)

    
    # Model prediction
    prediction = model.predict(single_pred)

    # Define a dictionary for crop prediction
    crop_dict = {
        1: "rice", 2: "maize", 3: "jute", 4: "cotton", 5: "coconut", 6: "papaya",
        7: "orange", 8: "apple", 9: "muskmelon", 10: "watermelon", 11: "grapes", 
        12: "mango", 13: "banana", 14: "pomegranate", 15: "lentil", 16: "blackgram", 
        17: "mungbean", 18: "mothbeans", 19: "pigeonpeas", 20: "kidneybeans", 
        21: "chickpea", 22: "coffee"
    }

    # Check if the predicted crop is in the dictionary
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated there.".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated."

    # Render the result in the template
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)  # In development