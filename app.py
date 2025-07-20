from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask (__name__)

model = joblib.load('heart_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        features = [
            float(request.form['age']),
            float(request.form['anaemia']),
            float(request.form['creatinine_phosphokinase']),
            float(request.form['diabetes']),
            float(request.form['ejection_fraction']),
            float(request.form['high_blood_pressure']),
            float(request.form['platelets']),
            float(request.form['serum_creatinine']),
            float(request.form['serum_sodium']),
            float(request.form['sex']),
            float(request.form['smoking']),
            float(request.form['time']),
            ]
    prediction = model.predict([features])
    result = "The patient is at risk" if prediction[0] == 1 else "The patient is NOT at risk"
    return render_template('index.html', prediction_text = result)

if __name__ == '__main__':
    app.run(debug = True)
