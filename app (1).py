
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Model Inference Service Running"

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    hours = data['hours']

    prediction = model.predict([[hours]])

    return jsonify({
        "predicted_marks": float(prediction[0])
    })

if __name__ == '__main__':
    app.run()
