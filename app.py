from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('fraud_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features'])

    if features.shape[0] != 14:
        return jsonify({'error': f'Expected 14 features, got {features.shape[0]}'})

    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({'isFraud': int(prediction)})


if __name__ == '__main__':
    app.run(debug=True)