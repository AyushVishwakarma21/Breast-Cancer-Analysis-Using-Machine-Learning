from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the scaler and model
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data in the order of the model's features
        features = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
            'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
            'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]
        try:
            data = [float(request.form[feature]) for feature in features]
        except Exception as e:
            return render_template('index.html', prediction_text='Invalid input: {}'.format(e))

        arr = np.array(data).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        prediction = model.predict(arr_scaled)[0]
        result = "Malignant" if prediction == 0 else "Benign"
        return render_template('index.html', prediction_text=f'The tumor is likely: {result}')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)