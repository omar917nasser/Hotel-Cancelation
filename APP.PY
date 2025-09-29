import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model_pipeline = pickle.load(open(r"E:\Coding\Celula Intern\Task 3\model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

X = 0  

@app.route('/predict', methods=['POST'])
def predict():
    global X
    X = X + 1

    form_data = request.form.to_dict()

    feature_order = [
        'number of adults', 'number of children', 'number of weekend nights',
        'number of week nights', 'type of meal', 'car parking space',
        'room type', 'lead time', 'market segment type', 'repeated',
        'P-C', 'P-not-C', 'average price ', 'special requests', 'date of reservation'
    ]
    
    data = {key: [form_data.get(key)] for key in feature_order}
    df = pd.DataFrame(data)

    numeric_cols = [
        'number of adults', 'number of children', 'number of weekend nights',
        'number of week nights', 'car parking space', 'lead time',
        'repeated', 'P-C', 'P-not-C', 'average price ', 'special requests'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    prediction = model_pipeline.predict(df)

    output = "Canceled 😞" if prediction[0] == 0 else "Not Canceled 😊"

    return render_template(
        'index.html',
        prediction_text=f'The booking will likely be: {output}, {X}'
    )

if __name__ == "__main__":
    app.run(debug=True)
