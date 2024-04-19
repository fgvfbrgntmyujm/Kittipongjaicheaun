from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)

# Function to classify iris flower type
def classify_iris(data):
    arr = np.array([data])
    pred = model.predict(arr)
    return 'Iris-setosa' if pred == 0 else 'Iris-versicolor'

# Sample data for demonstration
sample_data = [
    {'Sepal length': 5.1, 'Sepal width': 3.5, 'Petal length': 1.4, 'Petal width': 0.2, 'Predicted Class': 'Iris-setosa'},
    {'Sepal length': 4.9, 'Sepal width': 3.0, 'Petal length': 1.4, 'Petal width': 0.2, 'Predicted Class': 'Iris-setosa'},
    {'Sepal length': 4.7, 'Sepal width': 3.2, 'Petal length': 1.3, 'Petal width': 0.2, 'Predicted Class': 'Iris-setosa'}
]

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    pred = classify_iris([data1, data2, data3, data4])
    return render_template('after.html', data=pred, sepal_length=data1, sepal_width=data2, petal_length=data3, petal_width=data4)

@app.route('/All_result')
def all_results():
    # Prepare data to display in table or JSON format
    results = []
    for data in sample_data:
        result_row = {
            'Sepal length': data['Sepal length'],
            'Sepal width': data['Sepal width'],
            'Petal length': data['Petal length'],
            'Petal width': data['Petal width'],
            'Predicted Class': data['Predicted Class']
        }
        results.append(result_row)
    
    # Check if the request is coming from a Web App or Web Service
    if request.headers.get('Content-Type') == 'application/json':
        return jsonify(results)
    else:
        return render_template('all_results.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
