from flask import Flask, request, render_template, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model/pcos_svm_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # This serves your index.html file

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture raw form data for display in report
        blood_groups = {
            "11": "A+", "12": "A-", "13": "B+", "14": "B-", 
            "15": "AB+", "16": "O-", "17": "O+"
        }
        
        user_input_data = {
            "Age": request.form['Age'],
            "Weight": request.form['Weight'],
            "Height": request.form['Height'],
            "BMI": request.form['BMI'],
            "Blood Group": blood_groups.get(request.form['Blood_Group'], "Unknown"),
            "Cycle Type": "Regular" if request.form['Cycle'].upper() == 'R' else "Irregular",
            "Cycle Length": request.form['Cycle_length'],
            "Marriage Status (Years)": request.form['Marriage_Status'],
            "Pregnant": "Yes" if request.form['Pregnant'] == '1' else "No",
            "No. of Abortions": request.form['No_of_abortions'],
            "Weight Gain": "Yes" if request.form['Weight_gain'] == '1' else "No",
            "Hair Growth": "Yes" if request.form['Hair_growth'] == '1' else "No",
            "Skin Darkening": "Yes" if request.form['Skin_darkening'] == '1' else "No",
            "Hair Loss": "Yes" if request.form['Hair_loss'] == '1' else "No",
            "Pimples": "Yes" if request.form['Pimples'] == '1' else "No",
            "Fast Food": "Yes" if request.form['Fast_food'] == '1' else "No",
            "Regular Exercise": "Yes" if request.form['Reg_Exercise'] == '1' else "No"
        }

        cycle = request.form['Cycle']
        cycle_numeric = 0 if cycle.upper() == 'R' else 1

        data = [
            int(request.form['Age']),
            float(request.form['Weight']),
            float(request.form['Height']),
            float(request.form['BMI']),
            int(request.form['Blood_Group']),
            cycle_numeric,
            int(request.form['Cycle_length']),
            int(request.form['Marriage_Status']),
            int(request.form['Pregnant']),
            int(request.form['No_of_abortions']),
            int(request.form['Weight_gain']),
            int(request.form['Hair_growth']),
            int(request.form['Skin_darkening']),
            int(request.form['Hair_loss']),
            int(request.form['Pimples']),
            int(request.form['Fast_food']),
            int(request.form['Reg_Exercise'])
        ]

        input_array = np.array(data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)

        result = {
            "prediction": "PCOS" if prediction == 1 else "No PCOS",
            "probability_PCOS": f"{prediction_proba[0][1] * 100:.2f}%",
            "probability_No_PCOS": f"{prediction_proba[0][0] * 100:.2f}%",
            "date": datetime.now().strftime("%B %d, %Y")
        }

        return render_template('result.html', result=result, user_data=user_input_data)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict_from_server', methods=['GET'])
def predict_from_server():
    try:
        # Path to the dataset on the server
        dataset_path = os.path.join('data', 'PCOD-10.csv')  # Adjust the path as needed

        # Load the dataset
        if not os.path.exists(dataset_path):
            return jsonify({"error": f"Dataset not found at {dataset_path}"})

        dataset = pd.read_csv(dataset_path)

        # Ensure the dataset has all required columns
        required_columns = ['Age', 'Weight', 'Height', 'BMI', 'Blood_Group', 'Cycle', 'Cycle_length',
                            'Marriage_Status', 'Pregnant', 'No_of_abortions', 'Weight_gain',
                            'Hair_growth', 'Skin_darkening', 'Hair_loss', 'Pimples',
                            'Fast_food', 'Reg_Exercise']
        if not all(col in dataset.columns for col in required_columns):
            return jsonify({"error": "Dataset is missing one or more required columns"})

        # Preprocess the dataset
        dataset['Cycle'] = dataset['Cycle'].apply(lambda x: 0 if x.upper() == 'R' else 1)
        input_data = dataset[required_columns]
        input_scaled = scaler.transform(input_data)

        # Make predictions
        predictions = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)

        # Add predictions to the dataset
        dataset['prediction'] = ['PCOS' if pred == 1 else 'No PCOS' for pred in predictions]
        dataset['probability_PCOS'] = [f"{prob[1] * 100:.2f}%" for prob in probabilities]
        dataset['probability_No_PCOS'] = [f"{prob[0] * 100:.2f}%" for prob in probabilities]

        # Save the results to a new CSV file
        output_file = os.path.join('data', 'prediction_results.csv')
        dataset.to_csv(output_file, index=False)

        return jsonify({"message": "Predictions completed", "result_file": output_file})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='127.0.0.1')
