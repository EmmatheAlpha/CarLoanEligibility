import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


from flask import Flask, request, render_template, request

app = Flask(__name__)

# Load the model and scaler (ensure these files exist in the specified paths)
model = pickle.load(open('models/model.pkl', 'rb'))
standard_scaler =pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])  # Ensure the method is POST
def predict_datapoint():
    if request.method == "POST":
        # Get form data
        age = float(request.form.get('age'))
        gender = float(request.form.get('gender'))
        income = float(request.form.get('income'))
        credit_score = float(request.form.get('credit_score'))
        credit_history_length = float(request.form.get('credit_history'))
        no_of_existing_loans = float(request.form.get('existing_loans'))
        loan_amount = float(request.form.get('loan_amount'))
        loan_tenure = float(request.form.get('loan_tenure'))
        ltv_ratio = float(request.form.get('ltv_ratio'))
        employment_profile = float(request.form.get('employment_profile'))



        # Scale the input data
        new_data_scaled = standard_scaler.transform([[
            income, credit_score, credit_history_length,
            loan_amount, loan_tenure, ltv_ratio
        ]])

        full_features = [age,  gender, new_data_scaled[0][0],  new_data_scaled[0][1],  new_data_scaled[0][2], no_of_existing_loans, new_data_scaled[0][3],  new_data_scaled[0][4], new_data_scaled[0][5], employment_profile]

        # Make prediction using the full feature set
        result = model.predict([full_features])

        # Pass the result to the template
        return render_template('home.html', results=result)
    
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host = "0.0.0.0")
    