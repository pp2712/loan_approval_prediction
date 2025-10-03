# loan_approval_prediction
This project builds a Loan Approval Prediction System using Logistic Regression on a given dataset of loan applicants. It walks through data exploration, preprocessing, training, evaluation, and deployment of a machine learning model to predict whether a loan will be approved or not.

📌 Project Workflow
1. Import Libraries

Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

2. Load Dataset

Reads loan dataset (loan_datasets.csv)

Displays shape, info, summary statistics, and first few rows

3. Exploratory Data Analysis (EDA)

Summary statistics and missing values check

Distribution plots for key variables (Applicant Income, Loan Amount, Loan Status)

Correlation heatmap for numerical features

4. Data Preprocessing

Encodes categorical variables using LabelEncoder

Handles missing values (median for numeric, mode for categorical)

Drops irrelevant columns (Loan_ID)

Scales features using StandardScaler

5. Train-Test Split

Splits data into training (80%) and testing (20%) sets with stratified sampling

6. Model Training

Trains a Logistic Regression classifier

Hyperparameters: max_iter=1000, random_state=42

7. Model Evaluation

Computes Accuracy, Precision, Recall, F1-Score

Generates Classification Report

Plots Confusion Matrix

8. Predictions & Insights

Displays sample predictions on test set

Extracts feature importance (coefficients)

Plots top 10 most important features

9. Model Saving

Saves the trained model, scaler, and encoders using joblib

Includes a helper function predict_loan() to make predictions on new data

🗂 Dataset

The dataset used should be named loan_datasets.csv and placed in the project directory.

Expected columns include:

Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status

📊 Example Outputs
Distribution of Loan Status

Shows how many loans were approved vs. not approved.

Feature Importance

Identifies which features contribute most to loan approval predictions.

⚙️ How to Run

Clone the repository:

git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction


Install required libraries:

pip install -r requirements.txt


Place your dataset (loan_datasets.csv) in the project folder.

Run the script:

python loan_prediction.py

🔮 Using the Saved Model

You can load the saved model to predict new data:

import pandas as pd
import joblib

# Load saved objects
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Example new applicant data (replace with real values)
new_data = pd.DataFrame([{
    "Gender": 1,
    "Married": 0,
    "Dependents": 2,
    "Education": 0,
    "Self_Employed": 0,
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": 2,
    "Total_Income": 7000
}])

# Scale and predict
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)[:, 1]

print("Prediction:", "Approved" if prediction[0] == 1 else "Not Approved")
print("Probability:", probability[0])

📌 Requirements

Python 3.8+

pandas

numpy

matplotlib

seaborn

scikit-learn

joblib

🚀 Future Improvements

Try other ML algorithms (Random Forest, XGBoost, Neural Networks)

Hyperparameter tuning with GridSearchCV

Deploy model as a Flask/Django web app or Streamlit dashboard

📜 License

This project is licensed under the MIT License.
