# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Step 2: Load the Dataset
# Assuming the CSV file is in the current directory
df = pd.read_csv('loan_datasets.csv')

# Display basic info
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Step 3: Exploratory Data Analysis (EDA)
# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Loan_Status', data=df)
plt.title('Distribution of Loan Status')
plt.show()

# Visualize key numerical features
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['ApplicantIncome'], kde=True)
plt.title('Applicant Income Distribution')
plt.subplot(1, 2, 2)
sns.histplot(df['LoanAmount'].dropna(), kde=True)
plt.title('Loan Amount Distribution')
plt.tight_layout()
plt.show()

# Correlation heatmap (numerical features only)
numerical_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 4: Data Preprocessing
# Encode categorical variables
label_encoders = {}
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Handle any NaNs by converting to str
        label_encoders[col] = le

# Note: Loan_Status 'Y' -> 1 (approved), 'N' -> 0 (not approved)

# Handle missing values
# Numerical columns: impute with median
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Total_Income']
imputer_num = SimpleImputer(strategy='median')
df[num_cols] = imputer_num.fit_transform(df[num_cols])

# Categorical columns: impute with mode (already encoded, but for completeness)
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Drop irrelevant columns
df = df.drop('Loan_ID', axis=1, errors='ignore')

# Feature scaling (optional for Logistic Regression, but included)
scaler = StandardScaler()
feature_cols = df.columns.drop('Loan_Status')
df[feature_cols] = scaler.fit_transform(df[feature_cols])

print("\nPreprocessed Dataset Shape:", df.shape)
print("\nMissing Values After Preprocessing:")
print(df.isnull().sum().sum())

# Step 5: Feature Selection and Splitting
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTrain Set Shape:", X_train.shape)
print("Test Set Shape:", X_test.shape)

# Step 6: Model Training
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Step 8: Predictions and Insights
# Sample predictions on test set
sample_preds = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nSample Predictions:")
print(sample_preds.head(10))

# Feature importance (coefficients from Logistic Regression)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print("\nFeature Importance (Top 10):")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='Coefficient', y='Feature')
plt.title('Top 10 Feature Coefficients')
plt.show()

# Step 9: Save Model (Optional)
import joblib
joblib.dump(model, 'loan_approval_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("\nModel, scaler, and encoders saved!")

# To load and predict on new data (example function)
def predict_loan(new_data_df):
    # Assume new_data_df has the same columns as X
    new_data_scaled = scaler.transform(new_data_df)
    pred = model.predict(new_data_scaled)
    pred_prob = model.predict_proba(new_data_scaled)[:, 1]
    return pred, pred_prob

# Example usage (create a dummy new row matching features)
# new_row = pd.DataFrame({...})  # Fill with values
# prediction, prob = predict_loan(new_row)
# print(f"Prediction: {'Approved' if prediction[0] == 1 else 'Not Approved'}, Probability: {prob[0]:.4f}")