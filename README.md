import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your dataset
data = pd.read_csv('telecom_churn_dataset.csv')  # Replace 'telecom_churn_dataset.csv' with your actual dataset file path

# Data Preparation
# Assuming you have already preprocessed the data and selected relevant features
X = data.drop(columns=['churn'])  # Features (exclude the target variable 'churn')
y = data['churn']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# Model Evaluation
y_pred = log_reg_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
