# Import necessary libraries
import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Mount Google Drive
drive.mount('/content/drive')

# Define paths for dataset and output storage in Google Drive
data_folder = "/content/drive/MyDrive/Colab Notebooks/fraud_detection_project"
output_folder = os.path.join(data_folder, "output")
file_path = os.path.join(output_folder, "cleaned_creditcard.csv")

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load Preprocessed Dataset
df = pd.read_csv(file_path)

# Prepare Features & Target Variable
X = df.drop(columns=['Class'])
y = df['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balance the Dataset using SMOTE-Tomek
smote_tomek = SMOTETomek(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

# Visualize Class Distribution After Resampling
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_resampled, palette='coolwarm')
plt.title("Class Distribution After Resampling")

# Save and display the image
resampled_dist_path = os.path.join(output_folder, "resampled_class_distribution.png")
plt.savefig(resampled_dist_path)
plt.show()

print(f"Saved Image: {resampled_dist_path}")

# Standardize Features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Define Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=100, boosting_type='gbdt', random_state=42, n_jobs=-1),
    "CatBoost": CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0)
}

results = {}

# Train and Evaluate Models
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_pred)
    }

# Display Model Performance
results_df = pd.DataFrame(results).T
print("\n Model Performance:")
print(results_df)

# Visualize Model Performance
plt.figure(figsize=(12, 6))
results_df.plot(kind='bar', figsize=(12, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(loc='lower right')

# Save and display the image
performance_path = os.path.join(output_folder, "model_performance.png")
plt.savefig(performance_path)
plt.show()

print(f"Saved Image: {performance_path}")

# Hyperparameter Tuning for XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid, cv=3, scoring='f1', n_jobs=-1
)

grid_search.fit(X_train_resampled, y_train_resampled)

# Get Best Model & Evaluate
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\n Best Model: XGBoost with Hyperparameter Tuning")
print(f"Best Params: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_best):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_best):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_best):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_best):.4f}")

# Save the Best Model
model_path = os.path.join(output_folder, "fraud_detection_model.pkl")
joblib.dump(best_model, model_path)

print(f"\n Model saved successfully at: {model_path}")
