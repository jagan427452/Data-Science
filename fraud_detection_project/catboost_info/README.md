
Fraud Detection Project

Overview
This project uses machine learning models to detect fraudulent transactions in credit card data.

Directory Structure
creditcard.csv - Input dataset
output - Stores images and the trained model
original_class_distribution.png - Class distribution before resampling
resampled_class_distribution.png - Class distribution after resampling
model_performance.png - Comparison of model performances
fraud_detection_model.pkl - Saved best model

Steps to Run
Mount Google Drive using drive.mount('/content/drive')
Ensure the dataset creditcard.csv is in the fraud_detection_project folder
Run the provided Python script
Check the output folder for generated images and model

Models Used
Logistic Regression
Random Forest
XGBoost - Best Model after Tuning
LightGBM
CatBoost

Results and Performance
The XGBoost model gave the best results after hyperparameter tuning.

Notes
All images and model files are automatically saved in the output folder
You can use joblib.load('fraud_detection_model.pkl') to load the saved model
