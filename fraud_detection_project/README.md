# Fraud Detection Project

## Overview
This project detects fraudulent credit card transactions using machine learning models. The workflow is divided into two parts:
1. **Exploratory Data Analysis (EDA)** - Handles missing values, class imbalance, and visualizations.
2. **Model Training & Evaluation** - Trains models, evaluates performance, and selects the best one.

## Directory Structure
```
fraud_detection_project/
│── README.md                 # Project documentation
│── requirements.txt           # Dependencies list
│── creditcard.csv             # Input dataset
│── eda.py                     # Script for EDA and preprocessing
│── evaluation.py               # Model training and evaluation script
│── output/                     # Stores generated images and trained models
│   ├── original_class_distribution.png
│   ├── resampled_class_distribution.png
│   ├── model_performance.png
│   ├── cleaned_creditcard.csv  # Processed dataset after EDA
│   ├── fraud_detection_model.pkl
```

## Installation
Ensure you have **Python 3.8+** installed. Then, install all required dependencies by running:

```bash
pip install -r requirements.txt
```

## Running the Project
### 1. Run Exploratory Data Analysis (EDA)
First, clean the dataset and generate visualizations:
```bash
python eda.py
```
After execution, a cleaned dataset (`cleaned_creditcard.csv`) will be saved in the `output/` folder.

### 2. Train and Evaluate Models
Run the following command to train models and save the best one:
```bash
python evaluation.py
```

### 3. Check Output Files
After running, check the `output/` folder for:
- **EDA Images**: Class distribution before and after resampling.
- **Model Performance Chart**: A bar plot comparing model accuracy.
- **Saved Model**: `fraud_detection_model.pkl` for future predictions.

## Using the Trained Model
To make predictions using the trained model:
```python
import joblib

# Load the trained model
model = joblib.load("output/fraud_detection_model.pkl")

# Predict fraud on new data
sample_data = [[...]]  # Replace with actual data
prediction = model.predict(sample_data)
print("Fraud Detected" if prediction[0] == 1 else "No Fraud Detected")
```

## Handling Large Files in GitHub
If your dataset is too large, you can:
1. Use **Git Large File Storage (LFS)**
   ```bash
   git lfs install
   git lfs track "*.csv"
   git add creditcard.csv
   git commit -m "Added dataset using Git LFS"
   git push origin main
   ```
2. Store the file on **Google Drive** and access it programmatically.

## Security Best Practices
- Keep sensitive data in **private repositories**.
- Use **environment variables** instead of hardcoding credentials.
- Encrypt large files before uploading:
  ```bash
  zip --encrypt creditcard.zip creditcard.csv
  ```

## Notes
- The **XGBoost model** gave the best results after hyperparameter tuning.
- The dataset is processed and saved in the `output/` folder.
- The project is now modular, with separate **EDA** and **Evaluation** scripts.

## Authors
Developed for fraud detection using machine learning. Feel free to contribute or report issues.
