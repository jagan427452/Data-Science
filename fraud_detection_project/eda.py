# Import necessary libraries
import os
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define paths
data_folder = "fraud_detection_project"
output_folder = os.path.join(data_folder, "output")
file_path = os.path.join(data_folder, "creditcard.csv")

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load Dataset
df = pd.read_csv(file_path)

# Handle Missing Values
if df.isnull().sum().sum() > 0:
    df.fillna(df.median(), inplace=True)

# Visualize Original Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df['Class'], palette='coolwarm')
plt.title("Original Class Distribution")

# Save and display the image
original_dist_path = os.path.join(output_folder, "original_class_distribution.png")
plt.savefig(original_dist_path)
plt.show()

print(f"Saved Image: {original_dist_path}")

# Save cleaned dataset for further use
df.to_csv(os.path.join(output_folder, "cleaned_creditcard.csv"), index=False)

print("EDA completed and cleaned dataset saved.")
