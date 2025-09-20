import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# ------------------------------
# Configurations
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_PATH = "hf://datasets/sudha1726/tourism-package-prediction/tourism.csv"
REPO_ID = "sudha1726/tourism-package-prediction"
OUTPUT_DIR = "processed_data"

# Initialize Hugging Face API
api = HfApi(token=HF_TOKEN)

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from Hugging Face hub."""
    df = pd.read_csv(path)
    print("Dataset loaded successfully.")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, handle missing values, and preprocess dataset."""
    # Drop unnecessary columns
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())  # numeric → median
        else:
            df[col] = df[col].fillna(df[col].mode()[0])  # categorical → mode

    # Encode categorical variables
    categorical_cols = [
        'TypeofContact', 'Gender', 'CityTier',
        'Occupation', 'MaritalStatus',
        'Designation', 'ProductPitched'
    ]
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    print("Preprocessing complete (missing values handled).")
    return df

def split_and_save(df: pd.DataFrame, target_col: str):
    """Split dataset and save train/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    Xtrain.to_csv(f"{OUTPUT_DIR}/Xtrain.csv", index=False)
    Xtest.to_csv(f"{OUTPUT_DIR}/Xtest.csv", index=False)
    ytrain.to_csv(f"{OUTPUT_DIR}/ytrain.csv", index=False)
    ytest.to_csv(f"{OUTPUT_DIR}/ytest.csv", index=False)

    print("Train/test datasets saved locally.")
    return [
        f"{OUTPUT_DIR}/Xtrain.csv",
        f"{OUTPUT_DIR}/Xtest.csv",
        f"{OUTPUT_DIR}/ytrain.csv",
        f"{OUTPUT_DIR}/ytest.csv"
    ]

def upload_files(files: list, repo_id: str, repo_type: str = "dataset"):
    """Upload files to Hugging Face dataset repo."""
    for file_path in files:
        if os.path.exists(file_path):
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),
                repo_id=repo_id,
                repo_type=repo_type
            )
            print(f" {file_path} uploaded successfully.")
        else:
            print(f"File not found: {file_path}")

def main():
    df = load_dataset(DATASET_PATH)
    df = preprocess_data(df)
    files = split_and_save(df, target_col="ProdTaken")
    upload_files(files, REPO_ID)
    print(" Preprocessing and upload complete.")

if __name__ == "__main__":
    main()
