
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List


import pandas as pd
import numpy as np

raw_path = r"D:\sprints\Heart_Disease_Project\data\processed.cleveland.data"
out_path = r"D:\sprints\Heart_Disease_Project\data\heart_disease.csv"

headers = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","num"
]

# قراءة الملف بفواصل عادية
df = pd.read_csv(raw_path, header=None, names=headers)

# استبدال missing values
df = df.replace(["?", -9], np.nan)

# حفظ CSV نظيف
df.to_csv(out_path, index=False)

print("Saved clean dataset at:", out_path)
print(df.head())















UCI_EXPECTED_COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","target"
]

def load_heart_data(data_path: str = "data/heart_disease.csv",
                    allow_synthetic: bool = True,
                    synthetic_path: str = "data/sample_heart_disease.csv") -> pd.DataFrame:
    # Load the UCI Heart Disease dataset from a CSV at a relative path.
    # If not found and allow_synthetic=True, it falls back to a synthetic CSV.
    data_file = Path(data_path)
    if data_file.exists():
        df = pd.read_csv(data_file)
    else:
        if allow_synthetic and Path(synthetic_path).exists():
            df = pd.read_csv(synthetic_path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{data_path}'. Please place the UCI Heart Disease CSV at this path."
            )

    # Normalize column names (lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize target column name if needed
    if "num" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"num":"target"})
    for alt in ["condition","disease","hd","heartdisease"]:
        if alt in df.columns and "target" not in df.columns:
            df = df.rename(columns={alt:"target"})

    # Convert target to binary if it's ordinal 0..4
    if "target" in df.columns:
        try:
            df["target"] = (df["target"].astype(float) > 0).astype(int)
        except Exception:
            mapping = {"present":1, "yes":1, "absent":0, "no":0}
            df["target"] = df["target"].astype(str).str.lower().map(mapping).fillna(df["target"]).astype(int)
    else:
        raise ValueError("No 'target' column found. Ensure your CSV includes 'target' (or 'num').")

    # Replace '?' with NaN where common
    for col in ["ca","thal"]:
        if col in df.columns:
            df[col] = df[col].replace("?", np.nan)

    # Cast categorical integers to numeric
    for col in ["sex","cp","fbs","restecg","exang","slope","ca","thal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure numeric types where expected
    for c in [c for c in df.columns if c != "target"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(how="all")
    return df

def split_features_target(df: pd.DataFrame, target_col: str = "target"):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y

def get_feature_groups(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Returns (numeric_features, categorical_features) lists based on heuristics.
    categorical_candidates = {"sex","cp","fbs","restecg","exang","slope","ca","thal"}
    numeric_features = []
    categorical_features = []

    for c in X.columns:
        if c in categorical_candidates:
            categorical_features.append(c)
        else:
            if pd.api.types.is_numeric_dtype(X[c]):
                numeric_features.append(c)
            else:
                categorical_features.append(c)

    return numeric_features, categorical_features
