import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(df):

    # Drop ID column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Handle missing values
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Feature engineering
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 120],
        labels=["young", "adult", "middle_aged", "senior"]
    )

    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"]
    )

    df["risk_score"] = (
        df["hypertension"] +
        df["heart_disease"] +
        (df["avg_glucose_level"] > 140).astype(int) +
        (df["age"] > 55).astype(int)
    )

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )