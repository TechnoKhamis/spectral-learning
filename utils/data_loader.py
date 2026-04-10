import pandas as pd
import numpy as np


def load_data(red_path="./data/winequality-red.csv",
              white_path="./data/winequality-white.csv"):
    """
    Load and preprocess the Wine Quality dataset.

    Combines red and white wine CSVs, drops nulls/duplicates,
    standardizes features to mean=0, std=1.

    Returns
    -------
    X_scaled : ndarray (n_samples, 11)
    y        : ndarray (n_samples,) — quality scores 3-8
    feature_names : list of str
    """
    try:
        red = pd.read_csv(red_path, sep=";")
        white = pd.read_csv(white_path, sep=";")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset not found: {e}. "
                                "Place winequality-red.csv and winequality-white.csv in ./data/")

    # combine red + white
    df = pd.concat([red, white], ignore_index=True)

    # clean
    df = df.dropna()
    df = df.drop_duplicates()

    feature_names = [c for c in df.columns if c != "quality"]
    X = df[feature_names].values
    y = df["quality"].values

    # standardize: mean=0, std=1
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1   # guard against zero-variance features
    X_scaled = (X - mean) / std

    return X_scaled, y, feature_names