import pandas as pd
import numpy as np

def load_data():
    try:
        red_wine = pd.read_csv("./data/winequality-red.csv", sep=";")
        white_wine = pd.read_csv("./data/winequality-white.csv", sep=";")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # combine
    df = pd.concat([red_wine, white_wine], ignore_index=True)

    # clean
    df = df.dropna()
    df = df.drop_duplicates()

    # separate features and target
    X = df.drop(columns=['quality']).values
    y = df['quality'].values

    # standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_scaled = (X - mean) / std

    return X_scaled, y