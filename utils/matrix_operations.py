import numpy as np

def center_matrix(X):
    """Subtract mean from each feature column"""
    mean = X.mean(axis=0)
    return X - mean, mean

def normalize(X):
    """Standardize X to mean=0 and std=1"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std

def covariance_matrix(X):
    """Calculate covariance matrix of X"""
    return np.cov(X, rowvar=False)

def reconstruct(U, S, Vt):
    """Reconstruct matrix from SVD components"""
    return U @ np.diag(S) @ Vt