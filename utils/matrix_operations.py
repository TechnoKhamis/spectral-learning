import numpy as np

def center_matrix(X):
    """
    Subtract the column-wise mean from X.
    Parameters: X (ndarray) — input matrix (n_samples, n_features)
    Returns: X_centered (ndarray), mean (ndarray)
    """
    mean = X.mean(axis=0)
    return X - mean, mean

def normalize(X):
    """
    Standardize X to mean=0, std=1.
    Parameters: X (ndarray)
    Returns: X_normalized (ndarray), mean (ndarray), std (ndarray)
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1  # avoid division by zero
    return (X - mean) / std, mean, std

def covariance_matrix(X):
    """
    Compute the covariance matrix of a centered matrix X.
    Parameters: X (ndarray) — centered matrix (n_samples, n_features)
    Returns: cov (ndarray) — (n_features, n_features) covariance matrix
    """
    return np.cov(X, rowvar=False)

def reconstruct(U, S, Vt):
    """
    Reconstruct original matrix from full SVD components.
    Parameters: U, S (1D singular values), Vt
    Returns: reconstructed matrix (ndarray)
    """
    return U @ np.diag(S) @ Vt