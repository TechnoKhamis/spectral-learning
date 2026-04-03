import numpy as np
from utils.matrix_operations import center_matrix, covariance_matrix

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.mean = None

    def fit(self, X):
        # center the data
        X_centered, self.mean = center_matrix(X)

        # covariance matrix
        cov_matrix = covariance_matrix(X_centered)

        # eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store explained variance
        self.explained_variance = eigenvalues
        self.explained_variance_ratio = eigenvalues / eigenvalues.sum()

        # select top k components
        self.components = eigenvectors[:, :self.n_components].T

        return self

    def transform(self, X):
        # center then project
        X_centered, _ = center_matrix(X)
        return X_centered @ self.components.T