import numpy as np
from utils.matrix_operations import center_matrix, reconstruct

class SVD:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None
        self.Vt_k = None
        self.explained_variance_ratio = None

    def fit(self, X):
        # center the matrix
        X_centered, self.mean = center_matrix(X)

        # decompose
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.U = U
        self.S = S
        self.Vt = Vt

        # variance explained
        explained_variance = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio = explained_variance / explained_variance.sum()

        # select top k
        self.Vt_k = Vt[:self.n_components, :]

        return self

    def transform(self, X):
        # center then project
        X_centered, _ = center_matrix(X)
        return X_centered @ self.Vt_k.T

    def reconstruct(self):
        # rebuild matrix from components
        return reconstruct(self.U, self.S, self.Vt)