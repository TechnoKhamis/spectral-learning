import numpy as np
from utils.matrix_operations import center_matrix, reconstruct


class SVD:
    """
    Singular Value Decomposition for dimensionality reduction, implemented from scratch.

    Decomposes centered matrix X into U, Σ, Vᵀ and projects data onto the
    top-k right singular vectors.

    Parameters
    ----------
    n_components : int
        Number of singular vectors to retain.
    variance_threshold : float or None
        If set, auto-selects n_components to explain this fraction of variance.
    """

    def __init__(self, n_components=2, variance_threshold=None):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.U = None
        self.S = None    # all singular values
        self.Vt = None   # all right singular vectors
        self.Vt_k = None # top-k right singular vectors
        self.explained_variance_ratio = None
        self.cumulative_variance = None
        self.mean = None

    def fit(self, X):
        """
        Fit SVD on X.
        Steps: center → numpy.linalg.svd → compute variance ratios → select top k.
        """
        # 1. center
        X_centered, self.mean = center_matrix(X)

        # 2. full SVD decomposition  X = U Σ Vᵀ
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.U = U
        self.S = S
        self.Vt = Vt

        # 3. variance explained — singular values relate to eigenvalues as λ = σ²/(n-1)
        explained_variance = (S ** 2) / (X.shape[0] - 1)
        total = explained_variance.sum()
        self.explained_variance_ratio = explained_variance / total
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)

        # 4. auto-select k based on threshold if provided
        if self.variance_threshold is not None:
            self.n_components = int(
                np.argmax(self.cumulative_variance >= self.variance_threshold) + 1
            )

        # 5. store top-k right singular vectors
        self.Vt_k = Vt[:self.n_components, :]

        return self

    def transform(self, X):
        """Project X onto top-k right singular vectors."""
        X_centered, _ = center_matrix(X)
        return X_centered @ self.Vt_k.T

    def reconstruct(self, k=None):
        """
        Reconstruct the original matrix using top-k components.
        Useful for measuring information retained.
        Parameters: k — number of components (defaults to self.n_components)
        """
        k = k or self.n_components
        return self.U[:, :k] @ np.diag(self.S[:k]) @ self.Vt[:k, :]