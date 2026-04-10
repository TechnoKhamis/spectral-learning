import numpy as np
from utils.matrix_operations import center_matrix, covariance_matrix


class PCA:
    """
    Principal Component Analysis implemented from scratch.

    Uses eigenvalue decomposition of the covariance matrix to find
    directions of maximum variance in the data.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain.
    variance_threshold : float or None
        If set, auto-selects n_components to explain this fraction of variance.
        Overrides n_components when provided.
    """

    def __init__(self, n_components=2, variance_threshold=None):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.components = None           # top-k eigenvectors (k, n_features)
        self.explained_variance = None   # eigenvalues sorted descending
        self.explained_variance_ratio = None
        self.cumulative_variance = None
        self.mean = None
        self.feature_names = None

    def fit(self, X, feature_names=None):
        """
        Fit PCA on X.
        Steps: center → covariance matrix → eigendecomposition → sort → select k.
        """
        self.feature_names = feature_names

        # 1. center data
        X_centered, self.mean = center_matrix(X)

        # 2. covariance matrix (n_features x n_features)
        cov = covariance_matrix(X_centered)

        # 3. eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        # 4. sort descending by eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. variance ratios
        total = eigenvalues.sum()
        self.explained_variance = eigenvalues
        self.explained_variance_ratio = eigenvalues / total
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)

        # 6. auto-select k based on variance threshold if provided
        if self.variance_threshold is not None:
            self.n_components = int(
                np.argmax(self.cumulative_variance >= self.variance_threshold) + 1
            )

        # 7. store top-k components as rows
        self.components = eigenvectors[:, :self.n_components].T

        return self

    def transform(self, X):
        """Project X onto principal components."""
        X_centered, _ = center_matrix(X)
        return X_centered @ self.components.T

    def get_feature_loadings(self):
        """
        Return the feature loadings for each principal component.
        Shows which original features contribute most to each PC.
        Returns: dict mapping PC index to sorted (feature, loading) pairs
        """
        if self.components is None:
            raise RuntimeError("Call fit() before get_feature_loadings()")
        names = self.feature_names if self.feature_names is not None else \
            [f"feature_{i}" for i in range(self.components.shape[1])]
        loadings = {}
        for i, comp in enumerate(self.components):
            sorted_idx = np.argsort(np.abs(comp))[::-1]
            loadings[f"PC{i+1}"] = [(names[j], round(float(comp[j]), 4)) for j in sorted_idx]
        return loadings