import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_clustering(X_reduced, n_clusters=3, label=""):
    """
    Apply K-Means clustering on reduced-dimensional data and evaluate.

    Parameters
    ----------
    X_reduced  : ndarray — low-dimensional projection
    n_clusters : int — number of clusters
    label      : str — display name for print output

    Returns
    -------
    labels : ndarray — cluster assignments
    score  : float   — silhouette score (-1 to 1, higher = better separated)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_reduced)
    score = silhouette_score(X_reduced, labels)
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Silhouette Score ({n_clusters} clusters): {score:.4f}")
    return labels, score


def find_optimal_k(X_reduced, k_range=range(2, 8)):
    """
    Sweep over k values and return silhouette scores.
    Useful for justifying choice of n_clusters.

    Parameters
    ----------
    X_reduced : ndarray
    k_range   : iterable of int

    Returns
    -------
    scores : dict {k: silhouette_score}
    """
    scores = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_reduced)
        scores[k] = silhouette_score(X_reduced, labels)
    best_k = max(scores, key=scores.get)
    print(f"  Best k={best_k} (silhouette={scores[best_k]:.4f})")
    return scores