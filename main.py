"""
Spectral Learning — Main Pipeline
===================================
Applies PCA and SVD from scratch to the Wine Quality dataset.
Covers: preprocessing → dimensionality reduction → clustering → visualization.
"""

import sys
import numpy as np

from utils.data_loader import load_data
from models.pca_model import PCA
from models.svd_model import SVD
from utils.clustering import run_clustering, find_optimal_k
from utils.visualization import (
    plot_variance,
    plot_2d,
    plot_3d,
    plot_clusters,
    plot_cluster_subsets,
    plot_feature_loadings,
    plot_silhouette_sweep,
)

# ── 1. Load & preprocess ─────────────────────────────────────────────────────
print("=" * 55)
print("1. Loading data")
print("=" * 55)
X_scaled, y, feature_names = load_data()
print(f"   Dataset shape : {X_scaled.shape}")
print(f"   Quality range : {y.min()} – {y.max()}")
print(f"   Features      : {feature_names}")

# ── 2. PCA ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("2. PCA (variance_threshold=0.80 → auto k selection)")
print("=" * 55)
pca = PCA(variance_threshold=0.80)
pca.fit(X_scaled, feature_names=feature_names)
X_pca = pca.transform(X_scaled)

print(f"   Components selected  : {pca.n_components}")
print(f"   Variance per PC      : {np.round(pca.explained_variance_ratio[:pca.n_components], 4)}")
print(f"   Cumulative variance  : {np.round(pca.cumulative_variance[:pca.n_components], 4)}")
print(f"   Reduced shape        : {X_pca.shape}")

# ── 3. SVD ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("3. SVD (variance_threshold=0.80 → auto k selection)")
print("=" * 55)
svd = SVD(variance_threshold=0.80)
svd.fit(X_scaled)
X_svd = svd.transform(X_scaled)

print(f"   Components selected  : {svd.n_components}")
print(f"   Variance per SV      : {np.round(svd.explained_variance_ratio[:svd.n_components], 4)}")
print(f"   Cumulative variance  : {np.round(svd.cumulative_variance[:svd.n_components], 4)}")
print(f"   Reduced shape        : {X_svd.shape}")

# ── 4. Feature loading analysis (PCA interpretability) ───────────────────────
print("\n" + "=" * 55)
print("4. Feature loadings — which features drive each PC?")
print("=" * 55)
loadings = pca.get_feature_loadings()
for pc, pairs in list(loadings.items())[:3]:
    top3 = pairs[:3]
    print(f"   {pc}: " + ", ".join(f"{f} ({v:+.3f})" for f, v in top3))

# ── 5. Find optimal k for clustering ─────────────────────────────────────────
print("\n" + "=" * 55)
print("5. Finding optimal cluster count k (PCA space)")
print("=" * 55)
k_scores = find_optimal_k(X_pca[:, :3], k_range=range(2, 8))
best_k = max(k_scores, key=k_scores.get)

# ── 6. Clustering ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"6. K-Means clustering (k={best_k})")
print("=" * 55)
labels_pca, score_pca = run_clustering(X_pca[:, :3], n_clusters=best_k, label="PCA")
labels_svd, score_svd = run_clustering(X_svd[:, :3], n_clusters=best_k, label="SVD")
print(f"\n   PCA silhouette: {score_pca:.4f}")
print(f"   SVD silhouette: {score_svd:.4f}")

# ── 7. Visualizations ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("7. Generating visualizations")
print("=" * 55)

# use first 2/3 components for plotting regardless of k selected
X_pca2, X_svd2 = X_pca[:, :2], X_svd[:, :2]
X_pca3, X_svd3 = X_pca[:, :3], X_svd[:, :3]

print("  → Variance plot")
plot_variance(pca.explained_variance_ratio, svd.explained_variance_ratio)

print("  → 2D projections (quality labels)")
plot_2d(X_pca2, X_svd2, y, title="2D Projection — coloured by wine quality")

print("  → 2D projections (cluster labels)")
plot_2d(X_pca2, X_svd2, labels_pca, title="2D Projection — coloured by cluster")

print("  → 3D projections")
plot_3d(X_pca3, X_svd3, labels_pca)

print("  → Cluster scatter (PCA)")
plot_clusters(X_pca2, labels_pca, title=f"K-Means Clusters on PCA (k={best_k})")

print("  → Cluster subsets by wine quality")
plot_cluster_subsets(X_pca2, labels_pca, y)

print("  → Feature loadings")
plot_feature_loadings(loadings, n_components=min(3, pca.n_components))

print("  → Silhouette k sweep")
plot_silhouette_sweep(k_scores)

print("\n" + "=" * 55)
print("Done. All plots saved as PNG files.")
print("=" * 55)