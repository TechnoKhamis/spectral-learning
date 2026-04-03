import numpy as np
from utils.data_loader import load_data
from models.svd_model import SVD
from models.pca_model import PCA
from utils.clustering import run_clustering
from utils.visualization import plot_variance, plot_2d, plot_3d, plot_clusters

# 1. load data
X_scaled, y = load_data()
print(f"Data loaded: {X_scaled.shape}")

# 2. run SVD
svd = SVD(n_components=3)
svd.fit(X_scaled)
X_svd = svd.transform(X_scaled)
print(f"SVD reduced shape: {X_svd.shape}")

# 3. run PCA
pca = PCA(n_components=3)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print(f"PCA reduced shape: {X_pca.shape}")

# 4. clustering on SVD
print("\n--- SVD Clustering ---")
labels_svd, score_svd = run_clustering(X_svd, n_clusters=3)

# 5. clustering on PCA
print("\n--- PCA Clustering ---")
labels_pca, score_pca = run_clustering(X_pca, n_clusters=3)

# 6. visualizations
plot_variance(pca.explained_variance_ratio, svd.explained_variance_ratio)