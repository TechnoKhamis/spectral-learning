import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_variance(evr_pca, evr_svd):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PCA
    axes[0].plot(np.cumsum(evr_pca), marker='o')
    axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[0].set_title('PCA Cumulative Variance')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Cumulative Variance Explained')
    axes[0].legend()

    # SVD
    axes[1].plot(np.cumsum(evr_svd), marker='o')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[1].set_title('SVD Cumulative Variance')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance Explained')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('variance_plot.png')
    plt.show()

def plot_2d(X_pca, X_svd, labels, title='2D Projection'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PCA 2D
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
    axes[0].set_title('PCA 2D Projection')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    plt.colorbar(scatter1, ax=axes[0])

    # SVD 2D
    scatter2 = axes[1].scatter(X_svd[:, 0], X_svd[:, 1], c=labels, cmap='tab10', alpha=0.6)
    axes[1].set_title('SVD 2D Projection')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    plt.colorbar(scatter2, ax=axes[1])

    plt.tight_layout()
    plt.savefig('2d_plot.png')
    plt.show()

def plot_3d(X_pca, X_svd, labels):
    fig = plt.figure(figsize=(14, 6))

    # PCA 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='tab10', alpha=0.6)
    ax1.set_title('PCA 3D Projection')

    # SVD 3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X_svd[:, 0], X_svd[:, 1], X_svd[:, 2], c=labels, cmap='tab10', alpha=0.6)
    ax2.set_title('SVD 3D Projection')

    plt.tight_layout()
    plt.savefig('3d_plot.png')
    plt.show()

def plot_clusters(X_reduced, labels, title='Clustering'):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=labels, palette='tab10', alpha=0.6)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('clusters_plot.png')
    plt.show()