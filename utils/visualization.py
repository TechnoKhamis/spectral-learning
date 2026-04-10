import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

# consistent style
sns.set_theme(style="whitegrid", font_scale=1.0)
COLORS = sns.color_palette("tab10")


# ── 1. Variance explained ────────────────────────────────────────────────────

def plot_variance(evr_pca, evr_svd, save_path="variance_plot.png"):
    """
    Side-by-side cumulative variance plots for PCA and SVD.
    Draws a dashed line at the 95% threshold.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cumulative Variance Explained — PCA vs SVD", fontsize=14, fontweight="bold")

    for ax, evr, title in zip(axes, [evr_pca, evr_svd], ["PCA", "SVD"]):
        cumvar = np.cumsum(evr)
        components = np.arange(1, len(cumvar) + 1)
        ax.plot(components, cumvar, marker="o", linewidth=2, color=COLORS[0])
        ax.bar(components, evr, alpha=0.35, color=COLORS[1], label="Per-component variance")
        ax.axhline(0.95, color="red", linestyle="--", linewidth=1.2, label="95% threshold")
        k95 = int(np.argmax(cumvar >= 0.95) + 1)
        ax.axvline(k95, color="red", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.annotate(f"k={k95} → 95%", xy=(k95, 0.95), xytext=(k95 + 0.3, 0.88),
                    fontsize=9, color="red")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance Explained")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ── 2. 2D projections ────────────────────────────────────────────────────────

def plot_2d(X_pca, X_svd, labels, title="2D Projection", save_path="2d_plot.png"):
    """
    Side-by-side 2D scatter: PCA vs SVD, coloured by cluster or quality labels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, X, name in zip(axes, [X_pca, X_svd], ["PCA", "SVD"]):
        sc = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10",
                        alpha=0.55, s=15, linewidths=0)
        ax.set_title(f"{name} — 2D", fontsize=11)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.colorbar(sc, ax=ax, label="Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ── 3. 3D projections ────────────────────────────────────────────────────────

def plot_3d(X_pca, X_svd, labels, save_path="3d_plot.png"):
    """
    Side-by-side 3D scatter: PCA vs SVD.
    Requires n_components >= 3.
    """
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("3D Projection — PCA vs SVD", fontsize=14, fontweight="bold")

    for i, (X, name) in enumerate(zip([X_pca, X_svd], ["PCA", "SVD"])):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels,
                   cmap="tab10", alpha=0.45, s=10)
        ax.set_title(f"{name} — 3D", fontsize=11)
        ax.set_xlabel("C1"); ax.set_ylabel("C2"); ax.set_zlabel("C3")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ── 4. Cluster visualizations ────────────────────────────────────────────────

def plot_clusters(X_reduced, labels, title="K-Means Clusters", save_path="clusters_plot.png"):
    """
    2D cluster scatter with centroids marked.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    unique = np.unique(labels)
    for k in unique:
        mask = labels == k
        ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                   label=f"Cluster {k}", alpha=0.55, s=15)
        cx, cy = X_reduced[mask, 0].mean(), X_reduced[mask, 1].mean()
        ax.scatter(cx, cy, marker="X", s=200, color="black", zorder=5)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


def plot_cluster_subsets(X_reduced, labels, quality, save_path="cluster_subsets.png"):
    """
    Multi-panel plot showing clusters broken down by wine quality score.
    Demonstrates robustness by visualizing different quality subsets.
    """
    quality_vals = np.unique(quality)
    n = len(quality_vals)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
    fig.suptitle("Cluster Patterns by Wine Quality Score", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, qv in enumerate(quality_vals):
        mask = quality == qv
        ax = axes[i]
        ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                   c=labels[mask], cmap="tab10", alpha=0.6, s=20)
        ax.set_title(f"Quality = {qv}  (n={mask.sum()})", fontsize=10)
        ax.set_xlabel("C1"); ax.set_ylabel("C2")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ── 5. Feature loadings / component analysis ─────────────────────────────────

def plot_feature_loadings(loadings_dict, n_components=3, save_path="feature_loadings.png"):
    """
    Horizontal bar charts showing which original features drive each PC.
    Provides interpretability of latent components.

    Parameters
    ----------
    loadings_dict : dict from pca.get_feature_loadings()
    n_components  : int — how many PCs to show
    """
    fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 5))
    if n_components == 1:
        axes = [axes]
    fig.suptitle("Feature Loadings per Principal Component", fontsize=14, fontweight="bold")

    for ax, (pc_name, pairs) in zip(axes, list(loadings_dict.items())[:n_components]):
        features = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]
        ax.barh(features, values, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(pc_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Loading value")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ── 6. Silhouette k sweep ────────────────────────────────────────────────────

def plot_silhouette_sweep(scores_dict, save_path="silhouette_sweep.png"):
    """
    Bar chart of silhouette scores across k values to justify cluster count choice.

    Parameters
    ----------
    scores_dict : dict {k: score} from find_optimal_k()
    """
    ks = list(scores_dict.keys())
    scores = list(scores_dict.values())
    best_k = ks[int(np.argmax(scores))]

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = ["#1f77b4" if k != best_k else "#ff7f0e" for k in ks]
    ax.bar(ks, scores, color=bar_colors, edgecolor="white")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Optimal k Selection via Silhouette Score", fontsize=13, fontweight="bold")
    ax.annotate(f"Best k={best_k}", xy=(best_k, max(scores)),
                xytext=(best_k + 0.2, max(scores) + 0.005), fontsize=10, color="#ff7f0e")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")