import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_clustering(X_reduced, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_reduced)
    score = silhouette_score(X_reduced, labels)
    
    print(f"Silhouette Score: {score:.4f}")
    
    return labels, score