import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score


def run_clustering(data):
    # Preprocess data
    X = data.drop("target", axis=1)  # Dropping the target for unsupervised learning
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clustering_algorithms = {
        "K-Means": KMeans(n_clusters=2),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=2),
        "DBSCAN": DBSCAN(eps=0.5),
    }

    results = []
    for name, algorithm in clustering_algorithms.items():
        labels = algorithm.fit_predict(X_scaled)

        if len(set(labels)) > 1:  # Check if the clustering was successful
            silhouette = silhouette_score(X_scaled, labels)
            davies_bouldin = davies_bouldin_score(X_scaled, labels)
        else:
            silhouette = -1  # Indicating poor clustering
            davies_bouldin = float("inf")

        results.append(
            {
                "Algorithm": name,
                "Silhouette Score": silhouette,
                "Davies-Bouldin Index": davies_bouldin,
            }
        )

    return results
