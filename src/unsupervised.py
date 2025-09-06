
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from src.utils import load_heart_data, split_features_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease.csv"
SYNTH_PATH = PROJECT_ROOT / "data" / "sample_heart_disease.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
def elbow_method(X_scaled, ks=range(1, 11)):
    inertias = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    return inertias


def main(show_plots=True):
    df = load_heart_data(str(DATA_PATH), allow_synthetic=True, synthetic_path=str(SYNTH_PATH))
    X, y = split_features_target(df)

    X_scaled = StandardScaler().fit_transform(X.fillna(X.median(numeric_only=True)))

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    ks = list(range(1, 11))
    inertias = elbow_method(X_scaled, ks)
    print("K-Means Elbow Method Inertias:")
    for k, inertia in zip(ks, inertias):
        print(f"  k={k}, inertia={inertia}")

    if show_plots:
        plt.figure()
        plt.plot(ks, inertias, marker="o")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.title("K-Means Elbow Method")
        plt.show()

    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    labels_km = kmeans.fit_predict(X_scaled)

    Z = linkage(X_scaled, method="ward")
    if show_plots:
        plt.figure(figsize=(8, 5))
        dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90.)
        plt.title("Hierarchical Clustering Dendrogram (truncated)")
        plt.show()

    agg = AgglomerativeClustering(n_clusters=2)
    labels_agg = agg.fit_predict(X_scaled)

    if y is not None:
        kmeans_match = (labels_km == y.values).mean()
        agg_match = (labels_agg == y.values).mean()
        print(f"KMeans vs target match rate: {kmeans_match:.4f}")
        print(f"Agglomerative vs target match rate: {agg_match:.4f}")

        comp = {
            "kmeans_vs_target_match_rate": float(kmeans_match),
            "agg_vs_target_match_rate": float(agg_match)
        }
        with open(RESULTS_DIR / "clustering_comparison.json", "w") as f:
            json.dump(comp, f, indent=2)

    if show_plots:
        plt.figure()
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("K-Means Clusters (PCA 2D)")
        plt.show()

    print("Unsupervised analysis complete.")


if __name__ == "__main__":
    main()