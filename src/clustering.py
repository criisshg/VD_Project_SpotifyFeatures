"""
clustering.py

Clustering K-Means del dataset Spotify Features.
Troba el nombre òptim de clusters (k) per elbow + silhouette,
entrena el model final i exporta etiquetes i perfils.

Nota: executar DESPRÉS de tsne_analysis.py per tenir tsne_coords.csv
i poder generar la figura de clusters sobre t-SNE.

Executar des de l'arrel del projecte:
    python src/clustering.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data_massage import prepare_dataset, FEATURE_COLUMNS

FIGURES_DIR = os.path.join("outputs", "clustering", "figures")
CSV_DIR     = os.path.join("outputs", "clustering")
K_RANGE = range(2, 13)
RANDOM_STATE = 42
N_INIT = 10
TSNE_CSV = os.path.join("outputs", "tsne", "tsne_coords.csv")


def compute_kmeans_scores(X_scaled):
    inertias = []
    sil_scores = []
    for k in K_RANGE:
        print(f"  k={k}...", end=" ", flush=True)
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        score = silhouette_score(X_scaled, labels, sample_size=10_000,
                                 random_state=RANDOM_STATE)
        sil_scores.append(score)
        print(f"inèrcia={km.inertia_:.0f}  silhouette={score:.4f}")
    return inertias, sil_scores


def pick_best_k(sil_scores):
    best_k = list(K_RANGE)[int(np.argmax(sil_scores))]
    print(f"Millor k per silhouette: {best_k}")
    return best_k


def plot_elbow(inertias):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(K_RANGE), inertias, marker="o", color="#1DB954", linewidth=2)
    ax.set_xlabel("Nombre de Clusters (k)")
    ax.set_ylabel("Inèrcia (Within-cluster SSE)")
    ax.set_title("Elbow Method — K-Means")
    ax.set_xticks(list(K_RANGE))
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "clustering_elbow.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_silhouette(sil_scores, best_k):
    ks = list(K_RANGE)
    colors = ["#1DB954" if k == best_k else "#535353" for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ks, sil_scores, color=colors)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"Silhouette Score per k  (millor: k={best_k})")
    ax.set_xticks(ks)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "clustering_silhouette.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def run_final_kmeans(X_scaled, best_k):
    print(f"Entrenant KMeans final amb k={best_k}...")
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km.fit_predict(X_scaled)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Mides dels clusters: { {int(k): int(c) for k, c in zip(unique, counts)} }")
    return labels


def plot_tsne_clusters(df_clean_with_labels):
    if not os.path.exists(TSNE_CSV):
        print(f"[WARN] {TSNE_CSV} no trobat — executa primer tsne_analysis.py. Saltant figura.")
        return
    tsne_df = pd.read_csv(TSNE_CSV)
    merged = tsne_df.merge(
        df_clean_with_labels[["track_id", "cluster_id"]],
        on="track_id",
        how="left",
    ).dropna(subset=["cluster_id"])
    merged["cluster_id"] = merged["cluster_id"].astype(int)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(
        x="tsne_x",
        y="tsne_y",
        hue="cluster_id",
        data=merged,
        palette="tab10",
        alpha=0.5,
        s=10,
        edgecolor=None,
        ax=ax,
    )
    ax.set_title("t-SNE — Colorat per Cluster K-Means", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dimensió 1")
    ax.set_ylabel("Dimensió 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
              title="Cluster", markerscale=2)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "clustering_tsne.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def save_cluster_labels(df_clean, labels):
    out = df_clean[["track_id", "genre"]].copy()
    out["cluster_id"] = labels
    path = os.path.join(CSV_DIR, "cluster_labels.csv")
    out.to_csv(path, index=False)
    print(f"Saved {path} — {len(out)} files")
    return out


def save_cluster_profiles(df_clean, labels):
    df = df_clean[FEATURE_COLUMNS].copy()
    df["cluster_id"] = labels
    profiles = df.groupby("cluster_id")[FEATURE_COLUMNS].mean().round(4)
    path = os.path.join(CSV_DIR, "cluster_profiles.csv")
    profiles.to_csv(path)
    print(f"Saved {path} — {len(profiles)} clusters")


if __name__ == "__main__":
    print("[clustering] Carregant dataset...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    df_clean, X_scaled, scaler = prepare_dataset(verbose=False)

    print("Calculant scores K-Means per k=2..12 [~10-20 min]...")
    inertias, sil_scores = compute_kmeans_scores(X_scaled)
    best_k = pick_best_k(sil_scores)

    plot_elbow(inertias)
    plot_silhouette(sil_scores, best_k)

    labels = run_final_kmeans(X_scaled, best_k)
    df_clean["cluster_id"] = labels

    plot_tsne_clusters(df_clean)
    save_cluster_labels(df_clean, labels)
    save_cluster_profiles(df_clean, labels)

    print("[clustering] Fet. Outputs a outputs/")
