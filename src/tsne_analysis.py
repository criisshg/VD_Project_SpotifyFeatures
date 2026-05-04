"""
tsne_analysis.py

Visualització t-SNE del dataset Spotify Features.
Usa una mostra de 10.000 files per velocitat (~2-4 min).

Executar des de l'arrel del projecte:
    python src/tsne_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from src.data_massage import prepare_dataset

FIGURES_DIR = os.path.join("outputs", "tsne", "figures")
CSV_DIR     = os.path.join("outputs", "tsne")
SAMPLE_N = 10_000
RANDOM_STATE = 42
PERPLEXITY = 40
MAX_ITER = 1000


def get_sample(df_clean, X_scaled):
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(df_clean), size=SAMPLE_N, replace=False)
    df_sample = df_clean.iloc[idx].reset_index(drop=True)
    X_sample = X_scaled[idx]
    print(f"Mostra: {SAMPLE_N} files (random_state={RANDOM_STATE})")
    return df_sample, X_sample


def run_tsne(X_sample):
    print(f"Executant t-SNE (perplexity={PERPLEXITY}, max_iter={MAX_ITER})... [~2-4 min]")
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    coords = tsne.fit_transform(X_sample)
    print(f"t-SNE fet. Shape: {coords.shape}")
    return coords


def plot_tsne_genre(coords, df_sample):
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=df_sample["genre"],
        palette="tab20",
        alpha=0.5,
        s=10,
        edgecolor=None,
        ax=ax,
    )
    ax.set_title("t-SNE — Colorat per Gènere", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dimensió 1")
    ax.set_ylabel("Dimensió 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
              title="Gènere", fontsize=7, title_fontsize=8, markerscale=2)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "tsne_genre.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_tsne_popularity(coords, df_sample):
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=df_sample["popularity"],
        cmap="viridis",
        alpha=0.5,
        s=10,
    )
    plt.colorbar(sc, ax=ax, label="Popularity")
    ax.set_title("t-SNE — Colorat per Popularitat", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dimensió 1")
    ax.set_ylabel("Dimensió 2")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "tsne_popularity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def save_tsne_coords(df_sample, coords):
    import pandas as pd
    out = df_sample[["track_id", "genre", "popularity"]].copy()
    out["tsne_x"] = coords[:, 0]
    out["tsne_y"] = coords[:, 1]
    path = os.path.join(CSV_DIR, "tsne_coords.csv")
    out.to_csv(path, index=False)
    print(f"Saved {path} — {len(out)} files")


if __name__ == "__main__":
    print("[tsne_analysis] Carregant dataset...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    df_clean, X_scaled, scaler = prepare_dataset(verbose=False)

    df_sample, X_sample = get_sample(df_clean, X_scaled)
    coords = run_tsne(X_sample)

    plot_tsne_genre(coords, df_sample)
    plot_tsne_popularity(coords, df_sample)
    save_tsne_coords(df_sample, coords)

    print("[tsne_analysis] Fet. Outputs a outputs/")
