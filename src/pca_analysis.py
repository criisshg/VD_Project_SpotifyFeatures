"""
pca_analysis.py

Anàlisi PCA del dataset Spotify Features.
Adaptat de la versió inicial del company (scatter gènere) afegint:
- import de prepare_dataset (evita re-escalar les dades)
- PCA amb 11 components per al scree plot
- Biplot amb loading vectors
- Export de coordenades a outputs/pca_coords.csv

Executar des de l'arrel del projecte:
    python src/pca_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from src.data_massage import prepare_dataset, FEATURE_COLUMNS

FIGURES_DIR = os.path.join("outputs", "pca", "figures")
CSV_DIR     = os.path.join("outputs", "pca")


def run_pca(X_scaled):
    pca = PCA(n_components=11, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_
    print(f"Varianza PC1: {var[0]*100:.2f}%")
    print(f"Varianza PC2: {var[1]*100:.2f}%")
    print(f"Varianza PC3: {var[2]*100:.2f}%")
    print(f"Varianza acumulada (3 comp): {sum(var[:3])*100:.2f}%")
    return pca, coords


def plot_scree(pca):
    var = pca.explained_variance_ratio_
    cumvar = np.cumsum(var)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, 12), var * 100, color="#1DB954", label="Variança per component")
    ax.plot(range(1, 12), cumvar * 100, marker="o", color="white",
            linewidth=2, label="Variança acumulada")
    ax.axhline(80, color="red", linestyle="--", linewidth=1, label="80% llindar")
    ax.set_xlabel("Component Principal")
    ax.set_ylabel("Variança explicada (%)")
    ax.set_title("Scree Plot — PCA Spotify Features")
    ax.set_xticks(range(1, 12))
    ax.set_xticklabels([f"PC{i}" for i in range(1, 12)])
    ax.legend()
    ax.set_facecolor("#121212")
    fig.patch.set_facecolor("#121212")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pca_scree.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_scatter_genre(df_clean, coords):
    pca_var1 = coords[:, 0].std() ** 2
    df_clean = df_clean.copy()
    df_clean["PC1"] = coords[:, 0]
    df_clean["PC2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=df_clean,
        x="PC1",
        y="PC2",
        hue="genre",
        palette="tab20",
        alpha=0.6,
        s=15,
        edgecolor=None,
        ax=ax,
    )
    var = coords[:, :2].var(axis=0)
    total_var = var.sum() / coords.var(axis=0).sum() * 100
    ax.set_title(
        f"Projecció PCA del Dataset de Spotify\nVariança Total Explicada (2 comp): {total_var:.2f}%",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel(f"PC1", fontsize=11)
    ax.set_ylabel(f"PC2", fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              title="Gènere Musical", markerscale=2)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pca_scatter_genre.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_biplot(coords, pca, df_clean):
    fig, ax = plt.subplots(figsize=(12, 9))
    genres = df_clean["genre"].unique()
    palette = sns.color_palette("tab20", len(genres))
    color_map = dict(zip(genres, palette))
    colors = df_clean["genre"].map(color_map)

    ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.3, s=8, edgecolors="none")

    scale = 3.0
    for i, feat in enumerate(FEATURE_COLUMNS):
        ax.annotate(
            "",
            xy=(pca.components_[0, i] * scale, pca.components_[1, i] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
        )
        ax.text(
            pca.components_[0, i] * scale * 1.1,
            pca.components_[1, i] * scale * 1.1,
            feat,
            fontsize=8,
            color="#333333",
            ha="center",
        )

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[g], label=g) for g in genres]
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Gènere",
        fontsize=7,
        title_fontsize=8,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Biplot — PC1 vs PC2 amb Loading Vectors")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pca_biplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def save_pca_coords(df_clean, coords):
    import pandas as pd
    out = df_clean[["track_id", "genre"]].copy()
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1]
    out["PC3"] = coords[:, 2]
    path = os.path.join(CSV_DIR, "pca_coords.csv")
    out.to_csv(path, index=False)
    print(f"Saved {path} — {len(out)} files")


if __name__ == "__main__":
    print("[pca_analysis] Carregant dataset...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    df_clean, X_scaled, scaler = prepare_dataset(verbose=False)

    pca, coords = run_pca(X_scaled)
    plot_scree(pca)
    plot_scatter_genre(df_clean, coords)
    plot_biplot(coords, pca, df_clean)
    save_pca_coords(df_clean, coords)

    print("[pca_analysis] Fet. Outputs a outputs/")
