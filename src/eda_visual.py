"""
eda_visual.py

EDA visual del dataset Spotify Features.
Genera histogrames, boxplots, heatmap de correlació i scatters.
Cada execució sobreescriu els PNGs anteriors (plt.savefig ho fa per defecte).

Executar des de l'arrel del projecte:
    python src/eda_visual.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_massage import prepare_dataset, FEATURE_COLUMNS

FIGURES_DIR = os.path.join("outputs", "eda", "figures")
CSV_DIR     = os.path.join("outputs", "eda")
SCATTER_SAMPLE = 5_000
RANDOM_STATE = 42


def plot_histograms(df_clean):
    """
    Histograma + KDE de cada feature numèrica.
    Útil per veure la forma de la distribució i detectar asimetries.
    """
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    axes = axes.flatten()
    for i, feat in enumerate(FEATURE_COLUMNS):
        sns.histplot(df_clean[feat], kde=True, ax=axes[i], color="#1DB954")
        axes[i].set_title(feat, fontsize=10)
        axes[i].set_xlabel("")
    axes[-1].set_visible(False)
    fig.suptitle("Distribució de les Features — Spotify", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eda_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_boxplots_genre(df_clean):
    """
    Boxplots de les 4 features principals per gènere. 
    Permet comparar com varien energy, danceability, valence i popularity entre gèneres.
    """
    target_feats = ["energy", "danceability", "valence", "popularity"]
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    for i, feat in enumerate(target_feats):
        sns.boxplot(x="genre", y=feat, hue="genre", data=df_clean, ax=axes[i],
                    palette="tab20", legend=False, linewidth=0.8)
        axes[i].set_title(feat, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=45)
        for tick in axes[i].get_xticklabels():
            tick.set_ha("right")
    fig.suptitle("Distribució per Gènere — Features principals", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eda_boxplots_genre.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_correlation_heatmap(df_clean):
    """
    Mapa de calor de la correlació de Pearson entre les 11 features. 
    Ajuda a identificar variables redundants abans de PCA.
    """
    corr = df_clean[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlació de Pearson — 11 Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eda_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_scatter_high_corr(df_clean):
    """
    Scatters de les dues parelles amb correlació alta
    (energy-loudness +0.83, energy-acousticness -0.73). 
    Mostra 5.000 punts aleatoris per llegibilitat.
    """
    sample = df_clean.sample(n=SCATTER_SAMPLE, random_state=RANDOM_STATE)
    r_el = sample["energy"].corr(sample["loudness"])
    r_ea = sample["energy"].corr(sample["acousticness"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(sample["energy"], sample["loudness"], alpha=0.3, s=8, color="#1DB954")
    axes[0].set_xlabel("energy")
    axes[0].set_ylabel("loudness")
    axes[0].set_title(f"energy vs loudness  (r = {r_el:+.2f})", fontweight="bold")

    axes[1].scatter(sample["energy"], sample["acousticness"], alpha=0.3, s=8, color="#1DB954")
    axes[1].set_xlabel("energy")
    axes[1].set_ylabel("acousticness")
    axes[1].set_title(f"energy vs acousticness  (r = {r_ea:+.2f})", fontweight="bold")

    fig.suptitle("Parelles amb alta correlació", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eda_scatter_high_corr.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_scatter_popularity(df_clean):
    """
    Scatters de popularitat vs danceability, energy i valence.
    Permet veure visualment si alguna feature prediu millor la popularitat.
    """
    sample = df_clean.sample(n=SCATTER_SAMPLE, random_state=RANDOM_STATE)
    pairs = ["danceability", "energy", "valence"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat in zip(axes, pairs):
        ax.scatter(sample[feat], sample["popularity"], alpha=0.3, s=8, color="#1DB954")
        ax.set_xlabel(feat)
        ax.set_ylabel("popularity")
        r = sample[feat].corr(sample["popularity"])
        ax.set_title(f"popularity vs {feat}\n(r = {r:+.2f})", fontweight="bold")

    fig.suptitle("Popularitat vs Features musicals", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eda_scatter_popularity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def save_genre_profiles(df_clean):
    """
    Calcula la mitjana de cada feature per gènere i ho exporta a CSV.
    Servirà per al radar chart del dashboard.
    """
    profiles = df_clean.groupby("genre")[FEATURE_COLUMNS].mean().round(4)
    path = os.path.join(CSV_DIR, "genre_profiles.csv")
    profiles.to_csv(path)
    print(f"Saved {path} — {len(profiles)} gèneres")


if __name__ == "__main__":
    print("[eda_visual] Carregant dataset...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    df_clean, X_scaled, scaler = prepare_dataset(verbose=False)

    plot_histograms(df_clean)
    plot_boxplots_genre(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_scatter_high_corr(df_clean)
    plot_scatter_popularity(df_clean)
    save_genre_profiles(df_clean)

    print("[eda_visual] Fet. Outputs a outputs/")
