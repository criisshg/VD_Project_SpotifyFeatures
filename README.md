# VD Project — Spotify Features

**Visualització de Dades · EE – UAB · Curs 2025-26 · Grup 03**

Laia Alcalde · Laia Camara · Cristina Huanca · Elena Gutiérrez · Iker Bolancé · Alex Mayer

---

## Sobre el projecte

**Featurefy** és una visualització de patrons acústics sobre el dataset *Spotify Features* (Kaggle, 232K cançons · 26 gèneres · 11 features). El projecte combina **anàlisi exploratòria, reducció de dimensionalitat (PCA, t-SNE) i clustering (K-Means)** per respondre tres hipòtesis: quines features expliquen la popularitat, si existeixen clústers acústics naturals més enllà del gènere, i si els gèneres tenen perfils diferenciables. Els resultats es presenten en un informe i addicionalment un **dashboard interactiu**.

> Instal·lació i ús: **[SETUP.md](SETUP.md)**.

---

## Estructura del repositori

```
VD_Project_SpotifyFeatures/
│
├── data/
│   ├── raw/
│   │   └── SpotifyFeatures.csv          # Dataset original — MAI modificar
│   └── processed/
│       └── spotify_clean.csv            # Dataset net generat per data_massage.py
│
├── src/
│   ├── data_massage.py                  # Neteja i preparació del dataset
│   ├── eda_visual.py                    # Anàlisi exploratòria visual
│   ├── pca_analysis.py                  # Reducció de dimensionalitat PCA
│   ├── tsne_analysis.py                 # Visualització t-SNE
│   ├── clustering.py                    # Clustering K-Means (multi-k)
│   ├── correlation_network.py           # Xarxa Plotly de correlacions Pearson
│   └── dashboard_api.py                 # API Flask + servidor del dashboard
│
├── outputs/                             # generat pels scripts — no es puja
│   ├── eda/
│   │   ├── figures/                     # Gràfiques de l'EDA
│   │   ├── correlation_matrix.csv       # Matriu de correlació Pearson
│   │   └── genre_profiles.csv           # Mitjana de features per gènere
│   ├── pca/
│   │   ├── figures/                     # Gràfiques del PCA
│   │   └── pca_coords.csv               # Coordenades PC1, PC2, PC3 per cançó
│   ├── tsne/
│   │   ├── figures/                     # Gràfiques del t-SNE
│   │   └── tsne_coords.csv              # Coordenades 2D per cançó (mostra 10k)
│   └── clustering/
│       ├── figures/                     # Gràfiques del clustering
│       ├── cluster_labels.csv           # Cluster assignat a cada cançó (k òptim)
│       ├── cluster_multi_k.csv          # Assignacions per a k = 2..12 (slider)
│       └── cluster_profiles.csv         # Perfil mitjà de features per cluster
│
├── config/                              # Metadades del dashboard
│   ├── cluster_profiles.json            # Etiquetes humanes dels clusters K-Means
│   ├── playlist_presets.json            # Targets predefinits per al Builder
│   └── README.md
│
├── dashboard/                           # Dashboard Featurefy
│   ├── index.html                       # Bundle standalone de Claude Design
│   ├── dashboard-boot.js                # Pont entre l'API i bundle
│   └── README.md
│
├── SETUP.md                             # Instruccions d'instal·lació i ús
├── requirements.txt                     # Dependències Python
├── .gitignore
└── VD Proyecto Spotify.pdf              # Especificació del projecte
```

## Pipeline (executar des de l'arrel)

```bash
python src/data_massage.py          # → data/processed/spotify_clean.csv
python src/eda_visual.py            # ~1 min
python src/pca_analysis.py          # ~1 min
python src/tsne_analysis.py         # ~2-4 min
python src/clustering.py            # ~10-20 min · depèn de tsne
python src/correlation_network.py   # xarxa per al dashboard
```

## Dashboard Featurefy

```
src/*.py  →  outputs/*.csv + config/*.json  →  API Flask  →  bundle HTML
```

Bundle generat amb Claude Design, servit per [src/dashboard_api.py](src/dashboard_api.py). Seccions: **Builder**, **Vinyl**, **Space** (PCA/t-SNE), **Radar de gèneres** i **Graf de correlació**.

Per arrencar-lo i regenerar el bundle, veure [SETUP.md](SETUP.md) i [dashboard/README.md](dashboard/README.md).
