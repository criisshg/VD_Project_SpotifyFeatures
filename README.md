# VD Project — Spotify Features
**Visualització de Dades · EE – UAB · Curs 2025-26 · Grup 03**

Laia Alcalde · Laia Camara · Cristina Huanca · Elena Gutiérrez · Iker Bolancé · Alex Mayer

---

## Abans de res — llegeix SETUP.md

Per clonar el repo, crear l'entorn virtual i instal·lar les dependències consulta **[SETUP.md](SETUP.md)**.
Els scripts fallen si no tens l'entorn activat i les llibreries instal·lades.

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
│   └── README.md                        # Documentació de data_massage.py
│
├── outputs/
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
├── config/                              # Metadades del dashboard (no hardcoded al HTML)
│   ├── cluster_profiles.json            # Etiquetes humanes dels clusters K-Means
│   ├── playlist_presets.json            # Targets predefinits per al Builder
│   └── README.md
│
├── design/                              # Dashboard Featurefy
│   ├── Featurefy Dashboard.html                     # Versió de treball
│   ├── Featurefy Dashboard (standalone).html        # Versió completa autocontinguda
│   ├── Featurefy Dashboard - standalone source.html # Source net
│   └── README.md
│
├── PLAN.md                              # Pla del projecte i roadmap del dashboard
├── SETUP.md                             # Instruccions d'instal·lació i ús
├── requirements.txt                     # Dependències Python
├── .gitignore
└── VD Proyecto Spotify.pdf              # Especificació del projecte
```

> La carpeta `outputs/` es genera automàticament en executar els scripts. No es puja al repo.

---

## Ordre d'execució dels scripts

Sempre des de l'arrel del projecte (no des de `src/`):

```bash
python src/data_massage.py          # genera data/processed/spotify_clean.csv
python src/eda_visual.py            # ~1 min
python src/pca_analysis.py          # ~1 min
python src/tsne_analysis.py         # ~2-4 min
python src/clustering.py            # ~10-20 min
python src/correlation_network.py   # xarxa Plotly per al dashboard
```

`clustering.py` necessita que `tsne_analysis.py` s'hagi executat abans (llegeix `outputs/tsne/tsne_coords.csv`).

---

## Dashboard Featurefy (`design/`)

El dashboard final viu a `design/Featurefy Dashboard (standalone).html`. És una versió autocontinguda i preparada per ser **connectada via API** amb els outputs dels scripts:

```
src/*.py  →  outputs/*.csv + config/*.json  →  API lleugera  →  dashboard HTML
```

Així, regenerar clusters o t-SNE només requereix re-executar els scripts; el dashboard s'actualitza sense tocar HTML.

Seccions previstes: **Builder** (playlist personalitzable amb sliders), **Space** (PCA / t-SNE acolorit per cluster), **Radar de gèneres**, **Graf de correlació** i **Vinil**.

---

## Gràfiques generades

### EDA (`outputs/eda/figures/`)

| Fitxer | Què mostra |
|---|---|
| `eda_histograms.png` | Distribució de cada una de les 11 features numèriques amb histograma i corba KDE. Permet detectar asimetries, bimodalitat i outliers en cada variable. |
| `eda_boxplots_genre.png` | Boxplots de `energy`, `danceability`, `valence` i `popularity` separats per gènere. Permet veure com varien aquestes features entre gèneres i detectar valors extrems. |
| `eda_correlation_heatmap.png` | Mapa de calor amb la correlació de Pearson entre les 11 features. Les cel·les en vermell intens indiquen correlació alta (+/-). Clau per decidir quines variables eliminar abans de PCA. |
| `eda_scatter_high_corr.png` | Scatter de les dues parelles amb correlació alta: `energy vs loudness` (+0.83) i `energy vs acousticness` (−0.73). Confirma visualment que actuen de forma redundant. |
| `eda_scatter_popularity.png` | Scatter de `popularity` vs `danceability`, `energy` i `valence`. Permet veure si alguna d'aquestes features prediu la popularitat d'una cançó. |
| `eda_violin_genre.png` | Violin plots de les 4 features principals per gènere. Combina boxplot i densitat: mostra la forma completa de la distribució, revelant bimodalitats i asimetries que el boxplot amaga. |
| `eda_correlation_network.png` | Xarxa de correlació entre features: nodes = features, arestes = correlació Pearson. Gruix i opacitat = força; verd = positiva, vermell = negativa. Versió visual i llegible del heatmap. |

### PCA (`outputs/pca/figures/`)

| Fitxer | Què mostra |
|---|---|
| `pca_scree.png` | Gràfica de barres amb la variança explicada per cada component principal (PC1–PC11) i la seva acumulació. Indica quants components cal retenir per capturar ~80% de la variança. |
| `pca_scatter_genre.png` | Projecció de totes les cançons al pla PC1–PC2, cada gènere en un color diferent. Permet veure si els gèneres formen grups separats en l'espai de menor dimensió. |
| `pca_biplot.png` | Igual que l'scatter però amb fletxes que indiquen la direcció i força de cada feature dins l'espai PCA. Les fletxes paral·leles indiquen correlació; les oposades, correlació negativa. |

### t-SNE (`outputs/tsne/figures/`)

| Fitxer | Què mostra |
|---|---|
| `tsne_genre.png` | Projecció no lineal de 10.000 cançons en 2D, cada gènere en un color. A diferència del PCA, el t-SNE preserva l'estructura local i pot revelar agrupacions que el PCA no veu. |
| `tsne_popularity.png` | Mateixa projecció t-SNE però colorada per valor de `popularity` (gradient de color). Permet veure si les cançons populars es concentren en zones concretes de l'espai. |

### Clustering (`outputs/clustering/figures/`)

| Fitxer | Què mostra |
|---|---|
| `clustering_elbow.png` | Inèrcia (distància interna dels clusters) per a cada valor de k (2–12). El punt on la corba fa un colze indica el nombre de clusters òptim. |
| `clustering_silhouette.png` | Silhouette score per a cada k. Un valor més alt indica clusters més cohesionats i separats entre ells. La barra verda marca el millor k. |
| `clustering_tsne.png` | Projecció t-SNE de la mostra de 10.000 cançons, cada punt coloretat pel cluster que li ha assignat el K-Means. Permet veure si els clusters tenen sentit geogràficament en l'espai t-SNE. |
