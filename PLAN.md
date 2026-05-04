# PLAN — VD Project Spotify Features
**Visualització de Dades · EE – UAB · Curs 2025-26 · Grup 03**

Laia Alcalde · Laia Camara · Cristina Huanca · Elena Gutiérrez · Iker Bolancé · Alex Mayer

---

## Estat actual

| Fase | Estat |
|---|---|
| Selecció del dataset | ✅ Fet |
| Data massage (`src/data_massage.py`) | ✅ Fet |
| EDA visual (`src/eda_visual.py`) | ✅ Fet — 7 figures + `genre_profiles.csv` |
| PCA (`src/pca_analysis.py`) | ✅ Fet — scree, scatter, biplot + `pca_coords.csv` |
| t-SNE (`src/tsne_analysis.py`) | ✅ Fet — genre, popularity + `tsne_coords.csv` |
| Clustering (`src/clustering.py`) | ✅ Fet — elbow, silhouette, tsne + CSVs |
| Dashboard interactiu (Claude Design) | 🔜 Següent pas |
| Informe final | 🔄 En curs (Alex) |

---

## Context i hipòtesis (Acta reunió 1 — 7 abril 2026)

El grup va decidir treballar amb el dataset de **Spotify Features** perquè ja el coneixíem d'Aprenentatge Computacional, cosa que facilita el data massage i l'anàlisi posterior. La narrativa ha de ser de **data storytelling**: el dashboard ha de respondre una pregunta, no simplement mostrar dades.

Tres hipòtesis a respondre visualment:

1. **Popularitat ↔ features musicals** — Quines característiques (danceability, energy, valence…) es correlacionen més amb la popularitat d'una cançó?
2. **Clustering** — Existeixen grups naturals de cançons amb perfils acústics similars, independentment del gènere assignat?
3. **Diferències per gènere** — Com varien les features d'àudio entre gèneres?

---

## Resum del data massage (ja fet)

- **Raw:** 232.725 files × 18 cols. `data/raw/SpotifyFeatures.csv` — mai es modifica.
- **Net:** 176.655 files × 15 cols. `data/processed/spotify_clean.csv`.
- Eliminats 55.951 duplicats per `track_id`.
- Convertit `duration_ms` → `duration_min`.
- Eliminat gènere `A Capella` (119 cançons, sota el llindar de 1000).
- Columnes descartades de l'anàlisi: `key`, `mode`, `time_signature`.
- **Alertes importants per les fases següents:**
  - `energy ↔ loudness` correlació **+0.83** → evitar mostrar-les com independents.
  - `energy ↔ acousticness` correlació **−0.73** → idem.
  - `duration_min` arriba a 92.5 min (Comedy/Soundtrack) → considerar clipping abans de PCA/t-SNE.

Tots els scripts nous importaran d'aquí:

```python
from src.data_massage import prepare_dataset
df_clean, X_scaled, scaler = prepare_dataset()
```

---

## Procés complet fins a l'entrega

### Fase 1 — EDA Visual (`src/eda_visual.py`)

Objectiu: entendre les distribucions i seleccionar quines features protagonitzen el dashboard.

Gràfiques a generar i exportar a `figures/`:

- Histogrames / KDE per a cada feature numèrica (detectar skew, outliers).
- Boxplots per gènere de: `energy`, `danceability`, `valence`, `popularity`.
- Heatmap de correlació Pearson entre les 11 features.
- Scatter plots de les parelles amb alta correlació: `energy vs loudness`, `energy vs acousticness`.
- Scatter `popularity vs danceability`, `popularity vs valence`, `popularity vs energy`.

---

### Fase 2 — PCA (`src/pca_analysis.py`)

Objectiu: reduir dimensions i entendre quines features expliquen la variància.

- Scree plot (variància explicada per component).
- Biplot PC1–PC2 amb vectors de loading de les features.
- Plot 2D colorat per gènere → veure si els gèneres se separen.
- Interpretar PC1 i PC2 pels loadings (per exemple si PC1 captura energy/loudness).
- **Exportar:** `data/processed/pca_coords.csv` (PC1, PC2, PC3 per fila + gènere).

---

### Fase 3 — t-SNE (`src/tsne_analysis.py`)

Objectiu: visualització no lineal per detectar agrupacions reals.

- t-SNE sobre `X_scaled` (perplexity ~30–50, max_iter ≥ 1000).
- Plot 2D colorat per gènere → comprovar hipòtesi 3.
- Plot 2D colorat per `popularity` en gradient → comprovar hipòtesi 1.
- Comparar amb PCA: on t-SNE agrupa diferent hi ha estructura local interessant.
- **Exportar:** `data/processed/tsne_coords.csv` (dim1, dim2 per fila + gènere).

---

### Fase 4 — Clustering (`src/clustering.py`)

Objectiu: trobar grups naturals de cançons (hipòtesi 2).

- K-Means: elbow method per triar k + silhouette score.
- DBSCAN (opcional si hi ha forma irregular als clusters).
- Visualitzar clusters sobre l'espai t-SNE (color = cluster).
- Caracteritzar cada cluster: mitjana de features per cluster → "perfil sonor".
- Comprovar si els clusters corresponen a gèneres o si creuen fronteres.
- **Exportar:** `data/processed/cluster_labels.csv` i `data/processed/genre_profiles.csv`.

---

### Fase 5 — Dashboard interactiu (Claude Design)

Un cop tenim totes les gràfiques i CSVs generats, el dashboard s'assemblarà amb **Claude Design** (claude.ai) com a entrega final.

**Arquitectura: 3 panells corresponents a les 3 hipòtesis**

**Panell 1 — "Popularitat i features"**
- Selector de feature (danceability, energy, valence…).
- Scatter plot feature seleccionada vs popularity amb línia de tendència.
- Top-10 cançons per popularitat dins del filtre.

**Panell 2 — "L'espai sonor" (PCA / t-SNE)**
- Plot interactiu 2D (toggle PCA ↔ t-SNE).
- Color per gènere o cluster (toggle).
- Hover: nom de la cançó, artista, features principals.
- Filtre per gènere per destacar subconjunts.

**Panell 3 — "Perfil per gènere"**
- Radar chart: perfil mitjà de features per gènere seleccionat.
- Comparació de 2 gèneres en paral·lel.
- Distribució de popularitat per gènere.

**Dades que el dashboard necessitarà:**

| Fitxer | Contingut | Panell |
|---|---|---|
| `data/processed/spotify_clean.csv` | Dataset complet net | Tots |
| `data/processed/pca_coords.csv` | PC1, PC2, PC3 + gènere + cluster | 2 |
| `data/processed/tsne_coords.csv` | dim1, dim2 + gènere + cluster | 2 |
| `data/processed/cluster_labels.csv` | track_id + cluster_id + perfil | 2 i 3 |
| `data/processed/genre_profiles.csv` | Mitjana de features per gènere | 3 |

**Com usar Claude Design:**
1. Tenir tots els CSVs de `data/processed/` generats.
2. Obrir claude.ai → nou projecte → adjuntar els CSVs.
3. Prompt: *"Crea un dashboard interactiu HTML amb Plotly que mostri els tres panells descrits, usant les dades dels CSVs adjunts. Paleta Spotify: verd `#1DB954`, fons fosc, responsive, tooltips."*
4. Iterar sobre el disseny fins tenir el resultat desitjat.
5. Exportar `dashboard/index.html` com a entrega final.

---

## Estructura de fitxers objectiu

```
VD_Project_SpotifyFeatures/
├── data/
│   ├── raw/
│   │   └── SpotifyFeatures.csv          # mai modificar
│   └── processed/
│       ├── spotify_clean.csv            # ✅ fet
│       ├── pca_coords.csv               # ⬜ pendent fase 2
│       ├── tsne_coords.csv              # ⬜ pendent fase 3
│       ├── cluster_labels.csv           # ⬜ pendent fase 4
│       └── genre_profiles.csv           # ⬜ pendent fase 4
├── src/
│   ├── data_massage.py                  # ✅ fet
│   ├── eda_visual.py                    # ⬜ pendent fase 1
│   ├── pca_analysis.py                  # ⬜ pendent fase 2
│   ├── tsne_analysis.py                 # ⬜ pendent fase 3
│   └── clustering.py                    # ⬜ pendent fase 4
├── figures/                             # imatges exportades per a l'informe
├── dashboard/
│   └── index.html                       # ⬜ entrega final (Claude Design)
├── PLAN.md                              # aquest fitxer
├── README.md
└── VD Proyecto Spotify.pdf
```

---

## Dependències

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
```
