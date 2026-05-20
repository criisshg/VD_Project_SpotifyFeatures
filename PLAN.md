# PLAN — VD Project Spotify Features
**Visualització de Dades · EE – UAB · Curs 2025-26 · Grup 03**

Laia Alcalde · Laia Camara · Cristina Huanca · Elena Gutiérrez · Iker Bolancé · Alex Mayer

---

## Estat actual

| Fase | Estat |
|---|---|
| Selecció del dataset | ✅ Fet |
| Data massage (`src/data_massage.py`) | ✅ Fet |
| EDA visual (`src/eda_visual.py`) | ✅ Fet — 7 figures + `genre_profiles.csv` + `correlation_matrix.csv` |
| PCA (`src/pca_analysis.py`) | ✅ Fet — scree, scatter, biplot + `pca_coords.csv` |
| t-SNE (`src/tsne_analysis.py`) | ✅ Fet — genre, popularity + `tsne_coords.csv` |
| Clustering (`src/clustering.py`) | ✅ Fet — elbow, silhouette, tsne + multi-k CSV |
| Xarxa de correlació (`src/correlation_network.py`) | ✅ Fet — xarxa Plotly per al dashboard |
| Dashboard Featurefy (`dashboard/`) | ✅ Fet — bundle de Claude Design servit per Flask |
| Connexió Dashboard ↔ scripts (API) | ✅ Fet — `src/dashboard_api.py` + `dashboard-boot.js` |
| Informe final | 🔄 En curs |

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

Tots els scripts nous importen d'aquí:

```python
from src.data_massage import prepare_dataset
df_clean, X_scaled, scaler = prepare_dataset()
```

---

## Procés complet fins a l'entrega

### Fase 1 — EDA Visual (`src/eda_visual.py`) ✅

7 figures a `outputs/eda/figures/`:

- Histogrames + KDE per a cada feature numèrica.
- Boxplots per gènere de `energy`, `danceability`, `valence`, `popularity`.
- Heatmap de correlació Pearson.
- Scatter plots de les parelles amb alta correlació (`energy vs loudness`, `energy vs acousticness`).
- Scatter `popularity` vs `danceability` / `energy` / `valence`.
- Violin plots de les 4 features principals per gènere.
- Xarxa de correlació en format estàtic (PNG).

Outputs CSV: `correlation_matrix.csv`, `genre_profiles.csv`.

---

### Fase 2 — PCA (`src/pca_analysis.py`) ✅

- Scree plot (variància explicada per component).
- Biplot PC1–PC2 amb vectors de loading.
- Scatter PC1–PC2 colorat per gènere.
- **Output:** `outputs/pca/pca_coords.csv` (PC1, PC2, PC3 per fila + gènere).

---

### Fase 3 — t-SNE (`src/tsne_analysis.py`) ✅

- t-SNE sobre `X_scaled` (perplexity ~30–50, max_iter ≥ 1000) sobre mostra de 10k cançons.
- Plot 2D colorat per gènere.
- Plot 2D colorat per `popularity` en gradient.
- **Output:** `outputs/tsne/tsne_coords.csv` (dim1, dim2 per fila + gènere).

---

### Fase 4 — Clustering (`src/clustering.py`) ✅

- K-Means multi-k (k = 2..12) per alimentar el slider del dashboard.
- Elbow method + silhouette score.
- Visualització dels clusters sobre l'espai t-SNE.
- Caracterització de cada cluster per centroide (perfil sonor).
- **Outputs:** `outputs/clustering/cluster_labels.csv`, `cluster_multi_k.csv`, `cluster_profiles.csv`.
- **Interpretació humana:** `config/cluster_profiles.json` (Spoken/Live, Ambient/Instrumental, Mainstream Pop/Rock).

---

### Fase 5 — Xarxa de correlació Plotly (`src/correlation_network.py`) ✅

Xarxa interactiva de correlacions Pearson entre features (nodes = features, arestes = correlació amb gruix/color). Pensada per a la secció **/05** del dashboard.

---

### Fase 6 — Dashboard Featurefy (`dashboard/`) ✅

El dashboard es genera amb **Claude Design** com a bundle standalone autocontingut. Els fitxers es guarden a `dashboard/`:

- `dashboard/index.html` — bundle de Claude Design amb manifest gzipped i base64 (CSS, fonts, Plotly i app.js dins).
- `dashboard/dashboard-boot.js` — script pont que substitueix les dades sample del bundle per les reals que serveix l'API.
- `dashboard/README.md` — documentació de la carpeta i workflow de regeneració.

**Seccions del dashboard:**

1. **Builder** — Playlist personalitzable amb sliders per feature. Targets a `config/playlist_presets.json`, ranking per distància L2 ponderada.
2. **Vinyl** — Visualització polar dels gèneres principals.
3. **Space** — Projecció PCA / t-SNE (toggle), acolorida per cluster o gènere. Slider K = {3, 5, 7} per al clustering.
4. **Radar de gèneres** — Perfil mitjà de features per gènere, comparació en paral·lel.
5. **Graf de correlació** — Xarxa Plotly interactiva (Fase 5).

---

### Fase 7 — Connexió Dashboard ↔ scripts (API) ✅

Arquitectura final:

```
src/*.py  →  outputs/*.csv + config/*.json  →  API Flask  →  dashboard HTML
                                              (src/dashboard_api.py)
```

`src/dashboard_api.py` carrega tots els CSV/JSON una sola vegada a l'arrencada i exposa 9 endpoints (`/api/kpis`, `/api/tracks`, `/api/tsne`, `/api/pca`, `/api/clusters?k={3,5,7}`, `/api/cluster-profiles?k={3,5,7}`, `/api/genre-profiles`, `/api/correlation`, `/api/presets`). El bridge `dashboard/dashboard-boot.js` fa fetch dels endpoints, mapeja les claus llargues dels CSV a les curtes que espera el bundle (`track_id→tid`, `acousticness→ac`, …) i muta `window.DATA` in-place.

Regenerar clusters o t-SNE només requereix re-executar els scripts i reiniciar el servidor Flask; no cal tocar HTML. Veure [SETUP.md](SETUP.md) per a les instruccions d'execució.

---

## Estructura de fitxers objectiu (real)

```
VD_Project_SpotifyFeatures/
├── data/
│   ├── raw/
│   │   └── SpotifyFeatures.csv          # mai modificar
│   └── processed/
│       └── spotify_clean.csv            # ✅
├── src/
│   ├── data_massage.py                  # ✅
│   ├── eda_visual.py                    # ✅
│   ├── pca_analysis.py                  # ✅
│   ├── tsne_analysis.py                 # ✅
│   ├── clustering.py                    # ✅
│   ├── correlation_network.py           # ✅
│   └── dashboard_api.py                 # ✅ API Flask + servidor del dashboard
├── outputs/
│   ├── eda/                             # ✅
│   ├── pca/                             # ✅
│   ├── tsne/                            # ✅
│   └── clustering/                      # ✅
├── config/
│   ├── cluster_profiles.json            # ✅ etiquetes dels clusters
│   └── playlist_presets.json            # ✅ targets per al Builder
├── dashboard/
│   ├── index.html                       # ✅ bundle de Claude Design
│   ├── dashboard-boot.js                # ✅ pont API ↔ bundle
│   └── README.md                        # ✅ docs de la carpeta
├── PLAN.md
├── README.md
├── SETUP.md
└── VD Proyecto Spotify.pdf
```

---

## Dependències

```bash
pip install -r requirements.txt
# Pipeline:  pandas, numpy, scikit-learn, matplotlib, seaborn, networkx, plotly
# Dashboard: flask>=3.0, flask-compress>=1.14
```
