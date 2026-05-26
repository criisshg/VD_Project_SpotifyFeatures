# Setup

## 1. Entorn

```bash
git clone <url-del-repo>
cd VD_Project_SpotifyFeatures
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Pipeline d'anàlisi

Sempre des de l'arrel del projecte:

```bash
python src/data_massage.py          # → data/processed/spotify_clean.csv
python src/eda_visual.py            # ~1 min
python src/pca_analysis.py          # ~1 min
python src/tsne_analysis.py         # ~2-4 min
python src/clustering.py            # ~10-20 min · depèn de tsne
python src/correlation_network.py
```

Outputs a `outputs/`. `clustering.py` necessita `tsne_coords.csv` previ.

## 3. Dashboard

```bash
python src/dashboard_api.py
# → http://localhost:5001/
```

> Port **5001** (el 5000 està ocupat per AirPlay a macOS).

L'API exposa endpoints `/api/kpis`, `/api/tracks`, `/api/tsne`, `/api/pca`, `/api/clusters?k={3,5,7}`, `/api/cluster-profiles`, `/api/genre-profiles`, `/api/correlation`, `/api/presets`. El pont [dashboard/dashboard-boot.js](dashboard/dashboard-boot.js) connecta el bundle amb aquests endpoints — veure [dashboard/README.md](dashboard/README.md) si es regenera el bundle.

## Requisits

- Python 3.10+
- `requirements.txt`: pandas, numpy, scikit-learn, matplotlib, seaborn, networkx, plotly, flask, flask-compress
