# Setup — Entorn de treball

## 1. Clonar el repositori

```bash
git clone <url-del-repo>
cd VD_Project_SpotifyFeatures
```

## 2. Crear l'entorn virtual

```bash
python3 -m venv .venv
```

Això crea una carpeta `.venv/` dins del projecte amb Python i pip aïllats.
**No es puja al repo** (està al `.gitignore`).

## 3. Activar l'entorn

```bash
source .venv/bin/activate
```

El prompt del terminal canviarà a `(.venv)` per indicar que està actiu.

## 4. Instal·lar les dependències

```bash
pip install -r requirements.txt
```

## 5. Executar els scripts d'anàlisi

Sempre des de l'arrel del projecte (no des de `src/`):

```bash
python src/data_massage.py          # genera data/processed/spotify_clean.csv
python src/eda_visual.py            # ~1 min
python src/pca_analysis.py          # ~1 min
python src/tsne_analysis.py         # ~2-4 min
python src/clustering.py            # ~10-20 min
python src/correlation_network.py   # xarxa Plotly per al dashboard
```

Els outputs (figures i CSVs) es generen automàticament a `outputs/`.
`clustering.py` necessita que `tsne_analysis.py` s'hagi executat abans.

## 6. Arrencar el dashboard

El frontend viu a `dashboard/` (bundle HTML standalone generat amb Claude Design)
i el serveix una API Flask lleugera definida a [src/dashboard_api.py](src/dashboard_api.py).

Des de l'arrel del repo, amb l'entorn activat:

```bash
python src/dashboard_api.py
```

Després obre [http://localhost:5001/](http://localhost:5001/) al navegador.

> Port **5001**, no 5000 — el 5000 està ocupat per l'AirPlay Receiver a macOS.

L'API carrega a memòria els CSV/JSON ja generats pel pipeline (pas 5) i els
exposa als endpoints `/api/kpis`, `/api/tracks`, `/api/tsne`, `/api/pca`,
`/api/clusters?k={3,5,7}`, `/api/genre-profiles`, `/api/correlation`,
`/api/cluster-profiles` i `/api/presets`. El fitxer pont
[dashboard/dashboard-boot.js](dashboard/dashboard-boot.js) fa fetch a aquests
endpoints i injecta les dades reals dins del bundle (`window.DATA`).

> Si el bundle (`dashboard/index.html`) es regenera des de Claude Design, cal
> tornar a injectar el script pont — veure [dashboard/README.md](dashboard/README.md).

## 7. Desactivar l'entorn en acabar

```bash
deactivate
```

## 8. Alliberar espai (opcional)

Si vols recuperar l'espai en disc que ocupen els paquetes instal·lats:

```bash
rm -rf .venv
```

La propera vegada que obris el projecte, torna al pas 2.

---

## Requisits

- Python 3.10+
- Dependències (a `requirements.txt`):
  - **Pipeline d'anàlisi**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `networkx`, `plotly`
  - **Servidor del dashboard**: `flask>=3.0`, `flask-compress>=1.14`
