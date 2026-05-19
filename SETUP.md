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

## 6. Obrir el dashboard

El dashboard final viu a `design/`. Per obrir-lo, simplement obre el fitxer al navegador:

```bash
open "design/Featurefy Dashboard (standalone).html"
```

És una versió autocontinguda, preparada per ser connectada via API amb els outputs dels scripts.

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
- Dependències (a `requirements.txt`): `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `networkx`, `plotly`
