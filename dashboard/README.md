# dashboard/

Frontend del dashboard **Featurefy** servit per l'API Flask de
[../src/dashboard_api.py](../src/dashboard_api.py).

L'HTML es genera amb **Claude Design** com a bundle standalone amb manifest
gzipped i datos sample embedded. La feina d'aquesta carpeta és:

1. Allotjar aquest bundle (`index.html`).
2. Substituir els datos sample del bundle pels datos reals de l'API mitjançant
   un script pont (`dashboard-boot.js`).

---

## Fitxers que **sí** s'utilitzen en runtime

| Fitxer | Què és |
|---|---|
| `index.html` | Bundle standalone descarregat de Claude Design. Conté CSS, fonts, Plotly i el codi de l'app en un manifest gzipped + base64. Té injectat al final del template un `<script src="dashboard-boot.js">` per a que es carregui després de l'`app.js` empaquetat. |
| `dashboard-boot.js` | Script pont. Fa fetch a `/api/*`, mapeja les claus llargues dels CSV (`track_id`, `acousticness`, …) a les claus curtes que espera l'`app.js` empaquetat (`tid`, `ac`, …), mutar el `window.DATA` existent (no reassignar — l'app.js fa `const D = window.DATA`) i crida `GenreBar.init() / Builder.init() / Vinyl.init() / SonicSpace.init() / Radar.init() / FeatureGraph.init()`. |

Aquests dos els serveix Flask via:

```python
@app.route("/")                       # → index.html
@app.route("/<path:filename>")        # → qualsevol asset estàtic de dashboard/
```

## Fitxers que es van usar **una sola vegada** (es poden esborrar)

| Fitxer | Per a què va servir |
|---|---|
| `_unpack.py` | Script puntual que extreu el manifest del bundle (base64 + gzip) i guarda cada asset a `_unpacked/`. Va servir per descobrir l'estructura del bundle i veure el codi de l'`app.js` empaquetat sense haver d'inspeccionar el HTML monolític. |
| `_inject.py` | Script puntual que va injectar `<script src="dashboard-boot.js"></script>` al final del template JSON-encoded dins de `index.html`. Ja no cal tornar a executar-lo mentre el bundle no es regeneri. |
| `_unpacked/` | Carpeta amb tots els assets desempaquetats del bundle (Plotly 4.5 MB, `app.js` de 60 KB, `data.js` sample de 2.2 MB, fonts WOFF2, 2 JSON de config). **No es serveix**, només va ser per inspecció. |

> Tots tres es poden `.gitignore`-jar si es vol mantenir el repo net. Es poden
> tornar a generar en qualsevol moment executant `python dashboard/_unpack.py`
> i `python dashboard/_inject.py` des de l'arrel del projecte.

---

## Com arrencar

Des de l'arrel del repo:

```bash
source .venv/bin/activate
python src/dashboard_api.py
# → obrir http://localhost:5001/
```

> Port **5001**, no 5000 — el 5000 està ocupat per l'AirPlay Receiver de macOS.

## Quan regenerar el bundle

Si tornes a Claude Design i descarregues una versió nova del dashboard:

1. Substitueix `index.html` per la nova descàrrega.
2. Comprova que la nova versió segueixi exposant `window.SKIP_APP_BOOT` i els
   mòduls `GenreBar / Builder / Vinyl / SonicSpace / Radar / FeatureGraph`
   (executa `python dashboard/_unpack.py` i grep al JS desempaquetat).
3. Re-injecta `dashboard-boot.js` al template:
   ```bash
   python dashboard/_inject.py
   ```
4. Si han canviat les claus curtes del FEATS array o l'estructura de
   `window.DATA`, ajusta el mapeig a `dashboard-boot.js`.
