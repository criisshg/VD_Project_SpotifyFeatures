# dashboard/

Frontend del dashboard **Featurefy** servit per [../src/dashboard_api.py](../src/dashboard_api.py).

## Fitxers en runtime

- `index.html` — bundle standalone generat amb Claude Design (CSS, fonts, Plotly i `app.js` empaquetats). Té injectat un `<script src="dashboard-boot.js">` al final del template.
- `dashboard-boot.js` — script pont: fa fetch a `/api/*`, mapeja les claus llargues dels CSV a les curtes que espera `app.js`, muta `window.DATA` i crida els `init()` dels mòduls.

## Fitxers auxiliars (descartables)

`_unpack.py`, `_inject.py` i `_unpacked/` són eines puntuals per inspeccionar i re-injectar el bundle. Es poden `.gitignore`-jar i regenerar quan calgui.

## Arrencar

```bash
source .venv/bin/activate
python src/dashboard_api.py
# → http://localhost:5001/
```

> Port **5001** (el 5000 està ocupat per AirPlay a macOS).

## Regenerar el bundle

Quan descarreguis una nova versió des de Claude Design:

1. Substitueix `index.html`.
2. Re-injecta el pont: `python dashboard/_inject.py`.
3. Si han canviat les claus curtes o l'estructura de `window.DATA`, ajusta el mapeig a `dashboard-boot.js`.
