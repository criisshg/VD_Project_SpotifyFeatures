# `src/data_massage.py`

Script de preparació del dataset Spotify Features. Llegeix el CSV cru,
fa l'EDA + neteja + transformacions, i genera el CSV net que servirà
de base per a totes les fases posteriors (EDA visual, PCA, t-SNE, clustering).

> **Important per a l'equip**: el CSV cru `data/raw/SpotifyFeatures.csv` no es
> modifica mai. Tot el treball posterior s'ha de fer sobre l'output
> `data/processed/spotify_clean.csv`.

## Requeriments

- Python 3.10+
- `pandas`, `numpy`, `scikit-learn`

```bash
pip install pandas numpy scikit-learn
```

## Com executar-lo

Des de l'arrel del repo (no des de `src/`, perquè les rutes són relatives):

```bash
python src/data_massage.py
```

Generarà (o sobreescriurà) `data/processed/spotify_clean.csv` i imprimirà
un informe textual per stdout.

## Què fa el pipeline

L'ordre dels passos és deliberat — el filtratge per gènere passa **després**
d'eliminar duplicats per `track_id` perquè els recomptes per gènere siguin
coherents.

1. **`load_raw_data`** — carrega `data/raw/SpotifyFeatures.csv`.
2. **`inspect_dataset`** — informe textual: forma, tipus, nuls, duplicats,
   `describe()` numèric, recompte per gènere.
3. **`drop_duplicates_and_nulls`** — elimina duplicats per `track_id`
   (la mateixa cançó apareix en diversos gèneres al CSV original) i imputa
   nuls (mitjana per a numèriques, mode per a categòriques).
4. **`convert_duration`** — afegeix `duration_min` (= `duration_ms / 60_000`)
   i elimina `duration_ms`.
5. **`check_ranges`** — comprova que cada feature numèrica cau dins del
   rang esperat (`EXPECTED_RANGES`). Imprimeix `[OK]` o `[WARN]`. No retalla
   valors; només informa.
6. **`filter_minority_genres`** — elimina les files dels gèneres amb menys de
   `MIN_GENRE_COUNT` cançons (per defecte 1000).
7. **`correlation_report`** — matriu de correlacions Pearson entre features +
   llistat de parelles amb `|corr| >= 0.7` (redundàncies a tenir en compte
   abans de PCA/t-SNE).
8. **Guardar** `data/processed/spotify_clean.csv` (valors interpretables,
   sense escalar).
9. **`scale_features`** — `StandardScaler` sobre les features numèriques.
   Retorna `(X_scaled, scaler)` en memòria; **no es persisteix** com a CSV
   perquè l'escalat és una operació de modelatge, no de dades.

## Output: `data/processed/spotify_clean.csv`

15 columnes:

- **Info** (4): `genre`, `artist_name`, `track_name`, `track_id`.
- **Features** (11): `popularity`, `acousticness`, `danceability`,
  `duration_min`, `energy`, `instrumentalness`, `liveness`, `loudness`,
  `speechiness`, `tempo`, `valence`.

Columnes del raw que **no** es porten al CSV net: `key`, `mode`,
`time_signature` (categòriques que no formen part de l'anàlisi numèric
inicial). Si més endavant les voleu codificar i incloure, afegiu-les a
`FEATURE_COLUMNS`.

## Com usar-lo des d'altres scripts

```python
from src.data_massage import prepare_dataset

df_clean, X_scaled, scaler = prepare_dataset()
# df_clean: DataFrame amb info + features (sense escalar)
# X_scaled: np.ndarray (n_files, 11) llest per a PCA / t-SNE / clustering
# scaler:   StandardScaler entrenat (per transformar dades futures amb la
#           mateixa mitjana/desviació)
```

També podeu passar paràmetres:

```python
prepare_dataset(
    raw_path="data/raw/SpotifyFeatures.csv",
    clean_path="data/processed/spotify_clean.csv",
    min_genre_count=500,   # llindar més permissiu
)
```

## Resultats de l'última execució (referència)

- Raw: **232.725** files × 18 columnes.
- Eliminats **55.951** duplicats per `track_id` → 176.774 files.
- 1 nul a `track_name` imputat.
- Tots els rangs `[OK]` excepte `duration_min`: arriba a **92,5 min**
  (cançons llargues reals — `Comedy`, `Soundtrack`, etc.). No és un bug;
  cal decidir si retallem (clipping) abans de PCA per evitar que dominin
  la variància.
- Filtratge per gènere amb `MIN_GENRE_COUNT=1000`: només `A Capella`
  (119 cançons) cau sota el llindar.
- Forma final: **176.655** files × 15 columnes.
- Correlacions `|corr| >= 0,7` (redundàncies abans de PCA/t-SNE):
  - `energy ↔ loudness`: **+0,83**
  - `energy ↔ acousticness`: **−0,73**

## Configuració

Els paràmetres principals són constants al capçal del fitxer:

| Constant            | Valor per defecte                              | Descripció                                      |
| ------------------- | ---------------------------------------------- | ----------------------------------------------- |
| `RAW_DATA_PATH`     | `data/raw/SpotifyFeatures.csv`                 | CSV cru d'entrada                               |
| `CLEAN_DATA_PATH`   | `data/processed/spotify_clean.csv`             | CSV net de sortida                              |
| `MIN_GENRE_COUNT`   | `1000`                                         | Llindar mínim de cançons per gènere             |
| `INFO_COLUMNS`      | `genre, artist_name, track_name, track_id`     | Columnes informatives (no s'escalen)            |
| `FEATURE_COLUMNS`   | 11 features numèriques (vegeu sobre)           | Columnes que entren a PCA / t-SNE / clustering  |
| `EXPECTED_RANGES`   | rangs teòrics per feature                      | Per a la validació de coherència (no retalla)   |

# Análisis de Datos de Spotify - Proyecto de Visualización

Este proyecto consiste en un pipeline completo de preparación de datos y análisis estadístico del dataset "Spotify Features", enfocado en la limpieza y la reducción de dimensionalidad para entender cómo se agrupan los géneros musicales.

## 1. Preparación de Datos (Data Massage)
Utilizamos el script `data_massage.py` para realizar una limpieza profunda del dataset original:
* **Limpieza:** Eliminación de duplicados por `track_id` e imputación de valores nulos.
* **Transformación:** Conversión de duraciones a minutos y filtrado de géneros minoritarios.
* **Normalización:** Aplicación de `StandardScaler` para que todas las métricas (tempo, energía, acústica, etc.) tengan el mismo peso en el análisis.

## 2. Reducción de Dimensionalidad (PCA)
Para visualizar la "personalidad" de las canciones en un plano 2D, aplicamos **Análisis de Componentes Principales (PCA)** sobre 11 atributos numéricos.

### Resultados del Análisis:
* **PC1 (Eje X):** Explica el **33.23%** de la varianza.
* **PC2 (Eje Y):** Explica el **16.23%** de la varianza.
* **Total Retenido:** Logramos capturar casi el **50% (49.46%)** de la información original usando solo dos dimensiones.

## 3. Visualización Final
El siguiente gráfico muestra la proyección de las canciones en el nuevo hiperplano de PCA, coloreadas según su género musical.

![Análisis PCA de Spotify](pca_spotify.png)

*Interpretación: Se observa cómo ciertos géneros forman clústeres definidos (como la música clásica o acústica), mientras que los géneros comerciales tienden a solaparse en el centro del gráfico.*

---
## Cómo ejecutar el proyecto
1. Clonar el repositorio.
2. Crear un entorno virtual con Python 3.11/3.12.
3. Instalar dependencias: `pip install pandas scikit-learn matplotlib seaborn`.
4. Ejecutar el análisis: `python pca_analysis.py`.
