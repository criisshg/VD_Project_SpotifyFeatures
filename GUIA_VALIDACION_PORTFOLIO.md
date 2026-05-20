# Guia de estudio y portfolio final - Spotify Features

Proyecto: **VD Project - Spotify Features**  
Asignatura: **Visualitzacio de Dades - EE UAB - Curso 2025/26**  
Grupo: **03**

Este documento sirve para estudiar el proyecto antes de la prueba de validacion y como base para redactar el portfolio final. No se centra en el frontend, sino en el contenido analitico: dataset, hipotesis, data massage, scripts de `src/`, graficas generadas, Plotly, outputs, interpretacion y decisiones metodologicas.

---

## 0. Prompt revisado para seguir trabajando

Prompt original depurado, por si hay que reutilizarlo:

> Haz una investigacion exhaustiva del proyecto `VD_Project_SpotifyFeatures` para preparar una prueba de validacion del grupo y orientar el portfolio final. Analiza el contexto del proyecto, las hipotesis definidas en las actas/PDFs, las columnas del dataset, el data massage, los scripts de `src/`, los outputs generados y especialmente las graficas. No centres el analisis en la estructura visual del frontend, salvo lo minimo necesario para explicar como consume los CSV/JSON. Explica para cada grafica que muestra, por que se eligio, que hipotesis ayuda a validar o refutar, que decision metodologica justifica, que archivo la genera y como defenderla oralmente. Ten en cuenta las plantillas de `plantillas-docs-projecteVD/Word` para orientar la estructura del portfolio final: resumen, introduccion, dataset e hipotesis, metodologia, resultados, discusion, conclusiones, limitaciones y referencias. El resultado debe ser un documento Markdown detallado, claro y util para estudiar.

---

## 1. Idea central del proyecto

El proyecto analiza un dataset de **Spotify Features** para estudiar si las canciones se pueden entender y agrupar a partir de sus caracteristicas musicales y acusticas.

La narrativa no deberia ser "hemos hecho muchas graficas", sino:

> A partir de descriptores acusticos de canciones de Spotify, exploramos si existen patrones musicales interpretables: relacion entre popularidad y features, diferencias entre generos y grupos naturales de canciones mas alla del genero declarado.

La profesora remarco en las actas que no basta con aplicar tecnicas automaticamente. Hay que justificar:

- Por que el dataset es adecuado.
- Por que se usan unas variables y no otras.
- Que decisiones de limpieza afectan los resultados.
- Que aporta cada visualizacion.
- Que limitaciones tiene el analisis.

---

## 2. Fuentes del proyecto consultadas

Documentacion y contexto:

- `VD Proyecto Spotify.pdf`: acta 1, seleccion del dataset e hipotesis iniciales.
- `VD Proyecto Spotify-2.pdf`: acta 2, discusion metodologica sobre duplicados, features, objetivo final y dataset limpio.
- `VD Proyecto Spotify-3.pdf`: acta 3, cierre de EDA/PCA/t-SNE/clustering, Plotly y arquitectura dashboard/API.
- `README.md`: estructura del repositorio y resumen de outputs.
- `PLAN.md`: roadmap, estado del proyecto e hipotesis.
- `src/README.md`: explicacion de `data_massage.py`.
- `plantillas-docs-projecteVD/Word/*`: estructura esperada para informe inicial y portfolio final.

Codigo principal:

- `src/data_massage.py`
- `src/eda_visual.py`
- `src/pca_analysis.py`
- `src/tsne_analysis.py`
- `src/clustering.py`
- `src/correlation_network.py`
- `src/dashboard_api.py`

Outputs principales:

- `outputs/data_massage_report.txt`
- `outputs/eda/*`
- `outputs/pca/*`
- `outputs/tsne/*`
- `outputs/clustering/*`
- `config/cluster_profiles.json`
- `config/playlist_presets.json`

---

## 3. Hipotesis del proyecto

Las hipotesis salen de las actas y del `PLAN.md`. Se pueden formular asi para el portfolio:

### H1. Popularidad y features musicales

**Pregunta:** que caracteristicas musicales se relacionan mas con la popularidad de una cancion?

**Variables clave:** `popularity`, `danceability`, `energy`, `valence`, `acousticness`, `loudness`.

**Visualizaciones relacionadas:**

- `eda_scatter_popularity.png`
- `eda_correlation_heatmap.png`
- `correlation_network.html`

**Resultado defendible:**

La popularidad no parece estar explicada por una sola feature de forma fuerte. Las correlaciones mas relevantes con popularidad son moderadas:

- `popularity` vs `acousticness`: -0.3502
- `popularity` vs `loudness`: +0.3140
- `popularity` vs `energy`: +0.2259
- `popularity` vs `danceability`: +0.2085

Interpretacion: las canciones mas populares tienden a ser algo menos acusticas, algo mas sonoras/energeticas y algo mas bailables, pero la relacion no es suficiente para afirmar causalidad ni prediccion fuerte.

### H2. Clustering de canciones

**Pregunta:** existen grupos naturales de canciones con perfiles acusticos similares, independientemente del genero declarado?

**Variables clave:** features numericas estandarizadas. En la interpretacion final de los clusters se priorizan `energy`, `acousticness`, `speechiness`, `liveness`, `instrumentalness`, `popularity` y `danceability`.

**Visualizaciones relacionadas:**

- `clustering_elbow.png`
- `clustering_silhouette.png`
- `clustering_tsne.png`
- `cluster_profiles.csv`
- `cluster_multi_k.csv`

**Resultado defendible:**

Con K-Means aparecen tres perfiles principales interpretables:

- `C0` - **Spoken / Live**
- `C1` - **Ambient / Instrumental**
- `C2` - **Mainstream Pop / Rock**

Los clusters no son "generos nuevos", sino agrupaciones acusticas. Su valor es que resumen perfiles sonoros transversales.

### H3. Diferencias entre generos

**Pregunta:** los generos musicales tienen perfiles acusticos diferenciables?

**Variables clave:** todas las features numericas, comparadas por `genre`.

**Visualizaciones relacionadas:**

- `eda_boxplots_genre.png`
- `eda_violin_genre.png`
- `pca_scatter_genre.png`
- `pca_biplot.png`
- `genre_profiles.csv`
- radar de generos del dashboard.

**Resultado defendible:**

Algunos generos tienen perfiles claros en medias y distribuciones:

- `Opera` y `Classical`: mucha acousticness y poca energia.
- `Comedy`: speechiness y liveness muy altos.
- `Reggaeton`, `Dance`, `Pop`: mayor danceability/popularity.
- `Soundtrack`: alta instrumentalness y baja valence.

Pero tambien hay solapamiento: el genero declarado no separa perfectamente el espacio musical.

---

## 4. Dataset y columnas

### 4.1 Dataset original

Archivo:

- `data/raw/SpotifyFeatures.csv`

Dimensiones iniciales:

- **232.725 filas**
- **18 columnas**

Columnas originales:

- Identificativas: `genre`, `artist_name`, `track_name`, `track_id`
- Numericas usadas en analisis: `popularity`, `acousticness`, `danceability`, `duration_ms`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`
- Categoricas/musicales no usadas inicialmente: `key`, `mode`, `time_signature`

### 4.2 Dataset procesado

Archivo:

- `data/processed/spotify_clean.csv`

Dimensiones finales:

- **176.655 filas**
- **15 columnas**
- **26 generos**

Columnas finales:

- Info: `genre`, `artist_name`, `track_name`, `track_id`
- Features: `popularity`, `acousticness`, `danceability`, `duration_min`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`

### 4.3 Por que se eliminan `key`, `mode` y `time_signature`

No se eliminan porque sean inutiles, sino porque no entran directamente en el primer analisis numerico:

- `key`: tonalidad musical, categorica.
- `mode`: mayor/menor, categorica.
- `time_signature`: compas, categorica/discreta.

Para PCA, t-SNE y K-Means se trabaja con features numericas comparables. Incluir esas columnas requeriria codificacion especifica y una justificacion adicional. Se pueden mencionar como extension futura.

---

## 5. Data massage

Script:

- `src/data_massage.py`

Funcion central:

```python
from src.data_massage import prepare_dataset

df_clean, X_scaled, scaler = prepare_dataset()
```

Esta funcion es el punto comun del pipeline. La usan EDA, PCA, t-SNE y clustering.

### 5.1 Pasos del pipeline

1. **Carga del CSV raw**
   - Lee `data/raw/SpotifyFeatures.csv`.
   - El raw no se modifica nunca.

2. **Inspeccion**
   - Forma del dataset.
   - Tipos de datos.
   - Nulos.
   - Duplicados.
   - Estadisticas numericas.
   - Conteo por genero.

3. **Eliminacion de duplicados**
   - Se eliminan duplicados por `track_id`.
   - Se pasa de 232.725 a 176.774 filas.
   - Se eliminan **55.951 duplicados**.

4. **Imputacion de nulos**
   - Solo hay 1 nulo en `track_name`.
   - Se imputa con la moda.

5. **Conversion de duracion**
   - `duration_ms` se transforma en `duration_min`.
   - Es mas interpretable para graficas e informe.

6. **Comprobacion de rangos**
   - Valida rangos esperados de las features.
   - No recorta valores, solo avisa.
   - `duration_min` llega a 92,549 min, por encima del rango esperado.

7. **Filtrado de generos minoritarios**
   - Se elimina `A Capella`, con 119 canciones.
   - Umbral: minimo 1000 canciones por genero.

8. **Informe de correlaciones**
   - Detecta redundancias antes de PCA/t-SNE/clustering.

9. **Guardado del CSV limpio**
   - Se guarda `data/processed/spotify_clean.csv`.

10. **Estandarizacion**
   - Usa `StandardScaler`.
   - Devuelve `X_scaled`.
   - No se guarda en CSV porque es una transformacion de modelado.

### 5.2 Decision critica: duplicados por `track_id`

En el acta 2 aparece como punto metodologico importante: una misma cancion puede aparecer asociada a mas de un genero. El pipeline decide quedarse con la primera aparicion.

Como defenderlo:

- El objetivo era evitar que una misma cancion pesara varias veces en PCA, t-SNE y K-Means.
- Para analisis basados en distancia, duplicar canciones puede sesgar densidades y centroides.
- La contrapartida es que se pierde informacion multi-etiqueta de genero.
- Es una limitacion reconocida, no un error.

Respuesta si preguntan:

> Eliminamos duplicados por `track_id` porque los modelos de distancia y varianza tratarian cada repeticion como una observacion independiente. Sabemos que esto simplifica una relacion multi-genero, por eso en el informe lo planteamos como limitacion y posible mejora: conservar una tabla secundaria track-genero o usar etiquetas multiples.

### 5.3 Decision critica: duraciones extremas

`duration_min` tiene un maximo de 92,549 minutos. El script lo detecta con `[WARN]`, pero no hace clipping.

Como defenderlo:

- El pipeline informa del outlier, no lo oculta.
- Mantenerlo conserva canciones largas reales, especialmente de `Comedy` o `Soundtrack`.
- El riesgo es que afecte metodos basados en varianza o distancia.
- Como se estandariza con `StandardScaler`, la escala se normaliza, pero los outliers siguen pudiendo influir.

Mejora futura:

- Comparar resultados con clipping o winsorizacion de `duration_min`.
- Evaluar si los clusters cambian.

---

## 6. Diccionario de features

| Feature | Significado | Rango aprox. | Uso en el proyecto |
|---|---:|---:|---|
| `popularity` | Popularidad estimada de la cancion | 0-100 | Hipotesis H1, ranking, KPIs |
| `acousticness` | Probabilidad de que la cancion sea acustica | 0-1 | Diferencia musica organica vs electrica |
| `danceability` | Facilidad para bailar segun ritmo/tempo/regularidad | 0-1 | Generos bailables, playlists |
| `duration_min` | Duracion en minutos | 0.256-92.549 | Outliers, PCA, perfiles |
| `energy` | Intensidad y actividad perceptual | 0-1 | Muy relacionada con loudness/acousticness |
| `instrumentalness` | Probabilidad de no tener voz | 0-1 | Identifica instrumental/ambient/soundtrack |
| `liveness` | Probabilidad de grabacion en directo | 0-1 | Cluster Spoken/Live |
| `loudness` | Volumen medio en dB | -52.457 a 3.744 | Redundante con energy |
| `speechiness` | Presencia de habla | 0-1 | Comedy/spoken/podcast-like |
| `tempo` | BPM | 30.379-242.903 | Ritmo, playlists |
| `valence` | Positividad musical | 0-1 | Estado emocional, playlists |

---

## 7. EDA visual

Script:

- `src/eda_visual.py`

Entrada:

- `df_clean`, `X_scaled`, `scaler` desde `prepare_dataset(verbose=False)`.

Salidas:

- `outputs/eda/figures/eda_histograms.png`
- `outputs/eda/figures/eda_boxplots_genre.png`
- `outputs/eda/figures/eda_violin_genre.png`
- `outputs/eda/figures/eda_correlation_heatmap.png`
- `outputs/eda/figures/eda_correlation_network.png`
- `outputs/eda/figures/eda_scatter_high_corr.png`
- `outputs/eda/figures/eda_scatter_popularity.png`
- `outputs/eda/genre_profiles.csv`

### 7.1 `eda_histograms.png`

**Que muestra:** distribucion de las 11 features numericas con histograma y KDE.

**Por que se usa:** antes de modelar hay que conocer la forma de cada variable: asimetrias, concentraciones, colas largas y posibles outliers.

**Que ayuda a defender:**

- `duration_min` tiene cola larga.
- Algunas variables entre 0 y 1 no siguen distribucion normal.
- La estandarizacion es necesaria para PCA/t-SNE/K-Means porque las escalas son distintas.

**Hipotesis relacionada:** apoyo general a H1, H2 y H3, porque permite entender las variables antes de correlacionar o agrupar.

**Frase para defensa:**

> Los histogramas son el primer control visual de calidad: no explican una hipotesis por si solos, pero justifican por que despues escalamos y por que vigilamos outliers como la duracion.

### 7.2 `eda_boxplots_genre.png`

**Que muestra:** boxplots por genero para `energy`, `danceability`, `valence` y `popularity`.

**Por que se usa:** compara medianas, dispersion y outliers entre generos.

**Que ayuda a defender:**

- Hay diferencias entre generos.
- Tambien hay solapamiento interno.
- No basta con mirar medias: algunos generos tienen mucha variabilidad.

**Hipotesis relacionada:** H3.

**Frase para defensa:**

> Usamos boxplots porque resumen rapidamente como cambia una feature por genero y permiten ver si las diferencias son estables o si solo dependen de algunos valores extremos.

### 7.3 `eda_violin_genre.png`

**Que muestra:** distribucion completa por genero para `energy`, `danceability`, `valence` y `popularity`, combinando densidad y cuartiles.

**Por que se usa:** el violin plot enseña formas de distribucion que el boxplot puede ocultar, como bimodalidad o asimetria.

**Que ayuda a defender:**

- Algunos generos no tienen un unico perfil homogeneo.
- El genero es una etiqueta amplia, no una clase perfectamente separada.

**Hipotesis relacionada:** H3.

**Frase para defensa:**

> El violin complementa al boxplot: si el boxplot resume, el violin permite ver la forma real de la distribucion.

### 7.4 `eda_correlation_heatmap.png`

**Que muestra:** matriz de correlacion de Pearson entre las 11 features.

**Por que se usa:** identifica relaciones lineales y redundancias.

**Correlaciones principales:**

| Par de features | Correlacion |
|---|---:|
| `energy` - `loudness` | +0.8227 |
| `acousticness` - `energy` | -0.7294 |
| `acousticness` - `loudness` | -0.6953 |
| `danceability` - `valence` | +0.5791 |
| `liveness` - `speechiness` | +0.5425 |
| `instrumentalness` - `loudness` | -0.4971 |
| `danceability` - `loudness` | +0.4571 |
| `energy` - `valence` | +0.4455 |

**Que ayuda a defender:**

- `energy` y `loudness` no son independientes.
- `energy` y `acousticness` se oponen claramente.
- La popularidad no tiene correlacion fuerte con una unica variable.

**Hipotesis relacionada:** H1 y decisiones previas a PCA/H2.

**Frase para defensa:**

> El heatmap no solo busca relaciones interesantes; tambien sirve para evitar contar dos veces la misma informacion cuando hay variables redundantes.

### 7.5 `eda_scatter_high_corr.png`

**Que muestra:** dos scatter plots para las parejas mas redundantes:

- `energy` vs `loudness`
- `energy` vs `acousticness`

**Por que se usa:** el heatmap da el numero, el scatter confirma visualmente la forma de la relacion.

**Que ayuda a defender:**

- `energy` y `loudness` crecen juntas.
- `energy` y `acousticness` se mueven en sentidos opuestos.

**Hipotesis relacionada:** preparacion metodologica para PCA y clustering.

**Frase para defensa:**

> El scatter valida que la correlacion no es solo un numero aislado: vemos la nube de puntos y comprobamos la tendencia.

### 7.6 `eda_scatter_popularity.png`

**Que muestra:** relacion de `popularity` con `danceability`, `energy` y `valence`.

**Por que se usa:** son features musicalmente intuitivas para estudiar popularidad.

**Que ayuda a defender:**

- No hay una separacion clara entre canciones populares y no populares solo con estas variables.
- Sirve para matizar la hipotesis H1.

**Hipotesis relacionada:** H1.

**Frase para defensa:**

> La popularidad no queda explicada por una sola feature acustica. Hay tendencias suaves, pero no una relacion fuerte ni causal.

### 7.7 `eda_correlation_network.png`

**Que muestra:** red estatica de correlaciones. Nodos = features, aristas = correlaciones con `|r| >= 0.3`.

**Por que se usa:** convierte el heatmap en una lectura relacional mas directa.

**Que ayuda a defender:**

- Detectar rapidamente grupos de variables conectadas.
- Explicar redundancias sin mirar una matriz completa.

**Hipotesis relacionada:** H1 y seleccion de features.

**Diferencia con Plotly:** esta version es estatica en PNG; `correlation_network.py` genera la version interactiva HTML con Plotly.

---

## 8. PCA

Script:

- `src/pca_analysis.py`

Entrada:

- `X_scaled`: matriz estandarizada de 176.655 canciones x 11 features.

Salidas:

- `outputs/pca/figures/pca_scree.png`
- `outputs/pca/figures/pca_scatter_genre.png`
- `outputs/pca/figures/pca_biplot.png`
- `outputs/pca/pca_coords.csv`

### 8.1 Que es PCA en este proyecto

PCA reduce las 11 features originales a componentes principales que capturan la mayor varianza posible.

En vez de ver cada cancion en 11 dimensiones, la proyectamos a 2D o 3D:

- PC1: eje de maxima variacion.
- PC2: segundo eje, independiente de PC1.
- PC3: tercer eje, usado en CSV aunque no siempre se visualice.

PCA es lineal. Es bueno para explicar varianza global, pero puede no capturar estructuras locales complejas.

### 8.2 Varianza explicada

Resultados calculados sobre el dataset limpio:

| Componente | Varianza explicada | Acumulada |
|---|---:|---:|
| PC1 | 33.23% | 33.23% |
| PC2 | 16.23% | 49.46% |
| PC3 | 10.62% | 60.08% |
| PC4 | 8.94% | 69.02% |
| PC5 | 7.78% | 76.80% |
| PC6 | 6.61% | 83.42% |

Interpretacion:

- Con PC1 y PC2 se conserva casi la mitad de la informacion.
- Para superar el 80% hacen falta unos 6 componentes.
- Por tanto, el scatter 2D es una simplificacion util, no una representacion completa.

### 8.3 `pca_scree.png`

**Que muestra:** barras de varianza explicada por componente y linea acumulada.

**Por que se usa:** justifica cuanta informacion se pierde al proyectar a 2D.

**Hipotesis relacionada:** H2 y H3.

**Frase para defensa:**

> El scree plot nos permite ser honestos: PC1-PC2 sirven para visualizar, pero no contienen toda la estructura del dataset.

### 8.4 Interpretacion de los componentes

Loadings principales:

**PC1**

- `loudness`: +0.464
- `energy`: +0.449
- `acousticness`: -0.416
- `danceability`: +0.345
- `valence`: +0.338
- `instrumentalness`: -0.317

Lectura: PC1 separa canciones mas energeticas/sonoras/bailables de canciones mas acusticas/instrumentales.

**PC2**

- `speechiness`: +0.641
- `liveness`: +0.611
- `popularity`: -0.298
- `acousticness`: +0.206
- `instrumentalness`: -0.202

Lectura: PC2 captura principalmente presencia de habla/directo frente a canciones mas convencionales/populares.

**PC3**

- `duration_min`: +0.630
- `valence`: -0.429
- `danceability`: -0.407
- `liveness`: +0.246
- `energy`: +0.231

Lectura: PC3 esta muy influido por duracion y por contraste con positividad/bailabilidad.

### 8.5 `pca_scatter_genre.png`

**Que muestra:** canciones proyectadas en PC1-PC2, coloreadas por genero.

**Por que se usa:** comprobar si los generos se separan en un espacio reducido.

**Que ayuda a defender:**

- Algunos generos se desplazan hacia zonas concretas.
- Muchos generos se solapan.
- El genero no explica por si solo toda la estructura acustica.

**Hipotesis relacionada:** H3.

### 8.6 `pca_biplot.png`

**Que muestra:** scatter PCA con flechas de loadings de las features.

**Por que se usa:** permite interpretar los ejes.

Como leerlo:

- Flechas largas: variables que explican mucho el plano.
- Flechas parecidas/paralelas: variables correlacionadas.
- Flechas opuestas: variables negativamente correlacionadas.
- Puntos hacia una flecha: canciones con valores altos en esa feature.

**Hipotesis relacionada:** H2 y H3.

**Frase para defensa:**

> El biplot conecta la nube de puntos con las variables originales; sin el biplot, el PCA seria una proyeccion dificil de interpretar.

---

## 9. t-SNE

Script:

- `src/tsne_analysis.py`

Entrada:

- Muestra aleatoria de **10.000 canciones**.
- `X_scaled`.

Parametros:

- `SAMPLE_N = 10_000`
- `RANDOM_STATE = 42`
- `PERPLEXITY = 40`
- `MAX_ITER = 1000`

Salidas:

- `outputs/tsne/figures/tsne_genre.png`
- `outputs/tsne/figures/tsne_popularity.png`
- `outputs/tsne/tsne_coords.csv`

### 9.1 Que es t-SNE en este proyecto

t-SNE proyecta datos de alta dimension a 2D preservando relaciones locales. A diferencia del PCA, no busca explicar varianza global, sino que canciones parecidas queden cerca.

Importante para defensa:

- t-SNE no sirve para interpretar ejes como PC1/PC2.
- Las distancias locales son mas fiables que las globales.
- La forma exacta puede cambiar con parametros y semilla.
- Se usa muestra de 10.000 por coste computacional.

### 9.2 `tsne_genre.png`

**Que muestra:** proyeccion t-SNE coloreada por genero.

**Por que se usa:** detectar agrupaciones no lineales y mezcla entre generos.

**Que ayuda a defender:**

- Si hay zonas dominadas por generos, indica perfiles acusticos similares.
- Si hay mezcla, indica que el genero declarado no separa completamente las canciones.

**Hipotesis relacionada:** H3.

**Frase para defensa:**

> t-SNE nos ayuda a ver vecindarios de canciones similares; no interpretamos sus ejes, interpretamos proximidades locales.

### 9.3 `tsne_popularity.png`

**Que muestra:** misma proyeccion t-SNE, pero coloreada por `popularity`.

**Por que se usa:** comprobar si las canciones populares se concentran en zonas acusticas concretas.

**Que ayuda a defender:**

- Si la popularidad aparece dispersa, no depende solo del perfil acustico.
- Si hay zonas mas populares, pueden relacionarse con perfiles mainstream.

**Hipotesis relacionada:** H1.

---

## 10. Clustering K-Means

Script:

- `src/clustering.py`

Entrada:

- `X_scaled`: features estandarizadas.
- `outputs/tsne/tsne_coords.csv` para visualizar clusters sobre t-SNE.

Salidas:

- `outputs/clustering/figures/clustering_elbow.png`
- `outputs/clustering/figures/clustering_silhouette.png`
- `outputs/clustering/figures/clustering_tsne.png`
- `outputs/clustering/cluster_labels.csv`
- `outputs/clustering/cluster_multi_k.csv`
- `outputs/clustering/cluster_profiles.csv`

### 10.1 Por que K-Means

K-Means agrupa canciones minimizando distancia interna dentro de cada cluster. Es adecuado como primera aproximacion porque:

- Es simple de explicar.
- Funciona con variables numericas estandarizadas.
- Genera centroides interpretables.
- Permite comparar distintos valores de `k`.

Limitacion:

- Asume clusters aproximadamente esfericos en el espacio de features.
- Es sensible a escala, por eso se usa `StandardScaler`.
- Los ids numericos de clusters son arbitrarios.

### 10.2 Seleccion de k

El script evalua:

- `K_RANGE = range(2, 13)`
- Inercia para elbow method.
- Silhouette score para separacion/cohesion.

Ademas exporta `cluster_multi_k.csv` para el slider del dashboard con:

- `k3`
- `k5`
- `k7`

### 10.3 `clustering_elbow.png`

**Que muestra:** inercia para k=2..12.

**Por que se usa:** busca el punto donde aumentar k deja de mejorar mucho la compactacion.

**Hipotesis relacionada:** H2.

**Frase para defensa:**

> El elbow no da una verdad absoluta; es una ayuda visual para elegir un k razonable equilibrando simplicidad e inercia.

### 10.4 `clustering_silhouette.png`

**Que muestra:** silhouette score por k.

**Por que se usa:** mide si los clusters son compactos y separados.

**Hipotesis relacionada:** H2.

**Frase para defensa:**

> Usamos silhouette como criterio complementario al elbow, porque la inercia siempre baja al aumentar k.

### 10.5 `clustering_tsne.png`

**Que muestra:** puntos t-SNE coloreados segun cluster K-Means.

**Por que se usa:** visualizar si los clusters tienen coherencia espacial en una proyeccion no lineal.

**Cuidado importante:**

K-Means se entrena sobre las features estandarizadas, no sobre t-SNE. t-SNE solo sirve para visualizar.

**Frase para defensa:**

> El clustering no se calcula en el plano t-SNE; lo proyectamos ahi para poder verlo.

### 10.6 Perfiles de clusters k=3

Archivo:

- `outputs/clustering/cluster_profiles.csv`
- `config/cluster_profiles.json`

Tamanos:

| Cluster | Tamano | Interpretacion |
|---|---:|---|
| C0 | 10.253 | Spoken / Live |
| C1 | 44.725 | Ambient / Instrumental |
| C2 | 121.677 | Mainstream Pop / Rock |

Centroides principales:

| Cluster | Popularity | Acousticness | Danceability | Energy | Instrumentalness | Liveness | Speechiness | Loudness | Tempo | Valence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 20.34 | 0.7926 | 0.5618 | 0.6591 | 0.0015 | 0.7286 | 0.8678 | -12.03 | 98.18 | 0.4159 |
| C1 | 27.20 | 0.8361 | 0.3530 | 0.2006 | 0.4687 | 0.1667 | 0.0511 | -18.12 | 105.50 | 0.2138 |
| C2 | 40.98 | 0.2122 | 0.6086 | 0.6797 | 0.0776 | 0.2034 | 0.0931 | -7.04 | 123.12 | 0.5421 |

Interpretacion:

- **C0 Spoken / Live:** speechiness y liveness muy altos, poca instrumentalness, popularidad baja. Puede incluir comedy, spoken, live o contenido performativo.
- **C1 Ambient / Instrumental:** alta acousticness, baja energy/danceability, instrumentalness relativamente alta, loudness bajo. Perfil tranquilo/instrumental.
- **C2 Mainstream Pop / Rock:** mayor popularity, energy, danceability, loudness y valence; baja acousticness. Perfil de cancion mainstream.

### 10.7 Por que las etiquetas estan en JSON

Archivo:

- `config/cluster_profiles.json`

Razon:

- Los ids `C0`, `C1`, `C2` vienen de K-Means y son arbitrarios.
- Los nombres humanos son interpretaciones de centroides.
- Si se reentrena K-Means, los ids pueden cambiar.
- Separar configuracion del frontend evita hardcoding.

Frase para defensa:

> Los clusters no nacen con nombre; los nombramos despues leyendo los centroides. Por eso las etiquetas estan en configuracion externa y no incrustadas en el HTML.

---

## 11. Red de correlaciones con Plotly

Script:

- `src/correlation_network.py`

Salida:

- `outputs/eda/figures/correlation_network.html`
- `outputs/eda/correlation_matrix.csv`

### 11.1 Que hace el script

Construye una red interactiva de correlaciones Pearson entre las 11 features:

- Nodos: features.
- Aristas: correlaciones con `|r| >= 0.3`.
- Color verde: correlacion positiva.
- Color rojo: correlacion negativa.
- Grosor: fuerza de la correlacion.
- Opacidad: fuerza de la correlacion.
- Etiqueta: valor de `r`.
- Hover en nodo: lista de correlaciones ordenadas por magnitud.

### 11.2 Por que Plotly

Plotly se usa porque permite:

- Figura HTML interactiva.
- Hover explicativo.
- Visualizacion sin recalcular datos.
- Integracion directa en dashboard.
- Lectura mas amigable que una matriz 11x11.

### 11.3 Decisiones del codigo

Archivo de entrada:

- `data/processed/spotify_clean_sample.csv`

Razon comentada en el codigo:

- Pearson sobre 20.000 tracks es mas robusto que sobre 26 medias por genero.

Umbral:

- `R_THRESHOLD = 0.3`

Razon:

- Si se dibujan todas las correlaciones, la red se satura.
- El umbral deja solo relaciones interpretables.

Layout:

- Circular y fijo.

Razon:

- Evita que el layout cambie en cada ejecucion.
- Facilita comparar y explicar.

### 11.4 Como defender esta grafica

Respuesta corta:

> La red de correlaciones resume el heatmap en forma interactiva. No reemplaza la matriz, pero facilita ver que variables tiran juntas o en direcciones opuestas. Las aristas mas importantes son energy-loudness, acousticness-energy y danceability-valence.

Posible pregunta:

**Por que Pearson?**

Respuesta:

> Porque buscamos relaciones lineales entre features numericas como primera aproximacion. Sabemos que no captura relaciones no lineales, por eso complementamos con t-SNE y clustering.

---

## 12. Dashboard y API, solo lo necesario

El frontend no es el centro de la validacion, pero hay que entender como se conectan datos y visualizaciones.

Arquitectura:

```text
src/*.py
  -> outputs/*.csv + config/*.json
  -> src/dashboard_api.py
  -> dashboard/index.html + dashboard/dashboard-boot.js
```

### 12.1 `src/dashboard_api.py`

Carga una vez al arrancar:

- `spotify_clean.csv`
- `tsne_coords.csv`
- `pca_coords.csv`
- `cluster_multi_k.csv`
- `cluster_profiles.csv`
- `genre_profiles.csv`
- `correlation_matrix.csv`
- `cluster_profiles.json`
- `playlist_presets.json`

Endpoints:

- `/api/kpis`
- `/api/tracks`
- `/api/tsne`
- `/api/pca`
- `/api/clusters?k=3|5|7`
- `/api/cluster-profiles?k=3|5|7`
- `/api/genre-profiles`
- `/api/correlation`
- `/api/presets`

Idea clave:

> El navegador no recalcula PCA, t-SNE ni K-Means. Solo visualiza outputs ya generados por Python.

### 12.2 `dashboard/dashboard-boot.js`

Hace de puente entre la API y el bundle del dashboard:

- Hace `fetch` de los endpoints.
- Convierte nombres largos a claves cortas (`track_id -> tid`, `acousticness -> ac`, etc.).
- Prepara muestras para que Plotly no renderice demasiados puntos.
- Inserta los datos reales en `window.DATA`.
- Inicializa los modulos visuales.

Esto es importante porque evita duplicar datos en HTML y permite regenerar outputs sin tocar el frontend.

### 12.3 Builder de playlists

Archivo:

- `config/playlist_presets.json`

Funcion:

- Define perfiles manuales: `Late Night Focus`, `Sunday Morning`, `Workout Peak`, etc.
- Cada preset tiene un vector objetivo.
- Las canciones se ordenan por distancia L2 ponderada.

Importante:

- No son clusters descubiertos por el modelo.
- Son perfiles curados para interaccion.
- Esta separacion evita vender como "descubierto" algo que es una decision de diseno.

Frase para defensa:

> Los presets del Builder son perfiles definidos manualmente; los clusters, en cambio, salen de K-Means. Los mantenemos separados para no mezclar analisis con curacion.

---

## 13. Resumen de resultados numericos importantes

### 13.1 Dataset

| Metrica | Valor |
|---|---:|
| Filas raw | 232.725 |
| Columnas raw | 18 |
| Duplicados por `track_id` eliminados | 55.951 |
| Filas tras deduplicar | 176.774 |
| Generos eliminados por bajo soporte | 1 (`A Capella`, 119 canciones) |
| Filas finales | 176.655 |
| Columnas finales | 15 |
| Generos finales | 26 |
| Popularidad media | 36.29 |

### 13.2 Generos con mayor popularidad media

| Genero | Popularidad media |
|---|---:|
| Pop | 67.06 |
| Rap | 59.52 |
| Rock | 58.77 |
| Hip-Hop | 58.52 |
| Dance | 57.35 |
| Indie | 53.53 |
| Children's Music con apostrofo curvo | 52.30 |
| Alternative | 50.26 |

Nota: el dataset contiene dos etiquetas parecidas, `Children's Music` y `Children’s Music`, con distinto apostrofo. Esto es una posible limitacion de calidad de datos.

### 13.3 Extremos por genero en medias

| Feature | Minimo por genero | Maximo por genero |
|---|---|---|
| `popularity` | Children's Music = 4.246 | Pop = 67.065 |
| `acousticness` | Ska = 0.088 | Opera = 0.945 |
| `danceability` | Soundtrack = 0.262 | Reggaeton = 0.730 |
| `duration_min` | Children's Music = 2.379 | World = 5.273 |
| `energy` | Opera = 0.169 | Ska = 0.837 |
| `instrumentalness` | Comedy = 0.001 | Soundtrack = 0.790 |
| `liveness` | Soundtrack = 0.137 | Comedy = 0.725 |
| `loudness` | Classical = -21.752 | Reggaeton = -5.910 |
| `speechiness` | Soundtrack = 0.044 | Comedy = 0.854 |
| `tempo` | Comedy = 98.216 | Ska = 130.813 |
| `valence` | Soundtrack = 0.113 | Reggae = 0.680 |

---

## 14. Como estructurar el portfolio final

Las plantillas de `plantillas-docs-projecteVD/Word` sugieren una estructura tipo articulo:

- Titulo
- Autores
- Resumen / Abstract
- Palabras clave / Index Terms
- Introduccion
- Secciones de metodo y resultados
- Conclusion
- Agradecimientos si procede
- Bibliografia
- Apendice si hace falta

Propuesta adaptada al proyecto:

### 14.1 Titulo

Ejemplos:

- **Featurefy: Visualizacion de patrones acusticos en Spotify**
- **Spotify Features: analisis visual de popularidad, generos y perfiles acusticos**
- **Explorando Spotify mediante visualizacion de datos y clustering acustico**

### 14.2 Resumen

Debe incluir en pocas lineas:

- Dataset usado.
- Objetivo.
- Tecnicas: EDA, correlaciones, PCA, t-SNE, K-Means, Plotly.
- Resultado principal: patrones por genero, clusters interpretables y popularidad solo moderadamente relacionada con features.

Borrador:

> Este proyecto explora un dataset de Spotify Features con 176.655 canciones procesadas y 11 variables acusticas. El objetivo es analizar si las caracteristicas musicales permiten explicar diferencias entre generos, relacionarse con la popularidad y descubrir perfiles de canciones mediante clustering. Tras un proceso de limpieza, estandarizacion y analisis exploratorio, se generan visualizaciones de distribucion, correlacion, PCA, t-SNE y K-Means. Los resultados muestran redundancias claras entre energy, loudness y acousticness, diferencias acusticas entre algunos generos y tres perfiles principales de canciones: Spoken/Live, Ambient/Instrumental y Mainstream Pop/Rock.

### 14.3 Palabras clave

- Spotify Features
- Visualizacion de datos
- PCA
- t-SNE
- K-Means
- Correlacion
- Musica
- Data storytelling

### 14.4 Introduccion

Debe responder:

- Que problema se estudia.
- Por que Spotify Features es interesante.
- Que hipotesis guian el trabajo.
- Que se entrega: pipeline + outputs + dashboard.

### 14.5 Dataset y data massage

Incluir:

- Dataset raw y procesado.
- Columnas.
- Eliminacion de duplicados.
- Imputacion.
- Conversion de duracion.
- Filtrado de `A Capella`.
- Estandarizacion.
- Riesgos metodologicos.

Grafica o tabla recomendada:

- Tabla de columnas.
- Tabla raw vs clean.
- Resumen de decisiones.

### 14.6 Metodologia visual y analitica

Dividir por bloques:

1. EDA y distribuciones.
2. Correlaciones.
3. PCA.
4. t-SNE.
5. K-Means.
6. Red Plotly.
7. Dashboard/API como capa de consumo.

### 14.7 Resultados y discusion

Orden recomendado:

1. Distribuciones: entender datos.
2. Correlaciones: redundancias y popularidad.
3. Generos: diferencias y solapamientos.
4. PCA/t-SNE: estructura global/local.
5. Clustering: perfiles interpretables.
6. Dashboard: uso interactivo de resultados.

### 14.8 Conclusiones

Ideas clave:

- El dataset permite estudiar patrones acusticos, pero no explica toda la popularidad.
- Algunas features son redundantes.
- Los generos tienen perfiles, pero no separacion perfecta.
- K-Means produce perfiles utiles e interpretables.
- El dashboard convierte outputs analiticos en exploracion interactiva.

### 14.9 Limitaciones

Incluir explicitamente:

- Deduplicacion por `track_id` elimina relaciones multi-genero.
- Duraciones extremas no se recortan.
- Pearson solo captura relaciones lineales.
- t-SNE depende de parametros y muestra.
- K-Means impone una geometria de clusters.
- Popularidad en Spotify puede depender de factores externos no incluidos: marketing, artista, fecha, playlists, contexto social.
- Diferencia entre `Children's Music` y `Children’s Music` por apostrofo.

---

## 15. Preguntas probables en la validacion

### Por que habeis elegido este dataset?

Porque combina identificadores de canciones con descriptores acusticos numericos. Eso permite formular hipotesis visuales sobre popularidad, generos y agrupaciones. Ademas, el grupo ya lo conocia de Aprenentatge Computacional, lo que facilitaba entender las variables y centrarse en visualizacion.

### Por que escalais las features?

Porque PCA, t-SNE y K-Means son sensibles a escala. Sin escalado, variables como `tempo`, `loudness` o `popularity` tendrian mas peso numerico que variables entre 0 y 1 como `energy` o `danceability`.

### Por que no escalais el CSV limpio?

Porque el CSV limpio debe conservar valores interpretables. El escalado es una transformacion para modelado y se devuelve en memoria como `X_scaled`.

### Por que eliminasteis duplicados por `track_id`?

Para que una misma cancion no cuente varias veces en modelos de distancia/varianza. Es una simplificacion metodologica con coste: se pierde informacion de generos multiples.

### Por que usais Pearson?

Porque es una primera medida clara de relacion lineal entre variables numericas. Se complementa con scatter plots, PCA, t-SNE y clustering para no depender solo de correlaciones lineales.

### Que significa que `energy` y `loudness` correlacionen +0.82?

Que canciones mas energeticas tienden a tener mayor volumen medio. No son la misma variable, pero contienen informacion muy parecida, por eso hay que evitar interpretarlas como independientes.

### Por que `loudness` queda fuera de algunas interpretaciones?

Porque es muy redundante con `energy`. Mantener ambas en todos los controles puede duplicar informacion. En el clustering/dashboard se priorizan features mas interpretables para el usuario.

### PCA y t-SNE hacen lo mismo?

No. PCA es lineal y explica varianza global; t-SNE es no lineal y preserva vecindarios locales. PCA es mas interpretable por ejes; t-SNE es mas util para ver agrupaciones locales.

### Por que t-SNE solo usa 10.000 canciones?

Por coste computacional. El dataset completo tiene 176.655 canciones y t-SNE es caro. Una muestra con semilla fija permite reproducibilidad y visualizacion viable.

### Los clusters son generos?

No. Son perfiles acusticos derivados de las features. Pueden coincidir parcialmente con generos, pero no son etiquetas musicales oficiales.

### Por que tres clusters?

Porque el analisis de K-Means con elbow/silhouette da una solucion interpretable y simple. Ademas, los centroides resultantes tienen sentido musical: Spoken/Live, Ambient/Instrumental y Mainstream Pop/Rock.

### Por que los nombres de clusters son interpretativos?

K-Means solo devuelve ids numericos. Los nombres se asignan leyendo centroides. Por eso estan en `config/cluster_profiles.json` y no hardcodeados en el HTML.

### Que aporta Plotly frente al heatmap?

Interactividad. Permite hover, lectura por nodos y exploracion de relaciones sin mirar toda la matriz.

### Que pasa si regenerais t-SNE o K-Means?

Los CSV de `outputs/` cambian. La API los recarga al arrancar y el dashboard consume los nuevos datos sin modificar el HTML.

---

## 16. Mapa rapido: script -> input -> output -> finalidad

| Script | Input principal | Output | Finalidad |
|---|---|---|---|
| `data_massage.py` | `data/raw/SpotifyFeatures.csv` | `spotify_clean.csv`, report | Limpieza, validacion, escalado |
| `eda_visual.py` | `prepare_dataset()` | figuras EDA, `genre_profiles.csv` | Distribuciones, generos, correlaciones |
| `pca_analysis.py` | `X_scaled` | figuras PCA, `pca_coords.csv` | Reduccion lineal e interpretacion de ejes |
| `tsne_analysis.py` | muestra de `X_scaled` | figuras t-SNE, `tsne_coords.csv` | Vecindarios y estructura local |
| `clustering.py` | `X_scaled`, t-SNE CSV | figuras clustering, labels, perfiles | Agrupar canciones por perfil acustico |
| `correlation_network.py` | sample limpio | HTML Plotly, matriz corr | Red interactiva de correlaciones |
| `dashboard_api.py` | CSV/JSON de outputs/config | endpoints Flask | Servir datos al dashboard |

---

## 17. Orden de ejecucion del pipeline

Desde la raiz del repo:

```bash
python src/data_massage.py
python src/eda_visual.py
python src/pca_analysis.py
python src/tsne_analysis.py
python src/clustering.py
python src/correlation_network.py
python src/dashboard_api.py
```

Nota:

- `clustering.py` debe ejecutarse despues de `tsne_analysis.py` si se quiere generar `clustering_tsne.png`.
- El dashboard se abre en `http://127.0.0.1:5001/`.

---

## 18. Guion breve para exposicion oral

1. **Contexto:** elegimos Spotify Features porque contiene descriptores acusticos numericos adecuados para visualizacion y analisis.
2. **Hipotesis:** popularidad vs features, diferencias por genero y existencia de clusters acusticos.
3. **Limpieza:** eliminamos duplicados por `track_id`, imputamos un nulo, convertimos duracion, filtramos `A Capella` y estandarizamos para modelos.
4. **EDA:** estudiamos distribuciones, outliers y perfiles por genero.
5. **Correlaciones:** detectamos redundancias fuertes: `energy-loudness` y `energy-acousticness`.
6. **PCA:** reducimos 11 dimensiones a componentes interpretables; PC1 separa energia/loudness de acousticness/instrumentalness.
7. **t-SNE:** usamos una muestra de 10.000 para visualizar vecindarios no lineales.
8. **K-Means:** obtenemos perfiles acusticos interpretables, no generos oficiales.
9. **Plotly/API:** convertimos resultados analiticos en exploracion interactiva sin recalcular en navegador.
10. **Conclusion:** el proyecto muestra patrones acusticos claros, pero tambien limitaciones: popularidad no depende solo de features y el genero no separa perfectamente las canciones.

---

## 19. Checklist final para el grupo

Antes de la validacion, cada miembro deberia poder explicar:

- Que columnas tiene el dataset raw y el limpio.
- Por que se eliminan duplicados.
- Por que se escala.
- Que correlaciones principales se encontraron.
- Que significa PC1 y PC2.
- Diferencia entre PCA y t-SNE.
- Que significa cada cluster.
- Por que Plotly se usa en la red de correlaciones.
- Que outputs consume el dashboard.
- Que limitaciones metodologicas hay.

Frase final para recordar:

> El proyecto no intenta demostrar que el genero o la popularidad se expliquen perfectamente con features acusticas; intenta visualizar que patrones aparecen, que relaciones son fuertes, donde hay solapamiento y como convertir ese analisis en una herramienta explorable.
