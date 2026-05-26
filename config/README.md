# Configuració del dashboard

Aquesta carpeta conté metadades explícites per a les etiquetes del dashboard que no haurien de viure com a text fixat al frontend.

## `cluster_profiles.json`

Etiquetes derivades de les dades per als perfils de clúster de K-means utilitzats a la secció `03 · Space`.

Els identificadors dels clústers (`C0`, `C1`, `C2`) provenen de la sortida de K-means, però les etiquetes llegibles per humans són interpretatives. S'assignen llegint els valors de les features dels centroides.

Interpretació actual:

- `C0` -> `Spoken / Live`
- `C1` -> `Ambient / Instrumental`
- `C2` -> `Mainstream Pop / Rock`

Important: si es torna a entrenar K-means, els identificadors numèrics dels clústers poden canviar i aquestes etiquetes s'haurien de regenerar.

## `playlist_presets.json`

Perfils objectiu curats per a la secció `01 · Builder`.

Aquests presets no són clústers descoberts. Són objectius acústics predefinits utilitzats per inicialitzar els sliders i ordenar les cançons per distància ponderada. Això manté els títols de les playlists explicables sense fingir que s'han inferit del dataset.

El flux de recomanació és:

```text
títol del preset -> vector de features objectiu -> distància L2 ponderada -> tracks més properes
```

Per a un mode més estricte derivat de les dades, crea presets addicionals a partir dels centroides dels clústers o dels perfils mitjans per gènere i marca'n el `source` com a `cluster_centroid` o `genre_mean`.
