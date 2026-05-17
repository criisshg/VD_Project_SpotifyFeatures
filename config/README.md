# Dashboard configuration

This folder contains explicit metadata for dashboard labels that should not live as hard-coded text inside the frontend.

## `cluster_profiles.json`

Data-derived labels for the K-means cluster profiles used in section `03 · Space`.

The cluster ids (`C0`, `C1`, `C2`) come from the K-means output, but the human-readable labels are interpretive. They are assigned by reading the centroid feature values.

Current interpretation:

- `C0` -> `Spoken / Live`
- `C1` -> `Ambient / Instrumental`
- `C2` -> `Mainstream Pop / Rock`

Important: if K-means is retrained, the numeric cluster ids may change and these labels should be regenerated.

## `playlist_presets.json`

Curated target profiles for section `01 · Builder`.

These presets are not discovered clusters. They are predefined acoustic targets used to initialize the sliders and rank songs by weighted distance. This keeps the playlist titles explainable without pretending they were inferred from the dataset.

The recommendation flow is:

```text
preset title -> target feature vector -> weighted L2 distance -> closest tracks
```

For a stricter data-derived mode, create additional presets from cluster centroids or genre mean profiles and mark their `source` as `cluster_centroid` or `genre_mean`.
