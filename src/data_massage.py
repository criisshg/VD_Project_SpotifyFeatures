"""
data_massage.py

Pipeline de preparació del dataset Spotify Features.

Substitueix l'antic data_prep.py i fa:
- Carrega del CSV original (raw, mai modificat).
- Informe d'inspecció: tipus, nuls, duplicats, descripció, recompte per gènere.
- Comprovació de coherència de rangs de les variables numèriques.
- Neteja: eliminació de duplicats per track_id i imputació de nuls
  (mitjana per a numèriques, mode per a categòriques).
- Conversió de duration_ms -> duration_min (unitat interpretable).
- Filtratge de gèneres minoritaris (drop de files segons MIN_GENRE_COUNT).
- Informe de correlacions per detectar redundàncies abans de PCA / t-SNE.
- Estandardització (StandardScaler) de les features numèriques.

L'output és un CSV net (data/processed/spotify_clean.csv) amb valors
interpretables sense escalar; l'escalat es retorna en memòria per als
scripts de modelatge (PCA, t-SNE, clustering).
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. Configuració
# -----------------------------

RAW_DATA_PATH = os.path.join("data", "raw", "SpotifyFeatures.csv")
CLEAN_DATA_PATH = os.path.join("data", "processed", "spotify_clean.csv")

# Llindar mínim de cançons per gènere (els gèneres amb menys s'eliminen).
MIN_GENRE_COUNT = 1000

INFO_COLUMNS = [
    "genre",
    "artist_name",
    "track_name",
    "track_id",
]

# duration_min substitueix duration_ms després de convert_duration().
FEATURE_COLUMNS = [
    "popularity",
    "acousticness",
    "danceability",
    "duration_min",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]

# Rangs esperats per a la comprovació de coherència. Validació, no clipping.
EXPECTED_RANGES = {
    "popularity":       (0, 100),
    "acousticness":     (0, 1),
    "danceability":     (0, 1),
    "energy":           (0, 1),
    "instrumentalness": (0, 1),
    "liveness":         (0, 1),
    "speechiness":      (0, 1),
    "valence":          (0, 1),
    "loudness":         (-60, 5),
    "tempo":            (0, 250),
    "duration_min":     (0, 30),
}


# -----------------------------
# 2. Càrrega
# -----------------------------

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Carrega el CSV original. Falla si no existeix."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No he trobat el fitxer a: {path}")
    return pd.read_csv(path)


# -----------------------------
# 3. Informe d'inspecció
# -----------------------------

def inspect_dataset(df: pd.DataFrame) -> None:
    """Imprimeix un informe textual del dataset. No el modifica."""
    print("=" * 60)
    print("INSPECCIÓ DEL DATASET")
    print("=" * 60)

    print(f"\nForma: {df.shape[0]} files × {df.shape[1]} columnes")

    print("\nTipus de dades:")
    print(df.dtypes)

    print("\nNuls per columna:")
    print(df.isnull().sum())

    n_dup_total = df.duplicated().sum()
    print(f"\nDuplicats (fila sencera): {n_dup_total}")

    if "track_id" in df.columns:
        n_dup_track = df.duplicated(subset="track_id").sum()
        print(f"Duplicats per track_id: {n_dup_track} "
              "(la mateixa cançó pot aparèixer en diversos gèneres)")

    print("\nEstadístiques numèriques:")
    print(df.describe().round(3))

    if "genre" in df.columns:
        print("\nRecompte de cançons per gènere (ascendent):")
        print(df["genre"].value_counts().sort_values(ascending=True))


# -----------------------------
# 4. Comprovació de rangs
# -----------------------------

def check_ranges(df: pd.DataFrame, expected: dict = EXPECTED_RANGES) -> None:
    """
    Per a cada columna del diccionari, imprimeix [OK] o [WARN] segons si
    min/max del DataFrame queden dins del rang esperat.
    """
    print("\n" + "=" * 60)
    print("COMPROVACIÓ DE RANGS")
    print("=" * 60)

    for col, (lo, hi) in expected.items():
        if col not in df.columns:
            print(f"[SKIP] {col}: no és al DataFrame encara")
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min < lo or col_max > hi:
            print(f"[WARN] {col}: observat [{col_min:.3f}, {col_max:.3f}] "
                  f"fora de l'esperat [{lo}, {hi}]")
        else:
            print(f"[OK]   {col}: [{col_min:.3f}, {col_max:.3f}] dins de [{lo}, {hi}]")


# -----------------------------
# 5. Neteja
# -----------------------------

def drop_duplicates_and_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicats per track_id (mantenim la primera ocurrència) i
    imputa nuls: mitjana per a numèriques, mode per a categòriques.
    """
    print("\n" + "=" * 60)
    print("NETEJA: DUPLICATS I NULS")
    print("=" * 60)

    n_before = len(df)
    df = df.drop_duplicates(subset="track_id", keep="first").copy()
    print(f"Eliminats {n_before - len(df)} duplicats per track_id "
          f"({n_before} -> {len(df)} files).")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            df[col] = df[col].fillna(df[col].mean())
            print(f"  Imputats {n_null} nuls a '{col}' amb la mitjana.")

    for col in categorical_cols:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"  Imputats {n_null} nuls a '{col}' amb el mode.")

    return df.reset_index(drop=True)


# -----------------------------
# 6. Conversió de duration_ms
# -----------------------------

def convert_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Afegeix duration_min i elimina duration_ms."""
    df = df.copy()
    df["duration_min"] = df["duration_ms"] / 60_000.0
    df = df.drop(columns=["duration_ms"])
    return df


# -----------------------------
# 7. Filtratge de gèneres minoritaris
# -----------------------------

def filter_minority_genres(df: pd.DataFrame,
                           min_count: int = MIN_GENRE_COUNT) -> pd.DataFrame:
    """Elimina les files dels gèneres amb menys de min_count cançons."""
    print("\n" + "=" * 60)
    print(f"FILTRATGE DE GÈNERES (min_count = {min_count})")
    print("=" * 60)

    counts = df["genre"].value_counts()
    kept = counts[counts >= min_count].index
    dropped = counts[counts < min_count]

    if len(dropped) == 0:
        print("Cap gènere descartat: tots superen el llindar.")
    else:
        print(f"Gèneres descartats ({len(dropped)}):")
        for genre, n in dropped.items():
            print(f"  - {genre}: {n} cançons")
        n_rows_dropped = int(dropped.sum())
        print(f"Total de files eliminades: {n_rows_dropped}")

    return df[df["genre"].isin(kept)].reset_index(drop=True)


# -----------------------------
# 8. Informe de correlacions
# -----------------------------

def correlation_report(df: pd.DataFrame, threshold: float = 0.7) -> None:
    """Imprimeix la matriu de correlacions i les parelles amb |corr| >= threshold."""
    print("\n" + "=" * 60)
    print("CORRELACIONS ENTRE FEATURES")
    print("=" * 60)

    corr = df[FEATURE_COLUMNS].corr()
    print("\nMatriu de correlacions (Pearson):")
    print(corr.round(2))

    print(f"\nParelles amb |corr| >= {threshold}:")
    pairs_found = False
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                print(f"  {cols[i]:<18} <-> {cols[j]:<18}  corr = {value:+.3f}")
                pairs_found = True
    if not pairs_found:
        print("  (cap parella supera el llindar)")


# -----------------------------
# 9. Escalat
# -----------------------------

def scale_features(features_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Estandarditza les features numèriques (mitjana 0, desviació 1)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df.values)
    return X_scaled, scaler


# -----------------------------
# 10. Pipeline end-to-end
# -----------------------------

def prepare_dataset(raw_path: str = RAW_DATA_PATH,
                    clean_path: str = CLEAN_DATA_PATH,
                    min_genre_count: int = MIN_GENRE_COUNT
                    ) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Executa el pipeline complet i retorna:
    - df_clean: DataFrame net amb valors interpretables (sense escalar).
    - X_scaled: matriu numpy de features estandarditzades.
    - scaler:   StandardScaler entrenat (per reescalar dades futures).
    """
    df_raw = load_raw_data(raw_path)
    inspect_dataset(df_raw)

    df = drop_duplicates_and_nulls(df_raw)
    df = convert_duration(df)
    check_ranges(df)
    df = filter_minority_genres(df, min_count=min_genre_count)

    df_clean = df[INFO_COLUMNS + FEATURE_COLUMNS].copy()
    correlation_report(df_clean)

    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df_clean.to_csv(clean_path, index=False)
    print(f"\nCSV net guardat a: {clean_path}")

    X_scaled, scaler = scale_features(df_clean[FEATURE_COLUMNS])
    return df_clean, X_scaled, scaler


# -----------------------------
# 11. Execució directa
# -----------------------------

if __name__ == "__main__":
    df_clean, X_scaled, scaler = prepare_dataset()

    print("\n" + "=" * 60)
    print("RESUM FINAL")
    print("=" * 60)
    print(f"Forma del DataFrame net: {df_clean.shape}")
    print(f"Forma de X_scaled:       {X_scaled.shape}")
