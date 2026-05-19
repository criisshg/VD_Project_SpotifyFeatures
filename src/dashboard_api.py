"""
dashboard_api.py

API Flask lleugera que serveix els CSV / JSON ja generats pel pipeline
i el dashboard HTML estàtic. Endpoints definits al pla:
/Users/crishg/.claude/plans/dame-los-pasos-para-noble-gosling.md
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_compress import Compress


ROOT = Path(__file__).resolve().parent.parent

DASHBOARD_DIR = ROOT / "dashboard"

CSV_TRACKS = ROOT / "data" / "processed" / "spotify_clean.csv"
CSV_TSNE = ROOT / "outputs" / "tsne" / "tsne_coords.csv"
CSV_PCA = ROOT / "outputs" / "pca" / "pca_coords.csv"
CSV_CLUSTERS = ROOT / "outputs" / "clustering" / "cluster_multi_k.csv"
CSV_GENRE_PROFILES = ROOT / "outputs" / "eda" / "genre_profiles.csv"
CSV_CORRELATION = ROOT / "outputs" / "eda" / "correlation_matrix.csv"
JSON_CLUSTER_PROFILES = ROOT / "config" / "cluster_profiles.json"
JSON_PRESETS = ROOT / "config" / "playlist_presets.json"


def _records(df: pd.DataFrame) -> list:
    return df.replace({np.nan: None}).to_dict(orient="records")


def _load_all() -> dict:
    """Carrega tots els fitxers una sola vegada a l'arrencada."""
    tracks = pd.read_csv(CSV_TRACKS)
    tsne = pd.read_csv(CSV_TSNE)
    pca = pd.read_csv(CSV_PCA)
    clusters = pd.read_csv(CSV_CLUSTERS)
    genre_profiles = pd.read_csv(CSV_GENRE_PROFILES)
    correlation = pd.read_csv(CSV_CORRELATION, index_col=0)

    with open(JSON_CLUSTER_PROFILES, encoding="utf-8") as f:
        cluster_profiles = json.load(f)
    with open(JSON_PRESETS, encoding="utf-8") as f:
        presets = json.load(f)

    top_idx = tracks["popularity"].idxmax()
    top_track = tracks.loc[top_idx]
    kpis = {
        "tracks": int(len(tracks)),
        "genres": int(tracks["genre"].nunique()),
        "features": 11,
        "avg_popularity": round(float(tracks["popularity"].mean()), 2),
        "top_genre": tracks["genre"].value_counts().idxmax(),
        "top_track": {
            "track_id": top_track["track_id"],
            "track_name": top_track["track_name"],
            "artist_name": top_track["artist_name"],
            "popularity": int(top_track["popularity"]),
            "genre": top_track["genre"],
        },
    }

    return {
        "tracks": tracks,
        "tsne": tsne,
        "pca": pca,
        "clusters": clusters,
        "genre_profiles": genre_profiles,
        "correlation": correlation,
        "cluster_profiles": cluster_profiles,
        "presets": presets,
        "kpis": kpis,
    }


print(f"[dashboard_api] Carregant dades des de {ROOT}...")
STORE = _load_all()
print(f"[dashboard_api] OK: {STORE['kpis']['tracks']} tracks, "
      f"{STORE['kpis']['genres']} gèneres carregats.")


app = Flask(__name__, static_folder=None)
Compress(app)


@app.route("/")
def index():
    if not (DASHBOARD_DIR / "index.html").exists():
        return (
            "<h1>dashboard/index.html no trobat</h1>"
            "<p>Descarrega el bundle de Claude Design a dashboard/index.html "
            "(veure Paso 3 del pla).</p>",
            404,
        )
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.route("/<path:filename>")
def static_assets(filename: str):
    return send_from_directory(DASHBOARD_DIR, filename)


@app.route("/api/kpis")
def kpis():
    return jsonify(STORE["kpis"])


@app.route("/api/tracks")
def tracks():
    return jsonify(_records(STORE["tracks"]))


@app.route("/api/tsne")
def tsne():
    return jsonify(_records(STORE["tsne"]))


@app.route("/api/pca")
def pca():
    return jsonify(_records(STORE["pca"]))


@app.route("/api/clusters")
def clusters():
    k = request.args.get("k", default=3, type=int)
    col = f"k{k}"
    df = STORE["clusters"]
    if col not in df.columns:
        return jsonify({"error": f"k={k} no disponible. Opcions: k3, k5, k7"}), 400
    out = df[["track_id", col]].rename(columns={col: "cluster_id"})
    return jsonify(_records(out))


@app.route("/api/genre-profiles")
def genre_profiles():
    return jsonify(_records(STORE["genre_profiles"]))


@app.route("/api/correlation")
def correlation():
    corr = STORE["correlation"]
    return jsonify({
        "features": list(corr.columns),
        "matrix": corr.values.round(4).tolist(),
    })


@app.route("/api/cluster-profiles")
def cluster_profiles():
    return jsonify(STORE["cluster_profiles"])


@app.route("/api/presets")
def presets():
    return jsonify(STORE["presets"])


if __name__ == "__main__":
    # Port 5001 perquè 5000 està ocupat per AirPlay Receiver a macOS.
    app.run(host="127.0.0.1", port=5001, debug=False)
