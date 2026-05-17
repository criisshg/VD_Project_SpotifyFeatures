"""
correlation_network.py

Xarxa interactiva de correlacions Pearson entre les 11 features acústiques
del dataset Spotify Features.

- Nodes: les 11 features (layout circular, posicions fixes).
- Arestes: Pearson entre parelles, filtrades per |r| >= R_THRESHOLD.
- Color: verd Spotify (#1DB954) si r > 0, vermell (#e05c5c) si r < 0.
- Gruix i opacitat proporcionals a |r|.
- Etiqueta amb el valor de r al mig de cada aresta.
- Hover sobre node: llistat de totes les seves correlacions ordenades per |r|.

Output:
    outputs/eda/figures/correlation_network.html  (figura interactiva)
    outputs/eda/correlation_matrix.csv            (matriu Pearson 11x11)

Aquest script serveix com a referència per passar a Claude Design i que
integri la xarxa a featurefy_v1.html com a secció "/ 05 CORRELATIONS".

Executar des de l'arrel del repo:
    python src/correlation_network.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


FEATURES = [
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

# Pearson sobre 20k tracks (spotify_clean_sample.csv) és molt més robust que
# sobre 26 mitjanes per gènere (genre_profiles.csv).
DATA_PATH = os.path.join("data", "processed", "spotify_clean_sample.csv")

OUT_HTML = os.path.join("outputs", "eda", "figures", "correlation_network.html")
OUT_CSV  = os.path.join("outputs", "eda", "correlation_matrix.csv")

R_THRESHOLD = 0.3

COLOR_POS = "#1DB954"
COLOR_NEG = "#e05c5c"
COLOR_NODE = "#bdbdbd"
COLOR_NODE_HL = "#f1c40f"
COLOR_TEXT = "#FFFFFF"
COLOR_BG = "#000000"


def load_correlation_matrix(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Falten columnes al CSV: {missing}")
    return df[FEATURES].corr(method="pearson")


def circular_positions(features: list[str]) -> dict[str, tuple[float, float]]:
    """Coords (x, y) en cercle unitat, una posició fixa per feature."""
    n = len(features)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    return {f: (float(np.cos(a)), float(np.sin(a))) for f, a in zip(features, angles)}


def edge_style(r: float, threshold: float) -> tuple[str, float, float]:
    """Color, gruix i opacitat d'una aresta en funció de |r|."""
    color = COLOR_POS if r > 0 else COLOR_NEG
    norm = (abs(r) - threshold) / max(1 - threshold, 1e-9)
    norm = max(0.0, min(1.0, norm))
    width = 1.0 + 7.0 * norm
    opacity = 0.30 + 0.65 * norm
    return color, width, opacity


def build_edge_traces(corr: pd.DataFrame, pos: dict, threshold: float):
    """
    Una line trace per aresta (cal una traça per controlar gruix i color
    individualment). Retorna també una traça de text amb el valor r al
    punt mig de cada aresta.
    """
    edge_traces = []
    label_x, label_y, label_text, label_color = [], [], [], []

    feats = corr.columns.tolist()
    for i, f1 in enumerate(feats):
        for f2 in feats[i + 1:]:
            r = float(corr.loc[f1, f2])
            if abs(r) < threshold:
                continue
            x0, y0 = pos[f1]
            x1, y1 = pos[f2]
            color, width, opacity = edge_style(r, threshold)
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color=color, width=width),
                    opacity=opacity,
                    hoverinfo="text",
                    hovertext=f"{f1} ↔ {f2}<br>r = {r:+.3f}",
                    showlegend=False,
                )
            )
            label_x.append((x0 + x1) / 2)
            label_y.append((y0 + y1) / 2)
            label_text.append(f"{r:+.2f}")
            label_color.append(color)

    label_trace = go.Scatter(
        x=label_x,
        y=label_y,
        mode="text",
        text=label_text,
        textfont=dict(size=11, color=label_color, family="JetBrains Mono, monospace"),
        hoverinfo="skip",
        showlegend=False,
    )
    return edge_traces, label_trace


def build_node_trace(corr: pd.DataFrame, pos: dict, threshold: float):
    """Nodes amb hover que llista totes les correlacions ordenades per |r|."""
    feats = corr.columns.tolist()
    xs = [pos[f][0] for f in feats]
    ys = [pos[f][1] for f in feats]

    hover_texts = []
    for f in feats:
        lines = [f"<b>{f}</b>", ""]
        rels = corr[f].drop(f).sort_values(key=lambda s: s.abs(), ascending=False)
        for other, r in rels.items():
            mark = "●" if abs(r) >= threshold else "○"
            lines.append(f"{mark} {other}: {r:+.3f}")
        hover_texts.append("<br>".join(lines))

    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        marker=dict(
            size=18,
            color=COLOR_NODE,
            line=dict(color="#ffffff", width=1),
        ),
        text=feats,
        textposition="bottom center",
        textfont=dict(size=12, color=COLOR_TEXT, family="Inter, Arial, sans-serif"),
        hoverinfo="text",
        hovertext=hover_texts,
        showlegend=False,
    )


def build_figure(corr: pd.DataFrame, threshold: float = R_THRESHOLD) -> go.Figure:
    pos = circular_positions(corr.columns.tolist())
    edge_traces, label_trace = build_edge_traces(corr, pos, threshold)
    node_trace = build_node_trace(corr, pos, threshold)

    fig = go.Figure(data=[*edge_traces, label_trace, node_trace])
    fig.update_layout(
        title=dict(
            text=f"How features pull each other  ·  |r| ≥ {threshold}",
            x=0.5,
            xanchor="center",
            font=dict(color=COLOR_TEXT, size=18, family="Inter, Arial, sans-serif"),
        ),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        xaxis=dict(visible=False, range=[-1.4, 1.4]),
        yaxis=dict(visible=False, range=[-1.4, 1.4], scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=60, b=20),
        height=720,
        hoverlabel=dict(
            bgcolor="#1f1f1f",
            font=dict(color=COLOR_TEXT, family="Inter, Arial, sans-serif"),
        ),
    )
    return fig


def edges_summary(corr: pd.DataFrame, threshold: float):
    """Llista (f1, f2, r) per a les arestes pintades — útil per al panell lateral."""
    rows = []
    feats = corr.columns.tolist()
    for i, f1 in enumerate(feats):
        for f2 in feats[i + 1:]:
            r = float(corr.loc[f1, f2])
            if abs(r) >= threshold:
                rows.append((f1, f2, r))
    rows.sort(key=lambda t: abs(t[2]), reverse=True)
    return rows


def main():
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    corr = load_correlation_matrix(DATA_PATH)
    corr.round(4).to_csv(OUT_CSV)

    fig = build_figure(corr, threshold=R_THRESHOLD)
    fig.write_html(OUT_HTML, include_plotlyjs="cdn", full_html=True)

    edges = edges_summary(corr, R_THRESHOLD)
    print(f"[ok] Matriu desada a {OUT_CSV}")
    print(f"[ok] Figura desada a {OUT_HTML}")
    print(f"[info] Arestes amb |r| >= {R_THRESHOLD}: {len(edges)}")
    print("[info] Top arestes (|r| descendent):")
    for f1, f2, r in edges[:10]:
        print(f"  {f1:>16}  ↔  {f2:<16}  r = {r:+.3f}")


if __name__ == "__main__":
    main()
