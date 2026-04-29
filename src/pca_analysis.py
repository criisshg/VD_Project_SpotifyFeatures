import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargamos el dataset limpio (el output de data_massage.py)
df_clean = pd.read_csv("spotify_clean.csv")

# 2. Separamos las variables numéricas (features) de las de texto (info)
info_cols = ["genre", "artist_name", "track_name", "track_id"]
feature_cols = [col for col in df_clean.columns if col not in info_cols]

# 3. Estandarizamos los datos (Fundamental antes de hacer PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[feature_cols])

# 4. Entrenamos el PCA pidiendo solo 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Extraemos el porcentaje de varianza explicada
var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
var_total = var_pc1 + var_pc2

print(f"Varianza PC1: {var_pc1:.2f}%")
print(f"Varianza PC2: {var_pc2:.2f}%")
print(f"Varianza Total retenida: {var_total:.2f}%")

# 6. Agregamos las nuevas dimensiones al DataFrame para graficar
df_clean['PC1'] = X_pca[:, 0]
df_clean['PC2'] = X_pca[:, 1]

# 7. Graficamos usando Seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_clean,
    x='PC1',
    y='PC2',
    hue='genre',      # Coloreamos por género
    palette='tab20',  # Paleta con colores variados
    alpha=0.6,        # Transparencia para ver zonas densas
    s=15,             # Tamaño del punto
    edgecolor=None
)

# 8. Detalles estéticos del gráfico
plt.title(f'Proyección PCA del Dataset de Spotify\nVarianza Total Explicada: {var_total:.2f}%', fontsize=16, fontweight='bold', pad=20)
plt.xlabel(f'PC1 ({var_pc1:.2f}% varianza)', fontsize=12)
plt.ylabel(f'PC2 ({var_pc2:.2f}% varianza)', fontsize=12)

# Movemos la leyenda fuera del gráfico
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Género Musical', markerscale=2)
plt.tight_layout()

# Guardamos el archivo y mostramos
plt.savefig('pca_spotify.png', dpi=300, bbox_inches='tight')
plt.show()