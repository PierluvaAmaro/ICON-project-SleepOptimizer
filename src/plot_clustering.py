import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("Generazione dei grafici per il K-Means...")

# 1. CARICAMENTO E PREPROCESSING
df_original = pd.read_csv('../data/sleep_health_dataset.csv')
df_enriched = pd.read_csv('../data/sleep_health_enriched.csv')
df_original.columns = df_original.columns.str.lower().str.replace(' ', '_')
df_totale = pd.merge(df_original, df_enriched, on='person_id')

X = df_totale.drop(columns=['person_id', 'sleep_disorder_risk'])
target_reale = df_totale['sleep_disorder_risk']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

X_processed = preprocessor.fit_transform(X)

# 2. K-MEANS & PCA PER RIDUZIONE DIMENSIONALE
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
df_totale['Cluster'] = kmeans.fit_predict(X_processed)

# Riduzione a 2 componenti per visualizzazione bidimensionale
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_processed)
df_totale['PCA1'] = X_pca[:, 0]
df_totale['PCA2'] = X_pca[:, 1]

# 3. CREAZIONE GRAFICI
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Grafico 1: I cluster geometrici trovati da K-Means
sns.scatterplot(
    data=df_totale, x='PCA1', y='PCA2', hue='Cluster', 
    palette='Set1', alpha=0.6, ax=axes[0], edgecolor=None
)
axes[0].set_title('Spazio dei Cluster K-Means (Proiezione PCA)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Componente Principale 1')
axes[0].set_ylabel('Componente Principale 2')

# Grafico 2: Come si distribuiscono le diagnosi reali nei cluster
sns.countplot(
    data=df_totale, x='Cluster', hue='sleep_disorder_risk', 
    palette='viridis', ax=axes[1]
)
axes[1].set_title('Distribuzione del Rischio Reale nei Cluster', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Cluster Assegnato')
axes[1].set_ylabel('Numero di Pazienti')
axes[1].legend(title='Rischio Reale')

plt.tight_layout()

# Salvataggio del grafico come immagine nella cartella del progetto
output_image = '../data/kmeans_results.png'
plt.savefig(output_image, dpi=300)
print(f"Grafico salvato con successo in: {output_image}")