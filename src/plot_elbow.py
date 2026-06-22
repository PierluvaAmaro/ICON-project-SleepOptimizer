import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("="*60)
print(" CALCOLO DEL METODO DEL GOMITO (ELBOW METHOD) ")
print("="*60)

# 1. CARICAMENTO E PREPROCESSING 
df_original = pd.read_csv('../data/sleep_health_dataset.csv')
df_enriched = pd.read_csv('../data/sleep_health_enriched.csv')
df_original.columns = df_original.columns.str.lower().str.replace(' ', '_')
df_totale = pd.merge(df_original, df_enriched, on='person_id')

X = df_totale.drop(columns=['person_id', 'sleep_disorder_risk'])

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

print("Preprocessing dei dati in corso...")
X_processed = preprocessor.fit_transform(X)

# 2. CALCOLO DELLA WCSS PER DIVERSI VALORI DI K
wcss = []
k_values = range(1, 8) # Testiamo da 1 a 7 cluster

print("\nCalcolo della WCSS per K da 1 a 7 (Iterativo)...")
for k in k_values:
    print(f"--> Analisi con K = {k}")
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_processed)
    # kmeans.inertia_ contiene la WCSS del modello addestrato
    wcss.append(kmeans.inertia_)

# 3. GENERAZIONE E SALVATAGGIO DEL GRAFICO
plt.figure(figsize=(9, 5))
plt.plot(k_values, wcss, marker='o', linestyle='--', color='#b22222', linewidth=2, markersize=8)

# Evidenziamo il punto K=3 
plt.axvline(x=3, color='#2e8b57', linestyle=':', label='K Ottimale Scelto (K=3)')

plt.title('Metodo del Gomito (Elbow Method) per Selezione K', fontsize=14, fontweight='bold')
plt.xlabel('Numero di Cluster (K)', fontsize=12)
plt.ylabel('WCSS (Inerzia Intra-cluster)', fontsize=12)
plt.xticks(k_values)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
output_image = '../data/kmeans_elbow.png'
plt.savefig(output_image, dpi=300)

print("\n" + "="*60)
print(f"Grafico del Gomito salvato con successo in: {output_image}")
print("="*60)