import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings

# Ignoriamo i warning per un output pulito
warnings.filterwarnings('ignore')

print("="*60)
print(" AVVIO MOTORE NON SUPERVISIONATO: HARD CLUSTERING (K-MEANS) ")
print("="*60)

# 1. CARICAMENTO DATI
df_original = pd.read_csv('../data/sleep_health_dataset.csv')
df_enriched = pd.read_csv('../data/sleep_health_enriched.csv')

df_original.columns = df_original.columns.str.lower().str.replace(' ', '_')
df_totale = pd.merge(df_original, df_enriched, on='person_id')

# Salviamo il target reale per la validazione finale, ma NON lo diamo all'algoritmo
target_reale = df_totale['sleep_disorder_risk']

# 2. SELEZIONE E PREPROCESSING DELLE FEATURE
# Rimuoviamo ID e Target. Manteniamo le feature numeriche, categoriali e quelle di Prolog
X = df_totale.drop(columns=['person_id', 'sleep_disorder_risk'])

# Per il K-Means (basato su Distanza Euclidea) la standardizzazione è OBBLIGATORIA
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

print("Preprocessing dei dati in corso (Standardizzazione e One-Hot Encoding)...")
X_processed = preprocessor.fit_transform(X)

# 3. ESECUZIONE HARD CLUSTERING ITERATIVO (Algoritmo di Lloyd)
# Impostiamo K=3 ipotizzando 3 fenotipi principali: Sani, Rischio Lieve, Rischio Grave
K = 3
print(f"\nAddestramento K-Means con K={K} cluster...")
kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=300, n_init=10, random_state=42)

# La funzione predict effettua l'hard assignment minimizzando la varianza intra-cluster
df_totale['Cluster_Assegnato'] = kmeans.fit_predict(X_processed)

# 4. ANALISI DEI FENOTIPI (PROFILAZIONE DEI CENTROIDI)
print("\n" + "="*60)
print(" SCOPERTA DEI FENOTIPI: ANALISI DEI CENTROIDI (VALORI MEDI) ")
print("="*60)

# Calcoliamo la media delle variabili chiave numeriche per ogni cluster per capire chi sono
feature_chiave = ['age', 'sleep_duration_hrs', 'caffeine_mg_before_bed', 
                  'heart_rate_resting_bpm', 'screen_time_before_bed_mins']

profilo_cluster = df_totale.groupby('Cluster_Assegnato')[feature_chiave].mean().round(2)
profilo_cluster['Numero_Pazienti'] = df_totale['Cluster_Assegnato'].value_counts()
print(profilo_cluster)

# 5. VALIDAZIONE INCROCIATA CON LA DIAGNOSI MEDICA (Target Reale)
print("\n" + "="*60)
print(" MATRICE DI INCROCIO: CLUSTER vs DIAGNOSI MEDICA REALE ")
print(" (Il modello ha scoperto i malati senza conoscerli a priori?) ")
print("="*60)

# Creiamo una tabella pivot (crosstab) tra i cluster trovati ciecamente e le classi reali
matrice_validazione = pd.crosstab(df_totale['Cluster_Assegnato'], target_reale)
matrice_validazione['Totale'] = matrice_validazione.sum(axis=1)

# Calcoliamo la purezza percentuale del cluster per la classe maggioritaria
purezza = matrice_validazione.drop(columns='Totale').max(axis=1) / matrice_validazione['Totale'] * 100
matrice_validazione['Purezza_Dominante (%)'] = purezza.round(2)

print(matrice_validazione)
print("\nElaborazione completata.")