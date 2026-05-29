import os
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print(" AVVIO MOTORE PROBABILISTICO: RETE BAYESIANA (BBN) ")
print("=" * 60)

# =========================================================
# 1. CARICAMENTO DATI
# =========================================================

# Rendiamo il percorso dinamico andando due passi indietro rispetto alla posizione dello script
script_dir = os.path.dirname(os.path.abspath(__file__))
path_dataset = os.path.join(script_dir, '..', 'data', 'datasets', 'sleep_health_dataset.csv')
path_enriched = os.path.join(script_dir, '..', 'data', 'datasets', 'sleep_health_enriched.csv')

df_original = pd.read_csv(path_dataset)
df_enriched = pd.read_csv(path_enriched)

# Uniformiamo i nomi colonne
df_original.columns = df_original.columns.str.lower().str.replace(' ', '_')
df_enriched.columns = df_enriched.columns.str.lower().str.replace(' ', '_')

# Merge dataset
df_totale = pd.merge(df_original, df_enriched, on='person_id')

# =========================================================
# 2. PREPARAZIONE DATI PER LA BBN
# =========================================================

dati_bbn = df_totale[
    [
        'misallineamento_circadiano',
        'tossicita_presonno',
        'stress_fisiologico',
        'sleep_disorder_risk'
    ]
].copy()

# Rimuoviamo eventuali NaN
dati_bbn = dati_bbn.dropna()

# Convertiamo tutto in stringhe/categorie discrete
dati_bbn = dati_bbn.astype(str)

print("\nTipi delle variabili:")
print(dati_bbn.dtypes)

print("\nValori nulli:")
print(dati_bbn.isnull().sum())

print("\nCategorie disponibili:")
for colonna in dati_bbn.columns:
    print(f"\n{colonna}:")
    print(dati_bbn[colonna].unique())

print("\nStruttura dati preparata con successo.")

# =========================================================
# 3. DEFINIZIONE DELLA RETE BAYESIANA (DAG)
# =========================================================

modello_bayes = DiscreteBayesianNetwork([
    ('misallineamento_circadiano', 'sleep_disorder_risk'),
    ('tossicita_presonno', 'stress_fisiologico'),
    ('stress_fisiologico', 'sleep_disorder_risk')
])

# =========================================================
# 4. TRAINING DEL MODELLO
# =========================================================

print("\nCalcolo delle CPT (Conditional Probability Tables)...")

modello_bayes.fit(
    dati_bbn,
    estimator=MaximumLikelihoodEstimator,
    # prior_type="BDeu"
)

# Validazione struttura
assert modello_bayes.check_model()

print("Modello Bayesiano validato con successo!")

# =========================================================
# 5. MOTORE INFERENZIALE
# =========================================================

inferenza = VariableElimination(modello_bayes)

# =========================================================
# 6. INFERENZA PREDITTIVA
# =========================================================

print("\n" + "=" * 60)
print(" INFERENZA PREDITTIVA (WHAT-IF ANALYSIS) ")
print("=" * 60)

# IMPORTANTE:
# usa esattamente le categorie presenti nel dataset.
# Qui assumiamo che esistano:
# - 'critico'
# - 'severo'

q1 = inferenza.query(
    variables=['sleep_disorder_risk'],
    evidence={
        'stress_fisiologico': 'critico',
        'misallineamento_circadiano': 'severo'
    }
)

print("\nScenario 1:")
print("Paziente con:")
print("- Stress fisiologico = 'critico'")
print("- Misallineamento circadiano = 'severo'")

print("\nDistribuzione probabilistica del rischio:")
print(q1)

# =========================================================
# 7. INFERENZA DIAGNOSTICA
# =========================================================

print("\n" + "=" * 60)
print(" INFERENZA DIAGNOSTICA (RAGIONAMENTO ALL'INDIETRO) ")
print("=" * 60)

# ATTENZIONE:
# usa la categoria esatta presente nel dataset
# es: 'Severe'

q2 = inferenza.query(
    variables=['tossicita_presonno'],
    evidence={
        'sleep_disorder_risk': 'Severe'
    }
)

print("\nScenario 2:")
print("Diagnosi osservata:")
print("- sleep_disorder_risk = 'Severe'")

print("\nProbabilità della tossicità pre-sonno:")
print(q2)

# =========================================================
# 8. FINE
# =========================================================

print("\nElaborazione inferenziale completata.")