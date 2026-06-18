import os
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print(" AVVIO MOTORE PROBABILISTICO: RETE BAYESIANA (BBN) ")
print("=" * 60)

# =========================================================
# 1. CARICAMENTO DATI (Percorsi dinamici monostep)
# =========================================================
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

# Rimuoviamo eventuali NaN e convertiamo in stringhe discrete
dati_bbn = dati_bbn.dropna().astype(str)

print("\nStruttura dati preparata con successo.")

# =========================================================
# 3. DEFINIZIONE DELLA RETE BAYESIANA (DAG)
# =========================================================
# Usiamo la classe specifica richiesta dalla tua versione di pgmpy
modello_bayes = DiscreteBayesianNetwork([
    ('misallineamento_circadiano', 'sleep_disorder_risk'),
    ('tossicita_presonno', 'stress_fisiologico'),
    ('stress_fisiologico', 'sleep_disorder_risk')
])

# =========================================================
# 4. TRAINING DEL MODELLO AUTOMATICO
# =========================================================
print("\nCalcolo delle CPT (Conditional Probability Tables)...")

# Omettiamo 'estimator': pgmpy istanzierà l'estimatore discreto di default in autonomia
modello_bayes.fit(dati_bbn)

# Validazione struttura
assert modello_bayes.check_model()
print("Modello Bayesiano validato con successo!")

# =========================================================
# 5. MOTORE INFERENZIALE
# =========================================================
inferenza = VariableElimination(modello_bayes)

# =========================================================
# 6. INFERENZA PREDITTIVA (WHAT-IF ANALYSIS)
# =========================================================
print("\n" + "=" * 60)
print(" INFERENZA PREDITTIVA (WHAT-IF ANALYSIS) ")
print("=" * 60)

q1 = inferenza.query(
    variables=['sleep_disorder_risk'],
    evidence={
        'stress_fisiologico': 'critico',
        'misallineamento_circadiano': 'severo'
    }
)

print("\nScenario 1 (Paziente con Stress 'critico' e Misallineamento 'severo'):")
print(q1)

# =========================================================
# 7. INFERENZA DIAGNOSTICA (RAGIONAMENTO ALL'INDIETRO)
# =========================================================
print("\n" + "=" * 60)
print(" INFERENZA DIAGNOSTICA (RAGIONAMENTO ALL'INDIETRO) ")
print("=" * 60)

q2 = inferenza.query(
    variables=['tossicita_presonno'],
    evidence={
        'sleep_disorder_risk': 'Healthy'
    }
)

print("\nScenario 2 (Diagnosi osservata: Rischio 'Healthy'):")
print("Probabilità delle cause scatenanti (Tossicità Pre-Sonno):")
print(q2)

print("\nElaborazione inferenziale completata.")

# 1. Controlla quanti dati 'alta' ci sono in totale nel dataset (Il Prior)
print(df_totale['tossicita_presonno'].value_counts(normalize=True))

# 2. Controlla la distribuzione reale della tossicità SOLO tra i sani
print(pd.crosstab(df_totale['tossicita_presonno'], df_totale['sleep_disorder_risk'], normalize='columns'))