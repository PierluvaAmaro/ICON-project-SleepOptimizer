import os
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
import warnings
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

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

df_original.columns = df_original.columns.str.lower().str.replace(' ', '_')
df_enriched.columns = df_enriched.columns.str.lower().str.replace(' ', '_')

# Merge dataset
df_totale = pd.merge(df_original, df_enriched, on='person_id')

# Directory per i grafici
img_dir = os.path.normpath(os.path.join(script_dir, '..', 'img'))
os.makedirs(img_dir, exist_ok=True)

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

# Rimozione di eventuali NaN 
dati_bbn = dati_bbn.dropna().astype(str)

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
# 4. TRAINING DEL MODELLO AUTOMATICO
# =========================================================
print("\nCalcolo delle CPT (Conditional Probability Tables)...")

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

# ========== GRAFICO 1: DAG (visualizzazione della struttura causale) ==========
print("\nGenerazione grafico DAG della BBN...")
G = nx.DiGraph()
G.add_edges_from(modello_bayes.edges())

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)

# Evidenziamo i nodi di interesse
highlight_nodes = {'sleep_disorder_risk', 'misallineamento_circadiano', 'tossicita_presonno', 'stress_fisiologico'}
node_colors = ['#c44e52' if n in highlight_nodes else '#4c72b0' for n in G.nodes()]

nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1400, font_size=10, arrowsize=20)
plt.title('DAG: Struttura della Bayesian Network')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'dag_bbn.png'), dpi=300)
plt.close()

# ========== GRAFICO 2: Bar Chart Prior vs Posterior per `sleep_disorder_risk` ==========
print("Generazione Bar Chart: prior vs posterior per 'sleep_disorder_risk'...")
try:
    # Otteniamo gli stati possibili dalla colonna dei dati
    states = sorted(dati_bbn['sleep_disorder_risk'].unique())

    # Prior empirico (baseline) dai dati
    prior_counts = dati_bbn['sleep_disorder_risk'].value_counts(normalize=True).reindex(states).fillna(0).values

    # Posterior dal motore inferenziale (usiamo q1 calcolata sopra)
    posterior = q1
    # Estrazione valori posteriori in modo robusto
    if hasattr(posterior, 'values'):
        posterior_vals = np.array(posterior.values).flatten()
        # If order not matching states, attempt to use state_names
        try:
            order = posterior.state_names['sleep_disorder_risk']
            # Align posterior_vals to our `states` order
            posterior_series = pd.Series(posterior_vals, index=order)
            posterior_counts = posterior_series.reindex(states).fillna(0).values
        except Exception:
            posterior_counts = posterior_vals
    else:
        # Fallback: empirical posterior from data conditioned on evidence
        evidence_df = dati_bbn[(dati_bbn['stress_fisiologico'] == 'critico') & (dati_bbn['misallineamento_circadiano'] == 'severo')]
        posterior_counts = evidence_df['sleep_disorder_risk'].value_counts(normalize=True).reindex(states).fillna(0).values

    # Plot grouped bar chart
    x = np.arange(len(states))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, prior_counts * 100, width, label='Prior (empirico, %)', color='#4c72b0')
    plt.bar(x + width/2, posterior_counts * 100, width, label='Posterior (BBN, %)', color='#55a868')
    plt.xticks(x, [s.replace('_', ' ').title() for s in states], rotation=45, ha='right')
    plt.ylabel('Probabilità (%)')
    plt.title("Prior vs Posterior per 'Sleep Disorder Risk' (evidence: stress='critico', misallineamento='severo')")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'inference_prior_vs_posterior_sleep_disorder_risk.png'), dpi=300)
    plt.close()
    print('Grafici generati in', img_dir)
except Exception as e:
    print('Errore durante la generazione dei grafici:', e)

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

# 2. Controlla la distribuzione reale della tossicità solo tra i sani
print(pd.crosstab(df_totale['tossicita_presonno'], df_totale['sleep_disorder_risk'], normalize='columns'))