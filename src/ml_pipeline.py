import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

# 1. CARICAMENTO DEI DATASET
print("Caricamento dei dataset in corso...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(script_dir, '..', 'data', 'datasets'))

csv_original = os.path.join(data_dir, 'sleep_health_dataset.csv')
csv_enriched = os.path.join(data_dir, 'sleep_health_enriched.csv')

df_original = pd.read_csv(csv_original)
df_enriched = pd.read_csv(csv_enriched)

df_original.columns = df_original.columns.str.lower().str.replace(' ', '_')
df_totale = pd.merge(df_original, df_enriched, on='person_id')
print(f"Dataset unito con successo. Dimensioni: {df_totale.shape}")

# Directory per i grafici (cartella `img/` alla root del progetto)
img_dir = os.path.normpath(os.path.join(script_dir, '..', 'img'))
os.makedirs(img_dir, exist_ok=True)

# 2. DEFINIZIONE TARGET E FEATURE
le_target = LabelEncoder()
y = le_target.fit_transform(df_totale['sleep_disorder_risk'].astype(str).str.lower().str.replace(' ', '_'))

colonne_escluse = ['person_id', 'sleep_disorder_risk']
features_prolog = ['misallineamento_circadiano', 'tossicita_presonno', 'stress_fisiologico']

X_baseline = df_totale.drop(columns=colonne_escluse + features_prolog)
X_arricchito = df_totale.drop(columns=colonne_escluse)

# 3. PREPROCESSOR
def ottieni_tipo_colonne(X):
    return X.select_dtypes(include=['object']).columns.tolist(), X.select_dtypes(exclude=['object']).columns.tolist()

cat_base, num_base = ottieni_tipo_colonne(X_baseline)
cat_arr, num_arr = ottieni_tipo_colonne(X_arricchito)

def crea_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

# 4. DEFINIZIONE DEI MODELLI DA CONFRONTARE
modelli = {
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
    "SVM (Linear)": SVC(kernel='linear', probability=True, max_iter=2000, random_state=42)
}

# 5. ESECUZIONE DELLA CROSS-VALIDATION MULTIPLA
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metriche = ['accuracy', 'f1_macro', 'neg_log_loss']

risultati = {}

print("\nAvvio del Benchmarking dei Modelli (5-Fold Cross Validation)...")
print("Questa operazione richiederà qualche minuto a causa dell'SVM.\n")

for nome, modello in modelli.items():
    print(f"--> Addestramento e valutazione di: {nome}")
    
    pipe_base = Pipeline(steps=[('preprocessor', crea_preprocessor(cat_base, num_base)), ('classifier', modello)])
    pipe_arr = Pipeline(steps=[('preprocessor', crea_preprocessor(cat_arr, num_arr)), ('classifier', modello)])
    
    scores_base = cross_validate(pipe_base, X_baseline, y, cv=cv, scoring=metriche, n_jobs=-1)
    scores_arr = cross_validate(pipe_arr, X_arricchito, y, cv=cv, scoring=metriche, n_jobs=-1)
    
    risultati[nome] = {'base': scores_base, 'arr': scores_arr}

# 6. STAMPA DELLA TABELLA COMPARATIVA MASSIVA
print("\n" + "="*95)
print(f"{'BENCHMARKING DEI MODELLI: BASELINE vs ONTOLOGY-BACKGROUND KNOWLEDGE (KB)':^95}")
print("="*95)
print(f"{'Modello':<18} | {'Metrica':<12} | {'Solo Dati Grezzi (Baseline)':<26} | {'Dati + Prolog KB (Ibrido)':<26}")
print("-" * 95)

for nome in modelli.keys():
    for m in metriche:
        nome_metrica = m.replace('neg_', '')
        
        v_base = -risultati[nome]['base'][f'test_{m}'] if 'log_loss' in m else risultati[nome]['base'][f'test_{m}']
        v_arr = -risultati[nome]['arr'][f'test_{m}'] if 'log_loss' in m else risultati[nome]['arr'][f'test_{m}']
        
        base_str = f"{v_base.mean()*100:6.2f}% ± {v_base.std()*100:4.2f}%" if 'log_loss' not in m else f"{v_base.mean():.4f} ± {v_base.std():.4f}"
        arr_str = f"{v_arr.mean()*100:6.2f}% ± {v_arr.std()*100:4.2f}%" if 'log_loss' not in m else f"{v_arr.mean():.4f} ± {v_arr.std():.4f}"
        
        print(f"{nome:<18} | {nome_metrica:<12} | {base_str:<26} | {arr_str:<26}")
    print("-" * 95)


# ==========================================
# 7. GENERAZIONE DEI GRAFICI
# ==========================================
print("\nGenerazione dei grafici in corso...")
sns.set_theme(style="whitegrid")

# --- Grafico 1, 2, 3: Grouped Bar Chart per ogni metrica ---
metrica_nomi_puliti = {'accuracy': 'Accuracy', 'f1_macro': 'F1-Macro', 'neg_log_loss': 'Log-Loss'}

for m in metriche:
    m_pulito = metrica_nomi_puliti[m]
    
    nomi_modelli = []
    valori_base = []
    valori_arr = []
    
    for nome in modelli.keys():
        v_b = -risultati[nome]['base'][f'test_{m}'].mean() if 'log_loss' in m else risultati[nome]['base'][f'test_{m}'].mean()
        v_a = -risultati[nome]['arr'][f'test_{m}'].mean() if 'log_loss' in m else risultati[nome]['arr'][f'test_{m}'].mean()
        
        
        if 'log_loss' not in m:
            v_b *= 100
            v_a *= 100
            
        nomi_modelli.append(nome)
        valori_base.append(v_b)
        valori_arr.append(v_a)

    x = np.arange(len(nomi_modelli))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, valori_base, width, label='Baseline (Dati Grezzi)', color='#4c72b0')
    rects2 = ax.bar(x + width/2, valori_arr, width, label='Ibrido (Dati + Prolog KB)', color='#55a868')

    ax.set_ylabel(f'{m_pulito} {"(%)" if "log_loss" not in m else ""}')
    ax.set_title(f'Confronto {m_pulito}: Baseline vs. Modello Ibrido Neuro-Simbolico')
    ax.set_xticks(x)
    ax.set_xticklabels(nomi_modelli)
    ax.legend()

    # Aggiungi etichette sui bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(os.path.join(img_dir, f'confronto_{m_pulito.lower().replace("-", "_")}.png'), dpi=300)
    plt.close()


# --- Grafico 4: Feature Importance (Random Forest Ibrido) ---
print("Calcolo della Feature Importance per il Random Forest Ibrido...")

# Addestramento RF Ibrido finale su tutto il dataset per ottenere le importances
pipe_arr_rf = Pipeline(steps=[('preprocessor', crea_preprocessor(cat_arr, num_arr)), 
                              ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42))])

pipe_arr_rf.fit(X_arricchito, y)

# Estrazione dei nomi delle feature dal preprocessor
preprocessor = pipe_arr_rf.named_steps['preprocessor']
cat_encoder = preprocessor.named_transformers_['cat']
cat_nomi = cat_encoder.get_feature_names_out(cat_arr)
feature_names = num_arr + list(cat_nomi)

# Estrazione importances
importances = pipe_arr_rf.named_steps['classifier'].feature_importances_

# Aggregazione delle importances per le feature categoriche
feature_importanza_aggregata = {}
for name, imp in zip(feature_names, importances):
    # Rimuove il suffisso del one-hot encoding 
    base_name = name.split('_')[0] if name.startswith(tuple(cat_arr)) else name 
    
    # Gestione specifica per le stringhe Prolog per non spezzare il nome
    if name.startswith('misallineamento_circadiano'): base_name = 'misallineamento_circadiano'
    elif name.startswith('tossicita_presonno'): base_name = 'tossicita_presonno'
    elif name.startswith('stress_fisiologico'): base_name = 'stress_fisiologico'

    feature_importanza_aggregata[base_name] = feature_importanza_aggregata.get(base_name, 0) + imp

# Ordina e plotta
importances_df = pd.DataFrame({
    'Feature': list(feature_importanza_aggregata.keys()),
    'Importance': list(feature_importanza_aggregata.values())
}).sort_values(by='Importance', ascending=True)

# Pulisce i nomi per il grafico
importances_df['Feature'] = importances_df['Feature'].str.replace('_', ' ').str.title()

plt.figure(figsize=(12, 8))
# Colora di rosso le feature estratte da Prolog per evidenziarle
colors = ['#c44e52' if 'Prolog' in feat or 'Fisiologico' in feat or 'Tossicita' in feat or 'Misallineamento' in feat else '#4c72b0' for feat in importances_df['Feature']]
plt.barh(importances_df['Feature'], importances_df['Importance'], color=colors)
plt.xlabel('Importanza Relativa (Indice di Gini)')
plt.title('Feature Importance: Random Forest Ibrido (Evidenziate le Feature Prolog)')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'feature_importance_rf.png'), dpi=300)
plt.close()

print("\nEsecuzione completata! Controlla la cartella 'img/' per i grafici generati.")