import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Ignoriamo i warning di non-convergenza dell'SVM dovuti al max_iter
warnings.filterwarnings("ignore")

# 1. CARICAMENTO DEI DATASET
print("Caricamento dei dataset in corso...")
df_original = pd.read_csv('../data/sleep_health_dataset.csv')
df_enriched = pd.read_csv('../data/sleep_health_enriched.csv')

df_original.columns = df_original.columns.str.lower().str.replace(' ', '_')
df_totale = pd.merge(df_original, df_enriched, on='person_id')
print(f"Dataset unito con successo. Dimensioni: {df_totale.shape}")

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
# Per l'SVM usiamo probability=True per poter calcolare la log_loss e max_iter per non bloccare il PC
modelli = {
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
    "SVM (Linear)": SVC(kernel='linear', probability=True, max_iter=2000, random_state=42)
}

# 5. ESECUZIONE DELLA CROSS-VALIDATION MULTIPLA
# Usiamo 5 Fold invece di 10 per accelerare i test con i 3 modelli
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