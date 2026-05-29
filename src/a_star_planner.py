import heapq
import math
import os
import joblib  # Per caricare il modello di Machine Learning salvato
import pandas as pd
from pgmpy.inference import VariableElimination
from bayesian_network import modello_bayes

# Nota: Assicurati che il file della rete bayesiana sia importabile o caricalo dinamicamente
# Se necessario, puoi importare il modello direttamente dal tuo script precedente:
# from bayesian_network import modello_bayes

# =========================================================
# CONFIGURAZIONE FATTORI DI SCALA PER L'EURISTICA h(n)
# =========================================================
# Servono ancora all'euristica per guidare l'A* verso una direzione sensata
SCALE_BMI = 5.0
SCALE_CAFFEINE = 100.0
SCALE_STEPS = 5000.0
SCALE_SLEEP = 3.0

# Valori ideali puramente indicativi usati solo dal calcolo della distanza h(n)
IDEAL_BMI = 22.0
IDEAL_CAFFEINE = 0.0
IDEAL_STEPS = 10000.0
IDEAL_SLEEP = 8.0

# =========================================================
# 1. FUNZIONE OBIETTIVO MACCHINE LEARNING (CLASSIFICATORE)
# =========================================================
def is_healthy_ml(state, ml_model, threshold=0.85):
    """
    Interroga un modello di ML (es. Random Forest, XGBoost) passato come input.
    Sulla base delle feature numeriche dello stato attuale, verifica se la 
    probabilità della classe 'Healthy' supera la soglia impostata.
    """
    bmi, caffeine, steps, sleep = state
    
    # Creiamo il vettore di feature con lo stesso ordine usato nel training del modello
    # NOTA: adatta i nomi delle colonne a quelli esatti del tuo modello ML
    features = pd.DataFrame([{
        'bmi': bmi,
        'caffeine': caffeine,
        'steps': steps,
        'sleep': sleep
    }])
    
    try:
        # Estraiamo la probabilità della classe 'Healthy' (assumendo sia ad indice 0 o mappata)
        # Se il modello restituisce un array di probabilità [P(Not_Healthy), P(Healthy)]
        probabilities = ml_model.predict_proba(features)[0]
        
        # Supponiamo che il tuo modello sia binario o che tu sappia l'indice della classe 'Healthy'
        # Per esempio, se le classi sono ['Healthy', 'Insomnia', 'Sleep Apnea']
        class_index = list(ml_model.classes_).index('Healthy')
        prob_healthy = probabilities[class_index]
        
        return prob_healthy >= threshold
    except Exception as e:
        # Fallback di sicurezza in caso di modello mockato o non caricato
        return bmi <= 25.0 and caffeine <= 30 and steps >= 8000 and sleep >= 7.0


# =========================================================
# 2. FUNZIONE OBIETTIVO RETE BAYESIANA (BBN)
# =========================================================
def is_healthy_bbn(state, inferenza_bbn, threshold=0.80):
    """
    Interroga il motore d'inferenza della Rete Bayesiana.
    Poiché la BBN accetta variabili discrete (stringhe), questa funzione 
    mappa lo stato numerico dell'A* nelle categorie logiche della rete.
    """
    bmi, caffeine, steps, sleep = state
    
    # --- DISCRETIZZAZIONE DELLO STATO CORRENTE PER LA BBN ---
    # Mappiamo la Caffeina in 'tossicita_presonno'
    if caffeine > 100:
        tossicita = 'critica'
    elif caffeine > 40:
        tossicita = 'alta'
    else:
        tossicita = 'bassa'
        
    # Mappiamo Passi e Sonno in 'stress_fisiologico' (Logica di Dominio)
    if steps < 4000 or sleep < 5.5:
        stress = 'critico'
    elif steps < 7000 or sleep < 6.5:
        stress = 'medio'
    else:
        stress = 'normale'
        
    # Mappiamo il Sonno residuo in 'misallineamento_circadiano'
    if sleep < 5.0:
        misallineamento = 'severo'
    elif sleep < 6.5:
        misallineamento = 'moderato'
    else:
        misallineamento = 'assente'
        
    try:
        # Eseguiamo la query probabilistica sulla BBN impostando lo stato come evidenza
        query_result = inferenza_bbn.query(
            variables=['sleep_disorder_risk'],
            evidence={
                'tossicita_presonno': tossicita,
                'stress_fisiologico': stress,
                'misallineamento_circadiano': misallineamento
            },
            show_progress=False
        )
        
        # Estraiamo il valore associato alla categoria 'Healthy'
        prob_healthy = query_result.values[query_result.get_value(sleep_disorder_risk='Healthy')]
        return prob_healthy >= threshold
        
    except Exception as e:
        # Fallback di sicurezza
        return bmi <= 25.0 and caffeine <= 30 and steps >= 8000 and sleep >= 7.0


# =========================================================
# EURISTICA: CALCOLO DISTANZA EUCLIDEA PESATA
# =========================================================
def heuristic(state):
    """Calcola h(n): Distanza stimata verso lo stato ideale."""
    bmi, caffeine, steps, sleep = state
    
    d_bmi = max(0.0, bmi - 24.9) / SCALE_BMI
    d_caffeine = max(0.0, caffeine - 30) / SCALE_CAFFEINE
    d_steps = max(0.0, 8000 - steps) / SCALE_STEPS
    d_sleep = max(0.0, 7.0 - sleep) / SCALE_SLEEP
    
    return math.sqrt(d_bmi**2 + d_caffeine**2 + d_steps**2 + d_sleep**2)


# =========================================================
# AZIONI DISPONIBILI (BUG DEL SONNO RISOLTO)
# =========================================================
ACTIONS = [
    {
        "desc": "Aumenta l'attività fisica (+1000 passi)",
        "effect": lambda b, c, st, sl: (round(b - 0.1, 2), c, min(st + 1000, 15000), sl),
        "cost": 10
    },
    {
        "desc": "Riduci il consumo di caffeina (-20mg)",
        "effect": lambda b, c, st, sl: (b, max(0, c - 20), st, sl),
        "cost": 15
    },
    {
        "desc": "Migliora l'igiene del sonno (+30 min di riposo)",
        "effect": lambda b, c, st, sl: (b, c, st, round(min(sl + 0.5, 10.0), 2)),
        "cost": 5
    },
    {
        "desc": "Ottimizzazione dieta rigida (Perdita peso mirata -0.5 BMI)",
        "effect": lambda b, c, st, sl: (round(max(18.5, b - 0.5), 2), c, st, sl), # <--- BUG RISOLTO: sl al posto di b
        "cost": 25
    }
]


# =========================================================
# ENGINE ALGORITMO A* INTERATTIVO
# =========================================================
def plan_therapeutic_path(start_state, healthy_check_func, model_object):
    """
    Esegue la ricerca A* accettando dinamicamente la funzione di controllo 
    e l'oggetto modello associato (Rete Bayesiana o Modello ML).
    """
    counter = 0
    open_set = []
    
    h_start = heuristic(start_state)
    heapq.heappush(open_set, (h_start, 0, counter, start_state, []))
    
    best_g_scores = {start_state: 0}
    
    print(f"Stato Iniziale: BMI={start_state[0]}, Caffeina={start_state[1]}mg, Passi={start_state[2]}, Sonno={start_state[3]}h")
    
    while open_set:
        f, g, _, current_state, path = heapq.heappop(open_set)
        
        # CHIAMATA DINAMICA ALLA FUNZIONE OBIETTIVO SCELTA
        if healthy_check_func(current_state, model_object):
            return path, current_state
        
        if g > best_g_scores.get(current_state, float('inf')):
            continue
            
        for action in ACTIONS:
            bmi, caffeine, steps, sleep = current_state
            next_state = action["effect"](bmi, caffeine, steps, sleep)
            next_g = g + action["cost"]
            
            if next_g < best_g_scores.get(next_state, float('inf')):
                best_g_scores[next_state] = next_g
                next_f = next_g + heuristic(next_state)
                counter += 1
                
                new_path = path + [(action["desc"], next_state)]
                heapq.heappush(open_set, (next_f, next_g, counter, next_state, new_path))
                
    return None, None


# =========================================================
# CORPO DI ESECUZIONE / SIMULAZIONE DI TEST
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("   A* PLANNER INTEGRATO CON RETE BAYESIANA REALE ")
    print("=" * 60)
    
    paziente_iniziale = (25.0, 20, 10000, 7.0)
    
    # 1. CREAZIONE DEL MOTORE DI INFERENZA CON LA TUA RETE REALE
    # Qui prendiamo il modello_bayes importato dal tuo altro file
    inferenza_bbn_reale = VariableElimination(modello_bayes)
    
    print("\n[ESECUZIONE]: Utilizzo della Rete Bayesiana per validare lo stato Healthy...")
    
    # 2. INIEZIONE DELLA DIPENDENZA
    # Passiamo l'oggetto 'inferenza_bbn_reale' direttamente al planner
    piano, stato_finale = plan_therapeutic_path(
        start_state=paziente_iniziale, 
        healthy_check_func=is_healthy_bbn,  
        model_object=inferenza_bbn_reale    # <--- Il parametro 'inferenza_bbn' riceve questo!
    )
    
    if piano:
        print("\n" + "=" * 60)
        print(" PIANO TERAPEUTICO ADATTIVO GENERATO ")
        print("=" * 60)
        for i, (azione, stato) in enumerate(piano, 1):
            print(f"Step {i}: {azione}")
            print(f"        -> Stato: BMI={stato[0]}, Caffeina={stato[1]}mg, Passi={stato[2]}, Sonno={stato[3]}h\n")
        print(f"Stato Finale Validato dal Modello: {stato_finale}")
    else:
        print("Nessun piano terapeutico trovato.")