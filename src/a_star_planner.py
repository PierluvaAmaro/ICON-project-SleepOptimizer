import heapq
import math
import os
import joblib  
import pandas as pd
from pgmpy.inference import VariableElimination
from bayesian_network import modello_bayes

# =========================================================
# CONFIGURAZIONE FATTORI DI SCALA (Per normalizzare h(n))
# =========================================================
SCALE_BMI = 5.0
SCALE_CAFFEINE = 100.0
SCALE_STEPS = 5000.0
SCALE_SLEEP = 3.0
SCALE_CAF_BED = 50.0      # Scala caffeina pre-sonno
SCALE_ALC_BED = 3.0       # Scala alcol pre-sonno
SCALE_SCREEN = 120.0      # Scala screen time
SCALE_TEMP = 5.0          # Scala temperatura stanza

# =========================================================
# 1. FUNZIONE OBIETTIVO MACHINE LEARNING (CLASSIFICATORE)
# =========================================================
def is_healthy_ml(state, ml_model, threshold=0.85):
    """Interroga il modello ML inviando tutte le 8 feature correnti."""
    bmi, caffeine, steps, sleep, caf_bed, alc_bed, screen, temp = state
    
    # Costruiamo il DataFrame con i nomi esatti delle colonne del dataset
    features = pd.DataFrame([{
        'bmi': bmi,
        'caffeine': caffeine,
        'steps': steps,
        'sleep': sleep,
        'caffeine_mg_before_bed': caf_bed,
        'alcohol_units_before_bed': alc_bed,
        'screen_time_before_bed_mins': screen,
        'room_temperature_celsius': temp
    }])
    
    try:
        probabilities = ml_model.predict_proba(features)[0]
        class_index = list(ml_model.classes_).index('Healthy')
        return probabilities[class_index] >= threshold
    except Exception as e:
        # Fallback se il modello non è caricato
        return bmi <= 25.0 and caf_bed == 0 and alc_bed == 0 and screen <= 30 and 18.0 <= temp <= 20.0


# =========================================================
# 2. FUNZIONE OBIETTIVO RETE BAYESIANA (BBN)
# =========================================================
def is_healthy_bbn(state, inferenza_bbn, threshold=0.80):
    """Mappa le 8 feature quantitative nei nodi discreti della BBN."""
    bmi, caffeine, steps, sleep, caf_bed, alc_bed, screen, temp = state
    
    # 1. Mappatura TOSSICITÀ PRE-SONNO (Caffeina tot, Caffeina pre-sonno e Alcol)
    if caf_bed > 40 or alc_bed > 2 or caffeine > 200:
        tossicita = 'critica'
    elif caf_bed > 0 or alc_bed > 0 or caffeine > 100:
        tossicita = 'alta'
    else:
        tossicita = 'bassa'
        
    # 2. Mappatura STRESS FISIOLOGICO (Passi e Temperatura della stanza)
    # Una stanza troppo calda (>23°C) o troppo fredda (<16°C) aumenta lo stress termico
    if steps < 4000 or temp > 23.0 or temp < 16.0:
        stress = 'critico'
    elif steps < 7000 or temp > 21.0 or temp < 17.5:
        stress = 'medio'
    else:
        stress = 'normale'
        
    # 3. Mappatura MISALLINEAMENTO CIRCADIANO (Ore di sonno ed Esposizione agli schermi)
    if sleep < 5.0 or screen > 90:
        misallineamento = 'severo'
    elif sleep < 6.5 or screen > 30:
        misallineamento = 'moderato'
    else:
        misallineamento = 'assente'
        
    try:
        query_result = inferenza_bbn.query(
            variables=['sleep_disorder_risk'],
            evidence={
                'tossicita_presonno': tossicita,
                'stress_fisiologico': stress,
                'misallineamento_circadiano': misallineamento
            },
            show_progress=False
        )
        prob_healthy = query_result.values[query_result.get_value(sleep_disorder_risk='Healthy')]
        return prob_healthy >= threshold
    except Exception as e:
        return bmi <= 25.0 and steps >= 8000 and sleep >= 7.0 and caf_bed == 0


# =========================================================
# EURISTICA: DISTANZA DALLO STATO IDEAL-SALUTARE
# =========================================================
def heuristic(state):
    """Calcola h(n) includendo i vincoli delle nuove variabili di igiene del sonno."""
    bmi, caffeine, steps, sleep, caf_bed, alc_bed, screen, temp = state
    
    # Calcolo dei delta verso i target ottimali
    d_bmi = max(0.0, bmi - 24.9) / SCALE_BMI
    d_caffeine = max(0.0, caffeine - 30) / SCALE_CAFFEINE
    d_steps = max(0.0, 8000 - steps) / SCALE_STEPS
    d_sleep = max(0.0, 7.0 - sleep) / SCALE_SLEEP
    
    # Nuovi target: Caffeina sera = 0, Alcol sera = 0, Screen Time <= 30 min, Temp ideale = 18.5°C
    d_caf_bed = max(0.0, caf_bed - 0) / SCALE_CAF_BED
    d_alc_bed = max(0.0, alc_bed - 0) / SCALE_ALC_BED
    d_screen = max(0.0, screen - 30) / SCALE_SCREEN
    d_temp = abs(temp - 18.5) / SCALE_TEMP  # Distanza assoluta (penalizza sia troppo caldo che troppo freddo)
    
    return math.sqrt(d_bmi**2 + d_caffeine**2 + d_steps**2 + d_sleep**2 + 
                     d_caf_bed**2 + d_alc_bed**2 + d_screen**2 + d_temp**2)


# =========================================================
# LE AZIONI TERAPEUTICHE (SPAZIO DEGLI STATI A 8 DIMENSIONI)
# =========================================================
ACTIONS = [
    {
        "desc": "Aumenta l'attività fisica (+1000 passi)",
        "effect": lambda b, c, st, sl, cb, ab, scr, t: (round(b - 0.1, 2), c, min(st + 1000, 15000), sl, cb, ab, scr, t),
        "cost": 10
    },
    {
        "desc": "Riduci il consumo di caffeina giornaliero (-20mg)",
        "effect": lambda b, c, st, sl, cb, ab, scr, t: (b, max(0, c - 20), st, sl, cb, ab, scr, t),
        "cost": 15
    },
    {
        "desc": "Migliora l'igiene del sonno (+30 min di riposo)",
        "effect": lambda b, c, st, sl, cb, ab, scr, t: (b, c, st, round(min(sl + 0.5, 10.0), 2), cb, ab, scr, t),
        "cost": 5
    },
    {
        "desc": "Ottimizzazione dieta rigida (Perdita peso mirata -0.5 BMI)",
        "effect": lambda b, c, st, sl, cb, ab, scr, t: (round(max(18.5, b - 0.5), 2), c, st, sl, cb, ab, scr, t),
        "cost": 25
    },
    # --- NUOVE AZIONI DI IGIENE DEL SONNO ---
    {
        "desc": "Elimina caffè/te pomeridiano o serale (-20mg pre-sonno)",
        "effect": lambda b, c, st, sl, cb, ab, scr, t: (b, c, st, sl, max(0, cb - 20), ab, scr, t),
        "cost": 12
    },
    {
        "desc": "Riduci il consumo di alcol a cena (-1 unità)",
        "effect": lambda b, c, st, sl, cb, ab, scr, t: (b, c, st, sl, cb, max(0, ab - 1), scr, t),
        "cost": 18  # Costo alto dovuto alle abitudini sociali
    },
    {
        "desc": "Spegni i dispositivi elettronici prima di dormire (-30 min screen time)",
        "effect": lambda b, c, st, sl, cb, ab, scr, t: (b, c, st, sl, cb, ab, max(0, scr - 30), t),
        "cost": 8
    },
    {
        "desc": "Regola il termostato della camera (Ottimizza verso i 18.5°C)",
        "effect": lambda b, c, st, sl, cb, ab, scr, t: (
            b, c, st, sl, cb, ab, scr, 
            round(t - 1.0 if t > 18.5 else t + 1.0, 1) if abs(t - 18.5) > 0.5 else 18.5
        ),
        "cost": 4   # Costo bassissimo, basta un click sul termostato
    }
]

# =========================================================
# ENGINE ALGORITMO A*
# =========================================================
def plan_therapeutic_path(start_state, healthy_check_func, model_object):
    counter = 0
    open_set = []
    
    h_start = heuristic(start_state)
    heapq.heappush(open_set, (h_start, 0, counter, start_state, []))
    
    best_g_scores = {start_state: 0}
    
    print("Stato Iniziale Paziente:")
    print(f" -> BMI: {start_state[0]} | Caffè Giornaliero: {start_state[1]}mg | Passi: {start_state[2]} | Sonno: {start_state[3]}h")
    print(f" -> Caffè Pre-Sonno: {start_state[4]}mg | Alcol Sera: {start_state[5]} u | Screen Time: {start_state[6]}m | Temp Stanza: {start_state[7]}°C\n")
    
    while open_set:
        f, g, _, current_state, path = heapq.heappop(open_set)
        
        if healthy_check_func(current_state, model_object):
            return path, current_state
        
        if g > best_g_scores.get(current_state, float('inf')):
            continue
            
        for action in ACTIONS:
            b, c, st, sl, cb, ab, scr, t = current_state
            next_state = action["effect"](b, c, st, sl, cb, ab, scr, t)
            next_g = g + action["cost"]
            
            if next_g < best_g_scores.get(next_state, float('inf')):
                best_g_scores[next_state] = next_g
                next_f = next_g + heuristic(next_state)
                counter += 1
                
                new_path = path + [(action["desc"], next_state)]
                heapq.heappush(open_set, (next_f, next_g, counter, next_state, new_path))
                
    return None, None

# =========================================================
# TEST BENCH
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("   A* CLINICAL PLANNER: MODELLO AD 8 VARIABILI ")
    print("=" * 60)
    
    # Configurazione di un paziente fortemente fuori range (Stato Critico)
    # Ordine: (BMI, caffeine, steps, sleep, caf_bed, alc_bed, screen, temp)
    paziente_critico = (23.5, 150, 13500, 5.0, 60, 2, 120, 23.5)
    
    inferenza_bbn_reale = VariableElimination(modello_bayes)
    
    print("[RUN]: Ricerca del percorso terapeutico ottimale tramite BBN...")
    piano, stato_finale = plan_therapeutic_path(paziente_critico, is_healthy_bbn, inferenza_bbn_reale)
    
    if piano:
        print("=" * 60)
        print(" PIANO DI IGIENE DEL SONNO E DELLO STILE DI VITA TROVATO ")
        print("=" * 60)
        for i, (azione, s) in enumerate(piano, 1):
            print(f"Step {i}: {azione}")
            print(f"        -> Stato: BMI={s[0]}, Caffè={s[1]}mg, Passi={s[2]}, Sonno={s[3]}h, CaffèSera={s[4]}mg, Alcol={s[5]}u, Screen={s[6]}m, Temp={s[7]}°C\n")
        print(f"Stato Finale Validato dalla Rete: {stato_finale}")
    else:
        print("Nessun piano terapeutico trovato in grado di soddisfare i requisiti minimi della BBN.")