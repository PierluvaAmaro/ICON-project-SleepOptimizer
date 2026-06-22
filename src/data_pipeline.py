import pandas as pd
import os

# 1. Caricamento del dataset
data_path = 'data/sleep_health_dataset.csv'
df = pd.read_csv(data_path)

print("=== ANALISI INIZIALE DEL DATASET ===")
print(f"Numero di righe: {df.shape[0]}")
print(f"Numero di colonne: {df.shape[1]}")
print("\nValori mancanti per colonna:")
print(df.isnull().sum()[df.isnull().sum() > 0]) 

# 2. Preprocessing base per Prolog
df.columns = df.columns.str.lower().str.replace(' ', '_')

# sostituzione spazi con underscore
categorical_cols = ['gender', 'occupation', 'country', 'chronotype', 'season', 'day_type', 'sleep_disorder_risk']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.replace(' ', '_')

# 3. Generazione del file Prolog 
kb_path = 'data/fatti_pazienti.pl'

print(f"\nGenerazione dei fatti Prolog in corso: {kb_path} ...")
with open(kb_path, 'w') as f:
    f.write("% === FATTI ESTRATTI DAL DATASET ===\n")
    f.write("% Formato: paziente(ID, Eta, Genere, Occupazione, BMI, OreSonno, QualitaSonno, Caffeina, Alcol, ScreenTime, Esercizio, Battito, ShiftWork, Cronotipo, Rischio).\n\n")
    
    for index, row in df.iterrows():
        # Estrazoione colonne chiave per la KB
        p_id = row['person_id']
        eta = row['age']
        genere = row['gender']
        occupazione = row['occupation']
        bmi = row['bmi']
        ore_sonno = row['sleep_duration_hrs']
        qualita = row['sleep_quality_score']
        caffeina = row['caffeine_mg_before_bed']
        alcol = row['alcohol_units_before_bed']
        screen = row['screen_time_before_bed_mins']
        esercizio = row['exercise_day']
        battito = row['heart_rate_resting_bpm']
        shift = row['shift_work']
        cronotipo = row['chronotype']
        rischio = row['sleep_disorder_risk'] # target
        
        # Scrittura del fatto Prolog
        fatto = f"paziente({p_id}, {eta}, {genere}, {occupazione}, {bmi}, {ore_sonno}, {qualita}, {caffeina}, {alcol}, {screen}, {esercizio}, {battito}, {shift}, {cronotipo}, {rischio}).\n"
        f.write(fatto)

print("Generazione completata con successo!")