# Analisi e Predizione del Rischio di Disturbi del Sonno
**Progetto per il corso di Ingegneria della Conoscenza (ICON)**

Questo progetto implementa un sistema di Intelligenza Artificiale basato su un **approccio neuro-simbolico** per l'analisi e la predizione del rischio di disturbi del sonno (Sleep Disorder Risk). Il sistema combina tecniche di Machine Learning (sub-simbolico), deduzione logica in Prolog (simbolico) e reti bayesiane (probabilistico) per superare i limiti dei modelli *black-box* tradizionali.

## Architettura del Sistema

L'architettura si divide in quattro moduli logici principali, eseguiti in sequenza:

1. **Estrazione di Conoscenza Simbolica (Prolog)**
   - Trasforma i dati grezzi del paziente in fatti logici.
   - Tramite regole inferenziali complesse (`kb.pl`), estrae feature semantiche latenti: *misallineamento circadiano*, *tossicità pre-sonno* e *stress fisiologico*.
   - Esporta un dataset arricchito per l'addestramento dei modelli.

2. **Apprendimento Non Supervisionato (Clustering)**
   - Esplorazione dello spazio delle feature tramite algoritmo **K-Means** (con K=3 ottimizzato via Elbow Method) e riduzione dimensionale **PCA**.
   - Dimostra l'esistenza di cluster fenotipici che separano naturalmente i pazienti sani dai soggetti a rischio, "scoprendo" le diagnosi in modalità *cieca*.

3. **Apprendimento Supervisionato (Machine Learning)**
   - Modelli testati: **Decision Tree, Random Forest, SVM (Linear)**.
   - Esegue un benchmarking rigoroso (5-Fold Stratified Cross-Validation) confrontando le performance della *Baseline* (solo dati grezzi) con il *Modello Ibrido* (dati grezzi + feature logiche Prolog).
   - Valutazione tramite Accuracy, F1-Macro e Log-Loss.

4. **Ragionamento Probabilistico (Rete Bayesiana)**
   - Struttura causale (DAG) modellata con `pgmpy` per gestire l'incertezza del dominio clinico.
   - Esegue inferenza esatta (Variable Elimination) per supportare il ragionamento diagnostico (abduzione) e l'analisi predittiva (*What-if analysis*).

## Struttura della Repository

```text
📦 Progetto
 ┣ 📂 data/                 # Dataset originali e arricchiti, file dei fatti (.pl)
 ┣ 📂 figures/              # Grafici generati (PCA, Feature Importance, metriche)
 ┣ 📂 src/
 ┃ ┣ 📜 data_pipeline.py    # Preprocessing e conversione CSV -> Fatti Prolog
 ┃ ┣ 📜 kb.pl               # Knowledge Base e regole logiche (T-Box)
 ┃ ┣ 📜 clustering_analysis.py # Analisi non supervisionata (K-Means/PCA)
 ┃ ┣ 📜 ml_pipeline.py      # Training e validazione incrociata dei modelli supervisionati
 ┃ ┗ 📜 bayesian_network.py # Costruzione ed esecuzione della Rete Bayesiana
 ┗ 📜 README.md             # Questo file
