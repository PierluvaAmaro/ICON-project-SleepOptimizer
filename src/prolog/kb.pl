% === KNOWLEDGE BASE: REGOLE INFERENZIALI COMPLESSE ===

% Carica automaticamente il file dei fatti generato dalla pipeline Python
:- consult('../data/fatti_pazienti.pl').

% ----------------------------------------------------
% RECOLA 1: MISALLINEAMENTO CIRCADIANO
% ----------------------------------------------------
% Identifica il conflitto tra turni lavorativi e cronotipo biologico
misallineamento_circadiano(ID, severo) :-
    paziente(ID, _, _, _, _, _, _, _, _, _, _, _, 1, morning, _), !.
misallineamento_circadiano(ID, severo) :-
    paziente(ID, _, _, _, _, _, _, _, _, _, _, _, 1, evening, _), !.
misallineamento_circadiano(ID, moderato) :-
    paziente(ID, _, _, _, _, _, _, _, _, _, _, _, 1, neutral, _), !.
misallineamento_circadiano(ID, assente) :-
    paziente(ID, _, _, _, _, _, _, _, _, _, _, _, 0, _, _).


% ----------------------------------------------------
% REGOLA 2: TOSSICITÀ COMPORTAMENTALE PRE-SONNO (Sistema a Punteggio)
% ----------------------------------------------------
score_caffeina(Caffeina, 2) :- Caffeina > 100, !.
score_caffeina(Caffeina, 1) :- Caffeina > 0, !.
score_caffeina(_, 0).

score_alcol(Alcol, 2) :- Alcol > 2, !.
score_alcol(Alcol, 1) :- Alcol > 0, !.
score_alcol(_, 0).

score_screen(Screen, 2) :- Screen > 60, !.
score_screen(Screen, 1) :- Screen > 20, !.
score_screen(_, 0).

tossicita_presonno(ID, Livello) :-
    paziente(ID, _, _, _, _, _, _, Caffeina, Alcol, Screen, _, _, _, _, _),
    score_caffeina(Caffeina, SC),
    score_alcol(Alcol, SA),
    score_screen(Screen, SS),
    Totale is SC + SA + SS,
    (Totale >= 4 -> Livello = critica ;
     Totale >= 2 -> Livello = alta ;
     Livello = bassa).


% ----------------------------------------------------
% REGOLA 3: STRESS FISIOLOGICO COMBINATO (Incrocio Biometrico/Stile di Vita)
% ----------------------------------------------------
score_cardio(Battito, 2) :- Battito > 75, !.
score_cardio(Battito, 1) :- Battito > 65, !.
score_cardio(_, 0).

score_peso(BMI, 2) :- BMI > 29.9, !.    % Obesità
score_peso(BMI, 1) :- BMI > 24.9, !.    % Sovrappeso
score_peso(_, 0).                       % Normopeso o Sottopeso

score_attivita(Esercizio, 0) :- Esercizio >= 4, !. % Alta attività mitiga lo stress
score_attivita(Esercizio, 1) :- Esercizio >= 1, !.
score_attivita(_, 2).                              % Sedentario aumenta il carico

stress_fisiologico(ID, Livello) :-
    paziente(ID, _, _, _, BMI, OreSonno, _, _, _, _, Esercizio, Battito, _, _, _),
    score_cardio(Battito, SC),
    score_peso(BMI, SP),
    score_attivita(Esercizio, SA),
    (OreSonno < 6.0 -> SO = 2 ; 
     OreSonno < 7.0 -> SO = 1 ; 
     SO = 0),
    Totale is SC + SP + SA + SO,
    (Totale >= 5 -> Livello = critico ;
     Totale >= 2 -> Livello = medio ;
     Livello = normale).


% ----------------------------------------------------
% PREDICATO DI ESPORTAZIONE IN BATCH
% ----------------------------------------------------
% Questo predicato permette a Python di estrarre tutte le feature in un unica query velocissima
estrai_feature_latenti(ID, Mismatch, Tossicita, Stress) :-
    misallineamento_circadiano(ID, Mismatch),
    tossicita_presonno(ID, Tossicita),
    stress_fisiologico(ID, Stress).

    % === PREDICATO PER GENERARE IL DATASET ARRICCHITO IN CSV ===
genera_dataset_arricchito :-
    % Apre il file di output in modalità scrittura
    open('../data/sleep_health_enriched.csv', write, Stream),
    % Scrive l'header del nuovo CSV
    write(Stream, 'person_id,misallineamento_circadiano,tossicita_presonno,stress_fisiologico\n'),
    % Avvia il ciclo di fallimento per elaborare tutti i pazienti
    (
        paziente(ID, _, _, _, _, _, _, _, _, _, _, _, _, _, _),
        estrai_feature_latenti(ID, Mismatch, Tossicita, Stress),
        % Scrive la riga nel file
        format(Stream, '~w,~w,~w,~w\n', [ID, Mismatch, Tossicita, Stress]),
        fail % Forza il backtracking per passare al paziente successivo
    ;
        true % Termina con successo quando non ci sono più pazienti
    ),
    close(Stream),
    writeln('Elaborazione completata! File creato in data/sleep_health_enriched.csv').