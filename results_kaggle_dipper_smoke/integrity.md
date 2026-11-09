# Integrità dello smoke DIPPER su Kaggle

Archivio ricevuto: `humanization_smokedipper.zip`. Analisi del 9 novembre 2026, eseguita sui file esportati senza rilanciare modelli.

## Provenienza e completezza

- SHA-256 dell'archivio: `29724b6cff64732468e186d962299483b32ecdc6a345f6b9dbe84c9481990172`.
- Archivio leggibile: 67 elementi, directory comprese; 3.767.802 byte non compressi. Nessun percorso assoluto o contenente `..`.
- Commit registrato da Kaggle: `1e1d340980c6f643e54bc0d328bbb035c10c5c56`.
- Ambiente registrato: Python 3.13.16, PyTorch `2.11.0+cu130`, Transformers 4.57.3, bitsandbytes 0.48.2; due Tesla T4 disponibili, inferenza su `cuda:0`.
- I file `config.json`, `cohorts/cohort_manifest.json` e i tre pannelli `smoke.jsonl`, `pilot.jsonl`, `study.jsonl` sono identici byte per byte a quelli ricevuti per lo smoke Qwen e per il pilot Qwen.
- SHA-256 della configurazione: `86db7eb81f3d75fbfb306601395b70442659a3ea27d38b2e534991b571556d3b`; coincide con i manifest runtime, riscritture e detector. L'hash del lockfile Kaggle registrato coincide con il file del repository.
- SHA-256 del pannello smoke: `9c6553dce584ce1ba8686efc48ef29745e16d8ae92fd26d66b0b6dcbb06eb21c`.
- Il pannello contiene 300 fonti umane di calibrazione e 3 fonti di test. I due insiemi di `source_group_id` sono disgiunti. Le fonti di test sono una per dominio, tutte generate da ChatGLM.

## Riscritture e metriche

- Manifest generazione `completed`: 6 riscritture previste, 6 completate, tutte `ok`; tre `identity` e tre `dipper`, un tentativo per coppia sistema/fonte.
- Le tre riscritture `identity` sono identiche all'originale. Tutti i sei hash degli originali presenti nei metadati coincidono con il testo effettivo.
- Le tre generazioni DIPPER terminano con EOS; token di input 211, 212, 206 e token di output 145, 158, 143. I conteggi restano sotto i rispettivi limiti 512. Non risulta un arresto imposto dal limite dei token di output.
- Tempo di generazione DIPPER registrato nelle tre righe: 34,90 secondi complessivi. Non include scaricamento e caricamento del modello.
- `rewriting/pairs.jsonl` e `semantic/pairs.jsonl` sono identici; SHA-256 `5317f6865ea74b9dd06e326400d8160a5ad4f4716ef162a42a4f36f862bb651f`.
- Tutti e sei gli stadi semantici risultano `executed`: lexical, embedding, BERTScore, NLI, coseno allineato e BERTScore allineato. Ciascun CSV copre esattamente i sei ID attesi, senza ID duplicati o mancanti; i manifest si riferiscono allo stesso hash delle coppie. Il tempo complessivo registrato dagli stadi è 39,50 secondi, esclusi i download e gli avvii dei processi.
- L'hash dell'allineamento coincide con quello dichiarato dallo stadio BERTScore allineato. Gli hash del codice di generazione, metriche semantiche e similarità allineata coincidono con i file del commit corrente.
- Gli score `identity` nei sei CSV semantici coincidono esattamente con quelli dello smoke Qwen, escludendo soltanto i contatori di lavoro cumulativo della cache NLI.
- I campi `local_*` vuoti sono previsti: le riscritture reali non hanno una singola frase modificata designata in anticipo, diversamente dalle perturbazioni sintetiche. Gli score di documento e quelli allineati sono disponibili.
- Stile: 6 coppie ricevute e valutate, nessuna esclusa, scale costruite sui soli 300 umani di calibrazione, 105 feature estratte. Esportazione continua: 6 coppie e nessuna soglia semantica applicata.

## Detector e ricalcolo delle statistiche

I tre detector coprono 312 testi ciascuno: 300 umani di calibrazione, 3 umani di test, 3 originali macchina e 6 riscritture. Le 936 righe hanno score finiti nell'intervallo [0, 1]. `summarize_detection` è stato rieseguito sui soli score salvati: tutte le soglie coincidono e le 32 righe di riepilogo e le 8 righe macro coincidono con l'archivio entro la tolleranza di lettura CSV di 10^-12. Questa verifica non esegue inferenza.

Le 927 righe che riguardano calibrazione, umani di test, originali e `identity` sono identiche allo smoke Qwen, compresi metadati, hash dei testi e score. Anche `thresholds.json` è identico byte per byte. Le soglie rimangono:

| Detector | Soglia AI, superamento stretto |
|---|---:|
| OpenAI RoBERTa base | 0,9706076979637146 |
| Fakespot RoBERTa base, testo grezzo | 0,9611806273460388 |
| RADAR Vicuna | 0,794653058052063 |

Tutti e tre gli originali sono rilevati da ciascun detector. DIPPER non evade nessuno dei tre detector su questi testi: ASR 0/3 per ogni modello e 0/3 per la maggioranza. Anche `identity` ha ASR 0/3. Le soglie sono calibrate sui 300 umani; il 5% empirico di falsi positivi su questi dati di calibrazione non è una garanzia su altri testi. Gli intervalli bootstrap degeneri sul campione di tre non dimostrano che il tasso di evasione della popolazione sia zero.

## Condizione effettivamente valutata e limiti

Il preset è DIPPER XXL, revisione `c1fbf7a958a2aab022e9e6f81f7a3139f9e6ee3c`, tokenizer T5 XXL revisione `3db67ab1af984cf10548a73467f0e5bca2aaaeb2`, a 4 bit NF4, diversità lessicale 60 e diversità d'ordine 0. L'impostazione `enabled: false` nel preset non impedisce questa esecuzione: il sistema è stato selezionato esplicitamente in `active_systems` e risulta nei sei output attesi.

La variante usa l'intero breve passaggio come singola finestra senza prefisso (`entire_passage_no_prefix`), come documentato dal runner. Non replica il protocollo originale di DIPPER con finestre contestualizzate di tre frasi. Questi risultati riguardano tale configurazione quantizzata e greedy; manca un confronto con la stessa configurazione in piena precisione.

Configurazione, revisioni dei valutatori, detector e dati sono invariati rispetto a Qwen. L'NLI rimane il DeBERTa-v3 base previsto dal notebook. L'archivio registra revisioni e percorsi locali dei valutatori, ma non include i pesi né un hash indipendente di ogni file del modello in cache: la provenienza delle revisioni si basa sulla procedura di download e sulla configurazione registrata.

Non emerge un blocco tecnico per avviare il pilot DIPPER sugli stessi 42 testi già definiti nel pannello. Tre testi ChatGLM costituiscono una prova funzionale; non consentono una classifica affidabile degli humanizer o una valutazione della naturalezza. La qualità semantica richiede la lettura delle coppie e le annotazioni previste, oltre agli score continui. Il pilot conserva inoltre gli input rumorosi già individuati nel confronto Qwen; le eventuali analisi di sensibilità vanno dichiarate senza modificare a posteriori il pannello principale.
