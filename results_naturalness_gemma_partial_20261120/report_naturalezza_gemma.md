---
title: "Naturalezza: confronto parziale Qwen–Gemma"
subtitle: "Stessi 41 originali completi per entrambi i giudici — 20 novembre 2026"
lang: it-IT
fontsize: 10pt
papersize: a4
geometry: margin=2cm
colorlinks: true
---

**Risultato attuale.** Il confronto con un giudice di un'altra famiglia mantiene l'ordine numerico osservato con Qwen: **Qwen 3B, Qwen 14B, DIPPER tre frasi, DIPPER intero**. Gemma assegna voti più alti soprattutto ai due riscrittori Qwen, ampliandone il divario rispetto a DIPPER. L'effetto della posizione è più contenuto in Gemma, ma non scompare. Sono risultati parziali ed esplorativi, senza validazione umana indipendente.

## 1. Copertura e confrontabilità

Il checkpoint Gemma contiene **507 risposte valide, due invalide e 223 mancanti**, sulle 732 previste per 61 originali. Sono completi **41 originali**: dieci Essay, 14 Reuters e 17 WritingPrompts. Due originali hanno giudizi parziali e 18 non sono ancora iniziati.

Il confronto principale usa soltanto i **41 originali con tutte le dodici risposte valide per entrambi i giudici**. Per ciascun giudice sono 492 chiamate e **1.968 voti**, cioè 41 originali × quattro riscritture × dodici presentazioni. Le risposte parziali restano nei dati grezzi ma non contribuiscono al confronto principale. Il riferimento Qwen completo comprende 61 originali: qui viene ristretto agli stessi 41 di Gemma, senza confrontare campioni diversi.

Prompt, testi, mapping anonimi, quattro ordini e tre ripetizioni sono identici nelle richieste corrispondenti. Sono stati verificati anche i payload nativi Google. Non sono state avviate nuove chiamate API: i riepiloghi e il bootstrap sono stati ricalcolati dai dati ricevuti e coincidono con quelli del checkpoint.

I voti 1–5 misurano scorrevolezza, formulazione idiomatica e registro. Non misurano origine umana, fedeltà semantica o evasione dei detector. Il giudice vede le quattro riscritture e il genere, non l'originale da cui derivano.

## 2. Voti sugli stessi 41 originali

Si calcola prima la media dei dodici voti per originale e metodo, poi la media fra gli originali. Gli intervalli sono percentili al 95%, con **2.000 ricampionamenti appaiati per gruppo di fonte, seme 42**. Non si trattano i voti ripetuti come osservazioni indipendenti. Gli intervalli sono puntuali ed esplorativi, senza correzione per confronti multipli.

| Riscrittore | Giudice Qwen: media [IC 95%] | Giudice Gemma: media [IC 95%] |
|---|---:|---:|
| Qwen 3B | 4,122 [3,996; 4,232] | 4,539 [4,386; 4,681] |
| Qwen 14B NF4 | 3,892 [3,693; 4,059] | 4,289 [4,045; 4,490] |
| DIPPER intero | 3,270 [3,047; 3,498] | 3,343 [3,085; 3,602] |
| DIPPER tre frasi | 3,384 [3,167; 3,602] | 3,463 [3,236; 3,695] |

Il vantaggio dei riscrittori Qwen rispetto a DIPPER si ritrova anche con Gemma. Per esempio, la differenza 3B meno DIPPER intero è **0,852 punti con Qwen e 1,195 con Gemma**: lo spostamento del contrasto è +0,343 [0,185; 0,510]. Anche gli altri contrasti Qwen–DIPPER aumentano di circa 0,32–0,34 punti. Le due configurazioni DIPPER restano meno nettamente separate: gli intervalli delle loro differenze appaiate comprendono zero.

Questo controllo riduce il dubbio che la preferenza per Qwen sia esclusivamente dovuta all'uso di un giudice della stessa famiglia. **Non dimostra neutralità o assenza di bias condivisi fra LLM**, né certifica che un sistema sia più naturale per le persone. Il campione comune è ancora parziale e non bilanciato per dominio. La differenza fra questi risultati Qwen e il precedente riepilogo sui 61 dipende dal sottoinsieme analizzato, non da nuovi giudizi Qwen.

## 3. Posizione e ripetibilità

Il voto viene centrato sulla media delle presentazioni della stessa riscrittura. Lo scostamento medio per posizione, sugli stessi 41 originali, è:

| Posizione | Qwen: scostamento medio | Gemma: scostamento medio |
|---|---:|---:|
| Prima | −0,204 | −0,067 |
| Seconda | +0,089 | +0,100 |
| Terza | +0,067 | −0,028 |
| Quarta | +0,048 | −0,004 |

L'effetto residuo massimo in valore assoluto è numericamente inferiore in Gemma: **0,100 contro 0,204 punti**. Tuttavia la seconda posizione Gemma mantiene +0,100 [0,053; 0,143]. Il controbilanciamento espone tutti i metodi alle stesse posizioni; non elimina automaticamente il condizionamento del giudice.

A ordine fisso, i tre voti sono identici nel **98,5%** dei blocchi originale × metodo × ordine per Qwen e nel **96,6%** per Gemma. Queste percentuali descrivono ripetibilità, non accuratezza. Fra giudici, sui 1.968 slot appaiati, l'accordo esatto è **61,6%**, la differenza assoluta media è 0,394 punti e Spearman è 0,762. Sono indici descrittivi di voti ripetuti, non di 1.968 campioni indipendenti né di accordo con esseri umani.

## 4. Stato delle risposte e passaggio successivo

Le due risposte Gemma invalide hanno rispettivamente un JSON malformato e un identificativo di testo diverso dai quattro richiesti. Entrambe terminano regolarmente con `STOP`: non sono troncamenti. Restano marcate come invalide, senza trasformarle in pareggi, correggere i voti o rigenerarle in questa analisi.

Restano **223 richieste senza risposta salvata**. Una ripresa può raccogliere i pendenti mantenendo la politica attuale. I due slot invalidi già memorizzati rimangono però tali: senza sostituirli, anche completando tutti i pendenti si arriverebbe al massimo a **59 pannelli completi su 61 e 730 risposte valide su 732**. L'attuale stato parziale non significa che il budget sia esaurito: il registro contiene 948 tentativi su 4.000, inclusi 439 HTTP 503. Non sono stati aumentati budget o cambiati prompt in questa analisi.

Gemma continua a usare **MINIMAL, non OFF**. In tutte le 509 risposte il campo grezzo dei token di pensiero è assente; lo zero viene inferito dal bilancio totale − prompt − candidate e non sono presenti parti di pensiero non vuote. Questa evidenza non certifica assenza di ragionamento interno. Qwen usa il parametro richiesto `reasoning_effort=none`.

## 5. File disponibili

Le tabelle principali sono in `comparison/`: `system_summary_by_judge.csv` per i voti appaiati, `humanizer_contrast_shifts.csv` per il cambiamento delle differenze, `position_effects_by_judge.csv` per la posizione, `ordinal_agreement.csv` per la concordanza e `comparison_coverage.csv` per gli originali inclusi. I grafici dei voti e della posizione sono nella stessa cartella.

Il CSV lungo aggiornato e le lunghezze sono nell'esportazione **`naturalness_gemma_partial_20261120`**. Il filtro **`included_matched_current=True`** seleziona 1.968 voti per giudice sugli stessi 41 originali. Il vecchio pacchetto del pilota a dieci rimane distinto. L'archivio ricevuto è conservato senza modifiche; il confronto esteso definitivo e la validazione umana restano da completare.
