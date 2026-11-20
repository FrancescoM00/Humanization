# Naturalezza: dati del checkpoint Gemma parziale del 20 novembre 2026

Questo aggiornamento conserva i voti Qwen e aggiunge il checkpoint Gemma `(7)`. L'esportazione precedente, limitata al pilota Gemma da dieci originali, resta invariata in `exports/naturalness_review_data/`.

## Voti e filtri

`ratings_long.csv` ha **5.028 righe**, una per risposta osservata × metodo:

| Giudice | Righe | Voti validi | Voti primari per giudice | Voti nel confronto comune |
|---|---:|---:|---:|---:|
| Qwen | 2.992 | 2.988 | 2.928 su 61 originali | 1.968 su 41 originali |
| Gemma | 2.036 | 2.028 | 1.968 su 41 originali | 1.968 su 41 originali |

Qwen conserva una risposta invalida, Gemma due: producono rispettivamente quattro e otto righe senza voto. Le celle vuote non sono zeri. Le richieste mai ricevute non sono rappresentate come voti.

- `valid`: la risposta è valida e contiene il voto intero 1–5.
- `response_status`, `response_error`: stato e motivo originali.
- `complete_panel`: tutte le dodici richieste dell'originale sono valide per quel giudice.
- `included_primary`: voto valido e pannello completo **per il singolo giudice**.
- **`included_matched_current`**: voto appartenente ai **41 originali completi in entrambi**. È il filtro da usare per confrontare Qwen con Gemma. Produce 1.968 voti per ciascuno; non confrontare la media Qwen sui 61 con quella Gemma sui 41.

I campi identificativi sono `judge`, `source_id`, `domain`, `method`, `order_id`, `repeat_index` (0–2), `repetition` (1–3), `position` (1–4), `request_id`. Sono presenti voto, identificativo e hash della riscrittura, lunghezze e identificativi dei modelli richiesti/restituiti. La chiave comune ai due giudici è **`source_id, method, order_id, repeat_index`**: i request_id differiscono. Testi e posizioni coincidono nello stesso slot.

Le ripetizioni non sono osservazioni indipendenti: per i confronti aggregati l'unità resta l'originale, non il singolo voto.

## Copertura

- `gemma_source_coverage.csv`: tutti i 61 originali pianificati, con risposte valide, invalide e mancanti. Gemma ha **41 completi, due parziali e 18 non iniziati**; 507 risposte valide, due invalide e 223 mancanti.
- `gemma_complete41_sources.csv`: i 41 usati nel confronto attuale, tutti nei 61 Qwen. Sono dieci Essay, 14 Reuters e 17 WritingPrompts.
- `qwen_complete61_sources.csv`: riferimento Qwen completo.
- `gemma_pilot_sources.csv`: i dieci del pilota iniziale; è un sottoinsieme storico e non definisce il confronto attuale a 41 originali.

## Lunghezze

`rewrite_lengths.csv` contiene 492 riscritture, 123 originali × quattro metodi. Le colonne `in_qwen_complete61`, `in_gemma_complete_current` e `in_gemma_pilot` distinguono i sottoinsiemi. La chiave di unione con i voti è `source_id, method`.

`original_n_words` e `modified_n_words` sono i conteggi lessicali spaCy già salvati in `document_metrics.csv`; non sono stati cambiati. `length_ratio` è il loro rapporto. I conteggi aggiuntivi per spazi (`len(text.split())`) e caratteri Unicode (`len(text)`) hanno definizioni diverse, dichiarate nella configurazione. Tutti gli hash dei testi sono verificati contro il catalogo e i prompt effettivi.

## Prompt e ragionamento

- `judge_prompt.txt`: prompt di sistema originale, identico nei due giudici.
- `judge_configuration.json`: parametri e conteggi del checkpoint, con la distinzione fra corpo canonico e nativo Google.
- `request_examples.json`: stessa richiesta logica per entrambi, con testi e ordine effettivi e payload nativo Google.
- `gemma_reasoning_checks.csv`: controlli sui metadati delle **509 risposte**, incluse le due invalide.

Qwen è richiesto con `reasoning_effort="none"`; i suoi record non hanno un contatore separato dei token di ragionamento. Gemma è richiesto con `thinkingConfig.thinkingLevel="MINIMAL"`, che **non equivale a OFF**. Nelle 509 risposte il campo grezzo `thoughtsTokenCount` è assente; zero è inferito dal bilancio totale − prompt − candidate. Non ci sono parti di pensiero non vuote, ma compaiono marker vuoti. Il CSV conserva distinti valore mancante, inferito ed evidenza. Gli errori delle due risposte invalide riguardano il JSON, non il ragionamento.

Non sono state ricalcolate o corrette le risposte dei giudici e non sono state fatte nuove chiamate API. `export_manifest.json` documenta sorgenti e verifiche. Il report e le tabelle appaiate sono in `results_naturalness_gemma_partial_20261120/`.

Questo pacchetto serve per l'analisi. Per riprendere le chiamate su Kaggle usare il checkpoint originale Gemma `(7)` ricevuto, non lo ZIP con i CSV e il report.
