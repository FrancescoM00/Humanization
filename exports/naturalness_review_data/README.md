# File per la revisione degli esperimenti di naturalezza

Esportazione del 20 novembre 2026 dai giudizi già salvati. Si riferisce al protocollo con quattro testi, quattro ordini e tre ripetizioni. Non include i vecchi confronti AB/BA e non esegue nuove chiamate ai modelli. I voti provengono da LLM, non da annotatori umani.

## 1. CSV lungo dei voti

**`ratings_long.csv`** contiene una riga per risposta osservata × metodo, per entrambi i giudici:

| Giudice | Righe esportate | Voti validi | Voti inclusi nell'analisi primaria | Voti sul pilota comune da 10 originali |
|---|---:|---:|---:|---:|
| Qwen | 2.992 | 2.988 | 2.928, su 61 originali | 480 |
| Gemma | 480 | 480 | 480, su 10 originali | 480 |
| Totale | 3.472 | 3.468 | 3.408 | 960 |

Le quattro righe Qwen senza voto appartengono a **una sola risposta non valida**, che avrebbe dovuto contenere quattro voti. I valori restano vuoti: non sono sostituiti con zero. Le richieste pianificate ma mai ricevute non hanno righe in questo file.

Le 15 risposte valide Qwen dei due pannelli parziali producono 60 voti validi, conservati ma esclusi dall'analisi primaria. La validità di una risposta e la completezza del pannello sono quindi campi distinti.

| Colonna | Significato |
|---|---|
| `judge`, `provider`, `judge_model` | Giudice, servizio e modello richiesto. |
| `response_model`, `provider_model_version` | Identificativi restituiti dal servizio, se salvati. Un campo assente resta vuoto. |
| `source_id`, `source_group_id`, `domain` | Originale di partenza, gruppo e dominio. |
| `method` | Metodo che ha prodotto la riscrittura. |
| `request_id` | Identificativo della singola richiesta; differisce fra giudici. |
| `order_id` | Ordine pianificato, da `order_0` a `order_3`. |
| `repeat_index`, `repetition` | Stessa ripetizione con indice originale 0–2 e numerazione leggibile 1–3. |
| `position`, `anonymous_id` | Posizione 1–4 e nome anonimo T1–T4 mostrato al giudice. |
| `rating` | Voto intero 1–5; vuoto se la risposta è invalida. |
| `valid`, `response_status`, `response_error` | Validità e stato originale della risposta, con errore se presente. |
| `complete_panel` | Per questo giudice, tutte le 12 richieste dell'originale sono valide. |
| `included_primary` | Voto valido di un pannello completo. |
| `included_matched_pilot` | Voto primario riferito a uno dei 10 originali comuni del pilota Gemma. |
| `rewrite_id`, `text_sha256` | Identificativo della riscrittura e hash del testo effettivamente giudicato. |
| `original_n_words`, `modified_n_words`, `length_ratio` | Lunghezze lessicali già presenti nel catalogo e loro rapporto. |
| `modified_whitespace_words` | Conteggio aggiuntivo per spazi sul testo inviato al giudice. |

Per riprodurre il riepilogo Qwen a 61 originali filtrare `judge == "qwen"` e `included_primary == True`. Per confrontare i due giudici **sugli stessi dieci originali** usare `included_matched_pilot == True`: si ottengono 480 voti per ciascuno. Non confrontare direttamente la media Qwen sui 61 con la media Gemma sui 10.

La chiave per accoppiare gli stessi voti fra giudici è **`source_id, method, order_id, repeat_index`**. La posizione e l'hash del testo coincidono nei due lati. Le ripetizioni non sono campioni indipendenti: l'originale resta l'unità dei confronti aggregati.

## 2. Prompt e configurazione effettiva

- **`judge_prompt.txt`**: prompt di sistema originale in inglese, identico per i due giudici, senza traduzioni o modifiche.
- **`judge_configuration.json`**: parametri, scala, ordini, ripetizioni, campione, regole di validità, conteggi e gestione del ragionamento.
- **`request_examples.json`**: stessa richiesta logica per Qwen e Gemma, con testi effettivi e mapping dei metodi. Include sia la rappresentazione canonica sia il corpo nativo Google effettivamente usato.
- **`gemma_reasoning_checks.csv`**: controllo dei metadati per tutte le 120 risposte Gemma.

Qwen è richiesto con **`reasoning_effort="none"`**, temperatura 0, `response_format={"type":"json_object"}` e massimo 256 token di completamento. I record salvati non hanno un contatore separato dei token di ragionamento: il suo valore non viene inventato come zero.

Gemma è richiesto tramite Google con temperatura 0, massimo 256 token e **`thinkingConfig.thinkingLevel="MINIMAL"`**. **MINIMAL non equivale a OFF.** Il corpo Google non contiene `responseMimeType`: il formato è richiesto dal prompt; `response_format` appartiene alla rappresentazione canonica dell'adattatore, non al corpo nativo inviato a Google.

Nelle 120 risposte Gemma il campo grezzo `thoughtsTokenCount` è assente. Lo zero dei token di pensiero è **inferito** da `totalTokenCount - promptTokenCount - candidatesTokenCount`, che vale zero in tutti i casi. Non sono presenti parti di pensiero con testo non vuoto; sono presenti marker vuoti. Il CSV distingue il valore grezzo mancante da quello inferito, il tipo di evidenza e i marker. Questa registrazione non certifica che il modello non abbia svolto ragionamento interno.

## 3. Originali del pilota Gemma

**`gemma_pilot_sources.csv`** elenca i 10 originali, dominio, generatore del testo di partenza, conteggi delle richieste valide e appartenenza al riferimento Qwen. **Tutti e 10 sono fra i 61 completi Qwen.** I domini sono 1 Essay, 4 Reuters e 5 WritingPrompts.

**`qwen_complete61_sources.csv`** contiene l'elenco completo dei 61, per verificare direttamente l'intersezione.

## 4. Lunghezze delle riscritture

**`rewrite_lengths.csv`** contiene 492 righe: 123 originali eleggibili × quattro metodi. Include i sottoinsiemi dei 61 Qwen e dei 10 Gemma, indicati dalle colonne booleane.

Le lunghezze erano già nel catalogo `results_humanizer_metric_catalogue/document_metrics.csv`: le colonne **`original_n_words` e `modified_n_words` sono copiate senza cambiarne il conteggio**, basato sui token alfabetici spaCy. `length_ratio` è `modified_n_words / original_n_words`.

Sono aggiunti i conteggi per spazi (`len(text.split())`) e il numero di caratteri Unicode della riscrittura (`len(text)`). Il conteggio per spazi può differire da quello lessicale; non vanno scambiati. Il testo di ogni riscrittura è stato verificato tramite hash contro il catalogo e contro i prompt dei due giudici.

La chiave per unire lunghezze e voti è **`source_id, method`**; il file conserva anche `rewrite_id` e `text_sha256`.

`export_manifest.json` documenta archivi sorgente, hash, numerosità e verifiche dell'esportazione. Gli esiti e i prompt precedenti restano invariati.
