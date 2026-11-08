# Esperimenti di stile e conservazione semantica: Humanization

Pipeline riproducibile per gli esperimenti della tesi: stile umano/macchina, trasferimento fra domini e generatori, conservazione semantica e controlli positivi sulle riscritture. Il PDF fornito è stato usato come riferimento scientifico, non come istruzioni operative.

**Per continuare nell'app Codex:** leggere [CODEX_HANDOFF.md](CODEX_HANDOFF.md). Il trasferimento completo su GitHub è riuscito: questo branch `main` contiene codice e risultati; il branch `cloud-backup` conserva dataset, modelli, venv e cache tramite Git LFS (22 parti, 1,83 GB compressi). [Contenuti e istruzioni di recupero](TRANSFER_STATUS.md). Le directory originali `data/` e `.venv/` restano ignorate in Git normale: sono salvate negli archivi del branch dedicato.

**Controllo entity clean:** [confronto con la v1](results_semantic_clean/entity_policy_comparison.md), [revisione testuale](results_semantic_clean/clean_entity_text_audit.md) e [protocollo](ENTITY_CLEAN_EXPERIMENT.md). La patch è stata corretta e collegata tramite `--entity-policy clean`, con v1 come default. Sulle stesse 200 fonti sono applicabili 145 sostituzioni; 55 vengono saltate esplicitamente. Dati in `data/semantic_clean/`, risultati in `results_semantic_clean/`: il comparatore controlla tutte le 2.007 coppie estranee alla modifica, includendo OpAI e fixture. Il nome clean non certifica etichette semantiche gold; i limiti NER residui sono documentati.

**Scaricare i nuovi dati da GitHub:** [ZIP con pairs.jsonl e skipped.jsonl](exports/semantic_clean/semantic_clean_pairs.zip), [JSONL completo](exports/semantic_clean/pairs.jsonl) e [istruzioni](exports/semantic_clean/README.md). Queste copie sono sul branch `main`, accessibili dal browser e con `git pull`; tutte le cache scientifiche sono nel backup dati del branch `cloud-backup`. [Patch originale](patches/entity_clean_v2.original.patch) e [patch corretta](entity_clean_v2_corrected.patch) sono conservate entrambe.

**Confronto equo e robustezza:** [errori distribuiti e stile localizzato](results_semantic_fairness/interpretazione.md), con [protocollo e comandi](FAIRNESS_EXPERIMENT.md). Coseno e BERTScore sono ora allineati per frase e confrontati con gli stessi aggregatori NLI. Sul test appaiato, AUC del minimo: NLI **0,958**, coseno **0,607**, BERTScore **0,534**. Le corruzioni su 2–5 frasi mostrano che media e quota non supportata possono raggiungere o superare il minimo. Il minimo dipende dalla lunghezza; la normalizzazione per fasce mantiene AUC NLI 0,951. Guardie lessicali implementate e valutate, senza un miglioramento complessivo dimostrato.

La suite esporta [score continui e distribuzioni per sistema](results_semantic_fairness/continuous_suite/manifest.json), senza applicare la soglia storica OpAI ad altri humanizer. Sono state elaborate 3.500 coppie in totale, mantenendo DeBERTa-v3 **base float32**. [Stile sulle frasi effettivamente toccate](results_opai_touched_style/report.md): il segnale prima diluito emerge, ma l'ordine Polish < Paraphrase < Style non vale per l'intensità locale; va distinta dalla copertura dell'intervento. DIPPER e altri humanizer non sono ancora stati eseguiti.

L'estensione con controlli positivi e perturbazioni sottili è descritta in [SEMANTIC_CONTROLS.md](SEMANTIC_CONTROLS.md). Usa 300 controlli OpAI Polish/Paraphrase/Style, 562 corruzioni degli stessi output e 152 nuove sfide MGTBench, mantenendo gli esempi precedenti per confronto. La scelta dell'aggregatore usa calibrazione e test con sorgenti separate; le etichette restano proxy semantici. [Revisione testuale dei 40 casi](results_semantic_extension/review40.md), [audit di nove positivi](results_semantic_extension/positive_review9.md) e [movimenti delle feature nelle cinque operazioni OpAI](results_opai_style/report.md) documentano risultati e limiti. La revisione testuale è dell'assistente, non annotazione umana.

**Fase precedente, conservata per confronto:** [interpretazione dei controlli](results_semantic_extension/interpretazione.md) e [AUC e falsi allarmi](results_semantic_extension/aggregator_report.md). La calibrazione sceglieva il minimo bidirezionale: AUC test 0,958, AUC sui sottili appaiati 0,900; 11/153 allarmi sui positivi presunti con soglia calibrata. La soglia è storica, non una raccomandazione per la suite. Le cautele rimosse rimangono un punto debole (AUC circa 0,57, nove casi test). Il rescaling BERTScore e la matrice delle correlazioni sono inclusi.

L'esperimento è stato eseguito: [interpretazione dei risultati](results/interpretazione.md), [report completo](results/report.md) e [metriche](results/classifier_metrics.csv).

Aggiornamento richiesto: [studio con lunghezza abbinata](results_length_matched/report.md), [confronto con il precedente](results_length_matched/comparison_with_original.csv) e [pipeline semantica](SEMANTIC_EXPERIMENT.md). I risultati precedenti sono conservati; per lo stile il nuovo controllo è in `results_length_matched/`.

La fase semantica è completa: [interpretazione](results_semantic_full/interpretazione.md), [report con grafici e IC](results_semantic_full/report.md), [tutte le misure](results_semantic_full/scores.csv) e [asimmetria OpAI](results_semantic_full/opai_asymmetry.csv). Sono stati valutati 200 testi macchina con cinque perturbazioni, controlli identità e 800 coppie OpAI ufficiali: 2.407 coppie complessive. DeBERTa-v3 **base**, float32, è stato mantenuto; 88.900 confronti NLI unici su CPU, con picco osservato di 2,58 GiB per il processo NLI. Il blocco Hugging Face precedente è risolto.

Per verificare gli artefatti già prodotti, senza nuove inferenze: `.venv/bin/python scripts/validate_semantic_results.py`. I comandi per riprodurre l'intero esperimento sono in [SEMANTIC_EXPERIMENT.md](SEMANTIC_EXPERIMENT.md).

Per riprodurre il nuovo controllo: `.venv/bin/python scripts/run_length_matched_style.py`. Usa prefissi della stessa fonte terminati a fine frase con conteggi lessicali spaCy identici nella coppia, tra90 e180 parole e vicini a150. Il modello `word_count_only` serve a verificare il controllo; caratteri e numero di frasi restano diagnostici distinti.

## Riprodurre l'esperimento

Da `/workspace/Humanization` (o dalla radice del checkout):

```bash
bash scripts/setup.sh
.venv/bin/python scripts/download_mgtbench.py --output data/raw
.venv/bin/python scripts/run_style_experiment.py --raw-dir data/raw --output results
.venv/bin/python -m pytest -q
```

Il download rifiuta di sovrascrivere CSV già presenti. Per usare i dati già scaricati, saltare il secondo comando. I download usano il proxy e la verifica TLS; in questa macchina la rete può richiedere l'esecuzione autorizzata fuori dal sandbox. Non servono API key, GPU o modelli generativi. `setup.sh` installa dipendenze fissate, spaCy `en_core_web_sm` 3.8.0 e il corpus NLTK cmudict usato da textstat.

Per ricalcolare soltanto l'analisi su una cache compatibile:

```bash
.venv/bin/python scripts/run_style_experiment.py --reuse-features
```

La cache è verificata rispetto a campione, codice dell'estrattore, schema, modello e versioni. Il seed predefinito è 42; `--pairs 300` significa 300 umani **e** 300 macchine per dominio. `--bootstrap 500` controlla gli IC degli effetti e delle metriche. Le dipendenze dirette sono in `requirements.txt`; l'ambiente eseguito è fissato in `requirements-lock.txt`.

## Dati: cosa è stato effettivamente usato

TH-Bench [`5c1d99c`](https://github.com/DrenfongWong/TH-Bench/tree/5c1d99cefee49208206b6fa75d3a628ac9935f86) rimanda ai CSV MGTBench originali su [Google Drive](https://drive.google.com/drive/folders/1p4iBeM4r-sUKe8TnS4DcYlxvQagcmola). I file `clean` prodotti dal suo script degli attacchi sono soltanto macchina, campionati per lunghezza e troncati: non corrispondono direttamente al disegno richiesto.

Il download predefinito usa un [mirror di ricerca pubblico](https://github.com/derek33125/AITextDetection/tree/6f289c4cf4e3966e9442d104ea250bae43f3ff9e/Data_Generation/MGTBench) fissato al commit `6f289c4cf4e3966e9442d104ea250bae43f3ff9e`: `Essay_LLMs.csv`, `WP_LLMs.csv`, `Reuters_LLMs.csv`, 1.000 righe originali ciascuno, con umano e sette generatori allineati per riga. **L'uguaglianza byte per byte con l'archivio ufficiale non è stata verificata**, perché Google Drive è bloccato dalla policy corrente. Non viene sostituito il benchmark con un altro corpus.

Per usare i CSV originali già disponibili, passarne la directory a `--raw-dir`. In alternativa, `scripts/download_mgtbench.py --source official --output data/official` supporta Google Drive con `gdown==5.2.0` opzionale e accesso di rete a `drive.google.com` e `drive.usercontent.google.com`.

La pulizia locale è esplicita: `process_spaces` del loader TH-Bench; rimozione simmetrica della coda senza terminazione di frase plausibile; almeno 50 parole whitespace **dopo** questa operazione per entrambe le classi; deduplicazione globale casefold/whitespace; esclusione di output macchina identici a un testo umano. Non si tronca a un numero prefissato di parole e non si rimuovono selettivamente i segnali Wikipedia. La rimozione delle code usa punteggiatura finale con gestione di virgolette, decimali e abbreviazioni comuni: **è un'euristica**, non certifica la completezza grammaticale. Non sono stati verificati lingua e completezza delle frasi con un annotatore, né eliminati tutti i rifiuti o le formule di chatbot. Quindi il campione è una baseline MGTBench preparata localmente, non una riproduzione certificata del preprocessing descritto nel PDF. Gli artefatti residui sono un limite dell'interpretazione.

Il controllo iniziale aveva rilevato terminazioni senza punteggiatura plausibile in 430/900 testi macchina contro 66/900 umani: conservare questo artefatto rischiava di gonfiare il segnale. La prima esecuzione è salvata in `results_pilot/`. Per riprodurla:

```bash
.venv/bin/python scripts/run_style_experiment.py --keep-incomplete --output results_pilot
```

La pulizia avviene **prima** del campionamento, quindi i campioni pilot e principale possono differire: il loro confronto è un controllo descrittivo del preprocessing, non una stima causale appaiata del suo effetto.

Campionamento per ciascun dominio:

- 300 testi umani unici, ciascuno accoppiato con un solo output macchina della stessa riga;
- 43 output per ChatGPT-turbo, Claude, ChatGLM, Dolly, ChatGPT, GPT4All e 42 StableLM;
- selezione casuale con quote fissate, senza scegliere i testi sulla base delle feature o dei risultati;
- `pair_id` conserva l'allineamento; `group_id` collega anche prompt ripetuti, così train e test restano separati per origine.

`data/raw/SOURCE.json` e `results/data_manifest.json` registrano provenienza, hash SHA-256, esclusioni e conteggi. I testi grezzi e il campione locale non sono aggiunti a Git; le condizioni dei corpora di origine restano applicabili.

## Feature e confronti

105 feature fisse: lunghezza e variabilità delle frasi, punteggiatura, 16 categorie POS, 58 parole funzionali più frequenza totale, MATTR50 scritto a mano, misure lessicali e leggibilità textstat. Le definizioni sono in `results/feature_schema.json` e `stylometry/features.py`.

Le parole lessicali sono token `is_alpha`, casefolded, non lemmatizzati. MATTR usa tutte le finestre consecutive da 50 parole; sotto 50 restituisce NaN, senza cambiare la finestra. Le POS usano token non-spazio/non-punteggiatura e includono numeri e frammenti di contrazioni. Le parole funzionali sono forme letterali: `n't` non viene trasformato in `not`. I segni di punteggiatura sono conteggi di caratteri, compresi quelli in abbreviazioni. Textstat mantiene le proprie convenzioni di tokenizzazione. La sillabificazione è `en_US`, con cmudict locale; nessun download durante l'estrazione.

I sette proxy Wikipedia sono fissati dal capitolo 2 del PDF: vocabolario AI, enfasi, conclusioni, formule conversazionali, parallelismi negativi, Markdown ed em dash. Non misurano rule of three, profondità semantica, neutralità o correttezza delle citazioni. Quindi l'esperimento verifica **questi proxy**, non l'intera guida.

Modelli confrontati:

| Modello | Input |
| --- | --- |
| `style` | Feature stilometriche, escludendo conteggi globali, TTR/hapax e proxy Wikipedia |
| `wikipedia` | Sette proxy Wikipedia |
| `combined` | Unione dei due insiemi |
| `length_only` | Numero di parole, caratteri e frasi, come controllo del confondente lunghezza |

La regressione logistica usa L2, C=1 fisso, pesi bilanciati e soglia 0.5. Imputazione e standardizzazione sono apprese soltanto sul training. Si riportano ROC-AUC, balanced accuracy e macro-F1:

1. CV a cinque fold per gruppi dentro ciascun dominio;
2. addestramento su due domini e test sul terzo;
3. generatore escluso dal training, test sul suo output e sui corrispondenti umani, con origine dei testi esclusa dal training.

Le metriche aggregano predizioni out-of-fold, non medie interpretate come repliche indipendenti. IC bootstrap al 95% per gruppi sulle predizioni fisse: non includono tutta la variabilità dell'addestramento. I gruppi sono controllati con asserzioni e `train_group_overlap=0` nell'output.

## Effetti e lista corta

Hedges g usa differenza **macchina − umano** divisa per deviazione standard pooled, con correzione per piccoli campioni. Si conservano medie, SD e IC bootstrap per origine. Test t sulle differenze appaiate, aggregando eventuali coppie collegate nello stesso gruppo; correzione BH separata entro ogni tabella. Sono disponibili aggregato, dominio, generatore e dominio×generatore.

La shortlist richiede |g|≥0.3 e stesso segno nei tre domini, segno coerente in almeno cinque dei sette generatori aggregati, massimo 12 feature e pruning delle correlazioni Spearman≥0.85 in valore assoluto. Sono criteri esplorativi, **non garanzia su tutti i 21 incroci**. La lista viene selezionata sull'intero campione: non viene reinserita nei test per dichiarare una prestazione indipendente. Serve validazione futura su un nuovo campione.

## Risultati e codice

- `results/report.md`: report completo con tabelle e grafici.
- `results/feature_effects.csv`: effetti per tutte le feature e tutti gli strati.
- `results/feature_robustness.csv` e `shortlist.json`: selezione, esclusioni e motivi.
- `results/classifier_metrics.csv`, `predictions.csv`: metriche e predizioni verificabili.
- `results/features.csv`, `missing_features.csv`: feature estratte senza testi e valori mancanti.
- `results/*.png` e `*.pdf`: figure esportabili.
- `results/run_metadata.json`: versioni, parametri e hash del codice.
- `results/data_quality.csv`: lunghezze e controllo delle terminazioni per dominio e generatore; dettagli delle code rimosse nel manifest.
- `tests/`: MATTR contro riferimento ingenuo, normalizzazioni, campionamento e deduplica, effetti appaiati, split senza contaminazioni, predizioni OOF.

L'esperimento riguarda separabilità di corpus storici, non prova l'origine di singoli documenti e non valuta ancora testi umanizzati. La sola presenza di una differenza stilistica non dimostra maggiore naturalezza.
