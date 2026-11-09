# Humanization: stile e conservazione semantica

Esperimenti per la tesi su testi umani, generazioni macchina e riscritture. I risultati sono già disponibili: per consultarli non occorre rilanciare i modelli. Il PDF della tesi è materiale scientifico, non istruzioni operative. Versioni precedenti e strumenti rimossi durante la pulizia sono recuperabili dalla cronologia Git.

La **[sezione sperimentale della tesi](thesis/README.md)** raccoglie metodi, risultati, discussione e architettura del codice: [PDF](thesis/sezione_sperimentale.pdf), [LaTeX completo](thesis/sezione_sperimentale_completa.tex), [frammento integrabile](thesis/sezione_sperimentale.tex) e [Markdown](thesis/sezione_sperimentale.md).

Il passo successivo è pronto nel **[notebook Kaggle](kaggle/README.md)**: prova tecnica su 3 fonti, pilot su 42, humanization via prompt, tre detector e riuso della suite semantica/stilometrica su GPU. DIPPER è opzionale; i risultati reali di questo nuovo esperimento devono ancora essere raccolti su Kaggle.

## Risultati e protocolli

| Esperimento | Risultati | Riproduzione |
| --- | --- | --- |
| Stile con lunghezza abbinata | [Report](results_length_matched/report.md) | `scripts/run_length_matched_style.py` |
| Baseline stilometrica e controllo del preprocessing | [Interpretazione](results/interpretazione.md), [pilot](results_pilot/report.md) | `scripts/run_style_experiment.py` |
| Baseline semantica, Compress/Expand | [Interpretazione](results_semantic_full/interpretazione.md) | [SEMANTIC_EXPERIMENT.md](SEMANTIC_EXPERIMENT.md) |
| Positivi OpAI e perturbazioni sottili | [Interpretazione](results_semantic_extension/interpretazione.md), [review40](results_semantic_extension/review40.md) | [SEMANTIC_CONTROLS.md](SEMANTIC_CONTROLS.md) |
| Similarità allineate, errori distribuiti e lunghezza | [Interpretazione](results_semantic_fairness/interpretazione.md) | [FAIRNESS_EXPERIMENT.md](FAIRNESS_EXPERIMENT.md) |
| Stile nelle frasi OpAI toccate | [Report e casi](results_opai_touched_style/report.md) | `scripts/run_opai_touched_style.py` |
| Entity clean v2 e controllo numerico | [Interpretazione](results_semantic_clean/interpretazione.md), [34 date/importi/percentuali](results_semantic_clean/numeric34/report.md) | [ENTITY_CLEAN_EXPERIMENT.md](ENTITY_CLEAN_EXPERIMENT.md) |

La suite produce [score continui](results_semantic_fairness/continuous_suite/manifest.json); la soglia storica OpAI 0,04075 non si trasferisce ad altri humanizer. NLI mantiene un vantaggio sulle corruzioni testate, ma il minimo dipende dalla lunghezza e può essere superato dalla media su errori distribuiti. Le cautele restano problematiche.

## Dati, modelli e ripresa

- [Scarica pairs.jsonl e skipped.jsonl in ZIP](exports/semantic_clean/semantic_clean_pairs.zip), oppure apri i [file esportati](exports/semantic_clean/README.md): 2.297 coppie del run clean, con 145 sostituzioni applicate e 55 saltate.
- `main` contiene codice, test, report, CSV e figure. `cloud-backup` conserva dataset, modelli, venv e cache tramite Git LFS: [inventario](TRANSFER_STATUS.md) e [ripristino](backups/README.md).
- Per continuare in Codex leggere [CODEX_HANDOFF.md](CODEX_HANDOFF.md). La venv archiviata è uno snapshot Linux x86_64/Python 3.12, da ricreare su Windows, macOS o percorsi incompatibili.

La politica delle entità rimane `v1` per default; `--entity-policy clean` attiva le guardie nuove. Clean significa più conservativa, non gold semantica: rimangono errori NER e di sottotipo, descritti nell'[audit](results_semantic_clean/clean_entity_text_audit.md). Le review disponibili sono dell'assistente; le nuove annotazioni manuali dell'utente non sono contenute nella patch ricevuta. DIPPER, EditLens e nuovi servizi di humanization non sono stati eseguiti.

## Eseguire lo stile

Dalla radice del checkout, con Python 3.12:

```bash
bash scripts/setup.sh
.venv/bin/python scripts/download_mgtbench.py --output data/raw
.venv/bin/python scripts/run_style_experiment.py --raw-dir data/raw --output results
.venv/bin/python scripts/run_length_matched_style.py
.venv/bin/python -m pytest -q
```

Saltare il download se i CSV esistono; il downloader rifiuta sovrascritture. `setup.sh` installa dipendenze fissate, spaCy `en_core_web_sm` 3.8.0 e CMUdict. Dipendenze e ambiente sono in `requirements.txt` e `requirements-lock.txt`; per la semantica usare i file `requirements-semantic*` e i protocolli sopra.

Per ricalcolare l'analisi stilometrica da una cache compatibile usare `scripts/run_style_experiment.py --reuse-features`. Il seed è 42, `--pairs 300` indica 300 umani e 300 macchine per dominio, `--bootstrap 500` controlla gli IC. La cache è associata a campione, estrattore, schema, modello e versioni.

## Fonte e preparazione MGTBench

TH-Bench, commit [`5c1d99c`](https://github.com/DrenfongWong/TH-Bench/tree/5c1d99cefee49208206b6fa75d3a628ac9935f86), rimanda ai CSV MGTBench su [Google Drive](https://drive.google.com/drive/folders/1p4iBeM4r-sUKe8TnS4DcYlxvQagcmola). I suoi file `clean` degli attacchi contengono soltanto macchina e non corrispondono direttamente al disegno umano/macchina richiesto.

Il primo download è avvenuto da un [mirror pubblico di ricerca](https://github.com/derek33125/AITextDetection/tree/6f289c4cf4e3966e9442d104ea250bae43f3ff9e/Data_Generation/MGTBench), fissato a `6f289c4cf4e3966e9442d104ea250bae43f3ff9e`: `Essay_LLMs.csv`, `WP_LLMs.csv`, `Reuters_LLMs.csv`, 1.000 righe ciascuno, umano e sette generatori allineati. **I tre CSV successivamente forniti dall'autore come scaricati da Google Drive sono identici byte per byte**, anche rispetto agli hash registrati nei manifest dei run: [confronto e SHA-256](provenance/mgtbench_drive_comparison.md). Nessun risultato deve essere ricalcolato; i manifest storici conservano lo stato della verifica al momento dei run. Per CSV ufficiali già disponibili usare `--raw-dir`; il download `--source official` richiede `gdown==5.2.0` opzionale e accesso a Google Drive.

La pulizia locale applica `process_spaces`, elimina simmetricamente code senza terminazione plausibile, richiede almeno 50 parole whitespace dopo il taglio, deduplica casefold/whitespace ed esclude macchine identiche a umani. L'euristica gestisce virgolette, abbreviazioni e decimali; non certifica lingua, completezza o assenza di rifiuti/chatbot. Il preprocessing è locale, non TH-Bench certificato.

Per dominio: 300 umani unici, un output macchina per riga; 43 per ChatGPT-turbo, Claude, ChatGLM, Dolly, ChatGPT e GPT4All, 42 StableLM. `pair_id` allinea le coppie, `group_id` collega prompt ripetuti. Fonti ed esclusioni sono in `data/raw/SOURCE.json` e `results/data_manifest.json`. Pulire prima del campionamento può cambiare le fonti: il confronto con il pilot delle terminazioni non è causale. Il pilot si riproduce con `--keep-incomplete --output results_pilot`.

Il successivo controllo di lunghezza usa prefissi a fine frase con conteggi lessicali spaCy identici nella coppia, fra 90 e 180 parole e vicini a 150; `word_count_only` ha AUC 0,5. Caratteri e numero di frasi restano diagnostici distinti.

## Feature e analisi stilometrica

Le 105 feature comprendono lunghezza/variabilità delle frasi, punteggiatura, 16 POS, 58 parole funzionali e frequenza totale, MATTR50 manuale, lessico, leggibilità e sette proxy Wikipedia. Definizioni: [feature_schema.json](results/feature_schema.json) e [features.py](stylometry/features.py).

Parole lessicali: token `is_alpha`, casefold, senza lemmatizzazione. MATTR considera tutte le finestre di 50 parole e restituisce NaN sotto 50. POS: token non spazio/non punteggiatura, inclusi numeri e frammenti di contrazioni. Le parole funzionali sono forme letterali (`n't` resta distinto da `not`); la punteggiatura conta caratteri anche nelle abbreviazioni. Textstat mantiene le sue convenzioni, con sillabificazione `en_US` e CMUdict locale.

I proxy Wikipedia fissati dal capitolo 2 sono vocabolario AI, enfasi, conclusioni, formule conversazionali, parallelismi negativi, Markdown ed em dash. Non coprono rule of three, profondità semantica, neutralità o correttezza delle citazioni: si verificano questi proxy, non l'intera guida.

Logistica L2, C=1, classi bilanciate e soglia 0,5; imputazione e scale apprese sul training. Modelli: `style` (esclude conteggi globali, TTR/hapax e proxy), `wikipedia`, `combined`, `length_only`. Confronti: CV a cinque fold per fonte nel dominio, due domini→terzo, generatore escluso con le sue fonti escluse dal training. AUC, balanced accuracy e macro-F1 aggregano predizioni out-of-fold. Gli IC bootstrap per fonte sulle predizioni fisse escludono la variabilità dell'addestramento.

Hedges g: macchina−umano diviso SD pooled, correzione per piccoli campioni, IC per fonte. Test appaiati aggregano coppie collegate; BH per tabella, risultati per dominio/generatore/incrocio. Shortlist: |g|≥0,3, stesso segno nei tre domini e in almeno cinque generatori su sette, massimo 12 feature, pruning Spearman≥0,85. La selezione sull'intero campione è esplorativa e non garantisce i 21 incroci. Candidate: `pos_num`, `pos_cconj`, `mattr_50`, `pos_propn`.

I risultati misurano separabilità e cambiamenti stilistici; non certificano l'origine di un singolo testo o maggiore naturalezza. Condizioni dei corpora e attribuzioni del codice sono in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
