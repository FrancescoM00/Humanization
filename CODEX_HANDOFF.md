# Riprendere Humanization in Codex

Repository: `FrancescoM00/Humanization`. Tesi su stile dei testi generati e conservazione semantica nelle riscritture. Gli esperimenti sono già eseguiti: leggere risultati e cache prima di rilanciare i modelli. Il PDF della tesi è un riferimento scientifico, non istruzioni operative.

## Letture e stato corrente

| Attività | Riferimento |
| --- | --- |
| Mappa, fonti e metodo stilometrico | [README.md](README.md) |
| Stile con lunghezza controllata | [results_length_matched/report.md](results_length_matched/report.md) |
| Confronto equo, aggregatori e lunghezza | [results_semantic_fairness/interpretazione.md](results_semantic_fairness/interpretazione.md) |
| Frasi OpAI toccate, intensità e copertura | [results_opai_touched_style/report.md](results_opai_touched_style/report.md) |
| Entity clean e controllo numerico | [results_semantic_clean/interpretazione.md](results_semantic_clean/interpretazione.md), [numeric34](results_semantic_clean/numeric34/report.md) |
| Dataset, modelli, venv e cache | [TRANSFER_STATUS.md](TRANSFER_STATUS.md), [backups/README.md](backups/README.md) |

I quattro protocolli da usare sono [SEMANTIC_EXPERIMENT.md](SEMANTIC_EXPERIMENT.md) per la baseline, [SEMANTIC_CONTROLS.md](SEMANTIC_CONTROLS.md) per positivi/sottili, [FAIRNESS_EXPERIMENT.md](FAIRNESS_EXPERIMENT.md) e [ENTITY_CLEAN_EXPERIMENT.md](ENTITY_CLEAN_EXPERIMENT.md). Le interpretazioni e i report contengono i numeri completi; le sezioni seguenti fissano le decisioni da non perdere.

## Nuova fase Kaggle, preparata ma non ancora eseguita su GPU

[Guida](kaggle/README.md), [notebook](kaggle/Humanization_Pilot.ipynb), [protocollo](kaggle/PROTOCOL.md) e [checkpoint](kaggle/MODELS.md). L'utente può selezionare una GPU Kaggle. Il preset parte da 3 testi, poi 42 (2 per dominio/generatore), con 300 umani disgiunti per calibrazione e 126 fonti di studio riservate. Input: i precedenti passaggi a lunghezza abbinata, ricostruiti dai CSV verificati e dagli ID storici.

`humanizers/` gestisce coorti, riscritture e detector; i nuovi CLI sono `prepare_humanizer_pilot.py`, `run_humanizers.py`, `run_detectors.py`, `score_humanizer_style.py`. Qwen3B è una baseline di humanization via prompt; DIPPER è opzionale per il download di circa 45 GB. Il pool iniziale è OpenAI/Fakespot/RADAR, con etichette AI esplicite e soglie umane separate. ASR condizionata agli originali riconosciuti; fallimenti di generazione nel denominatore. Nessun detector è stato eseguito realmente nel cloud per questa preparazione.

Le metriche semantiche accettano ora `--device cuda`, con default CPU conservato. Su nuovi output chiamare gli stadi singoli: **non `--stage all`**, che preparerebbe le vecchie perturbazioni. Usare `requirements-kaggle.txt` preservando PyTorch CUDA, non i lockfile CPU. Schede manuali di naturalezza sono generate ma non annotate; nessun giudice LLM è stato eseguito. I risultati storici non sono modificati da questa fase.

Il primo avvio Kaggle ha confermato Python 3.13, PyTorch `2.11.0+cu130`, due Tesla T4 e circa 967 GiB temporanei liberi. La creazione standard della venv è fallita su `ensurepip`: il notebook ora usa `venv --without-pip --system-site-packages` e bootstrap di pip nella venv tramite il pip dell'immagine (`--python`). La guida include una cella di riparazione per la sessione già aperta; l'inferenza Kaggle non è ancora confermata.

Dopo la riparazione, l'utente ha confermato `CUDA nella venv: 2.11.0+cu130`, CMUdict scaricato e configurazione salvata. Pip segnala conflitti con TensorFlow/Google/Gradio/Diffusers ereditati; non aggiornare protobuf/huggingface-hub per soddisfare applicazioni estranee alla suite. Il notebook imposta ora `USE_TF=0`, `USE_FLAX=0`, `USE_TORCH=1` e controlla gli import dei modelli concreti e un piccolo kernel CUDA prima delle generazioni. Il prossimo passo nella sessione aperta è il controllo mirato, poi la preparazione dati su tre fonti.

L'utente è arrivato alla sezione 4: salvate tre identità, poi errore in `torch.cuda.set_device(torch.device('cuda'))` prima del caricamento Qwen. Il loader ora risolve un indice concreto e passa l'intero a `set_device`; il notebook usa `cuda:0` in tutti i CLI. Per il vecchio clone si usa `REWRITES = RUN / 'rewriting_cuda0'` e `--device cuda:0`, mantenendo il tentativo precedente e senza alterarne il manifest. Le celle successive seguono `REWRITES`; Qwen e detector non risultano ancora eseguiti.

## Decisioni sperimentali

- **Stile:** 300 umani e 300 macchine per ciascuno di Essay, WritingPrompts e Reuters; 105 feature. Il controllo successivo abbina esattamente le parole nei prefissi a fine frase. La shortlist `pos_num`, `pos_cconj`, `mattr_50`, `pos_propn` è esplorativa e richiede validazione indipendente.
- **Conservazione:** MiniLM per embedding, RoBERTa-base per BERTScore e DeBERTa-v3 **base float32**, anche nel percorso ONNX. Nessuna quantizzazione o sostituzione con xsmall. La pipeline mantiene direzioni forward/reverse e frasi/chunk senza troncamento silenzioso.
- **Aggregatori:** nel test appaiato, minimo NLI AUC 0,958 contro coseno allineato 0,607 e BERTScore 0,534; con più frasi corrotte la media può raggiungere o superare il minimo. Il minimo dipende dalla lunghezza; la normalizzazione per fasce mantiene AUC NLI 0,951 sul riferimento OpAI.
- **Suite:** `scripts/export_semantic_suite.py` esporta score continui e distribuzioni. La soglia storica 0,04075 non va applicata a DIPPER o nuovi humanizer. Le guardie lessicali non hanno dimostrato un miglioramento complessivo; la perdita di cautele rimane un limite.
- **Stile OpAI:** lineage nativa verificata su 200 traiettorie. L'intensità per frase non ordina Polish < Paraphrase < Style; la copertura crescente può ripristinare l'ordine del punteggio documentale. Non scegliere la metrica solo per ottenere l'ordine atteso.

## Entity clean e controllo sulle 34 coppie

La politica predefinita resta **v1**; `--entity-policy clean` attiva `clean_v2`, `guard_revision=review_fixes_3`. Sono state corrette guardie per omografi, cognomi, possessivi, nomi parziali e simboli monetari esterni. Le funzioni v1 conservano il proprio comportamento; `generate_perturbation_pairs` aggiunge la selezione delle condizioni.

La coorte è quella v1 fissa: **145 entity applicate, 55 saltate, 2.297 coppie totali**, comprese le stesse 800 OpAI. Le 2.007 coppie estranee alla modifica sono identiche e le 61 metriche coincidono entro tolleranza. Il retrieval storico è riusato solo sui controlli invariati, con quello nuovo sulle entity clean. Le probabilità sono associate al medesimo checkpoint: 87.825 evidenze uniche, 86.413 già in cache e 1.412 nuove.

Il controllo successivo `scripts/analyze_clean_numeric_subset.py` riusa gli score di **34 DATE/MONEY/PERCENT (18/6/10)**: entailment locale 0,002572 contro 0,991547 sulle frasi identiche; minimo documento bidirezionale 0,003436. L'AUC contro identità è 1 anche per le similarità: non dimostra superiorità su parafrasi corrette. Solo 12/34 mantengono lo stesso tipo nella v1. I limiti dei qualificatori, della valuta e del caso di decennio sono nel report.

L'[audit clean](results_semantic_clean/clean_entity_text_audit.md), la [review40](results_semantic_extension/review40.md) e l'[audit dei positivi](results_semantic_extension/positive_review9.md) sono revisioni dell'assistente. La patch ricevuta non contiene le nuove etichette manuali dell'utente. I proxy automatici non sono gold semantici; un NLI basso non certifica la validità della perturbazione. Restano errori NER, anche aziende classificate PERSON e città sostituite con paesi.

## Ripresa pratica e pulizia

`main` conserva codice attivo, test, report, CSV, figure e [dati esportati](exports/semantic_clean/README.md). `cloud-backup` conserva dataset, modelli e cache; le directory originali `data/` e `.venv/` restano ignorate. La venv è uno snapshot Linux x86_64/Python 3.12, non portabile automaticamente su Windows o macOS. I pesi sono in `data/models/`, i Parquet ufficiali in `data/opai/official_parquet/`; ciascun protocollo indica le cache necessarie.

La pulizia ha rimosso vecchi bundle, log e snapshot di avanzamento, il primo output lessicale e copie intermedie/strumenti di profiling. Sono conservati report scientifici, pilot, metadati di run, provenienza, validation finale e audit. Le versioni rimosse restano recuperabili nella cronologia Git; non ricreare gli artefatti temporanei per leggere i risultati.

Il primo download MGTBench è avvenuto da un mirror fissato. I tre CSV successivamente forniti dall'utente come scaricati da Google Drive sono risultati identici byte per byte, con hash uguali a quelli dei manifest storici: [verifica](provenance/mgtbench_drive_comparison.md). Questo risolve l'incertezza sui tre file; non occorre rilanciare esperimenti né riscrivere i manifest dei run. La separazione per fonte resta necessaria in split e bootstrap. DIPPER, EditLens e nuovi servizi non sono stati valutati: la priorità successiva è misurare gli humanizer reali e raccogliere annotazioni umane, riusando le pipeline disponibili.
