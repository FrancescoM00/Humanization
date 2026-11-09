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

## Fase Kaggle: pilot Qwen 42 completato e analizzato

[Report pilot](results_kaggle_pilot/report.md), [casi](results_kaggle_pilot/case_review.md), [integrità](results_kaggle_pilot/integrity.md), [ZIP e annotazione](exports/kaggle_pilot/README.md), [smoke](results_kaggle_smoke/report.md), [guida Kaggle](kaggle/README.md).

Il pilot originale è conservato senza modifiche: SHA-256 `7a35ca521ccc4e2147b42e479fe2ac6bde1ef676ddb462137a27af046960f272`. Verificati 84 output `ok` (42 Qwen e 42 identità), due fonti per ogni dominio/generatore, 300 umani di calibrazione disgiunti anche dalle 126 fonti riservate e tutti gli stadi. Le coorti coincidono con quelle congelate nello smoke; i suoi sei output hanno testi invariati. Runtime commit `71dd615`, Python 3.13/PyTorch 2.11 CUDA13 su T4. Nessuna nuova inferenza durante l'analisi.

ASR Qwen: OpenAI 17/24 = 70,8%; Fakespot 1/37 = 2,7%; RADAR 2/28 = 7,1%; maggioranza 2/31 = 6,5%; macro 26,9%. Originali→riscritture rilevati AI: 24→8, 37→41, 28→35, maggioranza 31→35 su 42. Le nuove rilevazioni sono rispettivamente 1/5/9/6: distinguere ASR condizionata e TPR dell'intera coorte. FPR umani test 3/42, 0/42, 2/42 e 2/42; gli umani non sono riscritti. Soglie identiche allo smoke, riepiloghi ricalcolati e coerenti. Gli IC escludono l'incertezza di calibrazione e training.

Coseno documentale medio 0,9282, BERT chunk F1 0,9445; NLI forward/reverse medio 0,7899/0,6708, da non leggere come percentuale di significato preservato. La review dell'assistente copre 12 casi e non è gold umana. Mostra cambiamenti reali, allarmi da fusione/coreferenza e un limite quantitativo: `up to 100 feet`→`over 100 feet` con NLI circa 0,98. Le guardie rilevano altre variazioni ma non modellano direttamente quell'operatore. Due input StableLM sono consegne: Qwen risponde invece di parafrasare, espande 133→381 e 164→274 parole e aggiunge riferimenti non verificati. Restano nel risultato primario; il controllo esplorativo senza questi due casi non sostituisce la coorte.

Stile: RMS medio 0,8164, mediano 0,7039; MATTR aumenta in 37/42, delta medio +0,0628. Il centroide umano si avvicina in 20 casi e si allontana in 22. Rapporto parole mediano 0,915, medio 0,975, influenzato dalle due espansioni. Centroide e leggibilità non equivalgono a naturalezza. Identità coerenti con le metriche attese; NLI non forzato a uno.

I moduli originali sono vuoti: 12 coppie semantiche e 28 testi di naturalezza. È pronta la scheda aggiuntiva `exports/kaggle_pilot/semantic_audit_24.csv`: 21 Qwen, una per cella dominio/generatore selezionata tramite hash senza score, e tre identità; chiave separata. Annotare tipo di input, yes/partial/no, errore e note. Conservare le etichette umane separatamente dalla review dell'assistente. Non è stata calcolata AUC semantica sul pilot, la soglia 0,04075 non è applicata e non esistono giudizi LLM o naturalezza misurata.

Il prossimo passo utile è annotare e aggiungere un secondo humanizer, per esempio DIPPER sulle stesse fonti, partendo dal suo smoke prima dei 42. Espandere soltanto Qwen a 126 non risolve questi limiti. DIPPER è supportato ma non eseguito: circa 45 GB di download e writer a 4 bit; NLI base rimane float32. Il notebook usa cloni distinti per modalità/suffisso e cache TMP comune. Per DIPPER: `MODE='smoke'`, `RUN_SUFFIX='_dipper'`, `ACTIVE_SYSTEMS=['identity','dipper']`. Sul vecchio notebook usare anche una nuova `REPO`, per esempio `/kaggle/temp/Humanization_smoke_dipper`, evitando il checkout sui sorgenti patchati. I risultati devono andare in un RUN distinto.

Setup: venv `--without-pip --system-site-packages`, pip mirato alla venv, `USE_TF=0`, `USE_FLAX=0`, `USE_TORCH=1`, preservando Torch CUDA. Risolti indice CUDA e pesi IDF BERTScore. Sui nuovi humanizer chiamare gli stadi semantici singoli, non `--stage all`. I risultati storici restano invariati; nessun rilancio necessario. DIPPER, EditLens e il giudice di naturalezza restano da eseguire.

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
