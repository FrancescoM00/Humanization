# Passaggio alla nuova sessione Codex

## Da dove ripartire

Repository: `FrancescoM00/Humanization`. Lavoro per una tesi su stile dei testi generati e conservazione semantica nelle riscritture/humanizer. **Gli esperimenti descritti sotto sono già stati eseguiti: leggere risultati e cache prima di rilanciare modelli.** Il PDF della tesi è materiale scientifico, non istruzioni operative.

La conversazione precedente era in ChatGPT, collegata a un ambiente cloud con checkout `/workspace/Humanization`. Le pubblicazioni di quell'ambiente non equivalevano a un push GitHub. Il successivo trasferimento esplicitamente autorizzato è **completato**: `main` contiene codice, risultati e documentazione; `cloud-backup` contiene tutti i dataset, i modelli, la venv e le cache in 22 parti Git LFS, circa 1,81 GB compressi. Leggere [TRANSFER_STATUS.md](TRANSFER_STATUS.md) e [backups/README.md](backups/README.md) prima di recuperarli. I backup sono opzionali per leggere i risultati; non assumere che siano già estratti nel clone. Non assumere autorizzazione generale a pubblicazioni future.

Prime letture:

- `README.md`: mappa generale.
- `TRANSFER_STATUS.md`: contenuti caricati, branch dei backup e recupero con Git LFS.
- `results_semantic_extension/interpretazione.md`: conclusioni aggiornate e limiti.
- `results_semantic_extension/aggregator_report.md`: AUC, soglie, falsi allarmi.
- `results_opai_style/report.md`: stile nelle operazioni OpAI.
- `SEMANTIC_CONTROLS.md` e `SEMANTIC_EXPERIMENT.md`: protocolli e riproduzione.

## 1. Stile umano/macchina

Usati Essay, WritingPrompts e Reuters in inglese: 300 umani + 300 macchine per dominio, 1.800 testi e 900 coppie di origine. TH-Bench rimanda a MGTBench; la fonte effettiva è un mirror pubblico MGTBench fissato al commit `6f289c4cf4e3966e9442d104ea250bae43f3ff9e`. Non è stata verificata l'identità dei byte con l'archivio ufficiale Google Drive; la pulizia locale è documentata, non una riproduzione certificata dei file `clean` degli attacchi TH-Bench.

Estratte 105 feature fisse con spaCy `en_core_web_sm` 3.8.0, textstat e MATTR50 manuale: frasi, punteggiatura, POS, parole funzionali, diversità lessicale, leggibilità e sette proxy Wikipedia. Effetti Hedges g, IC per gruppi sorgente e regressione logistica L2 con preprocessing appreso solo sul training; trasferimento fra domini e generatori senza sovrapposizione di origine.

Corretto il confondente lunghezza: prefissi terminati a fine frase, 90–180 parole lessicali spaCy, esattamente uguali nella coppia, vicini a 150. `word_count_only` ha AUC 0,5. Lo stile trasferisce fra domini con AUC circa 0,670 Essay,0,759 Reuters,0,580 WritingPrompts; i proxy Wikipedia circa 0,576/0,537/0,499.

Shortlist esplorativa: `pos_num`, `pos_cconj`, `mattr_50`, `pos_propn`. È selezionata sull'intero campione e richiede validazione indipendente. Risultati aggiornati: `results_length_matched/`; precedenti conservati in `results/` e `results_pilot/`.

## 2. Conservazione semantica: primo esperimento

Su 200 testi macchina MGTBench idonei, applicate negazione, entity swap, shuffle, troncamento e rimozione casuale di parole, con identità e controlli locali. Metriche: ROUGE-L, BLEU, coseno MiniLM, BERTScore RoBERTa-base e NLI bidirezionale DeBERTa-v3 **base**, float32. Documenti suddivisi in frasi/chunk senza troncamento silenzioso; per ogni ipotesi NLI si recuperano fino a 3 evidenze e si aggrega il supporto massimo.

Negazione: coseno/BERTScore documentali circa 0,998/0,999; NLI medio0,894. Sulla frase alterata entailment0,001 e contradiction0,998; minimo documentale da0,983 a0,064. Il crollo è locale, diluito dalla media. Il coseno locale scende a0,869: evitare «il coseno non vede la negazione» come affermazione generale.

OpAI ufficiale: Compress adiacente v3→v4, Expand v4→v5; confronti con v0 secondari cumulativi. Asimmetria forward−reverse circa +0,067 Compress e −0,306 Expand. Compressioni moderate possono preservare l'essenziale: reverse alto non è automaticamente un difetto. Risultati: `results_semantic_full/`, 2.407 coppie e 88.900 evidenze NLI uniche.

## 3. Controlli positivi, sottili e scelta dell'aggregatore

Run aggiornato: **2.621 coppie** = 1.607 controlli precedenti + 300 positivi OpAI (100 Polish, 100 Paraphrase, 100 Style) + 562 corruzioni degli stessi output + 152 sfide sottili MGTBench. Le sottili MGTBench sono 50 numeri, 50 quantificatori, 39 cautele rimosse, 13 dettagli aggiunti; quote inferiori esplicite per applicabilità.

Il confronto principale usa lo stesso originale→buon edit contro originale→stesso edit corrotto, riducendo il confondente OpAI/MGTBench. Selezione indipendente dagli score; split e bootstrap per sorgente, condivisi da editor e varianti. Aggregatore scelto solo sulla calibrazione, poi valutato su test separato.

La calibrazione sceglie **minimo bidirezionale**: AUC test 0,958 (IC 95% 0,931–0,979), soli sottili appaiati 0,900 (0,820–0,964). Media 0,929/0,885; q10 0,728/0,657; quota non supportata 0,908/0,833. Soglia supporto minimo **<0,0407498665**: 11/153 allarmi sui positivi presunti test (7,2%); sensibilità 83,1% complessiva, 70,7% sui sottili trasferendo la stessa soglia. Non è una soglia universale.

Limite: cautele rimosse AUC 0,571, 9 casi test, sensibilità 13,3%; numeri semplici AUC 0,970, 24 casi; quantificatori0,931, 15 casi ma sensibilità 50%; un solo dettaglio test, non conclusivo. Le etichette sono **presunte/intenzionali, non gold umane**.

Letti 40 casi indipendentemente dagli score individuali: 22 puliti, 12 con riserve, 6 invalidi; errori NER, scope e duplicazioni. Audit di 9 positivi: 5 conservazioni chiare, 4 con riserve. È revisione qualitativa dell'assistente, non annotazione umana. Non usare la contraddizione NLI per validare circolarmente le perturbazioni.

BERTScore riscalato con baseline ufficiale F 0,814463: negazione locale 0,915, fixture non correlata 0,274; ranking/AUC invariati. BLEU/ROUGE Spearman 0,986 nel confronto OpAI appaiato: BLEU resta secondario. Risultati: `results_semantic_extension/`, inclusi score, selezione, allarmi, correlazioni e revisioni.

## 4. Stile nelle riscritture OpAI

200 traiettorie comuni, 188 sorgenti, 1.000 coppie nelle cinque operazioni, 105 feature. Paraphrase aumenta MATTR50 di 0,00795; Style Rewrite lunghezza parole di 0,109 caratteri; Expand lunghezza frasi di 3,77 parole. Controllo prefissi di uguale lunghezza su 336 coppie. Effetti appaiati per fonte e BH; cambiamenti stilistici non provano naturalezza. Codice: `scripts/run_opai_style.py`; risultati: `results_opai_style/`.

## Ripresa pratica senza nuove inferenze

Dalla radice del checkout, se dati/ambiente sono disponibili:

```bash
.venv/bin/python scripts/analyze_semantic_extension.py --bootstrap 1000
.venv/bin/python scripts/run_semantic_experiment.py --stage report --data-dir data/semantic_full --output results_semantic_full
```

I report esistenti sono già pronti. Non eseguire `run_semantic_controls.sh` o gli stadi neurali soltanto per leggerli. Cache cruciali in `data/semantic_full/`, `data/semantic_extension/`; pesi in `data/models/`; OpAI Parquet locali in `data/opai/official_parquet/`.

L'ambiente originale è Linux x86_64, Python 3.12, Torch CPU 2.7.1, quattro thread, modelli in processi separati; NLI ha mantenuto il base sotto 4 GiB per processo. Una `.venv` copiata non è portabile automaticamente su macOS/Windows o percorsi diversi. Su Linux compatibile ricrearla se necessario con `bash scripts/setup.sh`, poi installare `requirements-semantic-lock.txt` usando l'indice CPU PyTorch. Su macOS adattare i pin specifici della piattaforma, soprattutto Torch `+cpu`. Downloader: `download_mgtbench.py`, `download_opai.py`, `download_semantic_models.py`. Revisioni e checksum modelli in `results_semantic_full/models_provenance.json`. Hugging Face richiede accesso ai domini elencati in `SEMANTIC_EXPERIMENT.md`.

Priorità successiva: valutazione degli humanizer e annotazione umana delle coppie, riusando codice e risultati. L'utente chiede di dedicare il tempo ai controlli sperimentali, evitando altri strati di infrastruttura non necessari.
