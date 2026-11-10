# Report cumulativo della sperimentazione

Aggiornamento: **10 novembre 2026**. Questo documento raccoglie il percorso sperimentale, collega i report completi e distingue risultati ottenuti, prove preliminari e attività ancora da svolgere. I pilot baseline 3B e DIPPER sulle stesse 42 fonti e lo smoke v2 sono stati ricevuti e analizzati; lo study da 126 è stato ricevuto: 626 esiti validi su 630, detector/semantica/stile disponibili e naturalezza parziale per limiti API.

## 1. Obiettivo e stato dei quattro assi

La suite valuta le riscritture rispetto a quattro domande distinte: eludono i detector? Conservano il contenuto? Quanto cambiano lo stile? Risultano naturali? Un miglioramento su un asse non implica un miglioramento sugli altri.

| Asse | Lavoro completato | Stato e limite principale |
| --- | --- | --- |
| Evasione | Tre classificatori, Binoculars zero-shot, calibrazione umana, ASR condizionata, transizioni prima/dopo e controlli nulli | Quattro riscrittori sulle stesse 126 fonti, con analisi separata delle 84 nuove; evasione dipendente dal detector |
| Conservazione | Perturbazioni controllate, positivi OpAI, confronti allineati, aggregatori, lunghezza, guardie e applicazione ai nuovi output | Metriche operative; sui nuovi humanizer manca un riferimento semantico umano indipendente |
| Stile | Umano/macchina a lunghezza abbinata, regressione logistica, trasferimento, frasi OpAI toccate e distanze sulle riscritture | Differenze misurate; non equivalgono a qualità o naturalezza |
| Naturalezza | Schede a testo singolo e giudice Groq AB/BA realmente eseguito nello smoke v2 | Study: 364/744 risposte nuove valide, tre confronti parziali e molte preferenze instabili all'ordine; nessuna validazione umana |

La validazione iniziale degli strumenti e la loro applicazione agli humanizer sono esperimenti differenti. Una metrica efficace sulle corruzioni artificiali può avere falsi allarmi o mancare errori nelle riscritture reali.

## 2. Dati, provenienza e unità di analisi

La base stilometrica comprende **900 coppie umano/macchina, 1.800 testi**: 300 umani e 300 macchine per ciascuno dei domini inglesi Essay, Reuters e WritingPrompts. I generatori storici sono ChatGLM, ChatGPT, ChatGPT-turbo, Claude, Dolly, GPT4All e StableLM.

I tre CSV sono stati inizialmente acquisiti da un mirror fissato; quelli successivamente caricati dall'autore come scaricati dal Drive indicato sono risultati identici byte per byte. Il confronto è documentato in [provenance/mgtbench_drive_comparison.md](provenance/mgtbench_drive_comparison.md). La fonte effettivamente analizzata è la baseline MGTBench: non si presenta il campione come una replica di tutti i file di attacco TH-Bench.

Per controllare la lunghezza sono stati ricavati prefissi terminati a fine frase, di 90–180 parole lessicali, con conteggio **esattamente uguale entro ogni coppia**. L'oggetto dello studio è quindi il passaggio iniziale, non l'intero documento. La pulizia non certifica che ogni testo macchina sia prosa adeguata: il pilot ha evidenziato anche input formulati come consegne.

Le trasformazioni dello stesso originale condividono una fonte. Più coppie, più frasi o più score non diventano automaticamente nuove osservazioni indipendenti. Split e ricampionamenti tengono conto dei gruppi di origine.

## 3. Prima fase: costruzione e verifica delle misure di stile

Con spaCy piccolo, textstat e MATTR implementato nel progetto sono state estratte **105 feature**: lunghezza e variabilità delle frasi, punteggiatura, distribuzioni POS, parole funzionali, diversità lessicale, leggibilità e proxy della guida Wikipedia sui testi AI. Le definizioni e i denominatori sono salvati insieme ai risultati.

È stata corretta una confusione iniziale fra stile e lunghezza mediante il matching esatto delle parole. Dopo il matching, il classificatore basato soltanto sul numero di parole ha AUC 0,500. Le differenze fra le altre feature vengono analizzate con differenza standardizzata delle medie, intervalli e controlli per dominio/generatore. Hedges g esprime la differenza rispetto alla variabilità; Benjamini–Hochberg corregge i confronti multipli. Sono strumenti statistici adottati nel protocollo per descrivere la grandezza degli effetti e controllare i confronti multipli.

La regressione logistica sulle feature stilistiche dà AUC entro dominio di **0,876 in Essay, 0,928 in Reuters e 0,881 in WritingPrompts**. Escludendo il dominio di test dall'addestramento, scende rispettivamente a **0,670, 0,759 e 0,580**. Esiste quindi segnale stilometrico, ma una sua parte dipende dal dominio.

La shortlist esplorativa comprende `pos_num`, `pos_cconj`, `mattr_50` e `pos_propn`. È stata selezionata usando l'intero campione e non viene retroattivamente presentata come scelta indipendente dai test. I proxy Wikipedia risultano molto più deboli: AUC entro dominio 0,569/0,549/0,510. Questo riguarda gli indicatori effettivamente implementati, non l'intera guida o tutti i suoi giudizi qualitativi.

Per i dettagli: [report a lunghezza abbinata](results_length_matched/report.md), [esperimento iniziale](results/interpretazione.md).

## 4. Seconda fase: validazione della conservazione semantica

### Perturbazioni e direzionalità

La baseline semantica seleziona **200 testi macchina** e produce trasformazioni controllate: negazione, sostituzione di entità, ordine delle frasi, troncamento e cancellazione di parole, oltre ai controlli. Le coppie OpAI Compress ed Expand verificano la direzionalità. Il run storico comprende 2.407 coppie fra trasformazioni, controlli, fixture e 800 coppie OpAI; non sono 2.407 fonti indipendenti.

Sono stati implementati ROUGE-L, BLEU, embedding MiniLM, BERTScore RoBERTa-base e NLI bidirezionale con **DeBERTa-v3 base float32**, mantenuto anche nelle successive esecuzioni GPU. Per i testi oltre il contesto si usano frasi/chunk e aggregazioni dichiarate.

Con la negazione, coseno e BERTScore documentali rimangono circa 0,998 e 0,999; NLI forward medio è circa 0,894. Sulla frase modificata, invece, entailment è circa 0,00113 e contraddizione 0,9976. Il coseno locale scende a circa 0,869: non è corretto estendere la scarsa sensibilità documentale a ogni forma di similarità locale. La contraddizione del modello non è, da sola, una validazione indipendente della perturbazione.

Dopo il troncamento, il supporto originale → testo corto è circa 0,992, mentre nella direzione opposta è circa 0,669. Le operazioni OpAI hanno intensità differenti: le compressioni leggere possono mantenere gran parte dell'informazione e dare asimmetria contenuta. Le direzioni sono compatibili con perdita o aggiunta di informazione, ma non identificano da sole la causa di ogni calo.

[Baseline semantica](results_semantic_full/interpretazione.md), [protocollo](SEMANTIC_EXPERIMENT.md).

### Controlli positivi, aggregatori e confronto equo

Sono state aggiunte riscritture OpAI Polish, Paraphrase e Style Rewrite come **positivi presunti**, perturbazioni più sottili su numeri, quantificatori, cautele e dettagli aggiunti, e il confronto fra media, q10, minimo e quota non supportata. I positivi del benchmark non sono etichette umane perfette: anche una riscrittura conservativa può introdurre piccole variazioni.

Il confronto è stato reso più equo calcolando anche coseno e BERTScore su unità allineate, con supporto massimo fra frasi e gli stessi aggregatori sensibili al caso peggiore. Nel test appaiato su **141 positivi presunti e 265 alterazioni, provenienti da 48 fonti**, il minimo NLI ha AUC **0,958**, contro **0,607** per coseno allineato e **0,534** per BERTScore allineato. È un risultato sui controlli definiti, non un'accuratezza già dimostrata sugli humanizer reali.

Le corruzioni su più frasi mostrano che il minimo non domina sempre: con due negazioni, per esempio, la media raggiunge AUC 0,989 contro 0,965 del minimo. Sui positivi il minimo NLI diminuisce al crescere del numero di frasi, con correlazione Spearman −0,316; normalizzando per fasce apprese dalla calibrazione, l'AUC del minimo è 0,951. I confronti fra humanizer devono quindi considerare lunghezza e numero di unità.

È stato chiarito il riscalamento BERTScore rispetto alla baseline ufficiale, F1 0,814463 per il modello/layer usato. Il riscalamento rende leggibile la scala ma non cambia ranking o AUC. BLEU rimane una misura secondaria: nel confronto OpAI appaiato la correlazione con ROUGE-L è 0,986.

Le guardie lessicali contano variazioni di cautele, negazioni, numeri e quantificatori. Non risolvono automaticamente gli errori sottili: nel confronto controllato NLI+guardie dà AUC 0,938 contro 0,958 dell'NLI solo. La rimozione delle cautele resta un limite, con NLI AUC 0,571 sul piccolo sottogruppo esaminato.

La soglia storica 0,04075 non viene trasferita ai nuovi humanizer. La suite esporta punteggi continui e distribuzioni; un minimo basso è un segnale da ispezionare, non un verdetto automatico.

[Controlli positivi e sottili](results_semantic_extension/interpretazione.md), [confronto equo](results_semantic_fairness/interpretazione.md), [protocollo](FAIRNESS_EXPERIMENT.md).

### Entità clean e controllo numerico

La patch ricevuta è stata valutata e integrata come condizione separata, preservando la politica v1 e i risultati precedenti. Il run clean applica **145 sostituzioni su 200 fonti**, con 55 skip e 2.297 coppie totali. Le coppie estranee al cambiamento sono conservate; i risultati sono salvati in directory nuove. La revisione delle regole riduce alcuni errori ma non rende la NER o le perturbazioni una gold semantica.

Nel controllo successivo su **34 fonti** — 18 date, 6 importi, 10 percentuali — l'entailment locale delle alterazioni è 0,002572, contro 0,991547 sulle identità; il minimo documentale bidirezionale è 0,003436. Coseno e BERTScore documentali rimangono rispettivamente 0,998363 e 0,999082. Anche le similarità separano perfettamente queste alterazioni dalle identità: tale confronto facile non dimostra superiorità NLI sulle parafrasi corrette.

[Entity clean](results_semantic_clean/interpretazione.md), [controllo numerico](results_semantic_clean/numeric34/report.md), [protocollo](ENTITY_CLEAN_EXPERIMENT.md).

## 5. Terza fase: stile delle riscritture OpAI

Sono state analizzate le operazioni Polish, Paraphrase, Style Rewrite, Compress ed Expand sulle traiettorie OpAI, prima sul documento e poi sulle sole frasi toccate. Gli identificatori delle frasi sono ricostruiti dalla lineage posizionale nativa verificata, attraverso i campi `sentences` e `sentence_labels`; il rilascio non include direttamente `sentence_ids`. L'obiettivo è distinguere la quantità di testo modificata dall'intensità della modifica.

Le quattro feature della shortlist non mostrano variazioni uniformi per tutte le operazioni. Per evitare una lettura basata soltanto su singole feature è stata costruita una distanza RMS dei delta standardizzati e una distanza Jensen–Shannon delle POS.

Su **103 traiettorie test e 99 fonti**, la RMS per frase è 0,366 per Polish, 0,717 per Paraphrase e 0,656 per Style Rewrite: l'ordine completo atteso non emerge. Pesando per la copertura del testo, i valori diventano 0,152/0,369/0,425. Questo distingue intensità locale e quota modificata; non autorizza a scegliere una metrica soltanto perché restituisce l'ordine desiderato. Una distanza stilometrica non è una misura di naturalezza.

[Stile OpAI sul documento](results_opai_style/report.md), [frasi toccate](results_opai_touched_style/report.md).

## 6. Quarta fase: humanizer e detector reali su Kaggle

### Sistemi, soglie e controlli

Il primo sistema è **Qwen2.5-3B-Instruct con prompt fisso di humanization**: è una baseline di riscrittura via prompt, non un giudice di naturalezza o un servizio commerciale. Il secondo è **DIPPER XXL a 4 bit**, con diversità lessicale 60, ordine 0 e intero passaggio senza prefisso contestuale. Questa variante non replica il protocollo originale di DIPPER a finestre di tre frasi. Entrambi usano una sola generazione greedy per fonte, senza selezione mediante i detector.

I detector sono OpenAI RoBERTa-base, Fakespot RoBERTa-base nella variante raw e RADAR RoBERTa-large. Le revisioni e l'indice della classe AI sono espliciti. Le soglie si fissano su **300 umani di calibrazione, 100 per dominio**, separati per gruppo di fonte dai test. Il FPR empirico individuale di calibrazione è il 5%; non è una garanzia sui dati futuri. Il voto di maggioranza combina i verdetti binari, senza mediare gli score grezzi.

L'ASR usa come denominatore gli originali che il detector riconosceva già come AI. Un originale già sfuggito non diventa un successo dello humanizer. Le generazioni fallite restano insuccessi; gli errori di scoring non vengono interpretati come classificazioni umane. Sono riportati anche i movimenti opposti e il TPR sull'intera coorte.

Sono stati risolti i problemi osservati nella reale immagine Kaggle: bootstrap della venv senza `ensurepip`, conservazione di PyTorch CUDA, indice esplicito `cuda:0` e pesi BERTScore trasferiti sul dispositivo dei vettori. L'esecuzione usa subprocess nella venv, cache condivisa e directory distinte per modalità e sistema. L'NLI rimane base float32 anche con DIPPER quantizzato.

### Baseline Qwen2.5-3B-Instruct con prompt fissato: pilot completato

Il pilot comprende **42 fonti**, due per ciascuna combinazione di tre domini e sette generatori, con 42 riscritture e 42 identità tecnicamente riuscite.

| Detector | Successi / originali rilevati | ASR | Originali → riscritture rilevate AI, su 42 |
| --- | ---: | ---: | ---: |
| OpenAI | 17/24 | 70,8% | 24 → 8 |
| Fakespot raw | 1/37 | 2,7% | 37 → 41 |
| RADAR | 2/28 | 7,1% | 28 → 35 |
| Maggioranza | 2/31 | 6,5% | 31 → 35 |

L'evasione si concentra sul detector storico OpenAI. Fakespot e RADAR riconoscono anche diversi originali che prima sfuggivano. Il risultato non è riassumibile come una percentuale unica di testi che inganna ogni detector.

Coseno documentale medio 0,9282 e BERTScore 0,9445 coesistono con NLI forward/reverse medio 0,7899/0,6708. Questi ultimi non sono percentuali di significato preservato. La lettura dei casi mostra omissioni, aggiunte, falsi allarmi da fusione/coreferenza e un limite quantitativo: `up to 100 feet` diventa `over 100 feet` con NLI comunque vicino a 0,98.

Due input StableLM sono consegne: baseline 3B risponde invece di riscriverle, espandendo 133→381 e 164→274 parole. Restano nell'analisi primaria; l'esclusione esplorativa dei due non sostituisce il campione congelato. Le review dell'assistente non sono annotazioni umane.

La distanza RMS media è 0,8164; MATTR aumenta in 37/42 casi, ma il centroide umano si avvicina solo in 20/42. Lo stile cambia senza una direzione universalmente più umana. Sono disponibili le [schede da annotare](exports/kaggle_pilot/README.md); la naturalezza non era misurata nel pilot originario; i successivi giudizi Groq sono descritti nelle sezioni 9 e 10.

[Report baseline 3B](results_kaggle_pilot/report.md), [casi](results_kaggle_pilot/case_review.md), [archivio e moduli](exports/kaggle_pilot/README.md).

### DIPPER XXL NF4 sul passaggio intero senza prefisso: smoke e pilot completati

Lo smoke usa gli stessi tre input ChatGLM già esaminati per Qwen. Tutte le riscritture terminano con EOS; nessuna raggiunge il limite d'uscita. Ogni detector continua a riconoscerle AI: ASR 0/3. DIPPER accorcia i testi del 18–27%.

Reuters perde il riferimento ai problemi legali dell'azienda: NLI forward medio 0,984, reverse 0,679, minimo reverse 0,000334 proprio sulla frase omessa. Essay mostra anche un allarme da fusione di due frasi; WritingPrompts cambia il meccanismo del potere del protagonista ed elimina la risposta finale di Sarah. Sono esempi leggibili, non una stima dell'accuratezza semantica o una classifica degli humanizer.

Il pilot DIPPER sugli stessi 42 input è **completo**: 42 riscritture e 42 identità, tutte con stato `ok`, nessun arresto al limite di generazione. L'archivio ricevuto è `humanization_pilotdipper.zip`; configurazione, dati, soglie e scale restano quelli precedenti. Il confronto baseline 3B–DIPPER è appaiato per fonte.

| Detector | baseline 3B: successi / originali rilevati | DIPPER: successi / originali rilevati | Differenza ASR DIPPER−baseline 3B |
| --- | ---: | ---: | ---: |
| OpenAI | 17/24 | 4/24 | −54,2 punti percentuali |
| Fakespot raw | 1/37 | 7/37 | +16,2 punti percentuali |
| RADAR | 2/28 | 3/28 | +3,6 punti percentuali |
| Maggioranza | 2/31 | 2/31 | 0 punti percentuali |

L'ordine dipende dal detector. Il bootstrap appaiato esplorativo dà intervalli 95% della differenza di [−75,0; −29,2] punti per OpenAI, [+5,4; +27,0] per Fakespot, [−7,1; +17,9] per RADAR e [−12,9; +12,9] per maggioranza, mantenendo fissi modelli e calibrazione. I due successi di maggioranza riguardano fonti completamente diverse. Dopo DIPPER le rilevazioni complessive sono 25/42 OpenAI, 33/42 Fakespot, 33/42 RADAR e 32/42 maggioranza: alcune fonti prima non rilevate diventano AI, perciò ASR e TPR complessivo vanno letti separatamente.

DIPPER accorcia 41/42 testi, con rapporto parole medio 0,739 e mediano 0,766; 23 output scendono sotto l'80% della lunghezza iniziale. baseline 3B ha rapporto mediano 0,915. MATTR aumenta in 13/42 DIPPER contro 37/42 baseline 3B; il centroide umano si avvicina in 13/42 contro 20/42. La RMS media è 0,7765 per DIPPER e 0,8164 per baseline 3B, mentre le mediane invertono l'ordine. Distanza e leggibilità non determinano la naturalezza.

Coseno/BERTScore documentali medi DIPPER sono 0,9215/0,9478 e NLI forward/reverse 0,6840/0,5949. La lettura di 12 coppie mostra omissioni e variazioni puntuali, fra cui euro→franchi e 100→30 piedi, ma anche falsi allarmi del protocollo a premessa singola: nel racconto Maria/Sofia la fusione da 15 a cinque frasi accompagna forward 0,00262 pur mantenendo gran parte della trama. Le medie non diventano una classifica di fedeltà senza etichette indipendenti. Le due consegne StableLM già note restano nel risultato primario; l'analisi secondaria da 40 non risolve tutti i problemi di qualità degli input.

Sono pronte [schede comuni](exports/kaggle_dipper_pilot/README.md) senza score: 45 coppie semantiche (i due sistemi sulle stesse 21 fonti, più tre identità) e 24 testi di naturalezza su sei fonti comuni con umani, originali e riscritture. Le chiavi sono separate; nessuna etichetta è stata compilata. Le precedenti 24 coppie Qwen/identità sono incluse e non vanno contate due volte se già giudicate.

[Report pilot DIPPER e confronto](results_kaggle_dipper_pilot/report.md), [casi](results_kaggle_dipper_pilot/case_review.md), [confronto ASR appaiato](results_kaggle_dipper_pilot/paired_comparison.md), [ZIP e schede](exports/kaggle_dipper_pilot/README.md). Lo [smoke DIPPER](results_kaggle_dipper_smoke/report.md) rimane documentato separatamente.

## 7. Perché 3, 42 e 126 testi: significato e limiti della numerosità

La riduzione del campione riguarda la nuova fase humanizer, non tutti gli esperimenti: lo stile usa già 1.800 testi e la validazione semantica centinaia di fonti e numerose trasformazioni. Le numerosità della fase Kaggle sono state scelte per procedere dal collaudo all'esplorazione, **non mediante un calcolo che ne garantisca la sufficienza statistica**.

| Pannello | Fonti macchina per humanizer | Ripartizione | Stato/scopo |
| --- | ---: | --- | --- |
| Smoke | 3 | Una per dominio, tutte ChatGLM | Collaudo di caricamento, generazione e suite |
| Pilot | 42 | 2 × 3 domini × 7 generatori | Primo confronto bilanciato, completato per entrambi i sistemi |
| Study | 126 | 6 × 3 × 7 | Ricevuto: quattro riscrittori più identità, 626/630 esiti validi; naturalezza parziale |
| Estensione rinviata | 420 | 20 × 3 × 7 | Tecnicamente fattibile, rinviata fino alla verifica del protocollo v2 su 126 fonti |

I 300 umani di calibrazione servono a fissare soglie e scale. Non aggiungono 300 casi al denominatore dell'evasione. Due sistemi sugli stessi 42 input danno 84 riscritture ma mantengono **42 fonti nel confronto appaiato**. Identità e tre detector non moltiplicano le fonti indipendenti. Smoke, pilot e study sono annidati: non si sommano 3+42+126 come campioni distinti.

Nel pilot, un caso su 42 vale circa 2,38 punti percentuali. Nell'ASR i denominatori sono ancora più piccoli: per OpenAI un successo su 24 vale 4,17 punti. L'intervallo bootstrap già osservato per l'ASR OpenAI va circa dal 52,0% all'87,5%. In ogni cella dominio/generatore ci sono due fonti: un solo caso può cambiare il tasso di 50 punti, o più se il denominatore condizionato è uno. I risultati per cella sono quindi esplorativi. Le 126 fonti lasciano soltanto sei elementi per cella: i sottogruppi restano esplorativi.

Una verifica dei metadati delle 900 fonti, escludendo i 300 gruppi di calibrazione attuali, lascia **600 fonti macchina utilizzabili**, comprese le 126 riservate. Ogni cella ha almeno 23 fonti residue: un campione bilanciato da 420 fonti è fattibile, e il massimo perfettamente bilanciato con questa separazione è 483 = 23×21. Questi limiti di disponibilità non sono una dimostrazione di potenza statistica né un controllo semantico della qualità degli input.

Per estendere oltre il pannello previsto occorre conservare esplicitamente i 300 gruppi di calibrazione e creare una nuova coorte con criteri basati su metadati/ID, includendo gli input correnti. **Non basta aumentare `reserve_per_cell`**: il selettore attuale riserva prima i test e poi sceglie la calibrazione, che potrebbe così cambiare. La nuova coorte dovrà applicarsi a entrambi gli humanizer e mantenere tracciati versione e soglie.

La dimensione finale va fissata in base alla precisione desiderata per il confronto complessivo e per gli eventuali sottogruppi, considerando i denominatori ASR e il budget. Aumentare le fonti migliora la precisione, ma non risolve mancanza di annotazioni, bias del corpus, limiti dei detector o errori delle metriche. I pilot restano invariati; il pannello da 126 è ora eseguito e ricevuto, mentre le 420 fonti restano rinviate.

## 8. Composizione del repository e riproducibilità

| Percorso | Funzione |
| --- | --- |
| `stylometry/` | Caricamento, feature, matching della lunghezza, analisi statistica e report di stile |
| `semantics/` | Perturbazioni, OpAI, metriche, allineamento, aggregatori, guardie e stile delle frasi toccate |
| `humanizers/` | Coorti congelate, riscrittura, detector e riepiloghi dell'evasione |
| `scripts/` | Comandi per preparazione, scoring, confronti e esportazione; i protocolli indicano quelli pertinenti |
| `kaggle/` | Notebook, configurazione, revisioni dei modelli, protocollo e istruzioni GPU |
| `results*` | Output numerici, interpretazioni e metadati degli esperimenti completati |
| `exports/` | Archivi ricevuti/condivisibili e schede manuali, con chiavi separate |
| `provenance/` | Verifica della provenienza dei CSV forniti |
| `thesis/` | Sezione sperimentale in Markdown, LaTeX e PDF; precedente alla nuova fase Kaggle |
| `CODEX_HANDOFF.md` | Stato operativo, decisioni e indicazioni per riprendere il lavoro |

Le directory locali `data/` e `.venv/` non sono versionate in `main`; gli archivi e i report pertinenti sono esportati. Il ramo `cloud-backup` conserva i materiali pesanti secondo [TRANSFER_STATUS.md](TRANSFER_STATUS.md). L'ambiente Kaggle viene ricreato con dipendenze dedicate e PyTorch CUDA dell'immagine. Gli ZIP dei run non contengono pesi o venv.

Coorti, configurazioni, revisioni dei checkpoint, hash delle coppie e metadati rendono verificabili gli input effettivi e le condizioni. I controlli già completati coprono split, identità, denominatori, copertura dei testi e coerenza dei risultati salvati. Non sono necessarie nuove infrastrutture di verifica prima dell'analisi dei prossimi dati.

## 9. Protocollo v2 e smoke completato prima delle 126 fonti

Il feedback successivo ai pilot ha spostato il centro dell'esperimento dalle sole evasioni alla **variazione complessiva di rilevabilità**. Dopo la baseline 3B, Fakespot passa da 37 a 41 rilevazioni e RADAR da 28 a 35. Queste transizioni sono una candidata a risultato centrale da confermare; il 70,8% di ASR sul detector storico OpenAI non descrive il successo generale dello humanizer.

Il [protocollo v2](STUDY_V2_PROTOCOL.md) mantiene le 126 fonti già congelate e i 300 gruppi di calibrazione. Include gli input-consegna nel risultato primario e prepara una classificazione del tipo sul solo originale. Le analisi sulla sola prosa richiedono etichette di tipo e conteggi espliciti; i flag automatici non escludono fonti. Le 420 fonti sono rinviate. Per le ipotesi sviluppate osservando il pilot, le **84 fonti non viste** vanno riportate anche separatamente dalle 42 già analizzate.

### 9.1. Nuove condizioni effettivamente eseguite su Kaggle

Lo [smoke v2 completo](results_kaggle_v2_smoke/report.md), ricevuto il 9 novembre 2026, contiene tre fonti ChatGLM, una per dominio, e **15 esiti tutti `ok`**: identità, baseline Qwen2.5-3B FP16, Qwen2.5-14B NF4 con lo stesso prompt e DIPPER XXL NF4 nelle condizioni passaggio intero e finestre di tre frasi/prefisso riscritto cumulativo. Quest'ultima differisce comunque dal protocollo originale per segmentazione spaCy e generazione greedy con budget per finestra. Non si isola causalmente la dimensione del modello, la quantizzazione o la sola finestra.

Sono completi i classificatori, **Binoculars zero-shot**, controlli nulli, sei stadi semantici, stile, NLI contestuale e giudice Groq. [Archivio originale](exports/kaggle_v2_smoke/README.md), [audit](results_kaggle_v2_smoke/integrity.md) e [review dei dodici output](results_kaggle_v2_smoke/text_review.md) conservano dati e limiti. I riepiloghi sono stati ricalcolati dagli score ricevuti senza nuove inferenze. I pannelli e le scale stilometriche coincidono con quelli congelati; i testi storici 3B e DIPPER passaggio non sono cambiati.

| Riscrittore | ASR OpenAI | ASR Fakespot | ASR RADAR | ASR Binoculars | ASR maggioranza RoBERTa |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 3B | 2/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| 14B NF4 | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| DIPPER passaggio | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| DIPPER tre frasi | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |

Tutti i tre originali sono rilevati da ciascun detector; nessuno dei tre umani test lo è. Le frequenze su tre fonti sono un collaudo e non stimano prestazioni generali. Binoculars esegue realmente Falcon 7B base/instruct su due T4 FP16, con statistiche FP32 e copertura registrata. Usa il ratio grezzo, AI se minore di **0,9046559856180912**, calibrato su 300 umani con 15 falsi positivi; non è una probabilità. La maggioranza resta quella separata dei tre classificatori correlati. Fast-DetectGPT e humanizer commerciali non sono stati aggiunti.

### 9.2. Lunghezza, rumore e stile

I controlli nulli sono ora valutati da tutti e quattro i detector e non producono evasioni nei casi eleggibili. Su OpenAI, 14B evade in 3/3 casi e i prefissi alla stessa lunghezza in 0/3; baseline 3B evade in 1/2 confronti eleggibili contro 0/2 prefissi, con l'Essay espanso escluso dal controllo. Il denominatore del prefisso con rumore è talvolta inferiore perché manca una sostituzione lessicale applicabile. Non si presenta il rumore come gold semantico. L'accorciamento da solo non riproduce queste evasioni, senza autorizzare una conclusione causale generale.

DIPPER a tre frasi accorcia meno in tutti e tre i casi: rapporto parole medio **0,892** contro **0,771** del passaggio intero. Recupera contenuto, ma conserva errori, come ricavi trasformati in profitti in Reuters. Il 14B ha rapporto medio **0,753** e può aggiungere o cambiare affermazioni: più grande non significa automaticamente più fedele. Le distanze RMS medie sono 0,880 per baseline 3B, 1,560 per 14B, 0,571 per DIPPER passaggio e 0,487 per DIPPER tre frasi. Misurano variazione nelle feature, non qualità o naturalezza.

### 9.3. NLI contestuale e similarità allineate

La [prova CPU precedente](results_nli_context_pilot/report.md) rimane distinta: 19 coppie esplicative, 5.446 confronti con DeBERTa-v3 base float32. Nel caso Maria/Sofia il forward medio passa da 0,0038 con singole premesse esaustive a 0,7994 con finestre fino a tre e 0,9820 con il documento esatto. Il documento assegna però soltanto 0,6624 a un controllo identico; non viene adottato come soluzione automatica.

Nel nuovo smoke sono completati **3.594 confronti contestuali e 120 aggregati**, tutti con copertura completa su questi testi. Nel WritingPrompts DIPPER tre frasi il reverse storico è 0,307, quello con finestre fino a tre 0,754 e con documento 0,985: una premessa singola può penalizzare fusioni e coreferenza. Viceversa, il 14B Reuters ha forward circa 0,990 con finestre fino a tre pur introducendo modifiche verificabili. Le finestre sono annidate e prendono un massimo: l'aumento è in parte imposto dalla costruzione, non prova di accuratezza.

Coseno e BERTScore restano riportati sia sul documento sia come minimi allineati; il nuovo report confronta esplicitamente queste aggregazioni. Non si trasferisce l'AUC 0,958 dei controlli ai nuovi humanizer, non si interpreta NLI come percentuale di significato conservato e non si applica la soglia OpAI 0,04075.

### 9.4. Giudice di naturalezza: risposte reali, stabilità limitata

Il runtime Kaggle conferma **`qwen/qwen3.8-27b`** e **`reasoning_effort=none`**. Fra vecchia prova e nuovi confronti sono presenti **30 risposte valide su 32 tentativi POST**. Nel preflight, quattro preferenze su sei sono stabili all'ordine e tutte concordano col proxy dell'assistente; due sono discordanti. Il 100% è accordo AI–AI condizionato a 4/6 casi, non validazione umana; il kappa è indefinito perché tutte le preferenze stabili appartengono alla stessa classe.

Dei **nove nuovi confronti**, soltanto **cinque sono stabili AB/BA**. Gli altri quattro scelgono sempre il testo mostrato come **B in entrambi gli ordini**: il vincitore cambia con la posizione, senza un errore rilevato nel parser. Tutti e tre i confronti WritingPrompts sono instabili. Questa fragilità impedisce di presentare una graduatoria affidabile di naturalezza. I casi instabili restano mancanti con denominatore esplicito; non si ripetono chiamate fino a ottenere accordo e non si cambiano prompt sulla base di questi esiti.

### 9.5. Annotazione e avvio del pannello esteso

Il [sampler di annotazione v2](docs/ANNOTATION_V2.md) separa un reference scelto senza score e un diagnostico arricchito per NLI basso e coseno alto/NLI basso, con moduli per due annotatori. Nello smoke ci sono 12 coppie non identità, tutte reference e nessuna diagnostica; le schede sono vuote. Nel pilot ci sono soltanto 84 coppie non identità: il censimento disponibile non può diventare 100 casi nuovi. Il target diagnostico di 100 si applicherà al pannello esteso. Un'AUC sul solo campione diagnostico non stimerebbe ingenuamente la prestazione nella popolazione.

Il [notebook aggiornato](kaggle/README_Study_v2.md) prepara l'avvio sulle 126 senza ripetere lo smoke e senza cambiare i preset. Riusa venv e cache complete leggibili della sessione, colloca checkout e file temporanei sotto `/kaggle/working` e mantiene i pesi fuori dai risultati. Le modifiche di storage sono verificate localmente; al momento dell’analisi dello smoke il run sulle 126 non era ancora avviato. L'archivio ricevuto documenta anche le correzioni operative applicate durante il run, tramite hash di codice e sorgenti inclusi, oltre al commit iniziale.

Le riprese operative hanno permesso di completare lo scoring sulle 126 fonti: i nuovi download Falcon usano una cache scrivibile e il giudice mantiene le generazioni fallite come non valutabili, senza sostituirle. L'archivio è ora ricevuto e analizzato nella sezione seguente; le tre comparazioni Groq rimangono parziali per budget di richieste e limite giornaliero di token.

## 10. Study v2 ricevuto: quattro riscrittori su 126 fonti

[Report completo](results_kaggle_v2_study/report.md), [detector e controlli](results_kaggle_v2_study/detection_analysis.md), [semantica/stile](results_kaggle_v2_study/semantic_style.md), [naturalezza](results_kaggle_v2_study/naturalness.md), [casi letti](results_kaggle_v2_study/case_review.md), [audit](results_kaggle_v2_study/integrity.md), [archivio e annotazioni](exports/kaggle_v2_study/README.md).

### 10.1. Completezza e fallimenti

Lo ZIP contiene **630 esiti: 126 identità e 504 tentativi dei quattro riscrittori**. Sono validi 626 esiti, cioè 500 riscritture e tutte le identità. Le quattro generazioni troncate appartengono a tre fonti delle 84 nuove: la baseline 3B su Essay 223 e Reuters 992, StableLM, esegue la consegna interna e raggiunge 512 token senza EOS; entrambi i DIPPER su WritingPrompts 184, GPT4All, entrano in ripetizioni, con fallimento già nella prima finestra del contestuale. Non sono errori CUDA dimostrati e non sono stati rigenerati.

Coorti, 300 gruppi di calibrazione, scale e soglie coincidono con quelli congelati. I 42 output condivisi per baseline 3B e DIPPER passaggio sono identici ai pilot. Gli input-consegna restano nel risultato primario; nuove letture mostrano che il problema non si limita ai due input StableLM già noti. Un'analisi sulla sola prosa richiede etichette sugli originali e criteri espliciti, non esclusioni dopo aver visto gli output.

### 10.2. L'evasione cambia con il detector

| Preset | OpenAI RoBERTa | Fakespot | RADAR | Binoculars | Maggioranza RoBERTa |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 3B | 45/73 = 61,6% | 1/109 = 0,9% | 8/90 = 8,9% | 17/120 = 14,2% | 4/94 = 4,3% |
| 14B NF4 | 60/73 = 82,2% | 1/109 = 0,9% | 8/90 = 8,9% | 29/120 = 24,2% | 6/94 = 6,4% |
| DIPPER passaggio | 10/73 = 13,7% | 11/109 = 10,1% | 7/90 = 7,8% | 0/120 = 0,0% | 7/94 = 7,4% |
| DIPPER tre frasi | 6/73 = 8,2% | 8/109 = 7,3% | 11/90 = 12,2% | 1/120 = 0,8% | 6/94 = 6,4% |

Le frazioni sono ASR sugli originali già rilevati e mantengono i fallimenti come mancati successi. OpenAI rileva soltanto 73/126 originali: il risultato storico non è il titolo del successo generale. Binoculars rileva 120/126 originali e mostra evasioni non osservate nello smoke. La maggioranza rimane quella correlata dei soli tre RoBERTa.

Sulle **84 fonti non viste nei pilot**, baseline 3B e 14B evadono OpenAI in 28/49 e 41/49 casi, Binoculars in 9/80 e 21/80; nessuno evade Fakespot. Il vantaggio ASR 14B−3B su Binoculars è **+15 punti percentuali, IC bootstrap appaiato 95% [6,3; 23,8]**. I confronti mantengono fissi checkpoint e calibrazione; non isolano causalmente la dimensione del modello, perché cambiano anche i pesi e la quantizzazione.

La candidata a risultato nata dai pilot si precisa: **Fakespot diventa più efficace sulle riscritture a prompt**. Sulle 84 nuove fonti passa da 70/82 a 82/82 rilevazioni per 3B e da 72/84 a 84/84 per 14B, con aumenti appaiati di +14,6 [7,3; 23,2] e +14,3 [7,1; 22,6] punti. Su RADAR gli aumenti sulle 84 hanno intervalli che includono zero. Non si conclude che tutti i detector recenti migliorino: con Binoculars il verso è opposto. I TPR appaiati usano gli stessi output validi prima/dopo, senza confondere il denominatore con l'ASR.

### 10.3. Quanto conta accorciare

Sulle nuove fonti baseline-AI eleggibili, OpenAI viene evaso dalla baseline 3B in 24/41 casi contro 1/41 prefissi alla stessa lunghezza, e dal 14B in 36/42 contro 4/42. Binoculars: 3B 5/62 contro 0/62; 14B 19/68 contro 0/68. Il vantaggio non è riprodotto dal solo prefisso, sul sottoinsieme abbinato osservato; gli output allungati rimangono non eleggibili e non si assume un effetto causale completamente identificato.

Per DIPPER su OpenAI non emerge un vantaggio aggregato chiaro sul taglio: sulle 126 fonti eleggibili passaggio e prefisso hanno entrambi 10/72 evasioni, contestuale e prefisso entrambi 5/64. **Le evasioni sono però spesso su casi diversi**, quindi non si afferma che la stessa evasione sia spiegata dalla lunghezza. Sul passaggio ci sono otto successi soltanto del controllo e otto soltanto dello humanizer; nel contestuale cinque e cinque, disgiunti. Rumore minimo, combinazioni e intervalli sono nel report dedicato.

### 10.4. Stile e semantica sui veri output

| Preset | Output validi | Rapporto parole medio | RMS stilometrico medio | Coseno documento | BERTScore grezzo | NLI storico medio forward/reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 3B | 124 | 0,962 | 0,831 | 0,937 | 0,946 | 0,785 / 0,696 |
| 14B | 126 | 0,871 | 1,047 | 0,903 | 0,927 | 0,714 / 0,645 |
| DIPPER passaggio | 125 | 0,758 | 0,796 | 0,925 | 0,948 | 0,693 / 0,602 |
| DIPPER tre frasi | 125 | 0,920 | 0,651 | 0,951 | 0,964 | 0,863 / 0,731 |

RMS usa 91 feature retained delle 105 estratte, standardizzate sui 300 umani di calibrazione, non solo la shortlist iniziale. MATTR aumenta mediamente di 0,0616 per 3B e 0,0813 per 14B; diminuisce di 0,0286 e 0,0262 per i due DIPPER. Il centroide umano si avvicina in 60/124, 61/126, 29/125 e 45/125 casi: non c'è un miglioramento stilistico o di naturalezza automaticamente dimostrato. Le tendenze rimangono nelle 84 fonti nuove e sulle 123 con successo comune a tutti i sistemi.

DIPPER a tre frasi comprime meno del passaggio e ha un profilo semantico medio più favorevole. La review delle fonti nuove trova però sia omissioni recuperate sia errori: dollari/euro e importi alterati con coseno circa 0,939; anno 2019 trasformato nell'anno corrente e possibilità rafforzata nonostante NLI circa 0,99. Anche un supporto alto può mancare una deriva. Le letture sono dell'assistente e non stimano la frequenza degli errori.

NLI contestuale completa **199.126 confronti** con DeBERTa-v3 base float32. Le finestre e le singole esaustive sono complete; quattro aggregati reverse del documento restano mancanti per budget. Il documento penalizza anche identità (supporto medio circa 0,896 contro 0,992 con singole esaustive). Il retrieval storico differisce dalla ricerca esaustiva nuova; le finestre annidate prendono un massimo e crescono per costruzione. Non si sceglie l'aggregatore più favorevole dopo aver visto i testi. Restano disponibili minimi allineati di coseno e BERTScore riscalato; non si trasferiscono soglia OpAI 0,04075 o AUC 0,958 agli humanizer.

### 10.5. Naturalezza parziale e sensibile alla posizione

| Confronto | Risposte valide / attese | Tentativi POST | Coppie complete AB/BA | Preferenze stabili / 126 |
| --- | ---: | ---: | ---: | ---: |
| Baseline 3B / 14B | 136/248 | 256 | 68 | 32/126 |
| Baseline 3B / DIPPER tre frasi | 128/246 | 256 | 64 | 46/126 |
| DIPPER passaggio / tre frasi | 100/250 | 201 | 50 | 15/126 |

I primi due confronti esauriscono il budget; il terzo si ferma per HTTP 429 sul limite giornaliero di token. Le nuove comparazioni producono 364 risposte valide su 713 tentativi. Il preflight sui sei casi dei pilot registra altri 12 eventi API con timestamp diversi dallo smoke; non sono nuove fonti o annotazioni umane. Totale nell'archivio: 725 tentativi, 376 risposte valide e 209.610 token registrati. Non si attribuisce automaticamente ogni tentativo senza risposta a un particolare errore né si ricostruisce da questi numeri l'intera quota dell'account.

Delle 182 coppie con entrambi gli ordini, **89 sono discordanti e 85 di queste scelgono B in entrambi gli ordini**. B è preferito in 263/364 risposte. I vincitori fra i soli casi stabili non sono una graduatoria delle 126 fonti: il blocco API lascia un prefisso degli ID ordinati per hash e la consistenza all'ordine seleziona ulteriormente il campione. Anche sulle 84 nuove le preferenze stabili sono soltanto 20, 28 e nove. Il modello e l'opzione senza ragionamento sono confermati, ma manca validazione umana e la sensibilità alla posizione resta un limite forte.

Non sono state effettuate altre chiamate nell'analisi dello ZIP. Un'eventuale ripresa richiederebbe continuare gli ID mancanti, conservare i log e dichiarare esplicitamente l'aumento dei budget già esauriti; il solo reset giornaliero non basta a completare i primi due confronti. Prima di spendere altra quota conviene valutare il giudice con riferimenti indipendenti.

### 10.6. Schede pronte e limiti ancora aperti

Lo [ZIP cieco da annotare](exports/kaggle_v2_study/annotation_forms.zip) comprende due copie indipendenti delle stesse **183 coppie**, senza sistemi o score: 84 reference da 21 fonti e 99 diagnostiche da ulteriori 25. La selezione a gruppi completi porta il diagnostico a 99 pur avendo capacità maggiore; non è corretto dichiarare cento, duplicare esempi o presentare l'arricchimento per score come campione casuale. Le chiavi sono separate. Le schede sono vuote e i giudizi di due annotatori non aumentano le fonti indipendenti.

Il reference permette una revisione meno condizionata dagli score; il diagnostico concentra l'attenzione sui disaccordi informativi. La valutazione manuale deve distinguere significato, naturalezza e tipo di input. Non è ancora disponibile un'accuratezza gold delle metriche sui veri humanizer. I file compilati della tesi precedono questa fase: il report è aggiornato, non si dichiara una revisione LaTeX/PDF già avvenuta.

## 11. Attività successive e confini delle conclusioni

1. Usare il confronto appaiato completato per documentare i comportamenti dei due preset e i limiti emersi, senza modificare a posteriori i risultati.
2. Raccogliere annotazioni indipendenti di conservazione e naturalezza sulle schede comuni. Le schede preparate non sono risultati e le review dell'assistente non sostituiscono giudizi umani; la fusione delle frasi è un caso prioritario per misurare i falsi allarmi dell'NLI.
3. Usare il pannello da 126 ricevuto e le 84 fonti nuove per i confronti fissati, senza ripetere le inferenze. Valutare eventuale completamento Groq solo con budget espliciti e dopo aver considerato la fragilità all’ordine. Il passaggio a 420 resta rinviato.
4. Tenere espliciti i limiti: testi brevi e iniziali, generatori storici, pool RoBERTa, singolo preset per humanizer, input anomali e assenza di stima gold dell'accuratezza semantica sui nuovi output.
5. Aggiornare la sezione sperimentale della tesi quando il confronto e la valutazione manuale avranno risultati consolidati. EditLens rimane rinviato; il giudice LLM è eseguito nello smoke e parzialmente sulle 126, ma manca una validazione umana indipendente.

Il risultato attuale è una suite implementata e sottoposta a controlli, con due pilot da 42 fonti, lo smoke v2 e l’applicazione ai quattro riscrittori sulle 126 fonti; la naturalezza estesa resta parziale. Le conclusioni su prestazioni generali, naturalezza e graduatorie degli humanizer restano da sostenere con il campione esteso e le annotazioni previste.
