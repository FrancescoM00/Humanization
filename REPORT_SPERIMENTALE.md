# Report cumulativo della sperimentazione

Aggiornamento: **9 novembre 2026**. Questo documento raccoglie il percorso sperimentale, collega i report completi e distingue risultati ottenuti, prove preliminari e attività ancora da svolgere. I pilot Qwen e DIPPER sulle stesse 42 fonti sono stati ricevuti e analizzati; sono disponibili il confronto appaiato e le schede comuni da annotare.

## 1. Obiettivo e stato dei quattro assi

La suite valuta le riscritture rispetto a quattro domande distinte: eludono i detector? Conservano il contenuto? Quanto cambiano lo stile? Risultano naturali? Un miglioramento su un asse non implica un miglioramento sugli altri.

| Asse | Lavoro completato | Stato e limite principale |
| --- | --- | --- |
| Evasione | Tre detector reali, calibrazione umana, ASR condizionata, transizioni prima/dopo e voto di maggioranza | Qwen e DIPPER sulle stesse 42 fonti. Pool interamente RoBERTa, campione ancora pilota |
| Conservazione | Perturbazioni controllate, positivi OpAI, confronti allineati, aggregatori, lunghezza, guardie e applicazione ai nuovi output | Metriche operative; sui nuovi humanizer manca un riferimento semantico umano indipendente |
| Stile | Umano/macchina a lunghezza abbinata, regressione logistica, trasferimento, frasi OpAI toccate e distanze sulle riscritture | Differenze misurate; non equivalgono a qualità o naturalezza |
| Naturalezza | Preparazione delle schede a testo singolo, separate dalle chiavi dei sistemi | Giudizi ancora assenti; nessun risultato di naturalezza o giudice LLM grande già eseguito |

La validazione iniziale degli strumenti e la loro applicazione agli humanizer sono esperimenti differenti. Una metrica efficace sulle corruzioni artificiali può avere falsi allarmi o mancare errori nelle riscritture reali.

## 2. Dati, provenienza e unità di analisi

La base stilometrica comprende **900 coppie umano/macchina, 1.800 testi**: 300 umani e 300 macchine per ciascuno dei domini inglesi Essay, Reuters e WritingPrompts. I generatori storici sono ChatGLM, ChatGPT, ChatGPT-turbo, Claude, Dolly, GPT4All e StableLM.

I tre CSV sono stati inizialmente acquisiti da un mirror fissato; quelli successivamente caricati dall'autore come scaricati dal Drive indicato sono risultati identici byte per byte. Il confronto è documentato in [provenance/mgtbench_drive_comparison.md](provenance/mgtbench_drive_comparison.md). La fonte effettivamente analizzata è la baseline MGTBench: non si presenta il campione come una replica di tutti i file di attacco TH-Bench.

Per controllare la lunghezza sono stati ricavati prefissi terminati a fine frase, di 90–180 parole lessicali, con conteggio **esattamente uguale entro ogni coppia**. L'oggetto dello studio è quindi il passaggio iniziale, non l'intero documento. La pulizia non certifica che ogni testo macchina sia prosa adeguata: il pilot ha evidenziato anche input formulati come consegne.

Le trasformazioni dello stesso originale condividono una fonte. Più coppie, più frasi o più score non diventano automaticamente nuove osservazioni indipendenti. Split e ricampionamenti tengono conto dei gruppi di origine.

## 3. Prima fase: costruzione e verifica delle misure di stile

Con spaCy piccolo, textstat e MATTR implementato nel progetto sono state estratte **105 feature**: lunghezza e variabilità delle frasi, punteggiatura, distribuzioni POS, parole funzionali, diversità lessicale, leggibilità e proxy della guida Wikipedia sui testi AI. Le definizioni e i denominatori sono salvati insieme ai risultati.

È stata corretta una confusione iniziale fra stile e lunghezza mediante il matching esatto delle parole. Dopo il matching, il classificatore basato soltanto sul numero di parole ha AUC 0,500. Le differenze fra le altre feature vengono analizzate con differenza standardizzata delle medie, intervalli e controlli per dominio/generatore. Hedges g esprime la differenza rispetto alla variabilità; Benjamini–Hochberg corregge i confronti multipli. Sono strumenti statistici adottati nel protocollo, non procedure che si assume siano state trattate nei corsi dell'autore.

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

### Qwen: pilot completato

Il pilot comprende **42 fonti**, due per ciascuna combinazione di tre domini e sette generatori, con 42 riscritture e 42 identità tecnicamente riuscite.

| Detector | Successi / originali rilevati | ASR | Originali → riscritture rilevate AI, su 42 |
| --- | ---: | ---: | ---: |
| OpenAI | 17/24 | 70,8% | 24 → 8 |
| Fakespot raw | 1/37 | 2,7% | 37 → 41 |
| RADAR | 2/28 | 7,1% | 28 → 35 |
| Maggioranza | 2/31 | 6,5% | 31 → 35 |

L'evasione si concentra sul detector storico OpenAI. Fakespot e RADAR riconoscono anche diversi originali che prima sfuggivano. Il risultato non è riassumibile come una percentuale unica di testi che inganna ogni detector.

Coseno documentale medio 0,9282 e BERTScore 0,9445 coesistono con NLI forward/reverse medio 0,7899/0,6708. Questi ultimi non sono percentuali di significato preservato. La lettura dei casi mostra omissioni, aggiunte, falsi allarmi da fusione/coreferenza e un limite quantitativo: `up to 100 feet` diventa `over 100 feet` con NLI comunque vicino a 0,98.

Due input StableLM sono consegne: Qwen risponde invece di riscriverle, espandendo 133→381 e 164→274 parole. Restano nell'analisi primaria; l'esclusione esplorativa dei due non sostituisce il campione congelato. Le review dell'assistente non sono annotazioni umane.

La distanza RMS media è 0,8164; MATTR aumenta in 37/42 casi, ma il centroide umano si avvicina solo in 20/42. Lo stile cambia senza una direzione universalmente più umana. Sono disponibili le [schede da annotare](exports/kaggle_pilot/README.md); la naturalezza non è ancora misurata.

[Report Qwen](results_kaggle_pilot/report.md), [casi](results_kaggle_pilot/case_review.md), [archivio e moduli](exports/kaggle_pilot/README.md).

### DIPPER: smoke e pilot completati

Lo smoke usa gli stessi tre input ChatGLM già esaminati per Qwen. Tutte le riscritture terminano con EOS; nessuna raggiunge il limite d'uscita. Ogni detector continua a riconoscerle AI: ASR 0/3. DIPPER accorcia i testi del 18–27%.

Reuters perde il riferimento ai problemi legali dell'azienda: NLI forward medio 0,984, reverse 0,679, minimo reverse 0,000334 proprio sulla frase omessa. Essay mostra anche un allarme da fusione di due frasi; WritingPrompts cambia il meccanismo del potere del protagonista ed elimina la risposta finale di Sarah. Sono esempi leggibili, non una stima dell'accuratezza semantica o una classifica degli humanizer.

Il pilot DIPPER sugli stessi 42 input è **completo**: 42 riscritture e 42 identità, tutte con stato `ok`, nessun arresto al limite di generazione. L'archivio ricevuto è `humanization_pilotdipper.zip`; configurazione, dati, soglie e scale restano quelli precedenti. Il confronto Qwen–DIPPER è appaiato per fonte.

| Detector | Qwen: successi / originali rilevati | DIPPER: successi / originali rilevati | Differenza ASR DIPPER−Qwen |
| --- | ---: | ---: | ---: |
| OpenAI | 17/24 | 4/24 | −54,2 punti percentuali |
| Fakespot raw | 1/37 | 7/37 | +16,2 punti percentuali |
| RADAR | 2/28 | 3/28 | +3,6 punti percentuali |
| Maggioranza | 2/31 | 2/31 | 0 punti percentuali |

L'ordine dipende dal detector. Il bootstrap appaiato esplorativo dà intervalli 95% della differenza di [−75,0; −29,2] punti per OpenAI, [+5,4; +27,0] per Fakespot, [−7,1; +17,9] per RADAR e [−12,9; +12,9] per maggioranza, mantenendo fissi modelli e calibrazione. I due successi di maggioranza riguardano fonti completamente diverse. Dopo DIPPER le rilevazioni complessive sono 25/42 OpenAI, 33/42 Fakespot, 33/42 RADAR e 32/42 maggioranza: alcune fonti prima non rilevate diventano AI, perciò ASR e TPR complessivo vanno letti separatamente.

DIPPER accorcia 41/42 testi, con rapporto parole medio 0,739 e mediano 0,766; 23 output scendono sotto l'80% della lunghezza iniziale. Qwen ha rapporto mediano 0,915. MATTR aumenta in 13/42 DIPPER contro 37/42 Qwen; il centroide umano si avvicina in 13/42 contro 20/42. La RMS media è 0,7765 per DIPPER e 0,8164 per Qwen, mentre le mediane invertono l'ordine. Distanza e leggibilità non determinano la naturalezza.

Coseno/BERTScore documentali medi DIPPER sono 0,9215/0,9478 e NLI forward/reverse 0,6840/0,5949. La lettura di 12 coppie mostra omissioni e variazioni puntuali, fra cui euro→franchi e 100→30 piedi, ma anche falsi allarmi del protocollo a premessa singola: nel racconto Maria/Sofia la fusione da 15 a cinque frasi accompagna forward 0,00262 pur mantenendo gran parte della trama. Le medie non diventano una classifica di fedeltà senza etichette indipendenti. Le due consegne StableLM già note restano nel risultato primario; l'analisi secondaria da 40 non risolve tutti i problemi di qualità degli input.

Sono pronte [schede comuni](exports/kaggle_dipper_pilot/README.md) senza score: 45 coppie semantiche (i due sistemi sulle stesse 21 fonti, più tre identità) e 24 testi di naturalezza su sei fonti comuni con umani, originali e riscritture. Le chiavi sono separate; nessuna etichetta è stata compilata. Le precedenti 24 coppie Qwen/identità sono incluse e non vanno contate due volte se già giudicate.

[Report pilot DIPPER e confronto](results_kaggle_dipper_pilot/report.md), [casi](results_kaggle_dipper_pilot/case_review.md), [confronto ASR appaiato](results_kaggle_dipper_pilot/paired_comparison.md), [ZIP e schede](exports/kaggle_dipper_pilot/README.md). Lo [smoke DIPPER](results_kaggle_dipper_smoke/report.md) rimane documentato separatamente.

## 7. Perché 3, 42 e 126 testi: significato e limiti della numerosità

La riduzione del campione riguarda la nuova fase humanizer, non tutti gli esperimenti: lo stile usa già 1.800 testi e la validazione semantica centinaia di fonti e numerose trasformazioni. Le numerosità della fase Kaggle sono state scelte per procedere dal collaudo all'esplorazione, **non mediante un calcolo che ne garantisca la sufficienza statistica**.

| Pannello | Fonti macchina per humanizer | Ripartizione | Stato/scopo |
| --- | ---: | --- | --- |
| Smoke | 3 | Una per dominio, tutte ChatGLM | Collaudo di caricamento, generazione e suite |
| Pilot | 42 | 2 × 3 domini × 7 generatori | Primo confronto bilanciato, completato per entrambi i sistemi |
| Study | 126 | 6 × 3 × 7 | Già congelato, non ancora eseguito sui due sistemi |
| Possibile estensione | 420 | 20 × 3 × 7 | Proposta tecnicamente fattibile; nuova coorte da definire, non già preparata o eseguita |

I 300 umani di calibrazione servono a fissare soglie e scale. Non aggiungono 300 casi al denominatore dell'evasione. Due sistemi sugli stessi 42 input danno 84 riscritture ma mantengono **42 fonti nel confronto appaiato**. Identità e tre detector non moltiplicano le fonti indipendenti. Smoke, pilot e study sono annidati: non si sommano 3+42+126 come campioni distinti.

Nel pilot, un caso su 42 vale circa 2,38 punti percentuali. Nell'ASR i denominatori sono ancora più piccoli: per OpenAI un successo su 24 vale 4,17 punti. L'intervallo bootstrap già osservato per l'ASR OpenAI va circa dal 52,0% all'87,5%. In ogni cella dominio/generatore ci sono due fonti: un solo caso può cambiare il tasso di 50 punti, o più se il denominatore condizionato è uno. I risultati per cella sono quindi esplorativi. Anche 126 fonti lascerebbero solo sei elementi per cella.

Una verifica dei metadati delle 900 fonti, escludendo i 300 gruppi di calibrazione attuali, lascia **600 fonti macchina utilizzabili**, comprese le 126 riservate. Ogni cella ha almeno 23 fonti residue: un campione bilanciato da 420 fonti è fattibile, e il massimo perfettamente bilanciato con questa separazione è 483 = 23×21. Questi limiti di disponibilità non sono una dimostrazione di potenza statistica né un controllo semantico della qualità degli input.

Per estendere oltre il pannello previsto occorre conservare esplicitamente i 300 gruppi di calibrazione e creare una nuova coorte con criteri basati su metadati/ID, includendo gli input correnti. **Non basta aumentare `reserve_per_cell`**: il selettore attuale riserva prima i test e poi sceglie la calibrazione, che potrebbe così cambiare. La nuova coorte dovrà applicarsi a entrambi gli humanizer e mantenere tracciati versione e soglie.

La dimensione finale va fissata in base alla precisione desiderata per il confronto complessivo e per gli eventuali sottogruppi, considerando i denominatori ASR e il budget. Aumentare le fonti migliora la precisione, ma non risolve mancanza di annotazioni, bias del corpus, limiti dei detector o errori delle metriche. I pilot completati restano invariati e nessuna estensione è ancora eseguita.

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

## 9. Attività successive e confini delle conclusioni

1. Usare il confronto appaiato completato per documentare i comportamenti dei due preset e i limiti emersi, senza modificare a posteriori i risultati.
2. Raccogliere annotazioni indipendenti di conservazione e naturalezza sulle schede comuni. Le schede preparate non sono risultati e le review dell'assistente non sostituiscono giudizi umani; la fusione delle frasi è un caso prioritario per misurare i falsi allarmi dell'NLI.
3. Fissare l'estensione comune del campione prima di esaminarne gli output, mantenendo la calibrazione attuale. Il pannello da 126 è già disponibile; un'estensione più ampia richiede la nuova coorte descritta sopra.
4. Tenere espliciti i limiti: testi brevi e iniziali, generatori storici, pool RoBERTa, singolo preset per humanizer, input anomali e assenza di stima gold dell'accuratezza semantica sui nuovi output.
5. Aggiornare la sezione sperimentale della tesi quando il confronto e la valutazione manuale avranno risultati consolidati. EditLens e un eventuale giudice LLM grande restano attività distinte, non ancora eseguite.

Il risultato attuale è una suite implementata e sottoposta a controlli, con una prima applicazione reale a Qwen e a DIPPER. Le conclusioni su prestazioni generali, naturalezza e graduatorie degli humanizer restano da sostenere con il campione esteso e le annotazioni previste.
