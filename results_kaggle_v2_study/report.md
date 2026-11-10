# Study v2 su 126 fonti: rilevabilità, semantica, stile e naturalezza

Analisi del 10 novembre 2026 dell'archivio `humanization_v2_study_protocol_v2.zip`. **Lo studio contiene i risultati dei quattro riscrittori sulle 126 fonti congelate: 630 esiti, 626 validi e quattro troncati. Detector, controlli nulli, stile e sei stadi semantici sono presenti; i tre confronti Groq sono invece tutti parziali.** Sono stati analizzati gli output salvati, senza nuove generazioni, inferenze neurali, chiamate API o etichette umane.

Il risultato principale è la **dipendenza dell'evasione dal detector**: il preset 14B evade sia OpenAI sia una parte di Binoculars, mentre Fakespot rileva quasi tutte le riscritture e diventa più efficace sugli output dei due preset a prompt. Il controllo sulle 84 fonti non viste nei pilot sostiene questa distinzione. La sola lunghezza non riproduce il vantaggio delle riscritture a prompt nei confronti abbinati; DIPPER non mostra un vantaggio aggregato chiaro sul taglio rispetto al detector storico. Nessuno di questi risultati certifica conservazione semantica o naturalezza.

[Archivio e schede](../exports/kaggle_v2_study/README.md), [audit](integrity.md), [detector e controlli](detection_analysis.md), [semantica e stile](semantic_style.md), [giudice](naturalness.md), [dodici casi letti](case_review.md), [provenienza](provenance.json).

## Campione, preset e unità indipendenti

Il pannello comprende **126 fonti = 3 domini × 7 generatori × 6 fonti**, con 300 gruppi umani di calibrazione separati. I domini sono Essay, Reuters e WritingPrompts; i generatori sono quelli storici di MGTBench. Le 42 fonti dei pilot sono incluse: il confronto sulle **84 fonti aggiuntive** evita di presentare il campione completo come una replica indipendente delle ipotesi nate dai pilot. Neppure i quattro riscrittori, gli ordini AB/BA o i controlli moltiplicano le fonti indipendenti.

Configurazione e pannelli coincidono con quelli già congelati; hash study `84f9f6933b2ee72412d65eedbf8f1fbe70cf0139bcd3d6d395cb85250c0edb68`. Le riscritture 3B e DIPPER passaggio sulle 42 fonti condivise sono identiche ai pilot. Le scale stilometriche e le soglie non sono ricalibrate sui nuovi output.

| Sistema | Condizione effettiva | Esiti validi / attesi |
| --- | --- | ---: |
| Identità | Testo macchina invariato | 126/126 |
| Baseline 3B | Qwen2.5-3B-Instruct, prompt fissato, greedy FP16 | 124/126 |
| 14B | Qwen2.5-14B-Instruct, stesso prompt, greedy NF4 | 126/126 |
| DIPPER passaggio | XXL NF4, lexical 60/order 0, intero passaggio senza prefisso | 125/126 |
| DIPPER tre frasi | Stessi pesi, finestre di tre frasi, prefisso riscritto cumulativo | 125/126 |

Ogni sistema produce un solo tentativo per fonte, senza feedback dai detector. Il 14B differisce dal 3B anche per quantizzazione; i due DIPPER differiscono anche per contesto e budget complessivo di generazione. Il preset a tre frasi usa spaCy e greedy: non è una replica esatta della configurazione originale del paper.

Il runtime iniziale è il commit `3d02820`, PyTorch `2.11.0+cu130`, due Tesla T4. Il bundle di ripresa della naturalezza è conservato nell'archivio e gli hash distinguono il codice effettivo dal solo commit iniziale. La ripresa del download Falcon e la gestione esplicita degli output falliti non cambiano checkpoint, prompt o precisione dei modelli.

## Quattro fallimenti, su tre fonti nuove

I quattro esiti `truncated` appartengono tutti alle 84 fonti nuove:

- la baseline 3B su `Essay:223` e `Reuters:992`, entrambe fonti StableLM, risponde alla consegna interna e raggiunge 512 token senza EOS;
- entrambi i preset DIPPER su `WritingPrompts:184`, GPT4All, entrano in ripetizioni e raggiungono il limite. Il contestuale fallisce già nella prima finestra.

Sono fallimenti della generazione con questo preset, non errori CUDA dimostrati. Restano nel denominatore ASR, quando l’originale è già rilevato, come tentativi senza evasione riuscita; non vengono inviati al giudice o alle metriche come testi validi. Semantica e stile descrivono quindi **500 riscritture valide più 126 identità**, condizionatamente alla riuscita. I confronti comuni ai quattro riscrittori hanno 123 fonti, 81 delle quali nuove.

La politica già scritta mantiene gli input-consegna nel risultato primario. Anche testi con stato `ok` possono eseguire una consegna invece di parafrasarla: la review lo osserva su Essay 489. Non si escludono a posteriori soltanto i casi sfavorevoli. Un'analisi sulla sola prosa richiede classificazione degli originali e conteggi espliciti; la scheda source-only esportata è ancora vuota.

## Evasione sulle 126 fonti

ASR è il numero di riscritture che diventano non rilevate **fra gli originali già rilevati**. Non è il tasso di testi non rilevati su tutta la coorte. I fallimenti restano nel suo denominatore; le transizioni e i TPR appaiati sono calcolati separatamente sui soli output disponibili.

| Preset | OpenAI RoBERTa | Fakespot | RADAR | Binoculars | Maggioranza RoBERTa |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 3B | 45/73 = 61,6% | 1/109 = 0,9% | 8/90 = 8,9% | 17/120 = 14,2% | 4/94 = 4,3% |
| 14B | 60/73 = 82,2% | 1/109 = 0,9% | 8/90 = 8,9% | 29/120 = 24,2% | 6/94 = 6,4% |
| DIPPER passaggio | 10/73 = 13,7% | 11/109 = 10,1% | 7/90 = 7,8% | 0/120 = 0,0% | 7/94 = 7,4% |
| DIPPER tre frasi | 6/73 = 8,2% | 8/109 = 7,3% | 11/90 = 12,2% | 1/120 = 0,8% | 6/94 = 6,4% |

OpenAI è il detector storico addestrato su GPT-2 e riconosce solo 73/126 originali. L'82,2% non è quindi una misura generale del successo del 14B. Binoculars riconosce 120/126 originali e mostra qui evasioni assenti nello smoke: tre esempi non permettevano di stimarle. Il ratio resta grezzo, AI se basso, calibrato sugli umani; non è una probabilità. Il voto riguarda soltanto i tre RoBERTa correlati, non comprende Binoculars e non è indipendente.

Il vantaggio ASR 14B−3B su Binoculars è **+10,0 punti percentuali**, intervallo bootstrap appaiato 95% **[+3,3; +17,5]** sulle 126 fonti. Su OpenAI è +20,5 punti [+8,2; +32,9]. Sono confronti esplorativi fra preset fissati: non isolano un effetto causale della dimensione del modello. Gli intervalli mantengono fissi checkpoint e calibrazione, non includono tutta l'incertezza del metodo. Per zero successi il bootstrap può dare [0; 0]: non significa rischio nullo nella popolazione; il CSV conserva anche un intervallo binomiale supplementare.

### Le 84 fonti non viste nei pilot

| Preset | OpenAI | Fakespot | RADAR | Binoculars |
| --- | ---: | ---: | ---: | ---: |
| Baseline 3B | 28/49 = 57,1% | 0/72 = 0,0% | 6/62 = 9,7% | 9/80 = 11,3% |
| 14B | 41/49 = 83,7% | 0/72 = 0,0% | 6/62 = 9,7% | 21/80 = 26,3% |
| DIPPER passaggio | 6/49 = 12,2% | 4/72 = 5,6% | 4/62 = 6,5% | 0/80 = 0,0% |
| DIPPER tre frasi | 4/49 = 8,2% | 4/72 = 5,6% | 8/62 = 12,9% | 1/80 = 1,3% |

Il vantaggio ASR 14B−3B su Binoculars rimane **+15,0 punti [+6,3; +23,8]**; su OpenAI +26,5 [+12,2; +40,8]. Le evasioni Binoculars del 14B compaiono in tutti i sette generatori, ma con soli 10–12 originali rilevati per generatore non si propone una graduatoria dei sottogruppi.

### Fakespot diventa più efficace; RADAR richiede una conclusione più prudente

Sulle 126 fonti, fra gli output validi della baseline 3B Fakespot passa da **107/124 a 123/124 rilevazioni**, con 17 nuove rilevazioni e una evasione: TPR appaiato +12,9 punti [+7,2; +19,4]. Con 14B passa da 109/126 a 125/126: +12,7 [+7,1; +19,0]. Non si confrontano 109 originali su 126 con un TPR su 124 come se il denominatore fosse uguale.

Sulle 84 nuove, Fakespot passa da **70/82 a 82/82** per 3B e da **72/84 a 84/84** per 14B: aumenti +14,6 [+7,3; +23,2] e +14,3 [+7,1; +22,6] punti. Tutti gli output validi dei due preset a prompt sono riconosciuti. Le dodici nuove rilevazioni per sistema non sono soltanto StableLM: sei StableLM, tre Claude, due Dolly e una ChatGPT.

Per RADAR, sulle nuove fonti il passaggio è 60→64 rilevazioni su 82 per 3B e 62→66 su 84 per 14B. Gli intervalli della differenza includono zero. Il risultato sostenuto è dunque **un aumento chiaro su Fakespot**, non su qualunque detector recente. Su Binoculars il verso è opposto: il TPR del 14B scende da 80/84 a 59/84. [Tabelle e intervalli completi](detection_overall.csv).

## Controlli di lunghezza e rumore

I 1.764 esiti congiunti comprendono quattro riscrittori e dieci controlli per fonte. Il prefisso ha lo stesso conteggio alfabetico spaCy della riscrittura e può finire dentro una frase; nessun padding quando l'output è più lungo. Il rumore cambia al massimo una parola applicabile e non è una gold semantica. I fallimenti di riferimento e le sostituzioni impossibili restano non eleggibili.

Sulle **84 fonti nuove**, fra gli originali baseline-AI eleggibili allo stesso confronto:

| Detector | Preset | Prefisso di uguale lunghezza | Riscrittura | Differenza ASR, punti percentuali e IC 95% |
| --- | --- | ---: | ---: | ---: |
| OpenAI | Baseline 3B | 1/41 | 24/41 | +56,1 [+41,5; +70,7] |
| OpenAI | 14B | 4/42 | 36/42 | +76,2 [+61,9; +88,1] |
| Binoculars | Baseline 3B | 0/62 | 5/62 | +8,1 [+1,6; +14,5] |
| Binoculars | 14B | 0/68 | 19/68 | +27,9 [+17,6; +38,2] |

Il vantaggio delle riscritture a prompt non è riprodotto dal solo prefisso nei confronti osservati. Non si confrontano questi numeratori con quelli su tutta la coorte: il sottoinsieme eleggibile è diverso. Il controllo non identifica in modo causale ogni contributo stilistico e non copre gli output allungati.

Per DIPPER sul detector storico il risultato è diverso: su tutte le fonti eleggibili il passaggio ha 10/72 evasioni sia per il prefisso sia per la riscrittura; il contestuale 5/64 per entrambi. I successi **non sono gli stessi casi**: passaggio otto evasioni solo del controllo e otto solo dello humanizer; contestuale cinque e cinque, disgiunte. Sulle nuove fonti passaggio 7/48 contro 6/48 e contestuale 3/42 contro 3/42. Si conclude che non emerge un vantaggio aggregato chiaro sul taglio, non che le singole evasioni siano causalmente spiegate dalla lunghezza. I controlli con rumore e tutti i denominatori sono nel [CSV](null_comparisons.csv) e nel [report dedicato](detection_analysis.md).

## Stile: il 14B cambia di più le feature, il DIPPER contestuale comprime meno

Le medie seguenti usano gli output `ok`, non i quattro fallimenti. Le distanze RMS usano **91 feature retained delle 105 estratte**, standardizzate sui 300 umani di calibrazione: non soltanto la shortlist di quattro feature. MATTR è riportato separatamente e non è una misura di naturalezza.

| Preset | Output validi | Rapporto parole medio | RMS medio | Δ MATTR medio | Più vicino al centroide umano |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 3B | 124 | 0,962 | 0,831 | +0,0616 | 60/124 |
| 14B | 126 | 0,871 | 1,047 | +0,0813 | 61/126 |
| DIPPER passaggio | 125 | 0,758 | 0,796 | −0,0286 | 29/125 |
| DIPPER tre frasi | 125 | 0,920 | 0,651 | −0,0262 | 45/125 |

DIPPER a tre frasi produce output più lunghi del preset sul passaggio in 102/125 confronti, con differenza media appaiata del rapporto +0,162. Rimane comunque più corto dell'originale in 102/125 casi; il passaggio lo è in 120/125. I due preset a prompt aumentano spesso MATTR, ma il centroide umano si avvicina soltanto in circa metà dei casi. Cambiare più feature o avvicinarsi a un centroide non dimostra miglioramento della prosa.

Le tendenze rimangono sulle 84 fonti nuove e sulle 123 fonti con successo comune a tutti i sistemi. Questi controlli riducono l'effetto dei diversi denominatori di successo; non trasformano le feature in giudizi gold di qualità. [Tabelle](style_summary.csv), [confronti comuni](semantic_common_success_summary.csv).

## Semantica: il preset contestuale recupera contenuto, ma nessuno score certifica fedeltà

| Preset | Coseno documento medio | BERTScore documento grezzo medio | NLI storico medio forward / reverse | NLI window3 medio forward / reverse |
| --- | ---: | ---: | ---: | ---: |
| Baseline 3B | 0,937 | 0,946 | 0,785 / 0,696 | 0,927 / 0,904 |
| 14B | 0,903 | 0,927 | 0,714 / 0,645 | 0,937 / 0,880 |
| DIPPER passaggio | 0,925 | 0,948 | 0,693 / 0,602 | 0,913 / 0,736 |
| DIPPER tre frasi | 0,951 | 0,964 | 0,863 / 0,731 | 0,940 / 0,879 |

Si tratta di scale diverse, non percentuali di significato preservato e non accuratezze da ordinare direttamente. I CSV includono anche ROUGE-L/BLEU, BERTScore riscalato allineato, coseno allineato, minimi, q10 e quote non supportate. Non si ripete il confronto iniquo fra minimo NLI e sola media documentale. La soglia OpAI 0,04075 e l'AUC 0,958 delle perturbazioni non vengono trasferite allo studio.

DeBERTa-v3 **base float32** completa 199.126 evidenze contestuali, circa 28,3 minuti di inferenza su GPU esclusi caricamenti. Tutte le modalità singole e a finestre hanno copertura completa. Il documento come premessa ha quattro aggregati reverse incompleti, due per 3B e due per 14B: restano mancanti. Anche il documento esatto non è una soluzione universale; sulle identità il suo supporto medio è circa 0,896, rispetto a 0,992 con singole esaustive.

Il percorso storico usa retrieval per similarità; `single_all` esamina tutte le premesse singole. Il confronto fra storico e `window3` combina quindi differenze di retrieval e ampiezza della premessa. Le modalità nuove sono annidate e scelgono il massimo: il supporto non decresce ampliando l'insieme dei candidati per costruzione. L'aumento non dimostra accuratezza. Il confronto corretto fra contesti usa le stesse ipotesi e la stessa copertura, come nelle [tabelle dedicate](semantic_context_common_coverage.csv).

La lettura di dodici coppie sulle fonti nuove mostra entrambe le direzioni del problema:

- **Errore con similarità alta:** Reuters 289, DIPPER passaggio, cambia dollari in euro, 26,8 in 26,7 e 43,75 miliardi in 4,3 miliardi, con coseno circa 0,939.
- **Recupero del contesto:** Reuters 633, DIPPER a tre frasi, recupera un'informazione omessa dal passaggio; fusioni e collegamenti fra frasi rendono insufficiente una sola premessa. Il testo originale ha però già una cronologia problematica.
- **Errore con NLI alto:** Reuters 889, DIPPER a tre frasi, trasforma il terzo trimestre 2019 nel trimestre dell'anno corrente e una possibilità in un'affermazione più forte; forward storico circa 0,989 e con finestre 0,993.
- **Stato tecnico valido, compito sbagliato:** Essay 489, le riscritture a prompt eseguono una consegna interna. La baseline 3B aggiunge il riferimento a un presunto studio non presente nell'originale, senza verifica esterna della citazione.

Questa è una review dell'assistente selezionata per casi diagnostici, non annotazione umana né stima della frequenza degli errori. Le guardie lessicali segnalano variazioni, non ne certificano la scorrettezza. Il DIPPER contestuale ha score medi migliori e minore compressione del passaggio, ma non si dichiara più fedele in termini gold senza le annotazioni.

## Naturalezza: limiti di API e sensibilità all'ordine

I tre confronti hanno rispettivamente 124, 123 e 125 coppie giudicabili su 126, dopo aver mantenuto espliciti i fallimenti. Sono pianificate 744 risposte AB/BA, rispetto alle 756 massime in assenza di fallimenti. Il modello resta `qwen/qwen3.8-27b`, `reasoning_effort=none`, senza fallback.

| Confronto | Risposte valide / attese | Tentativi POST | Coppie con entrambi gli ordini | Stabili | Discordanti | Stato |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline 3B / 14B | 136/248 | 256 | 68 | 32 | 36 | Budget richieste esaurito |
| Baseline 3B / DIPPER tre frasi | 128/246 | 256 | 64 | 46 | 18 | Budget richieste esaurito |
| DIPPER passaggio / tre frasi | 100/250 | 201 | 50 | 15 | 35 | HTTP 429, limite giornaliero token |

Le nuove comparazioni producono **364 risposte valide su 713 tentativi POST**. Tutte le risposte salvate sono valide; i 349 tentativi aggiuntivi non sono altri giudizi. Il log dei singoli tentativi non contiene sempre lo stato HTTP, quindi non si attribuisce ogni tentativo senza risposta a un 429. L'ultimo errore del terzo confronto cita il limite provider di 200.000 token giornalieri; i contatori del provider non sono ricostruiti sommando ingenuamente il solo archivio.

La prova iniziale su sei fonti pilot ha effettuato altri 12 POST: piano e preferenze coincidono con lo smoke, ma timestamp e metadati dimostrano nuove richieste. Il totale nell'archivio è dunque **725 tentativi, 376 risposte valide e 209.610 token registrati**; i sei casi iniziali non sono nuove fonti indipendenti o validazione umana. I soli tre confronti estesi registrano 203.445 token. Non viene inventato un costo monetario.

La copertura delle preferenze stabili sul campione originario è soltanto **32/126, 46/126 e 15/126**. Tra le 182 coppie con entrambi gli ordini, 89 sono discordanti. **Ottantacinque delle 89 discordanti scelgono B in entrambi gli ordini**; complessivamente B viene scelto in 263 delle 364 risposte. È un segnale forte di sensibilità alla posizione con questa rubrica e modello, non un errore di parser riscontrato.

Nei soli casi stabili, le preferenze 3B/14B sono 13 contro 19; 3B/DIPPER contestuale 41 contro quattro più una parità; DIPPER passaggio/contestuale nove contro sei. **Non sono una graduatoria sulle 126 fonti**: le API si fermano su un prefisso degli ID ordinati per hash, poi la consistenza AB/BA seleziona ulteriormente i casi. Anche sulle 84 nuove rimangono soltanto 20, 28 e nove preferenze stabili. Non si trasformano mancate risposte o discordanti in parità, né si ripetono chiamate finché un sistema vince.

Un eventuale completamento dovrebbe riprendere i request ID mancanti e dichiarare esplicitamente un aumento dei budget ormai esauriti; non basta rilanciare la cella invariata dopo il reset giornaliero. In questa analisi non sono state fatte altre chiamate né aumentati i budget. Prima di spendere altra quota, occorre valutare quanto il giudice sia utilizzabile con riferimenti indipendenti e il disaccordo all'ordine già misurato. [Analisi completa](naturalness.md).

## Annotazione e stato del lavoro

Sono pronti due moduli indipendenti per gli annotatori, con **183 coppie**: 84 reference da 21 fonti e 99 diagnostiche da altre 25 fonti. La capacità diagnostica disponibile è maggiore di 100, ma il raggruppamento completo delle fonti e la selezione effettiva portano a 99: il manifest dichiara il target non raggiunto. Non si duplica o inventa un esempio per arrivare a cento. Le schede sono vuote; la review dell'assistente e Groq non diventano gold umana.

Lo [ZIP cieco delle schede](../exports/kaggle_v2_study/annotation_forms.zip) contiene soltanto moduli A/B e istruzioni; le chiavi restano separate. Gli annotatori lavorano indipendentemente. Il reference è selezionato senza score; il diagnostico è arricchito per disaccordi fra metriche e non consente un'AUC ingenua di popolazione. I giudizi di due annotatori non raddoppiano le fonti indipendenti.

Il passo successivo utile è la revisione manuale, distinguendo fedeltà, tipo di input e naturalezza, poi il confronto con gli score e i casi di disaccordo. Non è necessario rifare i modelli per leggere questi risultati. L'estensione a 420 fonti, EditLens, Fast-DetectGPT e servizi commerciali restano rinviati. La sezione sperimentale compilata della tesi precede questa fase: il report cumulativo è aggiornato, i file LaTeX/PDF non vengono dichiarati aggiornati senza una revisione specifica.

## Configurazione NLI primaria fissata successivamente

Il [run di calibrazione ricevuto il 10 novembre](../results_nli_context_calibration/report.md), su 139 positivi presunti, 297 alterazioni e 137 identità, seleziona **window2 + minimo bidirezionale** con la policy precedente alle nuove inferenze. È una scelta successiva allo study e precedente alla ricezione delle nuove annotazioni umane, non una modifica retroattiva dei suoi endpoint. Le [misure primarie](nli_primary_scores.csv) sono estratte dagli aggregati contestuali già calcolati: 626 score validi e quattro fallimenti, nessuna nuova inferenza; tutte le 84 reference sono coperte. [Distribuzioni](nli_primary_summary.csv). Nessuna soglia viene trasferita da OpAI e le altre modalità restano disponibili come secondarie.
