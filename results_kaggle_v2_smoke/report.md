# Smoke Kaggle v2: quattro riscrittori, Binoculars e controlli

Analisi del 9 novembre 2026 sullo ZIP `humanization_v2_smoke_protocol_v2.zip`. **Il percorso v2 è completato su tre fonti: tutti i 15 esiti sono `ok` e nessuno stadio di scoring risulta mancante.** Il nuovo giudice produce risposte valide, ma quattro dei nove confronti nuovi cambiano preferenza invertendo l'ordine dei testi. Lo smoke verifica quindi l'esecuzione della suite; non permette ancora una graduatoria affidabile degli humanizer.

Sono stati letti e ricalcolati i riepiloghi degli output ricevuti, senza nuove generazioni, inferenze neurali o chiamate Groq. [Archivio originale e provenienza](../exports/kaggle_v2_smoke/README.md), [audit tecnico](integrity.md), [lettura qualitativa dei dodici output](text_review.md).

## Campione e condizioni effettive

Le fonti sono le stesse tre dello smoke precedente: un testo ChatGLM per ciascuno di Essay, Reuters e WritingPrompts. Non sono tre repliche dei sette generatori. I 300 gruppi umani di calibrazione rimangono separati dai test; anche i pannelli da 42 e 126 fonti coincidono byte per byte con quelli già congelati.

| Condizione | Configurazione | Esiti |
| --- | --- | ---: |
| Identità | Originale invariato | 3/3 |
| Baseline 3B | Qwen2.5-3B-Instruct, prompt fissato, greedy FP16 | 3/3 |
| 14B | Qwen2.5-14B-Instruct, stesso prompt, greedy NF4 | 3/3 |
| DIPPER passaggio | XXL NF4, intero passaggio, nessun prefisso, lexical 60/order 0 | 3/3 |
| DIPPER tre frasi | Stessi pesi, finestre di tre frasi e prefisso riscritto cumulativo | 3/3 |

Le condizioni sono quelle di [STUDY_V2_PROTOCOL.md](../STUDY_V2_PROTOCOL.md). Il DIPPER contestuale conserva differenze dal rilascio originale: segmentazione spaCy, greedy e budget di 512 token per finestra. Non si presenta questo confronto come un effetto isolato della dimensione dei modelli o una replica esatta del paper. Il 14B differisce dal 3B anche per quantizzazione.

La sessione usa PyTorch `2.11.0+cu130` e due Tesla T4. Il commit iniziale è `c92e7c2`; durante il run sono state applicate correzioni operative, riconoscibili dagli hash del codice effettivo. Il solo commit iniziale non descrive tutte le patch. Le riscritture storiche 3B e DIPPER passaggio restano identiche ai precedenti smoke.

## Rilevabilità: evasioni solo sul detector storico

Tutti i detector riconoscono i tre originali macchina. Nella tabella ASR significa quindi numero di originali rilevati che diventano non rilevati, su tre: il denominatore coincide qui con tutta la coorte soltanto perché tutti gli originali sono rilevati.

| Riscrittore | OpenAI RoBERTa | Fakespot | RADAR | Binoculars | Maggioranza RoBERTa |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 3B | 2/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| 14B | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| DIPPER passaggio | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| DIPPER tre frasi | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |

L'identità ha zero evasioni per tutti. Il voto di maggioranza riguarda solo i tre classificatori RoBERTa: Binoculars rimane separato e non rende indipendenti i voti precedenti. Nessuno dei tre umani test è rilevato; tre esempi non permettono di stimare un FPR nullo.

**Binoculars è ora realmente eseguito**, con Falcon 7B base e instruct sulle due GPU, pesi FP16 e statistiche FP32. Il ratio grezzo è orientato verso AI quando scende: soglia calibrata `0.9046559856180912`, decisione stretta `score < soglia`, parità assegnata a umano. La calibrazione rileva 15 umani su 300. Il ratio non è una probabilità e non si usa la soglia predefinita del paper al posto di quella calibrata. Sono presenti 348 record, corrispondenti a 342 testi distinti, in circa 310 secondi di scoring; download e caricamento sono esclusi.

Il risultato è coerente con la dipendenza dal detector già osservata nel pilot da 42. Non dimostra che il 14B sia generalmente migliore, che Binoculars sia infallibile o che le riscritture non possano evadere altri detector. Le frequenze su tre fonti non sostituiscono il confronto sulle 126.

## Controlli nulli: il solo accorciamento non riproduce queste evasioni

Le 42 righe congiunte includono le riscritture e dieci controlli per fonte: identità, rumore lessicale minimo, prefissi della lunghezza di ciascun riscrittore e relativi prefissi con rumore. Il conteggio è alfabetico spaCy; un prefisso può terminare all'interno della frase, come dichiarato dal protocollo. Un output più lungo dell'originale non ammette quel confronto e non viene riempito artificialmente.

Su tutti e quattro i detector i controlli nulli hanno **zero evasioni nei casi eleggibili**. Per OpenAI:

- il 14B evade in 3/3 casi, mentre gli originali ridotti alle stesse lunghezze evadono in 0/3;
- la baseline 3B ha due confronti di lunghezza eleggibili: evade in 1/2 contro 0/2 dei prefissi; Essay è escluso da questo confronto perché la riscrittura è più lunga;
- per 14B il prefisso con rumore è eleggibile in 2/3 casi e ha 0/2 evasioni, contro 2/2 delle riscritture corrispondenti.

L'assenza di una sostituzione lessicale applicabile è distinta dall'ineleggibilità per lunghezza. Il [CSV dei controlli](null_control_comparison.csv) conserva i denominatori. Nei casi osservati il prefisso non riproduce le evasioni OpenAI; non si stima da questo un contributo causale generale della riscrittura. Questi controlli non certificano equivalenza semantica e non sostituiscono l'analisi sulle fonti estese.

## Lunghezza e stile

| Dominio | Originale | Baseline 3B | 14B | DIPPER passaggio | DIPPER tre frasi |
| --- | ---: | ---: | ---: | ---: | ---: |
| Essay | 160 | 192 | 135 | 122 | 142 |
| Reuters | 149 | 124 | 94 | 122 | 143 |
| WritingPrompts | 153 | 133 | 120 | 112 | 127 |

Il preset DIPPER a tre frasi riduce la compressione in tutti e tre i casi: rapporto parole medio circa 0,892 contro 0,771 del passaggio intero. Il 14B conserva circa il 75,3% delle parole e continua quindi ad accorciare, nonostante il vincolo del prompt. La baseline 3B alterna espansione e accorciamento.

La distanza stilometrica RMS media è circa 0,880 per 3B, 1,560 per 14B, 0,571 per DIPPER passaggio e 0,487 per DIPPER tre frasi. Le scale sono quelle umane già fissate, non ristimate sui nuovi output. In questo smoke il 14B cambia maggiormente le feature; una distanza maggiore non significa maggiore naturalezza né migliore conservazione. La minore distanza del DIPPER contestuale va letta anche insieme alla minore compressione. Le dodici osservazioni non sostengono un ordinamento generale.

## Semantica: distinguere errori reali e formato della premessa

Le sei famiglie di score sono complete sulle stesse 15 coppie: ROUGE-L/BLEU, coseno documentale, BERTScore documentale, NLI storico, coseno allineato e BERTScore allineato. Le guardie lessicali e l'export continuo sono presenti. La soglia storica OpAI `0.04075` non è applicata.

Le medie sui tre output per preset, tenendo distinte le scale, sono:

| Preset | Coseno documento | BERTScore documento grezzo | Minimo coseno allineato bidirezionale | Minimo BERTScore allineato riscalato | NLI medio forward / reverse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 3B | 0,934 | 0,940 | 0,593 | 0,440 | 0,693 / 0,662 |
| 14B | 0,882 | 0,917 | 0,523 | 0,470 | 0,666 / 0,629 |
| DIPPER passaggio | 0,949 | 0,965 | 0,522 | 0,530 | 0,760 / 0,618 |
| DIPPER tre frasi | 0,962 | 0,972 | 0,661 | 0,663 | 0,804 / 0,640 |

I minimi allineati sono già parte del confronto: non si attribuisce al tipo di metrica ciò che può dipendere dalla media documentale. Il valore numerico di scale diverse non misura una differenza di accuratezza; per questa servono etichette indipendenti.

Le similarità documentali restano elevate anche quando la lettura individua aggiunte, omissioni o cambi di forza delle affermazioni. Nel 14B Reuters il coseno è 0,867 e il BERTScore 0,908; il testo introduce il rallentamento delle transazioni attribuito al lavoro remoto e trasforma una situazione di bancarotta in una possibilità. Il DIPPER contestuale recupera elementi omessi dal preset sul passaggio, ma in Reuters cambia comunque ricavi in profitti e riformula impropriamente l'impatto sul sistema finanziario. I casi completi sono nella [review dell'assistente](text_review.md), che non è annotazione umana gold.

Il nuovo NLI contestuale impiega DeBERTa-v3 **base float32**, 3.594 confronti unici e circa 31 secondi di inferenza su GPU, esclusi caricamenti. Sono presenti 120 aggregati: 15 coppie × due direzioni × quattro modalità, tutti con copertura completa in questo smoke. Le premesse sono singole esaustive, finestre fino a due, fino a tre e documento esatto compatibile con il budget; sui testi estesi la copertura va verificata di nuovo.

L'effetto del contesto non è una prova automatica di accuratezza. Nel WritingPrompts DIPPER contestuale il reverse medio passa da circa 0,307 nel percorso storico a 0,754 con finestre fino a tre e 0,985 con documento come premessa. La lettura trova molta trama conservata e fusioni di frasi: il valore basso con premesse singole può essere un falso allarme di allineamento. Viceversa, nel 14B Reuters il forward con finestre fino a tre arriva circa a 0,990 nonostante cambiamenti verificabili: contesti più larghi possono anche supportare impropriamente una riscrittura.

Le finestre includono tutte quelle più corte e prendono il massimo supporto: l'aumento rispetto alle sole frasi è in parte imposto dalla costruzione. Non si trasferisce l'AUC 0,958 delle perturbazioni a questi output, non si chiama il punteggio percentuale di significato conservato e non si sceglie a posteriori la modalità che dà il risultato più favorevole. [Dati contestuali](contextual_summary.csv), [dati appaiati completi](per_pair_summary.csv).

## Naturalezza: esecuzione riuscita, preferenze sensibili all'ordine

Il provider conferma `qwen/qwen3.8-27b`. Le risposte usano `reasoning_effort=none`, senza cambio di modello implicito. Considerando anche la prova sui pilot precedenti, risultano 30 risposte valide e 32 tentativi POST. Le opzioni effettive, l'uso token e gli esiti sono registrati; la chiave non è esportata.

| Confronto | Coppie | Preferenze stabili AB/BA | Instabili |
| --- | ---: | ---: | ---: |
| Vecchia prova 3B/DIPPER contro proxy dell'assistente | 6 | 4 | 2 |
| Baseline 3B contro 14B | 3 | 2 | 1 |
| Baseline 3B contro DIPPER tre frasi | 3 | 2 | 1 |
| DIPPER passaggio contro tre frasi | 3 | 1 | 2 |

Nei nove confronti nuovi sono stabili **5/9** preferenze. I quattro casi instabili scelgono tutti **B in entrambi gli ordini**: il testo preferito cambia quando si scambiano le posizioni. Tutti e tre i confronti WritingPrompts sono instabili, insieme al confronto fra DIPPER su Reuters. Nelle 18 risposte nuove ci sono 13 scelte B e cinque A. Le preferenze sono coerenti con i voti ordinali restituiti: non emerge un errore di parsing che spieghi la discordanza.

Il [CSV di audit dell'ordine](naturalness_order_audit.csv) conserva preferenze e stabilità di tutti i 15 confronti, compresi i sei della vecchia prova.

Nei cinque casi stabili, il 3B vince su 14B in Essay e perde in Reuters; vince su DIPPER contestuale in Essay e Reuters; DIPPER contestuale vince sul passaggio in Essay. Sono descrizioni dei casi, non una graduatoria: gli instabili restano mancanti con il denominatore visibile. Non si ripete la chiamata fino a ottenere coerenza.

Il 100% di accordo sui quattro casi stabili della vecchia prova è **AI–AI**, condizionato alla copertura 4/6. I voti dell'assistente non sono validazione umana; il kappa è indefinito perché quelle quattro preferenze appartengono alla stessa classe. Per la tesi occorrono giudizi indipendenti e una misura esplicita di stabilità all'ordine anche sulle 126 fonti.

## Passaggio alle 126 fonti

Lo smoke non richiede altre inferenze. Il [notebook aggiornato](../kaggle/README_Study_v2.md) conserva i preset scientifici e corregge l'avvio nella sessione con cache di sola lettura: nuovo checkout e file temporanei sotto `/kaggle/working`, riuso della venv e degli snapshot completi leggibili, pesi semantici fuori dal RUN. La correzione di avvio è verificata localmente; il suo avvio reale sulle 126 non è ancora eseguito.

Il prossimo run usa `MODE = "study"`, mantiene i 300 gruppi di calibrazione e attende **126 fonti, 630 esiti e quattro riscrittori**. Le 84 fonti non viste nei pilot vanno anche riportate separatamente per le ipotesi nate sui 42 casi. Le consegne restano nel primario secondo la politica già scritta. Il campione reference e quello diagnostico per score rimangono distinti; le schede dello smoke sono vuote e non vengono inventate annotazioni.

Il pannello esteso potrà verificare variazioni di rilevabilità, dipendenza dal detector e controlli di lunghezza. La graduatoria di naturalezza e l'accuratezza semantica sui veri humanizer restano subordinate alla copertura e alle annotazioni, non al solo completamento tecnico.
