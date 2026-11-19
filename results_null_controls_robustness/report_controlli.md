---
title: "Evasione dei detector: controlli di lunghezza e rumore lessicale"
subtitle: "Report del singolo esperimento — rianalisi del 19 novembre 2026"
lang: it-IT
fontsize: 10pt
papersize: a4
geometry: margin=2cm
colorlinks: true
---

**Risultato principale.** Sul campione di 78 testi utilizzabile per tutti e quattro i riscrittori, i due Qwen superano il controllo di sola lunghezza contro OpenAI RoBERTa, mentre contro Binoculars l'evidenza rimane per Qwen 14B. Richiedendo di superare tutti e tre i controlli sugli stessi 33 testi, soltanto Qwen 14B contro OpenAI supera la correzione statistica. Quest'ultimo risultato riguarda un detector storico e un campione piccolo: non dimostra un'evasione generalizzata dei detector.

# 1. Obiettivo e dati

Una riscrittura può rendere un testo meno riconoscibile da un detector, ma può anche accorciarlo o modificarne semplicemente alcune parole. L'esperimento verifica quanto dell'evasione osservata si ritrovi con questi interventi elementari. La domanda è: **la riscrittura evade più di un semplice accorciamento, di una sostituzione lessicale minima e della loro combinazione?**

Il punto di partenza è lo studio già eseguito su **126 testi generati da macchina** di MGTBench: 42 per ciascuno dei domini inglesi Essay, Reuters e WritingPrompts. Nel report una *fonte* è il testo macchina originale da cui si ottengono le diverse riscritture. Non si parte da un testo umano da trasformare: si parte da un testo macchina e si osserva come cambia la sua rilevazione.

I quattro metodi sono Qwen2.5-3B-Instruct, usato come baseline con generazione greedy in FP16; Qwen2.5-14B-Instruct, usato con quantizzazione NF4; DIPPER XXL sull'intero passaggio; DIPPER XXL a finestre di tre frasi, con il contesto delle finestre già riscritte. L'ultima configurazione non applica DIPPER tre volte allo stesso testo: elabora successivamente gruppi di tre frasi.

I detector sono OpenAI RoBERTa-base, Fakespot, RADAR e Binoculars. Viene riportato anche il voto di maggioranza dei primi tre: non include Binoculars e non costituisce un quinto modello indipendente. Le soglie dei quattro detector rimangono quelle calibrate per un FPR obiettivo del 5% su **300 testi umani separati dal test**. Non sono adattate ai nuovi confronti. Il voto di maggioranza mantiene la sua regola storica di almeno due voti su tre, senza una nuova calibrazione al 5%.

Questa è una **rianalisi dei punteggi già salvati**: non sono state generate nuove riscritture e non sono state eseguite nuove inferenze dei detector. I risultati erano già stati esaminati; l'analisi è quindi retrospettiva.

# 2. I tre controlli

**Solo lunghezza.** Si conserva l'inizio dell'originale, fermandosi allo stesso numero di parole della riscrittura. Il conteggio usa i token alfabetici di spaCy, come nelle analisi stilometriche. Non si aggiungono parole. Il taglio può avvenire dentro una frase e può eliminare informazioni: il controllo non è una parafrasi di qualità. Consente di verificare se il semplice accorciamento riproduca l'evasione della riscrittura.

**Solo rumore lessicale.** Si applica al massimo una sostituzione da un dizionario fissato, per esempio `also` con `likewise` oppure `use` con `utilize`. La lunghezza lessicale dell'originale resta invariata. Vincoli sulle categorie grammaticali e sulle entità riconosciute limitano i cambiamenti, senza garantire la conservazione del significato. Quando non esiste una sostituzione consentita, il testo non viene contato come un caso di rumore effettivo.

**Lunghezza più rumore.** Si costruisce prima il prefisso alla lunghezza richiesta, poi si applica la sostituzione lessicale, se possibile. Questo controllo combina i due interventi.

I tre confronti riguardano dunque interventi diversi. Un risultato che supera tutti e tre fornisce evidenza più convincente rispetto al solo confronto con il prefisso. Per confrontarli sulla stessa popolazione serve però un campione in cui siano tutti applicabili.

# 3. Campioni ed esclusioni

Il controllo esatto di lunghezza è applicabile quando la riscrittura è valida e non è più lunga dell'originale. Il numero di testi utilizzabili cambiava fra metodi:

| Riscrittore | Output validi | Allungamenti fra i validi | Output incompleti | Eleggibili per lunghezza |
|---|---:|---:|---:|---:|
| Qwen 3B | 124 | 25 | 2 | 99 |
| Qwen 14B NF4 | 126 | 19 | 0 | 107 |
| DIPPER intero | 125 | 4 | 1 | 121 |
| DIPPER tre frasi | 125 | 17 | 1 | 108 |

Le 27 esclusioni di Qwen 3B comprendono quindi 25 allungamenti e due output incompleti. Confrontare direttamente le prestazioni dei quattro metodi sui campioni separati avrebbe mescolato l'effetto del metodo con la composizione del campione.

La rianalisi introduce le seguenti intersezioni, ottenute senza usare gli score dei detector per scegliere i testi:

| Confronto | Stesse fonti per tutti i riscrittori | Escluse rispetto alle 126 |
|---|---:|---:|
| Controllo esatto di lunghezza | 78 | 48 |
| Rumore lessicale applicabile e output validi | 69 | 57 |
| Lunghezza più rumore | 33 | 93 |
| Tutti e tre i controlli simultaneamente | 33 | 93 |
| Tutti gli output validi, includendo gli allungamenti | 123 | 3 |

Il rumore è applicabile a 70 originali su 126; gli altri 56 non contengono una sostituzione ammessa dal dizionario e dai vincoli fissati. Richiedendo anche output validi per tutti i riscrittori, restano 69 fonti. Il prefisso può eliminare la parola modificabile: per lunghezza più rumore i campioni individuali sono 55, 50, 55 e 52, e l'intersezione scende a 33. Questi sono campioni selezionati dall'applicabilità del controllo, non sottocampioni casuali.

# 4. Misure e analisi statistica

L'**Attack Success Rate (ASR)** misura quante volte una modifica fa classificare come umano un originale macchina che il detector riconosceva inizialmente come macchina. Il denominatore contiene soltanto gli originali inizialmente rilevati, all'interno del campione selezionato.

Per esempio, sulle 78 fonti comuni OpenAI ne rileva inizialmente 50. Un risultato di 33/50 significa che 33 di questi 50 originali diventano non rilevati dopo la riscrittura. Non significa 33 evasioni su tutte le 126 fonti. I denominatori cambiano fra detector perché cambia la loro capacità iniziale di rilevazione.

La quantità confrontata è **ASR della riscrittura meno ASR del controllo**, espressa anche in punti percentuali. Un valore positivo indica più evasioni con la riscrittura.

Gli intervalli di ciascun ASR sono **Wilson bilaterali al 95%**. Restano informativi anche con zero successi: 0/95 dà un intervallo da 0 a circa 3,89%, non [0, 0]. Il limite esatto di circa 3,1% corrisponde invece a un limite superiore unilaterale al 95%.

Il confronto fra controllo e riscrittura usa il **test esatto bilaterale di McNemar**. Poiché i testi sono gli stessi, si contano le coppie discordanti: quelle che evadono soltanto con il controllo e quelle che evadono soltanto con la riscrittura. Si applica poi **Holm**, che tiene conto del numero di confronti. Sono disponibili la correzione sui 20 confronti di ogni controllo, cioè cinque decisioni detector per quattro riscrittori, e quella sui 60 confronti dei tre controlli insieme. Il report usa quest'ultima per le indicazioni principali di significatività, con soglia 0,05. Gli intervalli Wilson rimangono descrittivi e non sono intervalli simultanei.

I test confrontano ciascun riscrittore con il proprio controllo: **non sono test diretti di superiorità di Qwen 14B rispetto a Qwen 3B**. La correzione entro ciascun campione non autorizza inoltre a scegliere successivamente il campione che produce il risultato migliore. Le altre partizioni sono verifiche di sensibilità.

L'ASR non coincide con la variazione netta del TPR. Il TPR comprende anche originali inizialmente non rilevati, che possono diventare rilevabili dopo la riscrittura. Le tabelle complete riportano anche questi conteggi sull'intero sottoinsieme selezionato.

# 5. Risultati del controllo di lunghezza: 78 fonti comuni

Ogni cella riporta **evasioni con il prefisso → evasioni con la riscrittura**, con lo stesso denominatore nei due casi.

| Detector | Qwen 3B | Qwen 14B NF4 | DIPPER intero | DIPPER tre frasi |
|---|---:|---:|---:|---:|
| OpenAI | 1/50 → 33/50 | 6/50 → 41/50 | 7/50 → 7/50 | 4/50 → 4/50 |
| Fakespot | 0/71 → 1/71 | 0/71 → 1/71 | 1/71 → 5/71 | 0/71 → 4/71 |
| RADAR | 0/56 → 4/56 | 0/56 → 2/56 | 0/56 → 3/56 | 0/56 → 5/56 |
| Binoculars | 0/76 → 9/76 | 1/76 → 19/76 | 1/76 → 0/76 | 0/76 → 1/76 |
| Maggioranza | 0/63 → 2/63 | 0/63 → 2/63 | 0/63 → 3/63 | 0/63 → 2/63 |

Con Holm sui 60 confronti, rimangono significativi:

| Detector e riscrittore | Differenza di ASR | p corretto |
|---|---:|---:|
| OpenAI, Qwen 3B | +64,0 punti percentuali | 2,79 × 10⁻⁸ |
| OpenAI, Qwen 14B | +70,0 punti percentuali | 3,26 × 10⁻⁸ |
| Binoculars, Qwen 14B | +23,7 punti percentuali | 0,000427 |

Per questi confronti, il semplice accorciamento non riproduce l'evasione della riscrittura. Gli altri confronti non raggiungono la significatività dopo la correzione. In particolare, Binoculars con Qwen 3B ha p corretto 0,207 usando i 60 confronti; con la sola famiglia da 20 avrebbe comunque p=0,0664. La numerosità disponibile non consente di trattarlo come un effetto statisticamente sostenuto sul campione comune.

La scelta dei testi cambia l'interpretazione. Sui vecchi sottoinsiemi separati, Holm sui 20 confronti di lunghezza lascia significativi anche Binoculars con Qwen 3B e RADAR con DIPPER tre frasi. Sul campione comune questi due risultati non restano significativi. Non è una contraddizione: cambiano le fonti e la numerosità.

# 6. Confronto con tutti e tre i controlli sulle stesse 33 fonti

Il criterio più severo richiede una differenza positiva e significativa contro ciascuno dei tre controlli, usando le stesse fonti per tutti i riscrittori. **Soltanto Qwen 14B contro OpenAI soddisfa questo criterio.**

Fra le 33 fonti, OpenAI ne rileva inizialmente 17. Qwen 14B evade in **16/17 casi**, cioè il 94,1%, con intervallo Wilson al 95% **[73,0%; 99,0%]**. Gli stessi 17 casi danno:

| Controllo | Evasioni | ASR e intervallo Wilson 95% | Vantaggio della riscrittura | p Holm, 60 confronti |
|---|---:|---:|---:|---:|
| Solo lunghezza | 0/17 | 0,0% [0,0%; 18,4%] | +94,1 punti | 0,00183 |
| Solo rumore | 1/17 | 5,9% [1,0%; 27,0%] | +88,2 punti | 0,00360 |
| Lunghezza più rumore | 4/17 | 23,5% [9,6%; 47,3%] | +70,6 punti | 0,02832 |

Il vantaggio osservato supera quindi i tre interventi elementari considerati. Il denominatore resta però piccolo e riguarda un sottoinsieme di un detector storico. Il 94,1% non va presentato come ASR sull'intero studio.

Per Binoculars, Qwen 14B supera anche il controllo di solo rumore nel suo campione individuale: 16/68 evasioni contro 1/68, con p corretto 0,0143 sui 60 confronti. Tuttavia non supera il criterio congiunto sui 33 testi comuni. Non si può quindi estendere a Binoculars la conclusione più forte ottenuta per OpenAI.

Per gli altri metodi il mancato superamento dei tre controlli indica **evidenza insufficiente**. Non dimostra che le riscritture siano equivalenti al rumore, né che l'effetto reale sia nullo.

# 7. Sensibilità all'inclusione degli allungamenti

Per verificare se l'esclusione delle riscritture più lunghe cambi i risultati, si includono tutti gli output validi: 124 per Qwen 3B, 126 per Qwen 14B e 125 per ciascuna configurazione DIPPER. Il confronto fra metodi utilizza la loro intersezione di **123 fonti**.

Per una riscrittura più lunga dell'originale non esiste un prefisso dell'originale che ne raggiunga la lunghezza. Il controllo già disponibile conserva quindi l'originale intero. Questi casi sono segnalati come **non abbinati per lunghezza**. La verifica include gli allungamenti, ma non isola perfettamente il loro effetto; un controllo esatto richiederebbe un nuovo disegno e nuove inferenze.

| Detector | Qwen 3B | Qwen 14B NF4 | DIPPER intero | DIPPER tre frasi |
|---|---:|---:|---:|---:|
| OpenAI | 1/71 → 45/71 | 7/71 → 60/71 | 10/71 → 10/71 | 5/71 → 6/71 |
| Binoculars | 0/117 → 17/117 | 1/117 → 29/117 | 1/117 → 0/117 | 0/117 → 1/117 |

Con Holm sui 60 confronti, entrambi i Qwen superano il controllo su OpenAI e Binoculars. I vantaggi sono rispettivamente +62,0 e +74,6 punti su OpenAI, +14,5 e +23,9 punti su Binoculars. In questa sensibilità compare anche RADAR con DIPPER tre frasi, con p corretto 0,0498: un esito vicino alla soglia, assente nel controllo esatto sul campione comune di 78 fonti. Non viene quindi promosso a conclusione stabile dell'esperimento.

L'esclusione degli allungamenti non aumenta necessariamente l'ASR in tutti i casi. Nei campioni individuali, per OpenAI l'ASR di Qwen 3B passa dal 62,9% sugli eleggibili al 63,4% includendo gli allungamenti; per Qwen 14B passa dall'84,4% all'82,2%. Il confronto descrive la sensibilità alla composizione del campione e non misura un effetto causale dell'allungamento.

Sono riportate separatamente anche le 84 fonti che originariamente non appartenevano al pilot. La loro intersezione per il controllo esatto di lunghezza comprende 50 fonti e mantiene gli stessi tre confronti significativi del pannello principale. L'intersezione per tutti i controlli scende a 22: nessun metodo supera lì il criterio congiunto dopo Holm. Queste analisi non costituiscono una nuova validazione indipendente, perché i risultati dello studio erano già stati osservati.

# 8. Interpretazione e limiti

L'esperimento sostiene un risultato specifico: **l'evasione dipende dal detector e dal riscrittore, e alcuni vantaggi rispetto ai controlli rimangono anche usando le stesse fonti**. Il risultato più netto rispetto a tutti i controlli è Qwen 14B contro OpenAI RoBERTa. Contro Binoculars esiste evidenza rispetto alla sola lunghezza e al solo rumore nei rispettivi campioni, ma non per la verifica congiunta più restrittiva.

Il campione comune rende il confronto più equo fra metodi, ma non elimina ogni selezione: le 78 fonti rappresentano testi che tutti i riscrittori riescono a non allungare, e le 33 richiedono anche una sostituzione lessicale consentita. Il controllo di rumore è molto debole e circoscritto a una sola sostituzione. Il prefisso può eliminare informazioni e terminare dentro una frase. I risultati riguardano quindi questi controlli specifici, non ogni possibile perturbazione.

Qwen 3B e Qwen 14B differiscono anche nella precisione numerica, FP16 contro NF4; i test effettuati non isolano l'effetto della dimensione del modello. La mancata significatività non dimostra equivalenza. Infine, una mancata rilevazione non prova naturalezza o conservazione del significato: questo esperimento misura soltanto il comportamento dei detector.

# 9. Script, risultati e riproduzione

L'analisi usa il codice della repository `FrancescoM00/Humanization`, branch `feature/detector-before-after-metrics`, commit **87f2542**. I 87 controlli sul codice eseguiti per questa modifica sono passati. Il materiale di questo report documenta tale esecuzione senza cambiarne i risultati.

| File o cartella | Contenuto |
|---|---|
| `scripts/analyze_null_controls.py` | Comando di rianalisi e grafici. |
| `humanizers/null_control_analysis.py` | Costruzione dei campioni, coppie e confronti. |
| `humanizers/null_control_statistics.py` | Wilson, McNemar esatto e Holm. |
| `results_null_controls_robustness/` | Tabelle, punteggi appaiati, grafici e protocollo. |

I file principali nella cartella dei risultati sono `length_common_sources.csv` per il controllo di lunghezza, `all_three_same_sources.csv` per i tre controlli sugli stessi testi, `expansion_sensitivity.csv` per gli allungamenti, `coverage.csv` ed `eligibility.csv` per le esclusioni. `comparisons.csv` contiene tutte le analisi; `panel_membership.csv` elenca gli ID utilizzati; `paired_scores.csv` conserva punteggi e decisioni; `experiment.json` descrive input e definizioni. I grafici sono `length_common_sources.png` ed `expansion_sensitivity.png`.

Il comando, dalla cartella del progetto con le dipendenze già installate, è:

```bash
python scripts/analyze_null_controls.py --output results_null_controls_robustness
```

Legge l'archivio dello studio già incluso nella repository. Non richiede GPU, chiavi API o download di modelli. I risultati storici sono conservati; i nuovi intervalli e i confronti corretti sostituiscono le precedenti interpretazioni basate soltanto sul bootstrap degli ASR.
