---
title: "Conservazione del significato nelle riscritture dei quattro humanizer"
subtitle: "Report del singolo esperimento — 19 novembre 2026"
lang: it-IT
fontsize: 10pt
papersize: a4
geometry: margin=2cm
colorlinks: true
---

**Risultato principale.** Il valutatore NLI a finestre fino a due frasi è stato applicato alle 500 riscritture valide dello studio, comprese quelle più lunghe dell'originale. Sul campione comune di 123 fonti, DIPPER intero presenta supporti inferiori rispetto alla configurazione a tre frasi. Le due configurazioni Qwen non risultano nettamente ordinate dal confronto descrittivo. Le guardie lessicali rilevano modifiche di numeri, date, entità, negazioni, quantificatori e cautele, ma producono anche falsi allarmi: le loro percentuali non rappresentano errori semantici accertati.

# 1. Obiettivo

L'esperimento valuta **quanto il significato di un testo macchina venga conservato dopo la riscrittura**. Per ogni originale si confrontano gli output dei quattro metodi già eseguiti. Il punto di partenza è quindi un testo generato da un LLM, non un testo umano da trasformare.

La valutazione combina due strumenti. L'NLI misura se le affermazioni di un testo trovano supporto nell'altro. Le guardie lessicali cercano cambiamenti specifici che un modello NLI può sottovalutare, soprattutto nella forza delle affermazioni: per esempio, eliminare *likely* può trasformare una previsione cauta in un'affermazione più netta.

La configurazione NLI era stata scelta su perturbazioni controllate e riscritture OpAI. Qui viene utilizzata su output reali dei quattro riscrittori. È un'applicazione del valutatore, **non una nuova validazione della sua accuratezza con etichette umane**.

# 2. Dati e metodi di riscrittura

Lo studio comprende 126 testi macchina di MGTBench, 42 per ciascuno dei domini inglesi Essay, Reuters e WritingPrompts. Nel report una **fonte** è il testo originale da cui derivano le quattro riscritture.

I metodi sono Qwen2.5-3B-Instruct, baseline con generazione greedy in FP16; Qwen2.5-14B-Instruct, eseguito con quantizzazione NF4; DIPPER XXL sull'intero passaggio; DIPPER XXL a finestre di tre frasi, mantenendo il contesto delle finestre già riscritte. La configurazione a tre frasi elabora gruppi successivi di frasi: non applica DIPPER tre volte allo stesso passaggio.

| Metodo | Tentativi | Output validi | Incompleti | Allungamenti inclusi fra i validi |
|---|---:|---:|---:|---:|
| Qwen 3B | 126 | 124 | 2 | 25 |
| Qwen 14B NF4 | 126 | 126 | 0 | 19 |
| DIPPER intero | 126 | 125 | 1 | 4 |
| DIPPER tre frasi | 126 | 125 | 1 | 17 |

I 504 tentativi producono quindi **500 output validi**. Le quattro generazioni incomplete riguardano tre fonti e sono dichiarate, senza assegnare loro un punteggio NLI artificiale.

Sono mantenuti due livelli di analisi. Il primo descrive tutti gli output validi di ciascun metodo. Il secondo usa le **123 fonti con tutte e quattro le riscritture valide**, per un totale di 492 output appaiati. Gli otto output validi delle altre tre fonti rimangono nelle analisi complete del singolo metodo. Le riscritture più lunghe sono incluse in entrambi i livelli; nel campione comune gli allungamenti sono rispettivamente 25, 17, 4 e 17.

# 3. Come funziona il valutatore NLI

Il modello è **DeBERTa-v3-base addestrato su MNLI, FEVER e ANLI**, repository `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, revisione `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`.

Nel confronto NLI una *premessa* è il testo che dovrebbe fornire evidenza, mentre un'*ipotesi* è l'affermazione da verificare. Il modello restituisce un supporto di entailment: quanto l'ipotesi sia implicata dalla premessa secondo il modello.

I documenti lunghi non vengono passati interamente al modello. Si riutilizzano le unità di frase o chunk già fissate, fino a 160 token NLI per unità. Per ogni ipotesi si considerano come premesse le finestre contigue di **una o due unità** dell'altro testo, rispettando il budget di 384 token per coppia. Si conserva il supporto più alto tra le premesse ammissibili. Le finestre di due unità aiutano quando una riscrittura fonde informazioni distribuite in frasi vicine.

Il confronto viene ripetuto nelle due direzioni:

- **Originale → riscrittura, forward:** ogni unità riscritta deve trovare supporto nell'originale. Un punteggio basso può indicare un'aggiunta o un'affermazione modificata.
- **Riscrittura → originale, reverse:** ogni unità originale deve trovare supporto nella riscrittura. Un punteggio basso può indicare informazioni omesse.

Queste letture sono diagnostiche, non prove automatiche della causa. Una frase difficile, un allineamento imperfetto o un errore del modello possono abbassare il supporto anche senza un errore semantico.

Per ogni documento si salvano **minimo e media dei supporti in ciascuna direzione**. Il minimo evidenzia l'unità meno supportata; la media descrive il supporto complessivo delle unità. Gli aggregati bidirezionali seguono le convenzioni della suite:

- **Minimo bidirezionale:** il più basso fra il minimo forward e il minimo reverse.
- **Media bidirezionale conservativa:** il più basso fra la media forward e la media reverse.

La seconda misura **non è la media aritmetica delle due direzioni**. Quest'ultima è conservata separatamente, come diagnostica aggiuntiva. Prima si calcolano gli aggregati entro ogni documento; successivamente se ne descrive la distribuzione fra documenti.

Gli score a finestre di due frasi erano già disponibili per tutte le coppie valide. La rianalisi verifica la corrispondenza dei testi e degli allineamenti e ricostruisce gli aggregati dalle unità salvate. La copertura è completa nelle due direzioni. Non sono state rieseguite inferenze DeBERTa; è stata eseguita la nuova estrazione spaCy delle guardie lessicali. Il controllo aggiuntivo di identità comprende 126 coppie in cui originale e modificato coincidono.

# 4. Distribuzioni del supporto NLI

La tabella principale usa le **stesse 123 fonti per tutti i metodi**. Ogni cella contiene **mediana [primo quartile; terzo quartile]**: l'intervallo tra quartili descrive la metà centrale dei documenti e non è un intervallo di confidenza.

| Metodo | Minimo bidirezionale | Media bidirezionale conservativa |
|---|---:|---:|
| Qwen 3B | 0,110 [0,013; 0,809] | 0,872 [0,786; 0,968] |
| Qwen 14B NF4 | 0,105 [0,013; 0,879] | 0,853 [0,739; 0,976] |
| DIPPER intero | 0,006 [0,002; 0,030] | 0,687 [0,563; 0,809] |
| DIPPER tre frasi | 0,090 [0,023; 0,630] | 0,849 [0,765; 0,940] |

Il divario tra minimo e media mostra che un buon supporto complessivo può convivere con almeno un'unità molto poco supportata. Per DIPPER intero il terzo quartile del minimo è circa 0,030: il supporto della peggiore unità è basso nella maggior parte dei documenti di questo campione. Questo non quantifica da solo quanta informazione sia stata persa né dimostra che ciascun minimo basso corrisponda a un errore reale.

Usando tutti gli output validi di ciascun metodo, le mediane risultano:

| Metodo | Numero di output | Minimo bidirezionale mediano | Media conservativa mediana |
|---|---:|---:|---:|
| Qwen 3B | 124 | 0,107 | 0,872 |
| Qwen 14B NF4 | 126 | 0,102 | 0,849 |
| DIPPER intero | 125 | 0,006 | 0,687 |
| DIPPER tre frasi | 125 | 0,102 | 0,851 |

I campioni della seconda tabella non sono identici: il confronto fra metodi fa quindi riferimento alla prima tabella e al bootstrap sulle fonti comuni. Quartili e risultati per dominio di entrambi i livelli sono conservati nei CSV.

Non si applica una soglia decisionale. Il riferimento storico **0,04075** viene usato soltanto per un conteggio descrittivo: sul campione comune risultano sotto tale valore 46/123 documenti per Qwen 3B, 47/123 per Qwen 14B, 95/123 per DIPPER intero e 44/123 per DIPPER tre frasi. Non sono conteggi di errori accertati. Il controllo di identità mantiene una mediana del minimo pari a circa **0,987**, senza segnalazioni lessicali.

# 5. Confronto appaiato con bootstrap per fonte

Per confrontare i metodi si ricampionano **5.000 volte le 123 fonti**, mantenendo insieme i quattro output di ciascuna fonte. In questo modo le differenze non dipendono dall'uso di testi originali diversi e le quattro riscritture dello stesso originale non vengono trattate come fonti indipendenti.

Per l'NLI la quantità stimata è la differenza fra le mediane documentali dei due metodi. Gli intervalli sono i percentili 2,5 e 97,5 delle differenze ricampionate. La tabella usa sempre **metodo B meno metodo A**:

| A → B | Differenza del minimo mediano | Intervallo bootstrap 95% |
|---|---:|---:|
| Qwen 3B → Qwen 14B | −0,005 | [−0,121; +0,134] |
| Qwen 3B → DIPPER intero | −0,104 | [−0,250; −0,040] |
| Qwen 3B → DIPPER tre frasi | −0,020 | [−0,136; +0,080] |
| Qwen 14B → DIPPER intero | −0,099 | [−0,284; −0,049] |
| Qwen 14B → DIPPER tre frasi | −0,015 | [−0,179; +0,067] |
| DIPPER intero → DIPPER tre frasi | +0,083 | [+0,046; +0,156] |

La configurazione DIPPER a tre frasi ha un minimo mediano superiore a quella sull'intero passaggio secondo questo valutatore. Anche per la media conservativa il confronto dà **+0,162**, con intervallo **[+0,117; +0,195]**. Per Qwen 14B meno Qwen 3B la differenza della media conservativa è **−0,019**, con intervallo **[−0,056; +0,015]**: non emerge un ordinamento netto dei due metodi.

Gli intervalli sono **puntuali e descrittivi**, senza correzione congiunta per i diversi confronti. Non vengono utilizzati per formulare nuove conclusioni confermative di superiorità. Il confronto misura differenze del punteggio NLI, non un miglioramento verificato della qualità semantica. Tutti i sei confronti, anche per dominio e per le singole direzioni, sono disponibili nei risultati.

# 6. Guardie lessicali: cosa viene controllato

Le guardie sono regole deterministiche indipendenti dagli score NLI. Per ogni famiglia vengono salvati inventari originali e modificati, elementi aggiunti o rimossi e differenze di conteggio.

**Numeri e date.** I numeri vengono normalizzati quando possibile, comprendendo una parte delle forme in lettere, percentuali e valute. Le date riconosciute da spaCy e i pattern di calendario espliciti sono controllati separatamente; alcune variazioni di formato vengono normalizzate. Gli inventari mantengono le frequenze di menzione: anche una ripetizione eliminata può produrre una segnalazione. Non esiste una normalizzazione generale di ogni forma numerica, data o conversione di unità.

**Entità.** Si confronta la presenza delle superfici riconosciute da spaCy, con normalizzazioni di maiuscole e punteggiatura. Le entità numeriche e temporali vengono trattate nelle famiglie dedicate. Ripetere un nome non crea una nuova entità, ma alias, pronomi o confini NER diversi possono produrre differenze apparenti. Non viene risolta la coreferenza.

**Negazioni.** Si salvano conteggio e inventario normalizzato, riconoscendo anche forme contratte. Le negazioni comprese in espressioni come *no more than* vengono trattate come parte del limite quantitativo, per evitare un doppio conteggio. La regola non risolve la portata logica della negazione.

**Quantificatori.** Si cercano cambiamenti di quantificazione e di limiti numerici. *Up to 100* e *over 100* ricevono rappresentazioni diverse, perché il primo pone un limite superiore inclusivo e il secondo un limite inferiore esclusivo. Le espressioni di limite sono associate a un numero vicino; altri quantificatori, come *some* e *all*, usano un inventario lessicale.

**Cautele.** Si segnala il calo del numero totale di espressioni modali e lessicali riconosciute, come *might*, *perhaps*, *possibly* e *likely*. Sostituire *may* con *might* senza cambiare il totale non produce una perdita. La regola non distingue sempre incertezza, capacità, permesso ed espressioni idiomatiche; inoltre può perdere cambiamenti di intensità a conteggio invariato.

# 7. Frequenza delle segnalazioni lessicali

Le percentuali seguenti hanno tutte lo stesso denominatore: **123 documenti per metodo**. Sono inclusi anche i documenti senza elementi originali di una famiglia, che possono comunque introdurne di nuovi. Le famiglie possono sovrapporsi nello stesso documento.

| Metodo | Numeri | Date | Entità | Negazioni |
|---|---:|---:|---:|---:|
| Qwen 3B | 38,2% | 45,5% | 40,7% | 26,0% |
| Qwen 14B NF4 | 48,0% | 51,2% | 51,2% | 39,8% |
| DIPPER intero | 44,7% | 39,8% | 65,9% | 15,4% |
| DIPPER tre frasi | 30,1% | 35,0% | 56,9% | 12,2% |

| Metodo | Quantificatori | Cautele diminuite | Almeno una segnalazione |
|---|---:|---:|---:|
| Qwen 3B | 65,0% | 8,9% | 94,3% (116/123) |
| Qwen 14B NF4 | 74,8% | 17,9% | 96,7% (119/123) |
| DIPPER intero | 57,7% | 21,1% | 95,9% (118/123) |
| DIPPER tre frasi | 48,8% | 13,8% | 87,0% (107/123) |

**Queste percentuali non misurano la quota di testi semanticamente errati.** Le regole cercano differenze ampie e possono reagire a parafrasi corrette o a errori del riconoscitore. La quota “almeno una” è l'unione delle famiglie, non la somma delle percentuali.

Le quote individuali hanno intervalli Wilson bilaterali al 95%. Per esempio, la riduzione di cautele è 8,9% **[5,1%; 15,3%]** per Qwen 3B, 17,9% **[12,1%; 25,6%]** per Qwen 14B, 21,1% **[14,9%; 29,2%]** per DIPPER intero e 13,8% **[8,8%; 21,0%]** per DIPPER tre frasi. Questi intervalli riguardano la frequenza della segnalazione, non la precisione della regola.

Anche il confronto delle quote usa il bootstrap appaiato. Qwen 14B meno Qwen 3B dà **+8,9 punti percentuali** di documenti con cautele diminuite, intervallo **[+3,3; +14,6]**. DIPPER tre frasi meno DIPPER intero dà **−8,9 punti** per almeno una segnalazione, intervallo **[−14,6; −4,1]**. Rimangono confronti descrittivi dei comportamenti delle regole, non differenze accertate nel numero di errori reali.

# 8. Esempi e limiti delle segnalazioni

Le evidenze salvate mostrano il caso ricercato **“up to 100 feet” → “over 100 feet”**, in una riscrittura Qwen 3B della fonte Essay:308. La guardia riconosce il cambio di limite, anche se il numero 100 rimane identico. Un ulteriore caso di cautela è **“it will likely be many years” → “It will be many years”**, nella fonte Reuters:413 riscritta da DIPPER tre frasi.

Sono presenti anche esempi benigni che spiegano perché le quote automatiche siano elevate:

- **Reuters:258, DIPPER:** *McDonnell Douglas Aerospace Business* diventa *the McDonnell Douglas Aerospace Business*. Il riferimento può rimanere lo stesso, ma cambia il confine dello span riconosciuto come entità.
- **WritingPrompts:961, Qwen 14B:** *couldn't help but feel inspired* diventa *It was impossible not to feel inspired*. Il conteggio di *could* diminuisce, ma l'espressione originale non è necessariamente una cautela epistemica.
- **WritingPrompts:285, Qwen 3B:** *One day* diventa *One fateful day*. Un diverso riconoscimento DATE può spostare *one* dall'inventario temporale a quello numerico, senza un cambiamento reale del valore.

Questi sono esempi diagnostici, non nuove annotazioni umane né una stima della precisione delle guardie. Le regole non sono state adattate a tali esempi dopo l'esecuzione. Per quantificare quanti allarmi corrispondano davvero a errori servono etichette umane sul campione di interesse.

Il minimo NLI dipende dal numero di ipotesi: con più unità aumenta l'occasione di trovare una frase difficile. Includere gli allungamenti è necessario per descrivere tutti gli output validi, ma non elimina tale dipendenza. Le lunghezze e il numero di ipotesi sono salvati; il bootstrap sulle stesse fonti non corregge automaticamente le differenze di lunghezza o segmentazione delle riscritture.

Le finestre di due unità aiutano sulle fusioni locali, ma non risolvono ogni riordinamento, fusione estesa o omissione. Gli inventari lessicali globali possono inoltre non rilevare un numero o un'entità riassegnati a un altro fatto quando l'insieme delle parole rimane invariato. Qwen 3B e Qwen 14B differiscono anche nella precisione numerica: non viene isolato l'effetto della sola dimensione del modello.

Il risultato sostenuto da questa analisi è quindi una **descrizione comparabile dei punteggi e delle segnalazioni** sui quattro metodi. Per trasformarla in una misura della reale fedeltà semantica occorre il confronto con annotazioni umane. Un punteggio 0,110 non significa che sia conservato soltanto l'11% del significato.

# 9. Codice, risultati e riproduzione

L'esecuzione documentata corrisponde al commit **3525d64** del branch `feature/final-semantic-evaluation` della repository `FrancescoM00/Humanization`. Questo report non modifica i risultati dell'esecuzione.

Gli script principali sono:

- `scripts/evaluate_humanizer_semantics.py`: avvia la rianalisi, esporta tabelle ed evidenze e costruisce i grafici.
- `semantics/humanizer_evaluation.py`: verifica le coppie NLI, ricostruisce gli aggregati e calcola distribuzioni e confronti appaiati.
- `semantics/humanizer_guards.py`: estrae gli inventari spaCy e applica le nuove guardie; quelle storiche restano invariate.

Tutti i risultati sono in **`results_humanizer_semantics_final/`**. `document_scores.csv` contiene i 500 documenti con minimo e media forward/reverse, aggregati bidirezionali, numero di ipotesi, lunghezze, evidenze dei minimi e guardie. `nli_distributions.csv` e `lexical_alert_rates.csv` riportano i riepiloghi, distinguendo campione completo e fonti comuni, anche per dominio. `paired_source_comparisons.csv` contiene differenze e intervalli bootstrap.

`lexical_alert_evidence.jsonl` conserva originale, riscrittura e differenze delle coppie segnalate. `coverage.csv`, `generation_status.csv` e `common_source_ids.csv` permettono di controllare copertura, esclusioni e ID comuni. `historical_reference_only.csv` contiene il solo confronto descrittivo con 0,04075; `identity_control.csv` contiene le identità. `experiment.json` documenta definizioni, regole e provenienza. I grafici sono `nli_distributions_common.png` e `lexical_alert_rates_common.png`.

Il comando, dalla cartella del progetto con dipendenze e modello spaCy `en_core_web_sm` già installati, è:

```bash
python scripts/evaluate_humanizer_semantics.py \
    --output results_humanizer_semantics_final --bootstrap 5000
```

La rianalisi usa CPU e l'archivio dello studio già incluso nella repository. Non richiede GPU, chiavi API o nuovi download dei modelli NLI. Conserva gli score storici ed esegue soltanto la nuova estrazione spaCy e i calcoli statistici.
