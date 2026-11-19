---
title: "Catalogo descrittivo delle metriche sulle riscritture dei quattro humanizer"
subtitle: "ROUGE-L, BLEU, BERTScore, coseno e Flesch — 19 novembre 2026"
lang: it-IT
fontsize: 10pt
papersize: a4
geometry: margin=2cm
colorlinks: true
---

**Risultato principale.** Il catalogo richiesto è completo sulle 500 riscritture valide dei quattro metodi, comprese quelle più lunghe dell'originale. Le tabelle di confronto usano le stesse 123 fonti. Qwen 14B presenta la minore sovrapposizione lessicale con l'originale; DIPPER a tre frasi conserva maggiormente le formulazioni secondo ROUGE-L e BLEU. Gli indici BERTScore e coseno rimangono elevati. Il Flesch Reading Ease diminuisce in mediana con i due Qwen e aumenta con le due configurazioni DIPPER. Sono descrizioni dei testi, non una graduatoria della qualità dell'humanization.

# 1. Obiettivo e campione

L'esperimento riunisce in un catalogo le metriche indicate dal professore: **ROUGE, BLEU, BERTScore, coseno tra embedding e Flesch**. Per ROUGE si utilizza la variante **ROUGE-L F1**; per Flesch l'indice principale è il **Flesch Reading Ease**, esplicitamente distinto dal Flesch–Kincaid Grade aggiuntivo.

Le prime quattro metriche confrontano un originale macchina con la sua riscrittura. Flesch viene invece calcolato separatamente sui due testi, per descriverne la leggibilità e la variazione. Non viene costruito un punteggio complessivo che sommi indici con significati e scale diversi.

Il dataset è quello dello studio su 126 testi macchina di MGTBench, 42 per ciascuno dei domini inglesi Essay, Reuters e WritingPrompts. Ogni *fonte* è il testo originale da cui derivano le diverse riscritture. I metodi sono Qwen2.5-3B-Instruct in FP16, Qwen2.5-14B-Instruct con quantizzazione NF4, DIPPER XXL sull'intero passaggio e DIPPER XXL a finestre di tre frasi con contesto cumulativo. Quest'ultima configurazione riscrive gruppi successivi di frasi, non applica tre volte DIPPER allo stesso testo.

| Metodo | Output validi | Generazioni incomplete | Allungamenti inclusi fra i validi |
|---|---:|---:|---:|
| Qwen 3B | 124 | 2 | 25 |
| Qwen 14B NF4 | 126 | 0 | 19 |
| DIPPER intero | 125 | 1 | 4 |
| DIPPER tre frasi | 125 | 1 | 17 |

In totale si analizzano **500 output validi su 504 tentativi**. Non si escludono le riscritture che allungano e non si imputano valori alle quattro generazioni incomplete.

I riepiloghi completi descrivono tutti gli output validi del singolo metodo. Le tabelle principali del report usano invece l'intersezione di **123 fonti con tutti e quattro gli output validi**, cioè 492 riscritture appaiate. Gli altri otto output validi rimangono nei riepiloghi individuali. Sul campione comune gli allungamenti sono rispettivamente 25, 17, 4 e 17.

# 2. Significato e implementazione delle metriche

**ROUGE-L F1.** Misura la somiglianza lessicale attraverso la sottosequenza comune più lunga, cioè parole condivise che mantengono lo stesso ordine anche senza essere tutte consecutive. Si usa l'F1 documentale con Porter stemming, che riduce alcune varianti morfologiche alla stessa radice. Il riferimento è l'originale, il candidato è la riscrittura. La scala è 0–1: un valore alto indica maggiore sovrapposizione, non necessariamente una riscrittura migliore.

**BLEU.** Misura la precisione di sequenze di parole condivise, considerando anche una penalizzazione per candidati troppo brevi. Si usa SacreBLEU con tokenizer `13a`, smoothing esponenziale ed effective order. L'originale è l'unico riferimento. Il punteggio viene calcolato separatamente sull'intero passaggio di ogni coppia e riportato sulla scala **0–100**. È la variante sentence-level applicata al documento: le medie e mediane qui riportate **non sono corpus BLEU**, che richiederebbe una diversa aggregazione dei conteggi.

**BERTScore F1 a chunk.** Confronta rappresentazioni contestuali dei token, usando **RoBERTa-base, layer 10**, senza pesi IDF e senza riscalatura nel valore documentale principale. Per non troncare i testi, il documento viene diviso in frasi/chunk. Ogni chunk riscritto seleziona la migliore corrispondenza nell'originale, e viceversa. Precision e recall vengono aggregate pesando i chunk per il numero di token; la loro media armonica produce l'F1 finale. Si tratta quindi di un'**approssimazione documentale a chunk**, non del BERTScore standard ottenuto passando un documento intero o troncato al modello.

**Coseno MiniLM.** Si usa `sentence-transformers/all-MiniLM-L6-v2`. Gli embedding normalizzati delle frasi/chunk vengono mediati con pesi proporzionali ai token e si calcola il coseno tra le rappresentazioni documentali. La scala teorica è −1–1. Un valore alto indica vicinanza nel relativo spazio di embedding; non garantisce che negazioni, quantità o cautele siano state conservate.

**Flesch Reading Ease.** È un indice di facilità strutturale di lettura, calcolato con `textstat` in inglese statunitense. Dipende dalla lunghezza media delle frasi e dalle sillabe per parola. Nella formula inglese:

$$
FRE = 206{,}835 - 1{,}015\frac{\text{parole}}{\text{frasi}}
      - 84{,}6\frac{\text{sillabe}}{\text{parole}}.
$$

I conteggi seguono le convenzioni di `textstat` e il dizionario locale delle pronunce. Valori più alti indicano maggiore facilità stimata. Il punteggio viene spesso letto su una scala indicativa 0–100, ma può uscire da tale intervallo: i risultati non vengono tagliati o normalizzati. Flesch non misura né la fedeltà al significato né quanto un testo sembri scritto da un essere umano.

Per ogni coppia si calcola **delta Flesch = Flesch della riscrittura meno Flesch dell'originale**. Nei CSV si conserva anche il Flesch–Kincaid Grade, che stima un livello scolastico statunitense e costituisce una misura aggiuntiva distinta dal Reading Ease.

# 3. Cosa è stato calcolato in questa esecuzione

L'archivio dello studio conteneva già punteggi completi di similarità per tutte le coppie valide. In questo esperimento **ROUGE-L, BLEU, Flesch Reading Ease e Flesch–Kincaid Grade sono stati ricalcolati su CPU** e confrontati con i valori precedenti. I punteggi neurali BERTScore e coseno sono stati riutilizzati dopo il controllo degli identificativi e della corrispondenza con i testi salvati. Non sono state eseguite nuove inferenze dei modelli neurali, nuove riscritture o chiamate API.

Il modello BERTScore è `FacebookAI/roberta-base`, revisione `e2da8e2f811d1448a5b465c236feacd80ffbac7b`; il modello MiniLM usa la revisione `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. I testi lunghi vengono suddivisi senza un troncamento documentale silenzioso.

Come controllo separato sono incluse anche 126 coppie di identità, dove originale e modificato coincidono. In queste coppie ROUGE-L vale 1, BLEU 100, BERTScore e coseno sono circa 1, e delta Flesch è 0. Minime differenze numeriche dei modelli intorno a 1 non vengono trasformate in nuovi segnali.

# 4. Tabella descrittiva delle similarità

La tabella usa sempre le **stesse 123 fonti**. Ogni cella contiene **mediana [primo quartile; terzo quartile]**. I quartili descrivono la dispersione dei documenti e non sono intervalli di confidenza.

| Metodo | ROUGE-L F1 | BLEU, 0–100 | BERTScore F1 a chunk | Coseno MiniLM |
|---|---:|---:|---:|---:|
| Qwen 3B | 0,560 [0,485; 0,633] | 27,2 [20,2; 35,0] | 0,950 [0,934; 0,958] | 0,944 [0,928; 0,957] |
| Qwen 14B NF4 | 0,417 [0,355; 0,479] | 14,2 [9,8; 21,7] | 0,928 [0,917; 0,938] | 0,913 [0,890; 0,940] |
| DIPPER intero | 0,658 [0,578; 0,759] | 35,2 [25,9; 49,3] | 0,950 [0,935; 0,961] | 0,932 [0,909; 0,957] |
| DIPPER tre frasi | 0,736 [0,676; 0,802] | 45,8 [39,8; 57,3] | 0,964 [0,955; 0,974] | 0,958 [0,936; 0,977] |

Per rendere confrontabili le descrizioni che usano la media, sono riportate anche le medie documentali sugli stessi 123 originali:

| Metodo | ROUGE-L F1 | BLEU, 0–100 | BERTScore F1 a chunk | Coseno MiniLM |
|---|---:|---:|---:|---:|
| Qwen 3B | 0,555 | 29,7 | 0,946 | 0,937 |
| Qwen 14B NF4 | 0,419 | 15,4 | 0,927 | 0,903 |
| DIPPER intero | 0,659 | 37,5 | 0,948 | 0,925 |
| DIPPER tre frasi | 0,741 | 48,5 | 0,964 | 0,951 |

Qwen 14B presenta meno sovrapposizione lessicale rispetto agli altri metodi. Questo è compatibile con una maggiore riformulazione, ma non dimostra né una migliore parafrasi né una maggiore perdita di significato. DIPPER tre frasi mantiene una maggiore sovrapposizione secondo entrambe le metriche lessicali. BERTScore e coseno rimangono numericamente elevati, ma i loro valori non sono percentuali di informazione conservata.

# 5. Leggibilità prima e dopo la riscrittura

Sulle 123 fonti comuni, il Flesch Reading Ease degli originali ha mediana **48,5**, con quartili **[29,6; 69,7]**. Questo riferimento è identico per tutti i metodi perché gli originali sono gli stessi.

| Metodo | Flesch della riscrittura: mediana [Q1; Q3] | Delta Flesch per coppia: mediana [Q1; Q3] |
|---|---:|---:|
| Qwen 3B | 39,2 [25,7; 62,0] | −5,9 [−11,3; −0,5] |
| Qwen 14B NF4 | 40,4 [24,8; 58,3] | −6,7 [−14,8; +0,5] |
| DIPPER intero | 50,0 [34,3; 69,1] | +0,9 [−4,2; +7,3] |
| DIPPER tre frasi | 53,0 [35,8; 74,2] | +3,9 [+0,2; +8,2] |

La mediana dei delta viene calcolata **dopo aver sottratto i due punteggi per ogni coppia**. Non coincide necessariamente con la mediana delle riscritture meno la mediana degli originali. Per esempio, per Qwen 3B il delta mediano è −5,9, anche se 39,2 meno 48,5 dà circa −9,3.

Secondo l'indice, i due Qwen producono in mediana testi di lettura strutturalmente meno facile. DIPPER intero mostra una variazione mediana piccola, mentre DIPPER tre frasi mostra un incremento. Questi risultati dipendono da lunghezza delle frasi, sillabazione e convenzioni del conteggio: non dimostrano che una riscrittura sia più naturale, più corretta o più adatta a ogni genere testuale. Frasi più semplici ottenute eliminando informazioni potrebbero aumentare Flesch senza migliorare la qualità della parafrasi.

# 6. BERTScore riscalato e NLI come diagnostiche aggiuntive

Il BERTScore documentale principale è grezzo. Un valore alto è comune anche per testi non perfettamente equivalenti e non va interpretato come un giudizio di correttezza. Per rendere disponibile una lettura più informativa della scala, il catalogo conserva anche il **BERTScore allineato per frase/chunk, già riscalato**, presente nell'archivio.

In questa variante ogni unità cerca la corrispondenza con F1 massimo nell'altro testo. Si applica la baseline inglese ufficiale di RoBERTa-base al layer 10, **0,814463**, mediante `(F1 − baseline)/(1 − baseline)`. Si aggregano poi media e minimo delle unità, mantenendo la direzione più debole. Le mediane sono:

| Metodo | Media allineata riscalata | Minimo allineato riscalato |
|---|---:|---:|
| Qwen 3B | 0,717 | 0,521 |
| Qwen 14B NF4 | 0,600 | 0,420 |
| DIPPER intero | 0,668 | 0,355 |
| DIPPER tre frasi | 0,807 | 0,609 |

Questo è un **protocollo diverso** dall'approssimazione documentale della tabella principale: cambia anche l'aggregazione. Non è stato ottenuto applicando una riscalatura arbitraria al solo F1 documentale finale. La scala riscalata non è una probabilità e può assumere valori negativi.

I file conservano inoltre il minimo e la media conservativa dell'NLI bidirezionale a finestre fino a due unità, già analizzato nell'esperimento precedente. Questi punteggi rimangono separati e non entrano in un indice combinato. Il confronto con essi è descrittivo: differenze tra metriche con scale, contesti e aggregatori diversi non misurano automaticamente quale metrica sia più accurata.

# 7. Interpretazione e limiti

Il catalogo distingue tre aspetti: **sovrapposizione lessicale**, **similarità nelle rappresentazioni neurali** e **facilità strutturale di lettura**. Nessuno di questi aspetti coincide da solo con la qualità dell'humanization.

ROUGE-L e BLEU possono penalizzare una parafrasi corretta che utilizza parole diverse. BERTScore e coseno possono restare elevati anche con alterazioni locali importanti. Flesch non controlla fatti, nomi, numeri, negazioni o omissioni. Un testo che rimane identico all'originale massimizza la similarità, ma non dimostra alcuna capacità di riscrittura.

Le tabelle usano gli stessi originali per rendere comparabili le descrizioni dei quattro metodi. Le riscritture mantengono comunque lunghezze diverse, e BLEU contiene una penalizzazione per la brevità: questa analisi non isola un effetto causale della lunghezza. Qwen 3B e 14B differiscono anche per precisione numerica, FP16 contro NF4; i risultati non isolano l'effetto della sola dimensione del modello.

Non si applicano soglie, test di significatività o nuove classificazioni di successo e fallimento. Non vengono dedotte superiorità statistiche dalle sole differenze delle medie. Le colonne di media, deviazione standard, mediana e quartili sono **descrittive**; non sono una validazione delle metriche rispetto a giudizi umani.

# 8. File, codice e riproduzione

I risultati sono in **`results_humanizer_metric_catalogue/`**, nel branch `feature/humanizer-metric-catalogue` della repository `FrancescoM00/Humanization`.

La **tabella unica riassuntiva** è `summary_common_median.csv`: contiene le quattro righe dei metodi e tutte le metriche sul campione comune. `summary_common_mean.csv` offre la stessa struttura per le medie. Nel PDF le metriche sono separate in tabelle più leggibili perché hanno scale e significati diversi.

`document_metrics.csv` contiene una riga per ciascuna delle 500 riscritture valide, con ID di fonte, metodo, metriche lessicali, punteggi neurali, Flesch originale e riscritto, delta e appartenenza al campione comune. `summary_descriptive.csv` contiene numerosità, media, deviazione standard, mediana e quartili, sia per tutti gli output validi sia per le fonti comuni, anche per dominio. `coverage.csv` documenta le esclusioni; `identity_control.csv` contiene le 126 identità.

`metric_definitions.json` descrive le metriche; `experiment.json` documenta input, versioni e verifiche rispetto ai punteggi archiviati. I grafici sono `catalogue_similarity.png` e `catalogue_readability.png`.

Il comando principale è `scripts/summarize_humanizer_metrics.py`; la logica di calcolo e riepilogo è in `semantics/metric_catalogue.py`. Dalla cartella del progetto:

```bash
python scripts/summarize_humanizer_metrics.py \
    --output results_humanizer_metric_catalogue
```

Servono le dipendenze del progetto e il corpus locale `cmudict` utilizzato da `textstat`. La rianalisi legge l'archivio dello studio già incluso e funziona su CPU, senza GPU, chiavi API o download dei modelli neurali.
