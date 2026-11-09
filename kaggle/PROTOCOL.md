# Protocollo del primo esperimento con humanizer e detector

## Obiettivo

Misurare, sugli stessi testi, quanto una riscrittura modifichi la detection, conservi il contenuto e cambi lo stile. La precedente validazione su corruzioni controllate non viene trasformata in una gold label dei nuovi output. La trascrizione fornita dal professore motiva detector eseguiti indipendentemente e una valutazione distinta della naturalezza.

## Coorti e congelamento

Si riutilizzano le 900 coppie del controllo stilometrico a lunghezza abbinata. Sono passaggi iniziali di 90–180 parole, a fine frase. Il nuovo campionamento usa soltanto metadati di fonte, dominio e generatore: non usa le feature numeriche, score dei detector o qualità delle riscritture.

Prima si riservano 6 fonti per ognuna delle 21 celle dominio/generatore (126 fonti); le prime 2 di ogni cella costituiscono il pilot di 42. La prova tecnica usa una fonte del pilot per dominio. L'ordine è un ranking deterministico basato su seed e identificatore. I 100 umani per dominio della calibrazione vengono poi scelti escludendo tutti i gruppi di fonte riservati allo studio, non soltanto i 42 già usati nel pilot.

Il notebook ricostruisce i testi selezionati attraverso righe e conteggi della tabella storica, verifica l'hash dei tre CSV e l'identificatore di ogni testo ricostruito. La selezione non rifà la ricerca degli abbinamenti storici. I file di coorte non vengono sostituiti con una selezione diversa nella stessa directory.

## Sistemi di riscrittura

L'identità è il controllo tecnico. Qwen2.5-3B-Instruct, con prompt di humanization fissato in `config.json`, è una baseline riproducibile di rewriting. DIPPER è opzionale, con diversità lessicale 60 e di ordine 0, singola finestra breve senza prefisso. Le revisioni sono immutabili e i sistemi producono un solo output greedy per testo: non si seleziona il migliore mediante il detector sul quale viene poi valutato.

Si conservano copie esatte dell'input, testo generato, limiti, completamento EOS, errori e output invariati. Una copia dell'originale è un output valido ma viene segnalata come invariata; un output vuoto, oltre il budget d'ingresso o interrotto al limite d'uscita è una generazione fallita. I testi non vengono troncati per ottenere uno score.

## Detector, soglie e ASR

Per ciascun detector si registra esplicitamente l'indice della classe AI. I testi che superano il contesto vengono divisi senza sovrapposizione e senza perdita silenziosa; il punteggio documentale è la media delle probabilità AI dei chunk, pesata per token non speciali. È una convenzione di aggregazione del progetto, non una nuova probabilità documentale calibrata universalmente.

Per ogni detector si fissa una soglia sui soli 300 umani di calibrazione. Si dichiara AI quando lo score è strettamente maggiore della soglia; con i pareggi assegnati a umano, la quota empirica di calibrazione non supera il 5% nominale. Questo non garantisce un FPR del 5% sui futuri dati. Si riporta separatamente il FPR osservato sugli umani di test e il TPR sui testi macchina originali, anche per dominio.

Sia $B_j$ il sottoinsieme di fonti il cui originale macchina è riconosciuto AI dal detector $j$. Per lo humanizer $h$:

$$
\mathrm{ASR}_{j,h}=\frac{\sum_{i\in B_j}\mathbf{1}\{\text{generazione valida e riscrittura classificata umana}\}}{|B_j|}.
$$

Il denominatore è fissato prima della riscrittura. Le generazioni fallite sono insuccessi nel risultato primario; un errore di scoring è un dato mancante da risolvere, non un successo. Se $|B_j|=0$, l'ASR è indefinita e resta mancante. Si mostrano numeratore, denominatore, completamento e intervallo bootstrap per gruppo di fonte. Le soglie rimangono fisse durante il bootstrap: gli intervalli non includono la loro incertezza di calibrazione.

Le differenze TPR/FNR prima e dopo sono calcolate anche sulla stessa coorte di coppie valide e complete, indicando il denominatore. La tabella per detector precede la sintesi: macro-media ASR dei detector con denominatore positivo e voto di maggioranza fissato sui tre verdetti calibrati. La macro-media segnala quanti detector contribuiscono; le probabilità grezze non vengono mediate fra checkpoint.

Il pool iniziale comprende tre checkpoint della famiglia RoBERTa con training diversi. Le loro decisioni non sono statisticamente indipendenti. OpenAI è una baseline storica GPT-2, Fakespot una variante a input grezzo, RADAR un detector addestrato anche su parafrasi avversariali. Non si generalizza il risultato a tutti i detector o a servizi proprietari.

## Conservazione e stile

Si riusano i checkpoint semantici fissati: MiniLM, RoBERTa-base e DeBERTa-v3 base float32. Coseno e BERTScore includono confronti frase/chunk allineati con gli stessi aggregatori dell'NLI. Si conservano forward/reverse, media, q10, minimo e quota non supportata, insieme a lunghezze e guardie lessicali.

Gli output nuovi non hanno frasi di errore annotate: non si inventano target locali. La soglia storica OpAI 0,04075 non viene applicata. I punteggi restano continui; minimo e lunghezze devono essere letti insieme. L'NLI non è una certificazione della verità o dell'assenza di errori.

Lo stile usa lo stesso estrattore a 105 feature e i quattro delta della shortlist. La distanza RMS usa le 91 candidate prefissate del precedente studio sulle frasi toccate, escludendo le feature costanti o non stimabili sul nuovo riferimento. Le scale sono ricalcolate sui soli umani di calibrazione a livello di passaggio intero, con uguale peso per fonte e SD campionaria. Non si importano le scale OpAI di frasi singole. Si riporta anche la distanza Jensen–Shannon delle POS; distanza e avvicinamento al centroide umano sono descrittivi, non misure automatiche di qualità o naturalezza.

## Naturalezza e annotazione

Il notebook produce un modulo semantico originale/riscrittura e uno di naturalezza a testo singolo. I form sono separati dalle chiavi del sistema; i testi identici vengono mostrati una volta nel modulo di naturalezza. Le etichette vuote sono da compilare e non sono risultati ottenuti.

La scala di naturalezza manuale è un primo controllo esplorativo, non una misura già validata. La proposta del professore di un giudice più grande, circa 20B/70B o un servizio equivalente, rimane distinta da Qwen3B usato per riscrivere. Leggibilità e perplexity, se aggiunta in seguito, non sostituiscono il quarto asse.

## Stato delle verifiche

Il notebook viene consegnato senza risultati di inference Kaggle. I controlli locali coprono separazione delle fonti, ricostruzione dei testi, identità, denominatore ASR, direzione delle etichette, copertura dei chunk e compatibilità dei file riusati. Il primo smoke GPU verifica caricamento e inferenza nella reale immagine Kaggle. L'analisi scientifica inizia soltanto dopo aver raccolto riscritture e score reali.
