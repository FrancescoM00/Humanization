# Smoke DIPPER su Kaggle: esito e confronto con baseline 3B

In questo report **baseline 3B** indica Qwen2.5-3B-Instruct con il prompt fissato e decoding greedy; **DIPPER** indica la configurazione XXL NF4 sul passaggio intero senza prefisso, lessico 60/ordine 0. I risultati non descrivono ogni configurazione dei due modelli.

**Lo smoke DIPPER è completo e permette di passare al pilot congelato di 42 fonti.** Sono riuscite tutte e tre le riscritture e tutti gli stadi di valutazione. Nei tre casi DIPPER non elude alcuno dei detector, comprime i passaggi e introduce alcune perdite di informazione. Il caso Reuters mostra un'asimmetria NLI interpretabile direttamente nel testo; altri minimi bassi risentono della fusione di frasi.

Analisi dei soli output ricevuti, senza nuove generazioni o inferenze. Sono disponibili [ZIP originale](../exports/kaggle_dipper_smoke/README.md), [score per coppia](per_pair_summary.csv), [confronto appaiato con baseline 3B](comparison_with_qwen_smoke.csv), [lettura dei testi](text_review.md), [integrità](integrity.md) e [provenienza](provenance.json). Le letture qualitative sono dell'assistente e non costituiscono annotazioni umane indipendenti.

## Protocollo effettivamente eseguito

Tre passaggi macchina di **ChatGLM**, uno per Essay, Reuters e WritingPrompts, identici agli input dello smoke Qwen. Sono prefissi di 90–180 parole terminati a fine frase, non documenti completi. I tre umani test e i 300 umani di calibrazione coincidono con quelli precedenti. Il campione tecnico non rappresenta i sette generatori; il pilot ne comprenderà due testi per ciascuna delle 21 celle dominio/generatore.

Il riscrittore è `kalpeshk2011/dipper-paraphraser-xxl`, revisione `c1fbf7a958a2aab022e9e6f81f7a3139f9e6ee3c`, caricato a 4 bit su `cuda:0`. Il tokenizer è `google/t5-v1_1-xxl`; diversità lessicale 60, diversità d'ordine 0, decoding greedy, un solo tentativo per fonte e nessun feedback dei detector. Il protocollo usa **l'intero passaggio senza prefisso contestuale**, una variante esplicita rispetto alle finestre di tre frasi della procedura originale di DIPPER. Non si generalizzano questi risultati a tutte le configurazioni del modello.

Le tre identità e le tre riscritture hanno stato `ok`. Gli input hanno 206–212 token registrati e gli output DIPPER 143–158, con EOS raggiunto: le omissioni non sono dovute al limite configurato di 512 token. La piattaforma riporta PyTorch 2.11.0+cu130, due T4 disponibili e una GPU usata per ciascun processo. L'NLI mantiene **DeBERTa-v3 base float32**; la quantizzazione riguarda DIPPER.

## Evasione dei detector

| Detector | Originali rilevati AI | Successi DIPPER | Successi baseline 3B sugli stessi input |
| --- | ---: | ---: | ---: |
| OpenAI RoBERTa-base | 3/3 | 0/3 | 2/3 |
| Fakespot raw | 3/3 | 0/3 | 0/3 |
| RADAR | 3/3 | 0/3 | 0/3 |
| Voto di maggioranza | 3/3 | 0/3 | 0/3 |

Tutti i detector continuano a classificare AI ogni riscrittura DIPPER. L'ASR e la macro-ASR sono zero; anche l'identità ha ASR zero. Nessuno dei tre umani test è classificato AI. Le soglie individuali sono identiche a quelle baseline 3B, calibrate sui medesimi 300 umani con FPR empirico del 5%; il voto di maggioranza ha FPR di calibrazione 5/300.

Gli score AI DIPPER restano fra 0,993584 e 0,999905, sopra le soglie fissate. Non è un errore di inversione delle classi: le etichette AI sono esplicite e originali e identità hanno risultati coerenti. **Zero evasioni su tre fonti non dimostra che DIPPER sia inefficace nella popolazione.** Gli eventuali intervalli bootstrap 0–0 sono degeneri per questo campione e non costituiscono una garanzia. Anche il vantaggio osservato di baseline 3B sul solo OpenAI va verificato sui 42 testi. I tre modelli condividono la famiglia RoBERTa e non rappresentano tutti i detector.

## Conservazione del significato

Forward indica originale → riscrittura; reverse indica riscrittura → originale. I valori sono aggregati dei supporti NLI alle frasi/chunk, non probabilità che l'intero documento conservi il significato. BERTScore documento è il punteggio raw con matching di chunk previsto dalla suite.

| Dominio | Coseno doc | BERTScore doc F1 | NLI F medio | NLI R medio | NLI F minimo | NLI R minimo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Essay | 0,982035 | 0,971569 | 0,793982 | 0,713215 | 0,004202 | 0,005618 |
| Reuters | 0,972092 | 0,978208 | 0,984372 | 0,678702 | 0,950440 | 0,000334 |
| WritingPrompts | 0,892378 | 0,946528 | 0,500706 | 0,460963 | 0,004366 | 0,001560 |

**Reuters è il caso più leggibile.** Sparisce la frase «In addition to the economic downturn, Akeco has also been facing a number of legal challenges». Il forward rimane alto perché la riscrittura è ampiamente sostenuta dall'input; il reverse cala perché non ricostruisce tutte le informazioni originali. Il minimo reverse 0,000334 riguarda proprio la frase eliminata sui problemi legali. Rimangono Akeco, il 1973, la pandemia e le difficoltà economiche. La corrispondenza fra omissione testuale e direzione del supporto dà un esempio concreto di informazione persa, senza usare l'NLI stesso come unica prova dell'errore.

**Essay mostra sia omissioni sia un limite del minimo.** Restano il confronto fra medicina e assistenza infermieristica e il riferimento a PICOT. Spariscono i riferimenti espliciti ai colloqui clinici e alle popolazioni. La lista degli esami e le immagini diagnostiche vengono però fuse da due frasi originali in una frase nuova: nessuna singola premessa contiene tutta la nuova lista. Il forward minimo 0,004202 su questa frase non dimostra che le immagini siano inventate: sono presenti altrove nell'originale. Il retrieval prende il massimo fra singole premesse e non ne combina il contenuto.

**WritingPrompts contiene una modifica del meccanismo narrativo.** Il potere olfattivo di Jack viene spiegato attraverso «their looks and their expressions», un canale nuovo rispetto all'input. Sarah diventa anche «lonely», mentre prima era descritta come «down and upset»; la sua risposta finale di richiesta di aiuto scompare. Alcuni supporti bassi corrispondono a queste modifiche, altri alla fusione di frasi o alla coreferenza Jack/He. Il minimo non quantifica direttamente la gravità dell'errore.

## Similarità allineate e confronto equo

| Dominio | Coseno allineato minimo bidirezionale | BERTScore allineato minimo riscalato |
| --- | ---: | ---: |
| Essay | 0,507943 | 0,463312 |
| Reuters | 0,485927 | 0,603377 |
| WritingPrompts | 0,571900 | 0,523539 |

Anche le similarità, quando applicate alle stesse unità con aggregazione sensibile al caso peggiore, mostrano variazioni sostanziali. Non è corretto descriverle come universalmente cieche alle modifiche. Reuters mantiene una similarità documentale elevata pur perdendo un'informazione specifica; lo score locale offre un segnale aggiuntivo. I numeri non sono direttamente confrontabili fra scale diverse e tre esempi senza etichette indipendenti non permettono AUC o graduatorie di accuratezza.

Non viene applicata la soglia semantica storica 0,04075, né una nuova soglia scelta su questi output. Le guardie restano descrittive: in WritingPrompts diminuiscono di due le occorrenze del lessico di modalità, ma i `could` del racconto esprimono anche abilità, non soltanto cautele epistemiche. In Essay e Reuters le guardie considerate non cambiano, pur in presenza di omissioni: non sono un controllo completo delle informazioni.

## Stile e lunghezza

| Dominio | Parole prima → dopo | Variazione | RMS stile DIPPER | RMS stile baseline 3B | Δ MATTR50 DIPPER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Essay | 160 → 122 | −23,8% | 0,505852 | 0,736460 | −0,042604 |
| Reuters | 149 → 122 | −18,1% | 0,479741 | 1,015054 | −0,026723 |
| WritingPrompts | 153 → 112 | −26,8% | 0,727539 | 0,889197 | +0,019051 |

DIPPER produce in tutti e tre i casi meno parole di baseline 3B e una distanza RMS dall'originale più piccola, sulla medesima scala stimata dagli umani di calibrazione. Questo descrive una variazione stilometrica, non una maggiore qualità. Le differenze di lunghezza e copertura restano parte dell'interpretazione, soprattutto in presenza di compressione.

MATTR50 cala in Essay e Reuters e aumenta nel racconto; baseline 3B lo aumentava in tutti e tre i casi. La distanza dal centroide umano diminuisce soltanto in Reuters e aumenta negli altri due. Né RMS, né centroide, né diversità lessicale equivalgono a naturalezza. I moduli di revisione ricevuti sono ancora vuoti.

Le tre generazioni DIPPER sommano **34,90 secondi** registrati; i sei stadi semantici/allineati sommano **39,50 secondi**. Sono tempi parziali, esclusi download e caricamento dei modelli e gli altri blocchi del notebook; non stimano il tempo dell'intera sessione.

## Prossimo run su Kaggle

Nel notebook aggiornato già usato, impostare nella prima cella:

```python
MODE = "pilot"
RUN_SUFFIX = "_dipper"
ACTIVE_SYSTEMS = ["identity", "dipper"]
```

Rieseguire le sezioni 1–9 dall'inizio. La sezione 3 deve riportare **300 umani di calibrazione e 42 fonti test**, due per dominio/generatore; la sezione 4 deve completare 84 esiti fra identità e DIPPER. Il file finale sarà `humanization_pilot_dipper.zip`. Il clone e il RUN vengono derivati da modalità e suffisso; la cache temporanea è comune e riusa i modelli se la sessione Kaggle è ancora attiva.

Mantenere invariati parametri di DIPPER, detector, soglie e valutatori: un output poco convincente è un risultato da analizzare, non una ragione per modificare il preset dopo aver visto lo smoke. Il pilot include i tre input appena esaminati e non è quindi una nuova prova indipendente da essi. Il confronto con il pilot baseline 3B sarà appaiato sulle stesse 42 fonti. Le due fonti StableLM già identificate come consegne rimangono nel campione primario; il loro tipo di input va annotato e discusso.
