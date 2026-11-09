# Primo smoke Kaggle: riscritture baseline 3B e suite completa

In questo report **baseline 3B** indica Qwen2.5-3B-Instruct con il prompt fissato e decoding greedy; **DIPPER** indica la configurazione XXL NF4 sul passaggio intero senza prefisso, lessico 60/ordine 0. I risultati non descrivono ogni configurazione dei due modelli.

**Lo smoke è completato e non mostra blocchi tecnici per il pilot su 42 fonti.** Sono presenti tutti gli output di riscrittura, detection, stile, metriche semantiche ed esportazione continua. La prova rivela anche limiti di interpretazione: un minimo NLI molto basso può segnalare informazione nuova oppure contenuto conservato ma distribuito su più frasi.

L'analisi usa esclusivamente lo ZIP restituito dall'autore. Non sono state prodotte nuove riscritture o inferenze. [Archivio originale](../exports/kaggle_smoke/README.md), [sintesi numerica per coppia](per_pair_summary.csv), [revisione dei testi](text_review.md), [controlli di integrità](integrity.md), [provenienza](provenance.json).

## Dati effettivamente valutati

Tre passaggi macchina, uno per Essay, Reuters e WritingPrompts, tutti del generatore **ChatGLM**. Lo smoke non copre i sette generatori: il pilot successivo contiene invece due fonti per ciascuna delle 21 combinazioni dominio/generatore. I passaggi di input derivano dal precedente matching a fine frase di 90–180 parole; la valutazione riguarda questi passaggi, non i documenti completi.

Sono state prodotte tre identità e tre riscritture Qwen2.5-3B-Instruct, tutte con stato `ok`, una sola generazione greedy per fonte e prompt fissato. La directory completa è `rewriting_cuda0/`; `rewriting/` conserva il tentativo interrotto prima del caricamento baseline 3B e non è usata nei punteggi finali.

I 300 umani di calibrazione, 100 per dominio, sono separati dai gruppi delle fonti test e da tutte le 126 fonti riservate allo studio successivo. Le schede manuali sono ancora vuote: sei coppie per la semantica e nove testi unici per la naturalezza. Le osservazioni qualitative di questo report sono dell'assistente e non costituiscono annotazioni umane indipendenti.

## Evasione: due successi sul solo detector OpenAI

| Detector | Originali rilevati AI | Falsi positivi umani test | Successi baseline 3B | ASR baseline 3B |
| --- | ---: | ---: | ---: | ---: |
| OpenAI RoBERTa-base | 3/3 | 0/3 | 2/3 | 0,667 |
| Fakespot raw | 3/3 | 0/3 | 0/3 | 0 |
| RADAR | 3/3 | 0/3 | 0/3 | 0 |
| Voto di maggioranza | 3/3 | 0/3 | 0/3 | 0 |

L'identità ha ASR zero su tutti i detector. Le due evasioni OpenAI riguardano Essay e Reuters; WritingPrompts resta rilevato. Le probabilità AI di OpenAI passano rispettivamente da 0,999798 a 0,030136 e da 0,999824 a 0,194186; la soglia del detector è 0,970608. Fakespot e RADAR continuano a classificare AI tutte le riscritture. La macro-media ASR dei tre modelli è 0,222222; il voto di maggioranza è una sintesi diversa.

Le soglie individuali rispettano il FPR empirico di calibrazione del 5%, ossia 15/300; il voto di maggioranza ha 5/300. Lo zero sui tre umani test non dimostra un FPR nullo nella popolazione. Analogamente, intervalli bootstrap degeneri 0–0 su tre insuccessi non costituiscono una certezza sull'efficacia futura. Queste percentuali servono a verificare il funzionamento e a formulare osservazioni sui tre casi.

## Conservazione: leggere insieme aggregati e testi

Valori documentali registrati; BERTScore documento è l'approssimazione con matching di chunk già descritta nel protocollo, senza riscalamento.

| Dominio | Coseno doc | BERTScore doc F1 | NLI forward medio | NLI reverse medio | Minimo NLI nelle due direzioni |
| --- | ---: | ---: | ---: | ---: | ---: |
| Essay | 0,915717 | 0,938150 | 0,773793 | 0,828919 | 0,008724 |
| Reuters | 0,946510 | 0,949220 | 0,830675 | 0,775987 | 0,000674 |
| WritingPrompts | 0,938949 | 0,931329 | 0,475452 | 0,381940 | 0,000824 |

**Essay:** baseline 3B aggiunge un paragrafo che spiega PICOT e afferma che entrambe le professioni lo usano per guidare ricerca e decisioni; aggiunge inoltre il riferimento alla cura olistica. Sono contenuti espliciti non presenti nel passaggio fornito. Questo non significa che siano falsi nel mondo: la richiesta del riscrittore era conservare le informazioni senza aggiungerne. Il minimo forward di 0,022663 corrisponde proprio alla definizione PICOT nuova. Non tutti gli altri confronti a basso supporto rappresentano invece aggiunte sostanziali.

**WritingPrompts:** scompare la capacità esplicita di Jack di distinguere persone buone o cattive; compaiono una giornata di sole e una relazione causale tra il dono e il carattere curioso/avventuroso. Gli score bassi sono coerenti con alcuni cambiamenti reali. La cache mostra però anche allarmi contestuali: il destinatario Jack resta nel contesto del dialogo, mentre una frase valutata isolatamente non lo nomina.

**Reuters:** nome, anno, pandemia, difficoltà finanziarie e legali rimangono. Ci sono restringimenti da annotare, come `real estate` → `housing`. I minimi più estremi dipendono però dalla redistribuzione delle informazioni: la riscrittura nomina Akeco nella prima frase, mentre l'originale introduce il nome nella seconda; un'altra frase conserva il riferimento alla crisi economica altrove nel testo. Il modello confronta ogni ipotesi con singole premesse e ne prende il massimo, senza combinarle. Può quindi giudicare correttamente insufficiente una premessa isolata, mentre il minimo aggregato produce un allarme documentale eccessivo. Non è una dimostrazione di perdita grave di contenuto nel caso Reuters.

La [revisione dettagliata](text_review.md) ricostruisce queste unità dalla cache NLI senza nuove inferenze. **Il minimo rimane un segnale da ispezionare, non un verdetto automatico.** Media e distribuzione delle unità sono utili per distinguere il caso WritingPrompts, con numerose variazioni, da Reuters.

## Anche le similarità allineate rilevano cambiamenti

| Dominio | Coseno allineato minimo bidirezionale | BERTScore allineato minimo riscalato |
| --- | ---: | ---: |
| Essay | 0,475302 | 0,271141 |
| Reuters | 0,792342 | 0,661663 |
| WritingPrompts | 0,511305 | 0,387884 |

Le similarità a livello documentale sono elevate, ma quelle allineate hanno escursioni significative. Questi tre casi non dimostrano che coseno o BERTScore siano ciechi alle riscritture, né consentono di stimare una superiorità diagnostica dell'NLI. Per confrontare l'accuratezza occorrono annotazioni indipendenti sugli errori reali.

## Stile e lunghezza

| Dominio | Parole prima → dopo | Variazione | RMS stilometrico | Δ MATTR50 |
| --- | ---: | ---: | ---: | ---: |
| Essay | 160 → 192 | +20,0% | 0,736460 | +0,166640 |
| Reuters | 149 → 124 | −16,8% | 1,015054 | +0,108933 |
| WritingPrompts | 153 → 133 | −13,1% | 0,889197 | +0,091987 |

baseline 3B cambia effettivamente lo stile e aumenta MATTR50 in tutti e tre i passaggi. Non ne segue una maggiore naturalezza: la scala RMS descrive una distanza e non una qualità. Le 91 dimensioni della distanza sono standardizzate sui soli umani di calibrazione. Tutti i controlli identità hanno RMS, POS-JS e delta della shortlist esattamente zero.

Le guardie mostrano anche falsi allarmi lessicali: in Reuters `was unable` → `couldn't` fa aumentare i conteggi di negazione e modalità senza invertire la polarità. Nel racconto, molti `could` descrivono abilità invece di cautela epistemica. Questi conteggi non devono diventare etichette semantiche automatiche.

## Integrità e tempi osservati

I sei ID coincidono fra coppie e tabelle. I riepiloghi dei detector sono stati ricalcolati dai 936 score salvati e coincidono. I sorgenti BERTScore corretti sono nello ZIP e corrispondono ai metadati di esecuzione. La differenza fra versione iniziale e patch è conservata; i risultati antecedenti non sono stati riscritti.

NLI è eseguito su CUDA con il checkpoint base float32. Nell'identità le medie sono circa 0,988–0,995: non sono forzate a uno. Coseno e BERTScore identità sono circa uno entro l'aritmetica floating point.

I tempi registrati per le tre generazioni baseline 3B sommano **26,26 secondi**, escludendo download e caricamento. I sei stadi semantici/allineati sommano **36,17 secondi**, sempre esclusi download e altri blocchi del notebook. Non sono tempi dell'intero esperimento e non misurano la naturalezza.

## Passaggio al pilot

Si può procedere con i 42 input già congelati, mantenendo modello, prompt, decoding, detector, calibrazione e metriche. Il pilot include le tre fonti dello smoke: non è una nuova validazione indipendente da esse. Non si correggono o troncano gli output per nascondere aggiunte e non si adatta il prompt dopo aver visto questi tre casi senza definire una condizione separata.

Nella prima cella del notebook già aperto cambiare:

```python
MODE = "pilot"
REPO = Path("/kaggle/temp/Humanization_pilot")
```

La nuova cartella clona il codice corretto senza sovrascrivere le patch dello smoke; `TMP` rimane invariato e riusa modelli e venv della sessione. Eseguire poi le celle dall'inizio. Nella sezione 3 devono risultare **300 umani di calibrazione e 42 fonti test**. A fine run scaricare `humanization_pilot.zip` e compilare le schede manuali senza usare NLI come gold label.
