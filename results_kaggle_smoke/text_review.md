# Revisione qualitativa delle tre riscritture Qwen dello smoke

Review dell'assistente, non annotazione umana né gold semantica. Ho letto prima le tre coppie senza consultare gli score, poi ho associato metriche e cache NLI. Nessuna nuova inferenza è stata eseguita. Tutti e tre gli originali provengono da **ChatGLM**, uno per dominio: lo smoke non valuta il trasferimento fra generatori.

## Essay — aggiunte riconoscibili, conservazione solo parziale

ID `humanizer_4feaa95c9c411db4a15d0b66`.

L'input annuncia il confronto tra pratica medica e infermieristica tramite PICOT; descrive le due professioni ma non spiega l'acronimo. La riscrittura aggiunge un intero paragrafo: «PICOT stands for Population, Intervention, Comparison, Outcome, and Timeframe» e afferma che entrambe le professioni lo usano per guidare ricerca e decisioni. È informazione aggiunta rispetto all'input, anche se potesse essere vera sul piano del dominio. Aggiunge inoltre il passaggio sul trattamento dei sintomi e la cura olistica; riduce il riferimento originale a «individuals and populations» ai pazienti.

Il numero di parole lessicali passa 160→192 (+20%). Coseno documentale 0,915717 e BERTScore documentale approssimato F1 0,938150. NLI forward medio 0,773793, minimo 0,022663; reverse medio 0,828919, minimo 0,008724.

La cache conferma che il minimo forward è proprio la spiegazione PICOT nuova. Migliore premessa: l'annuncio originale del confronto con PICOT; ipotesi: la definizione dell'acronimo. L'allarme è qui informativo. Il secondo peggior supporto forward (0,031678) riguarda la parafrasi di strumenti e imaging «assist in their work»→«aid in their diagnostic processes»: non è la stessa aggiunta e va interpretato con cautela. Il minimo reverse riguarda l'introduzione programmatica dell'essay contro l'asserzione sull'uso di PICOT; **non** è la perdita del riferimento alle popolazioni, cui il modello assegna invece 0,850913.

Stile: RMS 0,736460; MATTR50 +0,166640. Distanza dal centroide umano 0,997983→1,039183 (leggermente maggiore). Qwen evade solo OpenAI. Il cambiamento stilistico o l'evasione non certificano conservazione.

## Reuters — quadro informativo largamente conservato, minimi soprattutto contestuali

ID `humanizer_934be96f948fc08f2ea4e843`.

Nome Akeco, anno 1973, COVID-19, difficoltà economiche e legali, fallimento e calo dei ricavi restano presenti. Ci sono interpretazioni più specifiche: «real estate» diventa «housing» e il rientro dal «virtual world» diventa ritorno negli uffici dopo il lavoro da casa. Sono restringimenti/esplicitazioni da annotare; la coppia non presenta una distruzione generale delle informazioni.

Parole 149→124 (−16,8%). Coseno documento 0,946510, BERTScore approssimato F1 0,949220. NLI forward medio 0,830675, minimo 0,007338; reverse medio 0,775987, minimo 0,000674. Coseno allineato minimo 0,792342; BERTScore allineato riscalato minimo 0,661663.

**Ricostruzione esatta dei minimi usando `_nli_jobs`, tokenizer locale del medesimo checkpoint e `nli_probability_cache.jsonl`, senza inferenza:**

- Forward 0,007338: premessa originale prima frase «The major real estate firm, which had been a leader in the Japanese market, is facing involuntary bankruptcy…» contro ipotesi riscritta prima frase che nomina «Akeco». Il nome dell'azienda è nella **seconda** frase originale. Il contenuto necessario è distribuito tra due premesse; il massimo di NLI a singola premessa non lo combina.
- Reverse 0,000674: premessa «Moreover, Akeco has also encountered several legal hurdles.» contro ipotesi «In addition to the economic downturn, Akeco has also been facing a number of legal challenges.» Il calo economico è già presente in un'altra frase riscritta. Un valutatore documentale dispone quindi del contesto mancante nel confronto locale.
- Il secondo peggiore reverse 0,693917 riguarda fondazione nel 1973 e leadership per molti anni. La leadership prolungata è rimasta nella prima frase riscritta, mentre la fondazione è nella seconda: ancora una redistribuzione delle informazioni.

Questi minimi non dimostrano una perdita grave della coppia Reuters. Sono un caso concreto del limite già noto di evidenza a frase singola e coreferenza, più che una prova che il classificatore abbia trovato una corruzione reale.

Guardie: hedge +1 e negazione +1 derivano principalmente da «was unable»→«couldn't». È una variante di incapacità, non un'inversione della polarità. I conteggi lessicali sono descrittivi e non possono essere interpretati da soli come allarmi semantici corretti.

Stile: RMS 1,015054; MATTR +0,108933; distanza dal centroide umano 1,359723→0,858619. Qwen evade OpenAI ma non gli altri detector.

## WritingPrompts — omissioni e dettagli nuovi, con altri allarmi contestuali

ID `humanizer_bb791f89bb7b7e6a3547ddfb`.

La riscrittura elimina la capacità esplicita di Jack di decidere accuratamente se le persone siano buone o cattive. Aggiunge «One sunny day» e introduce la causalità «This gift made him both curious and adventurous»; l'originale descrive dono e carattere senza questa relazione causale. L'attenzione a chi ha bisogno di aiuto diventa anche una motivazione generale prima dell'incontro con Sarah. Questi sono cambiamenti effettivi rispetto al passaggio iniziale, indipendentemente dagli score.

Parole 153→133 (−13,1%). Coseno documento 0,938949, BERTScore approssimato F1 0,931329. NLI forward medio 0,475452, reverse 0,381940; minimi 0,000824 e 0,003241.

Il minimo forward riguarda la nuova causalità del dono e la ricerca di persone da aiutare: qui il basso supporto è coerente con una reale aggiunta. Il dettaglio «sunny day» è fra gli altri casi a basso forward (0,032205). L'omissione della capacità buono/cattivo compare nel secondo peggiore supporto reverse, 0,003424.

Tuttavia il minimo reverse assoluto (0,003241) non riguarda quell'omissione: confronta «Sarah explained that she was indeed facing a challenging period and that she required support» con «Sarah told Jack that she was going through a tough time and that she needed help». Il destinatario Jack è nel contesto della frase precedente della riscrittura, non nella frase confrontata. Anche questo è un limite di evidenza/coreferenza; non tutte le unità con NLI basso sono errori effettivi.

La frase riscritta iniziale fonde inoltre contenuti distribuiti nelle prime due frasi originali; il confronto migliore a premessa singola ha forward 0,109382 pur preservando in sostanza questi elementi. La fusione può quindi penalizzare la metrica senza distruggere il significato.

Stile: RMS 0,889197, MATTR +0,091987, centroide umano 1,450704→1,280073. Nessuna evasione di alcun detector. Le guardie sulle cautele sono ambigue perché diversi «could» esprimono abilità narrativa, non incertezza epistemica.

## Implicazioni prima del pilot 42

Non vedo un nuovo blocco software nella lettura dei tre casi: le metriche sono state prodotte e i testi sono output effettivi. È ragionevole procedere sul protocollo congelato, **senza interpretare il minimo NLI come verdetto automatico** e compilando le schede di revisione sui nuovi humanizer.

Il primo caso mostra aggiunte visibili e un allarme forward utile; Reuters mostra quanto il minimo possa scendere quando fatti conservati vengono redistribuiti fra frasi; WritingPrompts contiene sia cambiamenti veri sia falsi/ambigui allarmi contestuali. Non sono tre prove indipendenti che l'NLI abbia ragione ogni volta. Serve distinguere errore del riscrittore da limite della metrica.

Non correggere o troncare le riscritture per riportarle alla lunghezza originale: si nasconderebbero proprio le aggiunte osservate. Non cambierei il prompt dopo questi tre casi se il prossimo passo è misurare la baseline già definita. Se si decide di aggiungere evidenza da finestre adiacenti o una variante con più premesse, conservarla come metrica/protocollo separato, senza sostituire retroattivamente i punteggi dello smoke e senza presupporre che elimini tutti i falsi allarmi.

Non attribuire score di naturalezza a questa review: il testo può essere fluido pur alterando il contenuto, e nessun giudice umano indipendente è stato coinvolto qui.
