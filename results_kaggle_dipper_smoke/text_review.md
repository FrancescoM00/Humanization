# Lettura qualitativa dello smoke DIPPER

Questa è una lettura dell'assistente delle tre coppie originali–DIPPER e delle riscritture Qwen sugli stessi input. Non è un'annotazione umana, un riferimento gold o una valutazione cieca: le osservazioni sono state confrontate anche con gli score della suite. Non permette di stimare accuratezza, tasso di errori semantici o naturalezza nella popolazione dei testi. Le evidenze NLI sotto sono ricostruite dalla cache presente nell'archivio, senza eseguire nuovamente il modello.

I tre output DIPPER hanno `status=ok`, `finished_eos=true` e fra 143 e 158 token di output, sotto il limite configurato di 512. Le omissioni osservate non derivano dal raggiungimento del limite di generazione. Il preset valutato usa una sola finestra per l'intero brano, senza prefisso contestuale, con diversità lessicale 60 e ordine 0; è questa variante che si può confrontare con Qwen.

## Essay: omissioni e una frase che fonde evidenze

- Fonte: `Essay:198:ee27e99f1bc3dc32`.
- Coppia DIPPER: `humanizer_2e39571e7c6c84c0fb41ce3c`.
- Coppia Qwen: `humanizer_4feaa95c9c411db4a15d0b66`.

DIPPER conserva il contrasto generale fra medicina e assistenza infermieristica, il riferimento alle domande PICOT e i tipi di supporto emotivo, sociale, fisico e mentale. La frase «In contrast, nursing practice is the practice of caring for and helping to improve the health and well-being of individuals and populations» sparisce come affermazione distinta; la formulazione finale sui pazienti conserva parte del concetto, ma perde il riferimento alle popolazioni.

La lista «clinical interviews, medical history, physical examination, and laboratory tests» diventa «clinical history, physical examination, laboratory tests and diagnostic images». Il riferimento esplicito ai colloqui viene perso. Le immagini diagnostiche hanno invece un riscontro nell'originale, che nella frase successiva parla di «specialized medical equipment and imaging techniques». Non sono un'aggiunta arbitraria; le due frasi originali sono state compresse in una sola. Il riferimento generale alle apparecchiature specializzate non è riprodotto.

Questa fusione mette in evidenza un limite dell'aggregatore NLI: il supporto forward minimo è **0,00420** proprio sulla frase che elenca storia clinica, esami e immagini. La valutazione cerca una singola frase originale che implichi tutta l'ipotesi; nessuna delle due frasi separate contiene l'intera lista. Il massimo supporto registrato deriva perfino dalla frase introduttiva sulle domande PICOT, segno che tutti i candidati singoli ricevono un supporto molto basso. Il minimo non dimostra quindi che l'intera frase riscritta sia inventata. Il reverse minimo, **0,00562**, riguarda invece la frase originale sulle apparecchiature e sulle tecniche di imaging, che la nuova lista non riproduce integralmente.

Qwen, sullo stesso testo, espandeva il contenuto con la definizione «PICOT stands for Population, Intervention, Comparison, Outcome, and Timeframe» e con una spiegazione aggiuntiva della cura olistica. DIPPER evita quelle espansioni, ma comprime ed elimina altri dettagli. I due sistemi mostrano tipi di deriva differenti; lo smoke non li ordina per fedeltà complessiva.

## Reuters: omissione esplicita dei problemi legali

- Fonte: `Reuters:417:b07c03bd035b9249`.
- Coppia DIPPER: `humanizer_98c982a986eae239befcadb4`.
- Coppia Qwen: `humanizer_934be96f948fc08f2ea4e843`.

DIPPER conserva Akeco, il 1973, il mercato giapponese, il fallimento involontario, la pandemia e la difficoltà a soddisfare la domanda immobiliare. Tuttavia elimina completamente la frase finale: «In addition to the economic downturn, Akeco has also been facing a number of legal challenges.»

La direzione NLI corrisponde bene a questa omissione: il forward medio è **0,98437**, mentre il reverse medio è **0,67870**. Il reverse minimo, **0,000334**, si riferisce esattamente alla frase sui problemi legali: la riscrittura non contiene più tale informazione. Il basso reverse ha qui un riscontro testuale specifico, indipendente dal verdetto del modello.

Si osservano anche differenze più piccole. «return to work from the virtual world» diventa «return to the real world from the virtual world»: il riferimento al lavoro viene generalizzato. «a significant decline in its revenue» diventa «its income fell», che mantiene la diminuzione ma perde l'intensità espressa da «significant». Il supporto reverse sulla frase della domanda e dei ricavi è **0,19839**; non è possibile attribuire l'intero calo a uno solo di questi cambiamenti.

Qwen conservava la frase finale sui problemi legali: «Moreover, Akeco has also encountered several legal hurdles.» Introduceva però una lettura più specifica del ritorno al lavoro, «returning to their offices after working from home», e trasformava la domanda immobiliare in domanda di abitazioni. Il caso illustra perché la conservazione va controllata informazione per informazione.

## WritingPrompts: cambia il meccanismo del potere

- Fonte: `WritingPrompts:562:785eead74dc70cd2`.
- Coppia DIPPER: `humanizer_473b5a6d9575d26f3b8ffebd`.
- Coppia Qwen: `humanizer_bb791f89bb7b7e6a3547ddfb`.

L'originale dice che Jack ha un superpotere che gli permette di «\"smell\" what a person was like» e di percepirne emozioni e tratti, distinguendo se una persona è buona o cattiva. DIPPER inizia con «a special gift of smelling people», ma subito dopo spiega: «He could tell by their looks and their expressions whether they were good or bad.» Il giudizio morale viene mantenuto, mentre l'inferenza tramite aspetto ed espressioni è nuova e cambia il meccanismo dichiarato del potere.

Sarah, nell'originale, è «down and upset»; nella riscrittura è «sad and lonely». La solitudine non è indicata nell'input. Sparisce inoltre la sua risposta conclusiva: «Sarah told Jack that she was going through a tough time and that she needed help.» La riscrittura termina quando Jack le chiede se può aiutarla.

Il forward medio è **0,50071** e il reverse medio **0,46096**. Il supporto forward alla frase sul giudizio «by their looks and their expressions» è **0,00451**; il reverse minimo **0,00156** corrisponde alla risposta omessa di Sarah. Sono due allarmi con un riscontro testuale chiaro.

Altri valori bassi sono meno interpretabili come errori reali. La frase «He was a curious and adventurous lad, and he spent his days in exploring new places and meeting new people» fonde informazioni presenti in due frasi originali, passando da «Jack» a «He»: il supporto forward è solo **0,02883**. Il secondo supporto reverse più basso, **0,00310**, confronta «Jack felt a sense of warmth and compassion for Sarah, and he decided to ask her if he could help her» con «He felt a warm and sympathetic feeling for her, and he asked her if he could help her». Qui intervengono coreferenza e il passaggio dalla decisione alla domanda compiuta; un valore tanto basso non è una misura diretta della gravità della modifica.

Qwen conservava l'idea di percepire emozioni e tratti attraverso l'olfatto, ma eliminava il giudizio esplicito buono/cattivo. Aggiungeva inoltre «One sunny day» e una relazione causale fra il dono di Jack e la sua curiosità. DIPPER conserva il giudizio morale, ma altera il meccanismo del potere ed elimina la risposta finale di Sarah.

## Cosa permette di concludere questa lettura

Lo smoke documenta output completi del modello e modifiche semantiche effettive, soprattutto omissioni. Il caso Reuters dà un esempio leggibile di asimmetria direzionale. Essay e alcune frasi di WritingPrompts mostrano che un supporto minimo molto basso può dipendere anche dalla fusione di frasi e dalla coreferenza: non va trasformato automaticamente in una classificazione di testo semanticamente scorretto.

La naturalezza rimane da annotare con i moduli previsti. I giudizi qualitativi dell'assistente e i punteggi NLI non sostituiscono quella valutazione. Con tre input, tutti dello stesso generatore ChatGLM, il passo successivo utile è il pilot congelato sui 42 testi, mantenendo invariati configurazione e protocollo.
