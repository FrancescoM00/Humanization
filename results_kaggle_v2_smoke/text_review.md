# Lettura dei dodici output dello smoke v2

Questa lettura è dell'assistente e usa testi e score ricevuti insieme. Non è una valutazione umana cieca, una gold semantica o una misura dell'accuratezza delle metriche. Sono state lette tutte le **12 riscritture** dei quattro preset sulle tre fonti ChatGLM; le identità sono controlli tecnici. Non si stimano percentuali di errore degli humanizer da questi tre input.

Le fonti sono `Essay:198:ee27e99f1bc3dc32`, `Reuters:417:b07c03bd035b9249` e `WritingPrompts:562:785eead74dc70cd2`, già presenti nei pilot. I nomi brevi sotto identificano la baseline Qwen2.5-3B con prompt fissato, il Qwen2.5-14B NF4 con lo stesso prompt, DIPPER XXL NF4 sul passaggio intero e DIPPER XXL NF4 a finestre di tre frasi con prefisso cumulativo. Non rappresentano ogni configurazione delle rispettive famiglie.

## Tutti gli output, con i cambiamenti osservabili

| Fonte | Preset | Parole prima → dopo | Osservazione testuale |
| --- | --- | ---: | --- |
| Essay | Baseline 3B | 160 → 192 | Aggiunge definizione PICOT, suo impiego da entrambe le professioni e spiegazione della cura olistica; sono dettagli non espliciti nel passaggio |
| Essay | 14B NF4 | 160 → 135 | Conserva esami, immagini, supporto infermieristico e riferimento a individui/comunità; evita il nuovo paragrafo definitorio PICOT della baseline, con alcune generalizzazioni e qualificazioni |
| Essay | DIPPER passaggio | 160 → 122 | Perde il riferimento esplicito a popolazioni e colloqui clinici; fonde esami e immagini che l'originale distribuiva su due frasi |
| Essay | DIPPER tre frasi | 160 → 142 | Ripristina individui e popolazioni rispetto al preset sul passaggio; mantiene la fusione della lista diagnostica e l'omissione esplicita dei colloqui |
| Reuters | Baseline 3B | 149 → 124 | Conserva i problemi legali; restringe real estate a housing e specifica il ritorno negli uffici dopo lavoro da casa |
| Reuters | 14B NF4 | 149 → 94 | Conserva nome, anno e problemi legali, ma aggiunge un rallentamento iniziale delle transazioni dovuto al lavoro remoto e cambia un aumento osservato della domanda in un'aspettativa |
| Reuters | DIPPER passaggio | 149 → 122 | Elimina i problemi legali e generalizza return to work in return to the real world |
| Reuters | DIPPER tre frasi | 149 → 143 | Conserva i problemi legali e il ritorno al lavoro; cambia revenue in profits e riformula la causa iniziale come financial crisis |
| WritingPrompts | Baseline 3B | 153 → 133 | Elimina il criterio morale esplicito buono/cattivo; aggiunge giornata di sole e una relazione causale fra dono e curiosità |
| WritingPrompts | 14B NF4 | 153 → 120 | Conserva olfatto, distinzione kind/cruel e risposta finale di Sarah; aggiunge giornata di sole e intensificatori emotivi |
| WritingPrompts | DIPPER passaggio | 153 → 112 | Introduce il giudizio attraverso aspetto/espressioni e la solitudine di Sarah, eliminandone la risposta finale |
| WritingPrompts | DIPPER tre frasi | 153 → 127 | Conserva meccanismo olfattivo, distinzione buono/cattivo e risposta finale; varia alcune formulazioni emotive, per esempio warm glow al posto della compassione esplicita per Sarah |

I conteggi sono quelli alfabetici spaCy registrati nel run. Nessuno dei dodici testi è un output vuoto o una risposta con spiegazioni esterne alla riscrittura. Gli stati tecnici ricevuti sono `ok`; questo non equivale a conservazione del significato.

## 14B Reuters: più grande non significa automaticamente più fedele

Coppia `humanizer_d2cf262a9d82e46683eb7059`. L'originale presenta Akeco come «facing involuntary bankruptcy»; il 14B scrive «financial troubles that could lead to involuntary bankruptcy». Si attenua la formulazione della situazione corrente: non è una contraddizione logica certa, ma la forza dell'affermazione cambia.

Il cambiamento più netto è «While remote work initially slowed down real estate transactions»: il rallentamento iniziale delle transazioni non è esplicitato nell'input. Inoltre «the demand for real estate began to rise» diventa «the return to physical offices was expected to boost demand». L'aumento della domanda è trasformato in aspettativa, mentre il mancato soddisfacimento della domanda diventa mancata capitalizzazione di una tendenza. L'output conserva Akeco, 1973 e le difficoltà legali, ma non è una semplice riduzione senza cambiamenti informativi.

L'NLI storico assegna medie forward/reverse 0,581/0,281; con finestre fino a tre, 0,990/0,907. L'ampliamento della premessa risolve parte dei problemi di aggregazione, ma il forward quasi unitario coesiste con differenze verificabili. Non va interpretato come certificazione del contenuto. Groq preferisce questo output alla baseline 3B in entrambi gli ordini: è un giudizio di naturalezza, non di fedeltà, e i due esiti non sono in contraddizione.

## DIPPER contestuale: recupera omissioni ma conserva altri errori

In Reuters, coppia `humanizer_99604d05be42adde097732a2`, torna la frase «Akeco has also been facing a number of legal difficulties», eliminata dal preset sul passaggio intero. Nel racconto, coppia `humanizer_8e884fd6a8ecce1338e1500d`, tornano la risposta di Sarah e una descrizione coerente del potere olfattivo: «He could smell a person’s character».

Questi recuperi coincidono con una minore compressione: Reuters 143 parole invece di 122, racconto 127 invece di 112. Non si attribuisce tutto il risultato causalmente alla dimensione della finestra: cambiano anche contesto di generazione e budget complessivo per finestra, e sono soltanto tre esempi.

Il preset contestuale non è però equivalente in ogni dettaglio. In Reuters «a significant decline in its revenue» diventa «its profits fell sharply»: ricavi e profitti non sono la stessa quantità. La nuova apertura attribuisce inoltre il fallimento all'«impact of the financial crisis», riformulando il rapporto originale con il sistema finanziario nazionale. L'assenza dell'omissione legale non basta quindi a concludere conservazione perfetta.

Nel racconto, l'NLI storico ha reverse medio 0,307 pur in presenza della risposta finale e di molta trama conservata; con finestre fino a tre sale a 0,754 e con documento intero a 0,985. Il minimo reverse rimane 0,008 con finestre. Il caso conferma l'importanza dell'unità di premessa e della coreferenza, senza rendere automaticamente corretto il nuovo verdetto.

## Essay: aggiunta nuova e fusione sono fenomeni differenti

La baseline 3B ripropone «PICOT stands for Population, Intervention, Comparison, Outcome, and Timeframe» e il paragrafo sulla cura olistica già osservati nel primo smoke. Il 14B non introduce tale definizione e mantiene i riferimenti a «individuals and communities». Questo è un cambiamento del profilo degli errori, non una misura globale di superiorità.

Entrambi i preset DIPPER uniscono la lista degli esami e il riferimento alle immagini in una nuova frase. Le immagini sono sostenute dalla frase originale successiva: un basso supporto della singola premessa non dimostra che siano inventate. Nel preset a tre frasi rimangono invece assenti i colloqui clinici espliciti, mentre individui e popolazioni vengono ripristinati.

## Il giudice di naturalezza è sensibile all'ordine

I nove confronti del nuovo smoke hanno entrambe le risposte AB/BA valide, ma soltanto **cinque mantengono la stessa preferenza** dopo lo scambio. Quattro sono discordanti. Nei tre confronti WritingPrompts il giudice sceglie sempre l'etichetta visibile **B** in entrambi gli ordini: il vincitore cambia quando cambiano le posizioni. Lo stesso accade in Reuters per DIPPER passaggio contro DIPPER tre frasi.

Fra 3B e 14B, i due confronti stabili si dividono: Essay preferisce la baseline 3B, Reuters il 14B; WritingPrompts è discordante. Fra baseline 3B e DIPPER contestuale, i casi stabili Essay e Reuters preferiscono la baseline 3B; il racconto è discordante. Fra i due preset DIPPER, soltanto Essay è stabile e preferisce il contestuale.

Non si risolve il problema scegliendo l'ordine favorevole o descrivendo i cinque confronti residui come una graduatoria generale. Occorrono copertura, discordanza e annotazioni indipendenti. Le risposte del giudice e questa lettura dell'assistente non sono gold umana, e i controlli nulli di rilevazione non sono etichette di correttezza semantica.

## Implicazione per le 126 fonti

Non emerge dai dodici testi un blocco di formato che imponga nuovi tentativi o un aggiustamento del prompt: gli output sono testi leggibili del genere richiesto. Errori semantici, compressione e sensibilità all'ordine sono invece risultati da misurare, senza rimuovere i casi sfavorevoli o cambiare il preset dopo averli osservati.

L'estensione può verificare questi comportamenti sui preset dichiarati, conservando i controlli nulli e gli score NLI contestuali separati. Non è ancora giustificato un vincitore su fedeltà o naturalezza. Prima di usare la naturalezza come risultato conclusivo, la discordanza AB/BA deve restare esplicita e il riferimento indipendente deve essere raccolto.
