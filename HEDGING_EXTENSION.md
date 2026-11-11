# Estensione mirata delle alterazioni di cautela

Specifica fissata prima delle nuove inferenze, su richiesta dell'autore. Il [protocollo eseguibile](results_hedging_extension/protocol.json) mantiene il criterio storico: **supporto minimo bidirezionale < 0,040749866515398**, con DeBERTa-v3 base float32 e lo stesso recupero MiniLM top-3, segmentazione e budget. Non si ricalibrano soglia o aggregatore. La configurazione corrente `window2 + min` della suite rimane distinta e non viene nuovamente valutata qui.

## Domanda e campione

Si vuole verificare se il criterio storico trasferisca a nuove alterazioni della stessa categoria, distinguendo rimozione completa della possibilità e rafforzamento parziale. Sono pianificati 60 originali macchina diversi, preferibilmente 20 per dominio MGTBench, con una rimozione completa, un rafforzamento parziale e una parafrasi fedele per originale: 120 alterazioni e 60 controlli conservativi. Le identità non sostituiscono i positivi. Nessun testo viene selezionato in base ai punteggi NLI.

Si usano cautele già presenti nei testi nativi, senza aggiungerle artificialmente. Sono esclusi gruppi delle precedenti analisi semantiche e dei campioni stilometrici indicati nel protocollo, collegando righe con prompt o testo umano condiviso e rimuovendo duplicati testuali. Le riscritture restano collegate allo stesso originale; non si assume che umano e macchina della riga MGTBench abbiano contenuto equivalente.

## Trasformazioni e revisione

Le regole ammettono frasi affermative con modalità epistemica sufficientemente isolata. Escludono negazioni, citazioni, condizionali e letture di permesso o capacità non distinguibili con sufficiente chiarezza. La parziale deve rafforzare la proposizione mantenendo una qualificazione; togliere soltanto possibly da could possibly non viene etichettato automaticamente come errore di significato.

Ogni caso conserva originale, riscrittura, frase modificata, offset, regola e interpretazione intenzionale. È prevista lettura dell'assistente senza punteggi su tutte le triplette selezionate. Si registrano validità, ambiguità e note. Questa revisione è esplicitamente **dell'assistente**, non un'annotazione umana indipendente. Le schede umane vengono predisposte senza risultati. Le eventuali sostituzioni di casi con errori grammaticali o semantici evidenti avvengono prima delle inferenze, da una riserva deterministica dello stesso dominio, conservando la traccia. Le analisi sul pannello completo con etichette intenzionali e sul sottoinsieme revisionato valido sono separate.

## Punteggi e valutazione

Si esegue soltanto quanto necessario all'NLI: embedding per il recupero delle premesse, inferenza base già disponibile e aggregazione storica. I modelli locali e la cache probabilistica esistente vengono riutilizzati senza modificare gli esperimenti precedenti. L'export ONNX float32 è quello già verificato; il controllo numerico nativo previsto dallo script non usa i risultati per cambiare trasformazioni o soglia.

L'endpoint documentale è il minimo fra tutte le unità e le due direzioni, dopo il massimo entailment sulle evidenze recuperate per ciascuna unità. Il rischio per l'AUC è **meno il supporto**, non uno meno il supporto. La soglia storica è sul supporto: allarme soltanto sotto il valore esatto registrato. Non è una soglia di contraddizione. I punteggi locali e direzionali sono diagnostici separati: non ereditano la soglia documentale.

Si riportano AUC, sensibilità, falsi positivi, specificità, conteggi e intervalli al 95%, complessivi e separati per rimozione completa e parziale. I positivi sono le parafrasi fedeli collegate alle stesse fonti. Il bootstrap ha 2.000 estrazioni per gruppo originale, mantenendo insieme le versioni e dando uguale peso alle fonti entro classe. I conteggi delle varianti non sono fonti indipendenti. Gli intervalli sono puntuali ed esplorativi; non si sceglie una soglia migliore sul nuovo test.

Un rafforzamento può essere non supportato senza contraddire l'originale: si osservano anche forward, reverse e la frase modificata. L'analisi rimane un controllo mirato di cautele sintetiche e trasferimento del criterio storico, non una nuova validazione degli humanizer o una misura gold della fedeltà umana.
