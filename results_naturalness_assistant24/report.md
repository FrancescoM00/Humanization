# Valutazione esplorativa dell'assistente sui 24 testi di naturalezza

Sono stati letti e giudicati i 24 testi di `exports/kaggle_dipper_pilot/naturalness_paired_24.csv`, seguendo la rubrica in `ISTRUZIONI.md`. I risultati sono in [ratings.csv](ratings.csv): un voto intero da 1 a 5, una breve motivazione e `label_source=assistant` per ciascun ID. Il modulo originale resta vuoto e invariato.

## Natura dei giudizi e procedura

L'annotatore è **un singolo assistente AI**, non un lettore umano. Queste etichette sono un riferimento esplorativo dell'assistente e **non costituiscono gold umana**. L'attività ha usato soltanto il modulo senza ruoli e le istruzioni; non sono state aperte chiavi, tabelle degli score o review dei casi per assegnare i voti. L'assistente aveva tuttavia già ricevuto contesto generale sul progetto: la valutazione non è completamente cieca rispetto allo studio e le versioni testuali simili visibili nello stesso modulo possono influenzare il giudizio.

Ogni testo è stato valutato come passaggio autonomo nel proprio genere. La scala riguarda formulazioni, registro, ritmo e coerenza linguistica: 1 molto artificiale o innaturale, 3 intermedio, 5 molto naturale. Non si è cercato di predire l'origine del testo, verificare i fatti, ricostruire la fedeltà a un originale o assegnare voti dai detector, da Flesch o dall'NLI. La forma di scaletta o consegna non è penalizzata in quanto tale; sono valutate chiarezza e resa linguistica all'interno di quella forma.

## Distribuzione descrittiva

| Voto | Testi |
| --- | ---: |
| 1 | 0 |
| 2 | 3 |
| 3 | 9 |
| 4 | 10 |
| 5 | 2 |

I voti più bassi sono motivati soprattutto da costruzioni grammaticali incomplete, collocazioni poco idiomatiche o accumulo di ripetizioni. Quelli più alti descrivono un registro coerente e un andamento fluido. Si tratta di criteri di lettura applicati a questi passaggi, non di proprietà validate come indicatrici dell'origine umana. La scala è ordinale e il piccolo pannello non giustifica una graduatoria generale dei sistemi.

## Uso con un futuro giudice Groq

Un confronto con i risultati di un altro LLM misurerà **concordanza AI–AI**, non validazione umana della naturalezza. Per un giudizio a coppie, eventuali preferenze ricavate da questi voti devono conservare i pareggi e usare solo testi riferiti alla stessa fonte; la chiave può essere applicata dopo l'annotazione. Il confronto delle due presentazioni a ordine scambiato può misurare sensibilità alla posizione, ma non rende indipendenti i due verdetti.

Non è stato calcolato kappa fra annotatori: qui c'è un solo insieme di giudizi e non sono disponibili in questa attività etichette umane o risultati Groq appaiati. Per un controllo umano, raccogliere valutazioni separate da uno o più lettori, senza mostrare questi voti o quelli dei modelli prima dell'annotazione.

## Tracciabilità

Modulo letto: `exports/kaggle_dipper_pilot/naturalness_paired_24.csv`, SHA-256 `a2cb0e4917c2a22add238b74cc5da38a78f7df0623770225152a2934dc8bfaa7`. Nessun testo è riprodotto o modificato nel CSV dei giudizi; gli ID permettono il ricongiungimento. Nessuna chiamata API o nuova generazione è stata eseguita.
