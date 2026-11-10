# Naturalezza nello study da 126 fonti: copertura parziale e sensibilità alla posizione

Analisi del 10 novembre 2026 sui soli file ricevuti, senza nuove chiamate API. **Nessuno dei tre confronti estesi è completo.** Sono salvate 364 risposte valide su 744 richieste pianificate dopo i fallimenti delle generazioni. Il limite più importante per l'interpretazione è la combinazione di copertura parziale e preferenze che cambiano scambiando l'ordine dei testi: non è disponibile una graduatoria affidabile di naturalezza sulle 126 fonti.

## Copertura effettiva

I confronti usano il modello richiesto `qwen/qwen3.8-27b`, `reasoning_effort=none`, temperatura zero, massimo 200 token di completamento e i due ordini AB/BA. Tutte le 364 righe salvate hanno stato `ok` e `response_model` uguale al modello richiesto: non ci sono risposte con stato `invalid_response`. Piani, fingerprint, metadati delle richieste, report di eleggibilità e riepiloghi sono stati ricalcolati o confrontati con gli input. Non sono stati cambiati prompt, testi o giudizi.

| Confronto | Fonti selezionate | Coppie giudicabili | Escluse per generazione | Risposte valide / attese | Coppie con entrambi gli ordini | Stabili / discordanti | Coppie giudicabili ancora senza entrambi gli ordini |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 3B / 14B NF4 | 126 | 124 | 2 | 136/248 | 68 | 32 / 36 | 56 |
| Baseline 3B / DIPPER tre frasi | 126 | 123 | 3 | 128/246 | 64 | 46 / 18 | 59 |
| DIPPER passaggio / tre frasi | 126 | 125 | 1 | 100/250 | 50 | 15 / 35 | 75 |

Le preferenze stabili coprono soltanto **32/126 = 25,4%**, **46/126 = 36,5%** e **15/126 = 11,9%** delle fonti selezionate nei tre confronti. Fra le sole coppie con entrambi gli ordini, le quote stabili sono rispettivamente 47,1%, 71,9% e 30,0%. I denominatori rispondono a domande differenti e vanno riportati insieme.

I file `pairwise_summary.json` conservano tutte le 126 fonti per confronto, comprese quelle non giudicate. In questo archivio ogni coppia che ha risposte ne ha due: non ci sono casi con un solo ordine salvato. Le risposte coprono un prefisso della graduatoria per hash degli ID, saltando i fallimenti già dichiarati. La selezione non usa score di qualità; l'interruzione rende però la copertura parziale e non bilanciata per dominio/generatore. Inoltre limitarsi ai soli casi stabili applica un'ulteriore selezione sugli esiti del giudice. Le preferenze residue non descrivono automaticamente tutta la coorte.

## Fallimenti delle generazioni, senza rimpiazzi

La correzione `--failure-policy report` è stata effettivamente applicata. Gli ID sono selezionati prima del controllo degli stati; i fallimenti non sono sostituiti, rigenerati o trasformati in pareggi. Sono visibili questi casi:

| Fonte | Generatore | Output non `ok` | Conseguenza per i confronti |
| --- | --- | --- | --- |
| `Essay:223:eceeae3f088c81ef` | StableLM | Baseline 3B: `truncated` | Non giudicabile contro 14B o DIPPER tre frasi |
| `Reuters:992:d3932f02b4a9e76d` | StableLM | Baseline 3B: `truncated` | Non giudicabile contro 14B o DIPPER tre frasi |
| `WritingPrompts:184:b88567fa0a190a58` | GPT4All | Entrambi i preset DIPPER: `truncated` | Non giudicabile nei due confronti che coinvolgono DIPPER tre frasi |

Sono quattro output unici non validi su tre fonti, con sei coppie-confronto non giudicabili. I testi troncati non sono necessariamente vuoti: il loro stato è sufficiente a impedirne il voto. Il traceback precedente non permetteva di conoscere questi conteggi; ora sono documentati dai file. Le motivazioni operative della troncatura vanno lette nei metadati di riscrittura, senza attribuirle al giudice o a errori API.

La naturalezza è quindi **condizionata alla riuscita di entrambi gli output**, e rimane distinta dal tasso di fallimento del sistema e dalla politica ASR. Un sistema non può essere dichiarato migliore nascondendo le generazioni che non entrano nel giudizio.

## Perché le chiamate si sono fermate

| Confronto | Tentativi POST registrati | Risposte valide | Stato e motivo salvato | Token delle risposte salvate |
| --- | ---: | ---: | --- | ---: |
| Baseline 3B / 14B | 256 | 136 | `partial`, budget cumulativo esaurito | 77.797 |
| Baseline 3B / DIPPER tre frasi | 256 | 128 | `partial`, budget cumulativo esaurito | 73.283 |
| DIPPER passaggio / tre frasi | 201 | 100 | `partial`, HTTP 429 del provider | 52.365 |
| Totale confronti estesi | 713 | 364 | Tutti parziali | 203.445 |

Il budget contava correttamente tutti i tentativi, compresi i retry. Nei primi due confronti non basta quindi che `max_requests=256` superi 248 o 246 richieste nominali: le ripetizioni hanno consumato il margine. I log mostrano 348 richieste tentate due volte, 16 tentate una volta con risposta salvata e un tentativo finale senza risposta salvata. I file dei tentativi non conservano il codice HTTP di ogni singolo evento, quindi non si può attribuire ogni ripetizione al medesimo errore.

L'ultimo errore conservato per DIPPER riporta esplicitamente **limite di token al giorno**, con limite 200.000 e valore usato 199.936 al momento del rifiuto; il provider richiedeva di riprovare dopo circa tre minuti e sedici secondi. Questa è l'indicazione registrata allora, non una verifica della quota disponibile adesso. Non è un errore del modello di riscrittura né prova di quota GPU esaurita. I token `usage` delle risposte salvate e il contatore della quota del provider sono registrazioni differenti; il primo non ricostruisce da solo il consumo o il costo di tutti i tentativi.

I confronti estesi registrano tentativi dalle 00:12:54 alle 00:39:30 del 10 novembre, ora italiana: circa 26 minuti e 35 secondi, terminati con copertura parziale. Non sono il tempo necessario a completare il protocollo nominale.

## Sensibilità alla posizione

| Confronto | Coppie discordanti | Scelta B in entrambi gli ordini | Scelta A in entrambi gli ordini | Un ordine in parità |
| --- | ---: | ---: | ---: | ---: |
| Baseline 3B / 14B | 36 | 34 | 2 | 0 |
| Baseline 3B / DIPPER tre frasi | 18 | 18 | 0 | 0 |
| DIPPER passaggio / tre frasi | 35 | 33 | 1 | 1 |

Su 89 coppie discordanti, **85 scelgono B sia in AB sia in BA**: scambiando le posizioni cambia il testo vincente. Nelle 364 risposte, B è scelto 263 volte (72,3%), A 98 volte e la parità tre volte. Le preferenze per B non sono tutte errori — un testo può legittimamente vincere quando è B — ma il pattern B/B identifica direttamente l'instabilità rispetto alla presentazione.

Le risposte sono sintatticamente valide. In quattro casi il giudice esprime una preferenza stretta pur assegnando lo stesso voto intero 1–5 ai due testi; non ci sono preferenze opposte all'ordine stretto dei voti. Una scala ordinale grossolana può assegnare lo stesso livello a testi distinti: non vengono introdotte nuove regole post hoc per scartare quelle risposte o correggerne il vincitore.

## Cosa dicono i soli confronti stabili

| Confronto | Vittorie primo sistema | Vittorie secondo sistema | Parità stabili | Denominatore stabile |
| --- | ---: | ---: | ---: | ---: |
| Baseline 3B / 14B | 13 | 19 | 0 | 32 |
| Baseline 3B / DIPPER tre frasi | 41 | 4 | 1 | 46 |
| DIPPER passaggio / tre frasi | 9 | 6 | 0 | 15 |

Il giudice preferisce spesso la baseline 3B al DIPPER contestuale **nei casi stabili disponibili**, mentre il confronto con 14B è meno netto. Non si trasformano questi conteggi in vittorie sull'intero campione: le esclusioni per ordine sono molte, il giudizio API è incompleto e non ci sono annotazioni umane che validino tali preferenze. Naturalezza e conservazione rimangono assi diversi; un output scorrevole può avere omissioni o aggiunte.

### Le 84 fonti non viste nei pilot

| Confronto sulle sole 84 fonti | Giudicabili | Complete AB/BA | Stabili / discordanti | Ancora senza entrambi gli ordini | Vittorie primo / secondo / parità fra gli stabili |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 3B / 14B | 82 | 43 | 20 / 23 | 39 | 8 / 12 / 0 |
| Baseline 3B / DIPPER tre frasi | 81 | 40 | 28 / 12 | 41 | 24 / 3 / 1 |
| DIPPER passaggio / tre frasi | 83 | 34 | 9 / 25 | 49 | 5 / 4 / 0 |

Anche questo sottoinsieme ha copertura limitata: non risolve la selezione dovuta alla stabilità del giudice. I conteggi sono separati perché le ipotesi precedenti erano state sviluppate sui 42 pilot; non sono una nuova validazione umana o una replica completa.

## La prova iniziale sui vecchi pilot resta distinta

Lo ZIP contiene anche il preflight sui sei confronti 3B/DIPPER già pianificati: 12 risposte valide, 12 tentativi registrati, quattro preferenze stabili e due discordanti. Nei quattro stabili l'accordo con il proxy dell'assistente è 4/4; kappa indefinito per classe unica. **È concordanza AI–AI, non validazione umana.**

Il piano, il manifest e i voti sostanziali coincidono con lo smoke precedente; i tempi delle chiamate e alcuni tempi `usage` sono però diversi. Il primo tentativo qui è registrato alle 20:28:29 del 9 novembre, ora italiana, contro le 18:30:50 nello smoke originario. Non si deduplica una chiamata soltanto perché riusa lo stesso `request_id`, che identifica il payload: i log ricevuti documentano eventi differenti. Non permettono invece di stabilire se questa nuova esecuzione del preflight sia avvenuta direttamente nello study o prima, venendo poi copiata.

Il totale registrato nello ZIP è pertanto **725 tentativi e 376 risposte valide**, tenendo separati i 12/12 del preflight dai 713/364 dei confronti estesi. I token delle risposte salvate sommano 209.610 includendo il preflight. Non si convertono in costo monetario senza prezzi e contabilità verificati; le sei fonti del preflight non sono nuove fonti indipendenti.

## Dati esportati e conclusione utilizzabile

- [Riepiloghi sulle 126, 42 pilot e 84 nuove](naturalness_summary.csv): nove righe.
- [Audit di tutte le fonti per confronto](naturalness_sources.csv): 378 righe, con mancanti, ordini e preferenze canoniche.
- [Risposte dei confronti estesi](naturalness_orders.csv): 364 righe.
- [Coppie non giudicabili](naturalness_failures.csv): sei righe con stati dei sistemi.
- [Tentativi aggregati per richiesta](naturalness_request_attempts.csv): 365 richieste distinte tentate; gli esiti HTTP dei singoli tentativi non sono inventati.
- [Preflight separato](naturalness_preflight.csv): sei confronti.

La componente di naturalezza è eseguita e produce dati verificabili, ma **non sostiene ancora una graduatoria generale**. Il risultato metodologico da conservare è la bassa copertura delle preferenze stabili e la forte sensibilità alla posizione. Il prossimo controllo utile è il confronto con annotazioni indipendenti, mantenendo i casi instabili e i mancanti visibili. Non sono state avviate altre chiamate, aumentati budget o sostituiti modello e prompt per ottenere un risultato più favorevole.
