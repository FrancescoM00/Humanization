# Scelta della configurazione NLI sullo split di calibrazione

Specificazione del **10 novembre 2026**, successiva alla lettura dello study da 126 fonti e precedente alle nuove inferenze di questo confronto e alla ricezione delle annotazioni reference. Non è una preregistrazione retroattiva dello study. La selezione è uno sviluppo dello strumento su etichette proxy; la sua validazione umana resta successiva e distinta.

## Campione e dati esclusi

Si conservano lo split per fonte e il collegamento positivo/negativo già definiti nell'esperimento OpAI:

- **139 positivi presunti e 297 alterazioni intenzionali**, da 46 gruppi di fonte;
- i negativi sono soltanto `opai_matched_negative` con `evaluation_split="calibration"`; i positivi sono quelli richiamati da `metadata.matched_positive_id`;
- si aggiungono **137 identità**, una per originale distinto entro il gruppo di fonte dei positivi selezionati. Chunk e testo vengono copiati esattamente in entrambe le direzioni;
- totale **573 coppie**, senza aggiungere fonti indipendenti. Le identità non entrano nell'AUC.

Le 141 coppie positive e 265 alterate dello storico test, su 48 fonti, **non partecipano alla selezione**. La preparazione può leggere i metadati dello split per verificare la disgiunzione, non i suoi score. Sono esclusi anche i positivi non collegati ai negativi del confronto principale. Non si tratta di un nuovo campione gold: parafrasi OpAI e perturbazioni restano etichette operative presunte.

Gli input sono in [exports/nli_context_calibration/](exports/nli_context_calibration/README.md): coppie, allineamenti esatti, etichette, policy e manifest. La preparazione rifiuta di sostituire input diversi nella stessa directory.

## Modello e modalità

Si riusa il medesimo DeBERTa-v3 **base float32**, `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, revisione `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. Non si modifica il modello, non si usa quantizzazione e non si passa a xsmall. Il limite della coppia premessa/ipotesi è 384 token; nessun troncamento silenzioso.

Le quattro modalità sono quelle già implementate in [semantics/contextual_nli.py](semantics/contextual_nli.py):

| Modalità | Candidati a premessa |
| --- | --- |
| `single_all` | Tutte le unità singole |
| `window2` | Tutte le singole e le finestre contigue fino a due unità |
| `window3` | Tutte le singole e le finestre contigue fino a tre unità |
| `document` | Documento esatto, solo quando la coppia entra nel budget |

Le unità ipotesi sono identiche fra modalità. Si prende il massimo entailment fra i candidati ammissibili per ciascuna ipotesi, in entrambe le direzioni. Una modalità con ipotesi senza premessa ammissibile conserva la copertura incompleta e non produce un aggregato parziale spacciato per completo. Le finestre sono annidate: il supporto può aumentare per costruzione, senza dimostrare maggiore accuratezza.

## Sedici candidati e regola di scelta

Per ciascuna modalità si confrontano media, q10, minimo e quota delle ipotesi con supporto inferiore a 0,5: **16 candidati**. Lo score documentale di conservazione è il minimo delle due direzioni dell'aggregatore. Per la quota non supportata è `1 - max(quota_forward, quota_reverse)`, così valori alti indicano sempre conservazione presunta. Lo 0,5 definisce questa quota, non un verdetto di conservazione sui veri humanizer.

Un candidato è ammissibile come primario soltanto se:

1. ha copertura completa in entrambe le direzioni per tutte le 436 coppie etichettate e tutte le 137 identità;
2. la **media degli score documentali delle identità** è almeno 0,95;
3. il **decimo percentile della distribuzione degli score documentali delle identità** è almeno 0,90.

Questi ultimi due criteri sono soglie operative di coerenza, fissate prima delle nuove inferenze, non limiti universali di validità NLI. Il percentile fra documenti non va confuso con l'aggregatore q10 fra frasi. Le identità sono controlli necessari ma non dimostrano la correttezza sulle parafrasi, fusioni o omissioni.

Fra gli ammissibili si sceglie la massima **AUC aggregata sul solo split di calibrazione**, con positivi presunti = 1 e alterazioni = 0. Le parità entro `1e-12` preferiscono la premessa più corta, poi l'ordine media, q10, minimo, quota. Se nessuno passa, l'esito è `no_eligible_candidate`: non si forza una modalità primaria e non si allentano i requisiti dopo aver visto il risultato.

Si riportano tutti i candidati, ragioni di esclusione, distribuzione delle identità e AUC per famiglia di alterazione con i rispettivi positivi collegati. Gli intervalli AUC ricampionano **i 46 gruppi di fonte**, con 2.000 repliche e seed 42. Descrivono l'incertezza di sviluppo, senza correggere l'ottimismo della selezione. Non si calcola l'AUC globale eliminando solo i casi privi di copertura.

## Esecuzione e risultati ancora mancanti

La preparazione locale e il piano con tokenizer sono eseguiti; **le nuove probabilità NLI e la scelta primaria non sono ancora disponibili**. Il piano contiene 323.336 evidenze uniche deduplicate fra modalità. Le singole e finestre coprono tutte le unità. Il documento non raggiunge la copertura richiesta già nella pianificazione: rimane secondario e non viene salvato tramite troncamento.

Nel cloud corrente PyTorch è CPU e CUDA non è disponibile. Dal precedente run GPU, la previsione è circa **45–90 minuti**, con incertezza legata a lunghezza e numero dei confronti. Le vecchie cache coprono circa il 3% del piano: non si mescolano probabilità di runtime CPU/GPU per quel piccolo risparmio.

La [guida Kaggle](kaggle/README_NLI_Calibration.md) riusa pesi e venv quando disponibili ed esegue il vecchio scorer, seguito dall'analizzatore nuovo. Il pacchetto di ritorno comprende input, policy, metadati, CSV e selezione; le cache di probabilità e le evidenze estese rimangono sulla macchina per riprendere, fuori dallo ZIP leggero. Non si eseguono humanizer, detector o chiamate Groq.

## Annotazione successiva

Sono pronte le [84 coppie reference cieche](exports/kaggle_v2_study/reference/README.md), già selezionate senza score nello study. Conservano ID e testi, con chiavi separate. La scelta NLI va congelata prima di consultare le nuove etichette umane; i giudizi non verranno usati per scegliere il candidato più favorevole sullo stesso campione.

Le 99 diagnostiche restano un approfondimento separato. Non si trasferiscono allo study l'AUC di calibrazione o la soglia storica OpAI 0,04075. Anche dopo la scelta, gli score rimangono continui e le modalità secondarie rimangono disponibili. La taratura del giudice di naturalezza e la revisione statistica dei detector sono attività successive, non già eseguite da questo pacchetto.
