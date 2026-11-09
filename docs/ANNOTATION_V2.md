# Annotazione v2: riferimento casuale e casi diagnostici

La validazione degli humanizer richiede giudizi indipendenti sui loro output. Un
campione selezionato perché le metriche si contraddicono è utile per trovare
errori, ma non stima senza correzioni l'accuratezza nella popolazione.

`scripts/prepare_semantic_audit_v2.py` costruisce **due bracci separati** e li
mescola nelle schede, nascondendo a chi annota il braccio, gli score e il sistema:

- **Reference:** una fonte per cella dominio/generatore, scelta dal ranking hash
  con seed fissato, senza usare punteggi. Tutti i sistemi della fonte sono inclusi:
  con 21 celle e due humanizer sono 21 fonti e 42 coppie. La chiave registra la
  probabilità di inclusione della fonte `k/N` e il peso `N/k`. L'unità da
  ricampionare nelle analisi resta la fonte, non la singola riscrittura.
- **Diagnostic:** fonti rimanenti, disgiunte dal reference. Si privilegiano
  `coseno >= 0,9 e minimo NLI <= 0,1`, poi gli altri minimi NLI <= 0,1 e infine
  gli altri casi, con ciclo di preferenza 3:2:1. Dentro gli strati si intercalano
  le celle dominio/generatore e si ordinano i casi per NLI crescente e coseno
  decrescente; l'hash risolve i pareggi. Una fonte entra con tutte le sue
  riscritture, anche quando solo uno dei due sistemi soddisfa il criterio.

Queste soglie definiscono solo una strategia di selezione: **non sono soglie di
errore semantico**. I pesi reference non si applicano ai casi diagnostic. Non
calcolare un'AUC diagnostica non pesata presentandola come prestazione generale.
Il manifest conserva i motivi e la numerosità disponibile, senza creare gold.

## Dimensioni e indipendenza

Nel pilot ci sono **84 coppie non identità ma solo 42 fonti**. Il default sceglie
42 coppie reference e le 42 coppie residue diagnostic: è il censimento delle 84
coppie disponibili, non un campione da 100 nuove fonti o una selezione riuscita
di 100 casi diagnostici. Il manifest dichiara `diagnostic_target_met=false`.
Se tutte le 84 coppie sono giudicate, si può descrivere l'intero frame del pilot;
ciò non costituisce validazione su nuovi dati o una stima affidabile per ogni
generatore. Il braccio diagnostic da solo rimane selezionato per score.

Sul pannello da 126 fonti, con due sistemi riusciti per fonte, ci saranno 252
coppie: il piano di 42 reference più 100 diagnostic è fattibile. Non è ancora un
risultato ottenuto. Aggiungendo un terzo sistema cambiano i conteggi: l'opzione
`--diagnostic-size` indica coppie, `--reference-per-cell` indica fonti; i gruppi
non vengono spezzati per raggiungere artificialmente un numero tondo.

Due valutatori compilano due copie indipendenti, con stessi ID e ordine diverso.
Due giudizi della stessa coppia non raddoppiano le fonti. La chiave permette il
ricongiungimento con precedenti annotazioni tramite ID della coppia: non contare
due volte il giudizio dello stesso lettore su un testo già annotato. Le vecchie
schede da 24 e 45 coppie possono contenere gli stessi esempi.

Non è stato trovato nel repository un riferimento umano da cui verificare il
kappa 0,30 citato nel feedback. Non viene ricreato né riportato come risultato.
L'accordo andrà calcolato quando saranno disponibili i giudizi indipendenti,
specificando trattamento di `partial`, classi, mancanti e intervallo per fonte.

## Uso sul pilot disponibile

```bash
python scripts/prepare_semantic_audit_v2.py \
  --input-pairs exports/kaggle_pilot/humanization_pilot.zip \
  --input-pairs exports/kaggle_dipper_pilot/humanization_pilotdipper.zip \
  --scores results_kaggle_dipper_pilot/comparison_with_qwen.csv \
  --output exports/semantic_audit_v2/pilot_42 \
  --reference-per-cell 1 --diagnostic-size 100 --seed 42
```

L'input accetta JSONL o ZIP Kaggle, privilegiando `semantic/pairs.jsonl` negli
archivi storici. Gli input possono essere ripetuti per più sistemi. Le identità,
i controlli nulli e le generazioni fallite sono esclusi dal frame semantico e
contati nel manifest; il tasso di fallimento della pipeline va riportato
separatamente. Le coppie valide prive di score non sono scartate silenziosamente.

Per il run da 126 usare i suoi input e score reali, una nuova cartella e aggiungere
`--require-full-diagnostic` se si vuole interrompere la preparazione quando il
target non è raggiungibile. Non usare gli score del pilot come se fossero quelli
dello study. Le directory già congelate rifiutano contenuti diversi.

Lo ZIP `annotation_forms.zip` contiene soltanto i due form e le istruzioni; la
chiave con punteggi e bracci deve rimanere separata. Tutte le etichette preparate
sono vuote. Nessuna annotazione umana o inferenza viene eseguita dal comando.
