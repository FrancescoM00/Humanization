# Controlli nulli di lunghezza e rumore

I controlli verificano quanto cambi la detection quando si accorcia un originale o si applica una modifica lessicale minima. Non sono humanizer, riferimenti semantici gold o una dimostrazione causale dell'effetto dello stile. Le generazioni e i risultati storici non sono modificati.

## Preparazione

Usare la coorte congelata e una o più liste di riscritture complete:

```bash
python scripts/prepare_null_controls.py \
  --cohort RUN/cohorts/study.jsonl \
  --rewrites RUN/writer_3b/rewrites.jsonl \
  --rewrites RUN/writer_14b/rewrites.jsonl \
  --output RUN/null_controls --include-rewrites
```

È richiesta la piccola pipeline spaCy `en_core_web_sm` già prevista dal progetto. Si mantiene anche NER per proteggere le entità riconosciute durante il rumore. Il conteggio lessicale resta lo stesso della stilometria: token spaCy con `is_alpha`.

Il comando scrive `rewrites.jsonl`, compatibile con `run_detectors.py`, `eligibility.csv` e `null_controls_run.json`. Lo stesso output non viene sostituito se cambiano input o parametri. `--only NAME` seleziona riferimenti specifici; le identità degli input non sono usate per creare condizioni riferite a un humanizer.

## Condizioni

- `null_identity`: originale completo, separato dalle altre condizioni.
- `null_lexical_noise`: originale completo con al massimo una sostituzione lessicale del dizionario fisso.
- `null_length__NAME`: originale troncato esattamente al numero di parole alfabetiche prodotto da `NAME` sulla stessa fonte.
- `null_length_noise__NAME`: lo stesso prefisso, più il rumore lessicale minimo. Permette di distinguere il rumore sul testo completo da quello sul testo accorciato.

Il seed e l'identificatore del testo ordinano le sostituzioni ammissibili. Nessuno score viene usato per scegliere parole o testi. Il dizionario contiene solo sostituzioni da un token alfabetico a un token alfabetico; non modifica direttamente numeri, negazioni, modali o quantificatori. Nomi capitalizzati, token marcati PROPN/AUX/NUM/PART ed entità riconosciute da spaCy sono esclusi. La protezione NER è imperfetta e i sinonimi non sono equivalenti in ogni contesto: il rumore resta una perturbazione descritta, non una parafrasi certificata. Le modifiche effettive sono salvate nei metadati.

## Come è pareggiata la lunghezza

Il prefisso mantiene il numero esatto di parole alfabetiche della riscrittura. Si conservano punteggiatura e suffissi di contrazione attaccati all'ultima parola: per esempio non si lascia il solo `ca` quando il tokenizer divide `can't` in `ca` + `n't`. Il controllo può terminare a metà frase; `hard_cut` e `cut_at_sentence_boundary` lo dichiarano. Non è la stessa pulizia a fine frase applicata alle fonti iniziali. Il taglio può perdere contenuto ed è intenzionalmente un controllo di accorciamento, non un positivo di conservazione.

Se la riscrittura è più lunga dell'originale non è possibile costruire un prefisso della stessa lunghezza senza aggiungere parole. Si conserva allora l'originale con stato `ok`, ma `length_match_eligible=false`, `comparison_eligible=false` e `length_match_status=target_longer_than_original`. Nessun padding è aggiunto. Quei casi possono essere descritti nella tabella generale, ma non entrano nel confronto appaiato a lunghezza uguale.

Se la generazione di riferimento è fallita, il target non viene inventato: la riga del controllo è `reference_unavailable`, con target vuoto. Se non esiste una parola ammessa per il rumore, l'originale/prefisso rimane invariato e `noise_eligible=false`. Il manifest e la tabella di eligibilità conservano tutti questi casi, senza selezionare soltanto quelli favorevoli.

## Detector e confronto appaiato

Eseguire i detector sulla stessa coorte, con lo stesso config e i controlli nel file di riscrittura. Con `--include-rewrites` anche le riscritture di riferimento sono presenti, quindi i confronti usano una sola lista; le cache per testo identico possono evitare lavoro duplicato. Gli umani di calibrazione restano quelli congelati.

```bash
python scripts/run_detectors.py \
  --cohort RUN/cohorts/study.jsonl --rewrites RUN/null_controls/rewrites.jsonl \
  --config CONFIG --output RUN/null_controls/detection --device cuda:0

python scripts/prepare_null_controls.py \
  --cohort RUN/cohorts/study.jsonl \
  --rewrites RUN/writer_3b/rewrites.jsonl \
  --rewrites RUN/writer_14b/rewrites.jsonl \
  --output RUN/null_controls --include-rewrites \
  --detector-scores RUN/null_controls/detection/detector_scores.csv \
  --thresholds RUN/null_controls/detection/thresholds.json
```

La seconda invocazione non esegue inferenza. Scrive `null_control_comparison.csv` con il confronto controllo/riferimento entro **gli stessi originali eleggibili già riconosciuti AI**. Il denominatore non contiene le riscritture più lunghe, i riferimenti falliti o i controlli di rumore non applicabili. Si indicano separatamente fonti totali, eleggibili e non eleggibili. Score mancanti o associati a testo diverso causano un errore, non un'evasione.

Si possono fornire più `--detector-scores` per ricongiungere score già salvati dei riferimenti e dei controlli. Le chiavi duplicate devono avere score coerenti. Per detector con score grezzo orientato al contrario, come un rapporto Binoculars in cui valori bassi indicano AI, specificare esplicitamente `--lower-is-ai NOME` e usare le sue soglie congelate. Le uguaglianze alla soglia sono classificate umane in entrambe le direzioni. Il helper non rifà la calibrazione e non interpreta uno score grezzo come probabilità.

L'ASR generale di `run_detectors.py` continua a seguire la convenzione storica sull'intera coorte, contando stati non `ok` come insuccessi. Per attribuire una differenza al confronto con un controllo di lunghezza usare invece la tabella appaiata eleggibile: i fallback originali dei target più lunghi non sono un pareggiamento riuscito. Anche questa differenza rimane diagnostica, perché il taglio elimina informazioni, può interrompere una frase e cambia altre proprietà oltre al conteggio delle parole.
