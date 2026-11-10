# Annotazione delle 84 coppie reference

Questo pacchetto contiene le stesse 84 coppie in due ordini diversi: moduli A e B.
Sono 21 fonti, una per ciascuna combinazione dominio/generatore, con quattro
riscritture per fonte. La selezione reference è stata fatta senza usare gli score.
Quattro riscritture e due annotatori non moltiplicano le fonti indipendenti.

## Da dove iniziare

Se annoti da solo, compila soltanto `semantic_reference_annotator_A.csv`.
Il modulo B è destinato a un secondo annotatore, che lavora indipendentemente.
Non compilare entrambi come se fossero esempi nuovi. Puoi lavorare in più sessioni.
Conserva una copia del file vuoto e restituisci il CSV compilato mantenendo gli ID.
Non consultare chiavi, score, report, review dell’assistente o giudizi altrui.
La chiave è separata e non è inclusa in questo ZIP.

## Cosa compilare

- `input_type_prose_instructions_mixed`: prose, instructions oppure mixed,
  valutando il tipo del testo originale.
- `semantic_preserved_yes_partial_no`: yes se informazioni e forza delle
  affermazioni sono conservate; partial se il quadro principale resta ma ci sono
  alterazioni, omissioni o aggiunte rilevanti; no se il contenuto cambia ampiamente
  o la riscrittura svolge un altro compito.
- `error_type`: omission, unsupported_addition, number_value, number_bound,
  entity, negation, hedging_or_claim_strength, task_shift oppure other.
  Puoi indicare più tipi separati da punto e virgola. Se non c’è errore, lascia vuoto.
- `notes`: cita il dettaglio che giustifica il giudizio o spiega l’incertezza.

Fondere o dividere frasi è ammesso se il contenuto rimane. La lunghezza da sola
non decide la conservazione; togliere ripetizioni può conservare l’informazione
unica. Confronta la riscrittura con il passaggio: non verificare la verità esterna
e non attribuire un voto di naturalezza in questa scheda.

Non modificare `review_id`, `original` o `modified`. Se un caso è dubbio, descrivilo
nelle note; lascia il giudizio vuoto se non riesci a decidere, senza inventarlo.
I campi non compilati restano mancanti, non significano yes. I moduli iniziali sono
vuoti: non sono state prodotte etichette umane o gold automatiche.
