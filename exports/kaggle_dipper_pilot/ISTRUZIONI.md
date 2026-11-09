# Istruzioni per la revisione dei pilot

Le schede non contengono score né nomi dei sistemi. Compilare senza consultare report, detector, NLI, chiavi o commenti dell'assistente. Le valutazioni di più lettori devono essere conservate in file separati.

## Conservazione: semantic_paired_45.csv

Leggere originale e riscrittura. In `input_type_prose_instructions_mixed` indicare `prose`, `instructions` o `mixed`: un testo può contenere una consegna anziché prosa ordinaria.

In `semantic_preserved_yes_partial_no` usare:

- `yes`: conserva informazioni rilevanti e forza delle affermazioni; fusione/divisione delle frasi e normali variazioni di forma sono ammesse.
- `partial`: conserva il quadro principale, ma cambia o elimina informazioni rilevanti o aggiunge elementi non sostenuti dal passaggio.
- `no`: altera ampiamente il contenuto o cambia compito, per esempio rispondendo a una consegna anziché riscriverla.

In `error_type` annotare uno o più tipi: `omission`, `unsupported_addition`, `number_value`, `number_bound`, `entity`, `negation`, `hedging_or_claim_strength`, `task_shift`, `other`. Nelle note citare il cambiamento preciso ed eventuale incertezza. Una valuta, un titolo o l'autore di una battuta possono cambiare anche se il tema rimane uguale. La compressione può eliminare ripetizioni senza perdere altrettanta informazione unica: confrontare il contenuto, non soltanto le parole.

Si valuta la fedeltà all'originale; un'aggiunta non sostenuta dall'input non è automaticamente falsa nel mondo. Se l'originale è ambiguo o incoerente, segnalarlo nelle note. Non usare NLI come giudizio gold.

## Naturalezza: naturalness_paired_24.csv

Valutare soltanto il testo singolo. Compilare `naturalness_1_to_5`: 1 molto artificiale/innaturale, 3 intermedio, 5 molto naturale nel proprio genere. Considerare formulazioni, registro, ritmo e coerenza linguistica; motivare il voto nelle note.

Il voto non è una predizione dell'origine del testo, una valutazione della sua fedeltà a un originale o un fact-check. Non copiarlo da leggibilità, lunghezza o altri score.

## Salvataggio e riuso

Restituire i CSV con i giudizi compilati e gli ID invariati. Non modificare i passaggi. Un singolo lettore offre un primo controllo esplorativo; le schede vuote non sono risultati.

Se la precedente scheda Qwen da 24 coppie è già stata annotata, comunicarlo insieme al file: le stesse coppie sono incluse nelle 45 e i giudizi possono essere ricongiunti dalla chiave, senza contarli due volte come osservazioni indipendenti. Anche le nuove valutazioni di naturalezza possono riutilizzare testi già giudicati; il ricongiungimento va effettuato sul testo esatto e dichiarato.
