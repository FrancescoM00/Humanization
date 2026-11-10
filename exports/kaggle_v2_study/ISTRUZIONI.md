# Revisione semantica v2

Le due schede contengono le stesse coppie, con ordine diverso e giudizi vuoti.
Consegnare una sola copia a ciascun valutatore; i lettori lavorano indipendentemente,
senza consultare chiave, score, report, commenti dell'assistente o giudizi altrui.
Due annotatori della stessa coppia non creano due fonti indipendenti.

In `input_type_prose_instructions_mixed` usare prose/instructions/mixed.
In `semantic_preserved_yes_partial_no` usare yes (informazioni e forza conservate),
partial (quadro principale conservato ma informazioni rilevanti alterate, omesse
o aggiunte), no (contenuto ampiamente alterato o cambio di compito). Fusione e
divisione delle frasi sono ammesse se il contenuto rimane. Non giudicare soltanto
la lunghezza; rimuovere ripetizioni può conservare l'informazione unica.

In `error_type` indicare omission, unsupported_addition, number_value, number_bound,
entity, negation, hedging_or_claim_strength, task_shift o other, anche più d'uno.
Nelle note citare il punto preciso e l'eventuale ambiguità. Si valuta fedeltà al
passaggio, non verità esterna o naturalezza. Non copiare il verdetto da NLI.

Conservare ID e testi esatti. Le etichette vuote non sono risultati. La chiave
tenuta separatamente identifica il campione reference, scelto senza score, e
quello diagnostic, arricchito per casi problematici: non usarli come un unico
test casuale. La chiave permette anche di riconoscere coppie già annotate nelle
precedenti schede e di evitare il doppio conteggio di giudizi dello stesso lettore.
