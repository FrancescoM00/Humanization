# Pilot Kaggle: archivio e annotazione

**[ZIP originale](humanization_pilot.zip)**, preservato senza modifiche. SHA-256: `7a35ca521ccc4e2147b42e479fe2ac6bde1ef676ddb462137a27af046960f272`.

**[Report del pilot](../../results_kaggle_pilot/report.md)** e [casi commentati](../../results_kaggle_pilot/case_review.md).

## File da compilare

- **[Scheda semantica da 24 coppie](semantic_audit_24.csv)**: 21 riscritture, una per ciascuna combinazione dominio/generatore, più tre identità di controllo. Selezione tramite hash degli ID, senza usare gli score. Ordine mescolato; sistemi e punteggi sono nascosti.
- **[Scheda di naturalezza da 28 testi](naturalness_review_form.csv)**: il form originale del pilot, ancora senza giudizi.

**[Scarica i due moduli e le istruzioni in ZIP](annotation_forms.zip)**. L'archivio dei moduli non contiene le chiavi o i punteggi. Per Excel, importare i CSV come UTF-8 con separatore virgola e attivare il testo a capo per le colonne dei passaggi.

Per una revisione senza conoscere il sistema, consegnare al valutatore soltanto questi due file. Le chiavi [semantica](semantic_audit_24_key.csv) e [naturalezza](naturalness_review_key.csv) servono a ricongiungere i giudizi ai risultati dopo l'annotazione.

## Criteri semantici

Annotare `input_type_prose_instructions_mixed` come `prose`, `instructions` o `mixed`. La distinzione serve a identificare un eventuale cambio di compito, come rispondere a una consegna anziché riscriverla.

In `semantic_preserved_yes_partial_no`:

- `yes`: conserva le informazioni rilevanti e la forza delle affermazioni; variazioni di forma, fusione o divisione delle frasi sono ammesse.
- `partial`: conserva il quadro principale ma cambia o omette qualche informazione rilevante, oppure aggiunge contenuti non supportati dal passaggio.
- `no`: sostituisce il compito o altera ampiamente il contenuto.

In `error_type` indicare uno o più tipi: `omission`, `unsupported_addition`, `number_value`, `number_bound`, `entity`, `negation`, `hedging_or_claim_strength`, `task_shift`. Nelle note citare brevemente il punto preciso. Un'aggiunta non supportata dall'input non è automaticamente falsa nel mondo; qui si valuta la fedeltà al passaggio, non un fact-check esterno.

Non usare NLI, detector o giudizi della review dell'assistente come gold. È possibile annotare incertezza nelle note; non forzare una decisione sicura quando l'originale è ambiguo.

## Criteri di naturalezza

Valutare il testo singolo con `naturalness_1_to_5`: 1 molto artificiale/innaturale, 3 intermedio, 5 molto naturale nel suo genere. Considerare registro, formulazioni, ritmo e coerenza linguistica; annotare separatamente i motivi. Il voto non misura l'origine effettiva né la correttezza fattuale, e non va copiato da Flesch o dalla distanza stilometrica.

Un valutatore fornisce un primo controllo esplorativo; eventuali giudizi di più lettori vanno conservati separatamente. Le etichette ora presenti nei form sono tutte vuote.
