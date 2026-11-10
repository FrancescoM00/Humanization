# Primo passo: annotare le 84 coppie reference

Scarica [reference_annotation_forms.zip](reference_annotation_forms.zip): contiene i moduli A/B e le [istruzioni](ISTRUZIONI.md), senza chiavi, sistemi o score.

Sono **84 coppie su 21 fonti**, con quattro riscritture per fonte, già scelte nel reference dello study senza usare gli score. Non sono stati selezionati nuovi esempi e le 99 coppie diagnostiche rimangono fuori da questo primo passaggio. ID e testi sono preservati esattamente dai moduli originali.

Se sei l’unico annotatore, compila soltanto il [modulo A](semantic_reference_annotator_A.csv). Il [modulo B](semantic_reference_annotator_B.csv) è per un secondo lettore indipendente: contiene le stesse coppie in ordine diverso. Conserva ID e testi; compila soltanto tipo di input, conservazione, tipo di errore e note. Restituisci il CSV quando completato, anche con casi mancanti dichiarati.

La [chiave per l’analista](keys/reference_key.csv) è separata e **non va aperta prima di annotare**. Non è nello ZIP cieco. Il [manifest](manifest.json) registra il sottoinsieme e la provenienza. I moduli sono vuoti; nessuna annotazione umana o gold è stata generata.

Questa scheda valuta la fedeltà al testo originale, non la naturalezza o l’origine umana/AI. La validazione del giudice di naturalezza richiederà giudizi e dati separati.
