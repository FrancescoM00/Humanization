# Study v2: archivio ricevuto e schede da annotare

[Report delle 126 fonti](../../results_kaggle_v2_study/report.md), [audit](../../results_kaggle_v2_study/integrity.md), [provenienza](../../results_kaggle_v2_study/provenance.json).

L’[archivio originale](humanization_v2_study_protocol_v2.zip) è conservato senza modifiche: 27.194.720 byte, SHA-256 `7e55e8b510df4e26ca1f7d4a61648525a088523a8cb8983bb805fa9b16476cd8`. Contiene 630 esiti, 626 validi e quattro troncati, score dei detector e controlli, semantica e stile. I tre confronti Groq sono parziali: l’esistenza dello ZIP non certifica copertura completa. Modelli e venv non sono inclusi.

Per la revisione manuale usare [annotation_forms.zip](annotation_forms.zip): include istruzioni e le due copie indipendenti A/B, senza chiavi, sistemi o score. Le schede contengono le stesse **183 coppie**: 84 reference da 21 fonti e 99 diagnostiche da 25 ulteriori fonti. Le due copie non sono 366 coppie indipendenti; sono destinate a due annotatori che lavorano separatamente. Il diagnostico non raggiunge il target 100 per il raggruppamento completo delle fonti, non per mancanza generale di dati. Non duplicare un esempio per arrivare a cento.

Compilare i campi del tipo di input, conservazione yes/partial/no, tipo di errore e note secondo ISTRUZIONI.md. I moduli ricevuti sono vuoti: nessun giudizio umano è stato inventato. Il [manifest](sampling_manifest.json) descrive il campionamento. Il braccio reference è scelto senza score; il diagnostico è arricchito e non stima ingenuamente l’accuratezza nella popolazione.

La [chiave separata](keys/selection_key.csv) serve all’analista dopo l’annotazione: non aprirla prima di compilare i moduli e non condividerla con l’altro annotatore durante il lavoro. È fuori dallo ZIP cieco. La [scheda degli originali](source_screening.csv) serve a classificare gli input prose/instructions/mixed per un’eventuale analisi secondaria; i flag automatici non sono etichette umane e non escludono fonti dal primario.

**Primo passaggio dopo lo study:** usare il [pacchetto delle sole 84 reference](reference/reference_annotation_forms.zip), con [istruzioni dedicate](reference/README.md). Le 99 diagnostiche rimangono disponibili per un approfondimento successivo. Il sottoinsieme conserva esattamente ID e testi e non contiene etichette nuove.
