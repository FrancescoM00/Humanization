# Controlli nulli preparati sui pilot esistenti

[Scarica i controlli](controls.zip). Costruzione già eseguita sui testi dei pilot, senza nuove generazioni o score dei detector.

Sono presenti 336 righe: 42 riscritture della baseline 3B, 42 della configurazione DIPPER XXL sul passaggio intero, 42 identità nulle, 42 rumori lessicali minimi e quattro condizioni da 42 righe con prefisso/rumore sul prefisso per i due sistemi. Le identità ripetute dei run originali sono deduplicate.

Il confronto di lunghezza è eleggibile in 35/42 riferimenti della baseline 3B e 42/42 di DIPPER; sette riscritture 3B sono più lunghe dell'originale e non ricevono padding. Il rumore lessicale trova un edit eleggibile in 28/42 originali; gli altri restano dichiarati invariati. Prefissi ed edit non sono esempi gold di conservazione semantica.

**Non sono ancora disponibili score dei detector su questi controlli.** Non se ne ricava una quota di evasione attribuibile alla lunghezza. Lo smoke e lo study v2 li valutano con gli stessi detector e la stessa calibrazione delle riscritture.

[Eleggibilità](eligibility.csv), [manifest](null_controls_run.json), [metodo](../../docs/NULL_CONTROLS.md), [protocollo](../../STUDY_V2_PROTOCOL.md).
