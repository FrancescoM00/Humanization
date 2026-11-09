# Piano Groq pronto, giudizi ancora assenti

È stato preparato il piano su sei fonti dei pilot: baseline Qwen2.5-3B con prompt fissato contro DIPPER XXL sul passaggio intero, entrambi gli ordini AB/BA, 12 risposte attese e massimo 16 tentativi POST.

**Zero chiamate Groq e zero giudizi Groq sono stati eseguiti in questo ambiente.** La chiave è nei Secrets Kaggle secondo l'autore e sarà letta dal nuovo notebook. Il modello richiesto `qwen/qwen3.8-27b` e l'opzione senza ragionamento devono essere verificati dal catalogo autenticato e dalla prima richiesta; nessuna sostituzione automatica.

Le [24 valutazioni dell'assistente](../results_naturalness_assistant24/report.md) sono disponibili come proxy AI, non gold umana. Il piano di confronto fra riscrittori usa 12 di quei testi su sei fonti. Mancano ancora le risposte del giudice: accordo e kappa rimangono non definiti.

[Piano](judge_plan.json), [stato](judge_run.json), [copertura della validazione](rating_validation.json), [istruzioni](../docs/NATURALNESS_JUDGE.md).
