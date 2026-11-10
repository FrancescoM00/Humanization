# Calibrazione NLI: input pronti, scoring GPU da eseguire

Stato del 10 novembre 2026: preparati il [pannello di calibrazione](../exports/nli_context_calibration/README.md), la [regola di scelta](../NLI_CALIBRATION.md) e il [pacchetto Kaggle](../kaggle/README_NLI_Calibration.md). Sono 139 positivi presunti, 297 alterazioni e 137 identità su 46 fonti; il test storico 141/265 resta escluso.

Il piano costruito con il tokenizer richiede 323.336 evidenze uniche. **Non sono state eseguite le nuove inferenze**, non è stata selezionata una modalità primaria e non sono state create etichette umane. Il documento ha copertura insufficiente già nel piano; singole e finestre sono complete.

Il run Kaggle produrrà il confronto dei 16 candidati, distribuzioni sulle identità, AUC per famiglia, intervalli per fonte e la scelta secondo la policy congelata. Dopo aver ricevuto lo ZIP si aggiornerà questa directory con i risultati effettivi, mantenendo separata la verifica sulle 84 reference umane.
