# Pannello congelato per la scelta NLI

[Specifica della scelta](../../NLI_CALIBRATION.md), [guida Kaggle](../../kaggle/README_NLI_Calibration.md).

La preparazione comprende **139 positivi presunti, 297 alterazioni e 137 identità**, su 46 gruppi di fonte. I testi provengono soltanto dallo split di calibrazione del confronto OpAI appaiato. Le 141 coppie positive e 265 alterate del test storico non entrano nella scelta; la lettura dei metadati degli split serve a verificare la disgiunzione delle fonti.

- `pairs.jsonl` e `alignment.json`: 573 coppie e unità esatte condivise dalle modalità.
- `labels.csv`: classi operative presunte e collegamento fra negativi/positivi; nessuna etichetta umana nuova.
- `policy.json`: candidati, requisiti e regola di selezione fissati prima delle nuove inferenze.
- `preparation_manifest.json`: conteggi, esclusioni e provenienza.

Le identità non entrano nell'AUC e non sono fonti aggiuntive. Si selezionano modalità e aggregatore soltanto dopo lo scoring sul pannello preparato. **Nessun risultato della nuova selezione è ancora disponibile.**

Riproduzione dalla radice del repository:

```bash
.venv/bin/python scripts/calibrate_contextual_nli.py prepare \
  --pairs data/semantic_extension/pairs.jsonl \
  --alignment data/semantic_extension/alignment.json \
  --output exports/nli_context_calibration
```
