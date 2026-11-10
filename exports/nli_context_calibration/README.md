# Pannello congelato per la scelta NLI

[Specifica della scelta](../../NLI_CALIBRATION.md), [guida Kaggle](../../kaggle/README_NLI_Calibration.md).

La preparazione comprende **139 positivi presunti, 297 alterazioni e 137 identità**, su 46 gruppi di fonte. I testi provengono soltanto dallo split di calibrazione del confronto OpAI appaiato. Le 141 coppie positive e 265 alterate del test storico non entrano nella scelta; la lettura dei metadati degli split serve a verificare la disgiunzione delle fonti.

- `pairs.jsonl` e `alignment.json`: 573 coppie e unità esatte condivise dalle modalità.
- `labels.csv`: classi operative presunte e collegamento fra negativi/positivi; nessuna etichetta umana nuova.
- `policy.json`: candidati, requisiti e regola di selezione fissati prima delle nuove inferenze.
- `preparation_manifest.json`: conteggi, esclusioni e provenienza.

Le identità non entrano nell'AUC e non sono fonti aggiuntive. Gli input e la policy sono conservati senza modifiche. Il successivo [run di calibrazione ricevuto](../../results_nli_context_calibration/report.md) seleziona **window2 + minimo**; la verifica umana è ancora da completare.

L'[archivio originale ricevuto](nli_calibration_v1_received.zip) è conservato esattamente: 6.183.514 byte, SHA-256 `22669f3e6d67fc80b4b25fb98dcb975c4a70cce9b91695040e39dd08cf1f30d6`. Il bundle di input precedente resta congelato; il nuovo archivio contiene score e analisi, senza pesi o cache probabilistiche estese.

Riproduzione dalla radice del repository:

```bash
.venv/bin/python scripts/calibrate_contextual_nli.py prepare \
  --pairs data/semantic_extension/pairs.jsonl \
  --alignment data/semantic_extension/alignment.json \
  --output exports/nli_context_calibration
```
