# Audit di integrità dello smoke Kaggle

Audit in sola lettura su `data/kaggle_received/smoke_20261109`, senza nuovi pesi o inferenze. Archivio ricevuto SHA-256 `5034fb1ddc52797e4fa3954f8ef821136cb77a2ca542ee6e5f8cf4538344e525`.

## Esito

**Nessun blocco tecnico rilevato prima del pilot 42.** Tutti gli stadi previsti hanno output completi e coerenti. Le precauzioni scientifiche restano quelle del protocollo e il campione di tre fonti non permette stime di efficacia.

- `semantic/pairs.jsonl`, `rewriting_cuda0/pairs.jsonl` e `rewriting_cuda0/rewrites.jsonl` sono byte-identici: sei righe `ok`, tre identità e tre Qwen, sei ID unici.
- La cartella `rewriting/` contiene il precedente tentativo interrotto con sole tre identità. È conservata ma non contamina input o score del run completo.
- Pannelli smoke/pilot/study: 3/42/126 fonti test, ognuno con gli stessi 300 umani di calibrazione. Gruppi di fonte disgiunti; calibrazione esclusa anche da tutte le 126 fonti riservate.
- Le sei tabelle semantic/scores contengono gli stessi sei ID. Tutti i manifest di stadio riportano lo SHA corretto delle coppie.
- I metadati BERTScore/NLI/allineamento coincidono con gli hash dei due `patched_sources` inclusi. Gli AST dei sorgenti corretti coincidono con il codice corrente. Gli hash degli stadi lessicale/embedding corrispondono al codice originale `86d6992`; quello del rewriter corrisponde allo stesso commit. La patch modifica solo il trasferimento dei pesi BERTScore e la differenza di versione è documentata, non occultata.
- NLI: header `inference_device=cuda`, `inference_precision=float32`; revisione configurata `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. SHA del config uguale al checkpoint locale base già usato. Il codice corretto carica esplicitamente float32 su CUDA. I pesi non sono inclusi nell'archivio, quindi non è una nuova verifica SHA dei byte dei pesi Kaggle.

## Detector

Ricalcolati `detection_summary.csv` e `detection_macro_summary.csv` dai 936 score già registrati, con l'analizzatore corrente: corrispondenza entro 1e-10, senza inferenza.

| Detector | Soglia score > t | FPR calibrazione | Qwen successi / originali AI |
| --- | ---: | ---: | ---: |
| OpenAI | 0.97060769796 | 15/300 | 2/3 |
| Fakespot raw | 0.96118062735 | 15/300 | 0/3 |
| RADAR | 0.79465305805 | 15/300 | 0/3 |
| Voto maggioranza | almeno 2 voti | 5/300 | 0/3 |

Tutti i tre originali sono rilevati AI da ogni detector, nessuno dei tre umani test è falso positivo. Identità ASR 0. Macro-ASR Qwen 0.222222. Ogni modello ha 312 record: 300 cal H, 3 test H, 3 test M, 6 coppie riscritte/identità. Le identità riusano lo score del testo identico.

## Controlli identità e campi mancanti

- Coseno documento esattamente 1. BERTScore documento/allineato e coseno allineato distano da 1 al massimo 5.96e-8.
- RMS stilometrico, POS-JS, quattro delta shortlist e cambiamenti guardie sono esattamente 0; rapporto parole 1.
- NLI identità forward=reverse: medie Essay 0.987931, Reuters 0.991482, WP 0.994625; minimi 0.970583/0.986318/0.988711. Non sono imposti a 1.
- NaN nei `local_*` è corretto: nessuna frase target è annotata per i nuovi humanizer. Le metriche documentali/allineate e le sei distanze stilometriche sono presenti, senza feature mancanti nel RMS.
- Le schede manuali sono vuote: naturalezza 9 testi, semantica 6 coppie. Nessuna annotazione umana o naturalezza misurata va rivendicata.

## Numeri descrittivi Qwen

| Dominio | Coseno doc | BERTScore doc raw | NLI F mean/min | NLI R mean/min | RMS stile | Rapporto parole |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| Essay | .915717 | .938150 | .773793 / .022663 | .828919 / .008724 | .736460 | 1.200000 |
| Reuters | .946510 | .949220 | .830675 / .007338 | .775987 / .000674 | 1.015054 | .832215 |
| WP | .938949 | .931329 | .475452 / .000824 | .381940 / .003241 | .889197 | .869281 |

Tutti e tre gli smoke provengono da **ChatGLM**, una fonte per dominio: lo smoke non è bilanciato fra sette generatori; il pilot 42 lo è per costruzione. Minimi NLI bassi richiedono revisione qualitativa e non sono gold automatici. Gli intervalli bootstrap 0–0 su tre insuccessi non devono essere interpretati come certezza che la vera ASR sia zero. Nessun motivo tecnico per ricalibrare soglie o selezionare nuovi casi dopo aver visto questi risultati.
