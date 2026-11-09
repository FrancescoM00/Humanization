# Pilot DIPPER: archivio e revisione appaiata

**[ZIP originale ricevuto](humanization_pilotdipper.zip)**, conservato byte per byte. SHA-256 `21a66ed1cb1d3744be742a79f4fb7b924668d21200ed11033364ac48a18df31d`.

[Report del pilot e confronto con Qwen](../../results_kaggle_dipper_pilot/report.md), [12 casi commentati](../../results_kaggle_dipper_pilot/case_review.md), [confronto ASR appaiato](../../results_kaggle_dipper_pilot/paired_comparison.md).

## Schede comuni per i due sistemi

**[Scarica i moduli e le istruzioni in ZIP](annotation_forms.zip)**. Lo ZIP contiene solo i due form con giudizi vuoti e le istruzioni: nessuna chiave o punteggio. Queste schede sono state preparate dopo il run; non sono annotazioni ricevute dall'autore.

- [Semantica, 45 coppie](semantic_paired_45.csv): le stesse 21 fonti già scelte tramite hash per il modulo Qwen, una per dominio/generatore, ora con entrambe le riscritture; tre identità completano i controlli. Ordine rimescolato tramite hash, senza selezione basata sugli score. I nomi dei sistemi sono nascosti.
- [Naturalezza, 24 testi](naturalness_paired_24.csv): sei fonti, due per dominio, scelte fra le precedenti 21 tramite hash dell'ID. Ogni fonte contribuisce umano, originale macchina, Qwen e DIPPER. I testi sono mostrati singolarmente in ordine mescolato; giudizi e chiavi sono separati.

Le chiavi [semantica](semantic_paired_45_key.csv) e [naturalezza](naturalness_paired_24_key.csv) servono solo dopo la valutazione. Non consegnarle al valutatore insieme ai form. La chiave semantica riporta anche `previous_qwen_review_id`: le precedenti 24 coppie Qwen/identità sono incluse. Se erano già state annotate, ricongiungere quei giudizi tramite chiave evitando di contarli come due osservazioni indipendenti dello stesso valutatore. Al momento non sono state ricevute etichette compilate.

Per Excel importare i CSV in UTF-8, separatore virgola e testo a capo. Il testo dei passaggi va conservato esattamente. Per le istruzioni di annotazione usare [ISTRUZIONI.md](ISTRUZIONI.md).

La revisione delle 21 fonti è un primo riferimento comune e bilanciato, non una garanzia di precisione. Le sei fonti della naturalezza sono un controllo ancora più piccolo. Eventuali annotazioni di più valutatori vanno salvate separatamente; non usare i punteggi della suite o i commenti dell'assistente per decidere le etichette.
