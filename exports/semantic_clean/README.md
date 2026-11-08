# Dati scaricabili: entity clean v2

- [pairs.jsonl](pairs.jsonl): tutte le **2.297 coppie** del run definitivo.
- [skipped.jsonl](skipped.jsonl): le **55 sostituzioni non applicabili**.
- [semantic_clean_pairs.zip](semantic_clean_pairs.zip): archivio da circa **1,4 MB** con entrambi i file.

Per scaricare dal browser, aprire lo ZIP e usare **Download raw file**, oppure usare il [download diretto](https://github.com/FrancescoM00/Humanization/raw/refs/heads/main/exports/semantic_clean/semantic_clean_pairs.zip). Con `git pull` questi file arrivano direttamente nella cartella `exports/semantic_clean/` del clone.

Le sostituzioni clean applicate sono **145**, sulle stesse 200 fonti del run storico; il file completo comprende anche le altre condizioni, i controlli e 800 coppie OpAI. I file sono copie byte per byte di quelli in `data/semantic_clean/`; non contengono nuove annotazioni gold. Il [report](../../results_semantic_clean/interpretazione.md) spiega risultati e limiti NER.

Il backup integrale in `cloud-backup` include anche allineamenti, piano NLI e probabilità, per riprendere senza rifare le inferenze. [Recupero del backup](../../TRANSFER_STATUS.md).
