# Dati e backup su GitHub

Repository: [FrancescoM00/Humanization](https://github.com/FrancescoM00/Humanization).

- **`main`** contiene codice attivo, test, dipendenze fissate, report, CSV, figure, patch originale/corretta e manifest dei backup.
- **[`cloud-backup`](https://github.com/FrancescoM00/Humanization/tree/cloud-backup/backups)** conserva dataset, modelli, venv e cache in archivi Git LFS. Le parti pesanti non vengono scaricate dal normale checkout di `main`.

## File pronti da scaricare

[exports/semantic_clean/](exports/semantic_clean/README.md) contiene `pairs.jsonl`, `skipped.jsonl` e lo [ZIP da circa 1,4 MB](exports/semantic_clean/semantic_clean_pairs.zip). Sono le 2.297 coppie definitive del run clean, con 145 sostituzioni applicate e 55 saltate. Con `git pull` arrivano direttamente nella cartella `exports/` del clone.

I report e i protocolli sono indicizzati nel [README](README.md); per continuare in Codex leggere [CODEX_HANDOFF.md](CODEX_HANDOFF.md). Le versioni precedenti e i bundle rimossi durante la pulizia sono recuperabili dalla cronologia Git. La cronologia completa della chat non è contenuta nel repository.

## Inventario del backup

Lo snapshot aggiornato all'8 novembre 2026 contiene anche tutta `data/semantic_clean/`: allineamenti, copia del retrieval ricalcolato, piano NLI e probabilità. Il precedente campione fairness di 3.500 coppie e le altre fasi restano inclusi. Modelli e venv sono quelli già archiviati; l'aggiornamento più recente riguarda dati e cache di esecuzione.

| Componente | Contenuto originale | Parti | Byte compressi |
| --- | --- | ---: | ---: |
| Dataset | `data/`, escluso `data/models/` | 2 | 171.434.235 |
| Modelli | MiniLM, RoBERTa, DeBERTa-v3 base e ONNX in `data/models/` | 13 | 1.160.384.762 |
| Venv | Pacchetti, spaCy e CMUdict in `.venv/` | 6 | 518.027.092 |
| Cache | Snapshot delle cache di esecuzione del progetto | 1 | 15.194 |
| **Totale** | **4.365.132.616 byte originali** | **22** | **1.849.861.283** |

Ogni parte è al massimo 90 MiB. Dimensioni e SHA-256 sono nel [manifest](backups/manifest.json); il trasferimento iniziale era stato verificato con ripristino in una directory temporanea. Gli archivi escludono credenziali, `.git` e configurazioni private dell'esecutore.

## Recuperare i componenti

Le istruzioni canoniche per installare Git LFS, recuperare il ramo ed estrarre un componente sono in [backups/README.md](backups/README.md). Il normale ZIP del codice non comprende le parti LFS pesanti. Servono Python 3.12 e `zstd`; il restorer preserva componenti esistenti, salvo sostituzione esplicita.

La venv è uno **snapshot Linux x86_64/Python 3.12.14**, con interprete esterno e percorsi assoluti. Su Windows, macOS o percorsi diversi va ricreata dai lockfile; dati e modelli restano riutilizzabili. Questa limitazione è spiegata nel documento di ripristino.
