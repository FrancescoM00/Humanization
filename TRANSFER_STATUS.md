# Trasferimento su GitHub completato

Repository: [FrancescoM00/Humanization](https://github.com/FrancescoM00/Humanization).

## Contenuti disponibili

- **`main`**: tutto il codice sperimentale, test, dipendenze fissate, report, CSV, grafici, ZIP delle versioni precedenti, [riepilogo per Codex](CODEX_HANDOFF.md), manifest dei backup e script di ripristino.
- **[`cloud-backup`](https://github.com/FrancescoM00/Humanization/tree/cloud-backup/backups)**: archivi completi dei dataset, dei modelli, della venv e delle cache, memorizzati tramite **Git LFS**. Il caricamento di tutti gli oggetti LFS e del ramo è riuscito.

Le parti pesanti non sono presenti nel working tree di `main`: aprire il codice e leggere i risultati non richiede di scaricarle. Il normale ZIP del codice non include i backup pesanti; usare Git LFS come descritto sotto.

Aggiornamento dell'8 novembre 2026: pubblicato il controllo **entity clean v2**, con codice, test, patch originale e corretta, report e CSV. I dati pronti da scaricare sono in [exports/semantic_clean/](exports/semantic_clean/README.md) sul branch `main`: 2.297 coppie, 145 sostituzioni applicate e 55 saltate. Il backup dataset include l'intera `data/semantic_clean/`, con allineamenti e probabilità NLI; sono aggiornate anche le piccole cache di esecuzione. Pesi e venv sono gli stessi già archiviati. [Protocollo e risultati](ENTITY_CLEAN_EXPERIMENT.md).

Aggiornamento del 5 novembre 2026: aggiunti codice e risultati del confronto semantico equo e dello stile localizzato. Il componente dataset è stato aggiornato con tutte le 3.500 coppie, i nuovi allineamenti, le probabilità NLI e le cache stilometriche. Modelli e venv sono gli stessi già trasferiti. Leggere [FAIRNESS_EXPERIMENT.md](FAIRNESS_EXPERIMENT.md) per riprendere questa fase.

| Componente | Contenuto originale | Parti | Byte compressi |
| --- | --- | ---: | ---: |
| Dataset | `data/`, escluso `data/models/`: corpora, campioni, coppie, cache e probabilità | 2 | 153.492.058 |
| Modelli | Tutto `data/models/`, inclusi MiniLM, RoBERTa, DeBERTa-v3 base e ONNX | 13 | 1.160.384.762 |
| Venv | Tutto `.venv/`, inclusi pacchetti, spaCy e CMUdict | 6 | 518.027.092 |
| Cache | `.cache/`, `.pytest_cache/` e `semantic_run.log` del progetto | 1 | 14.228 |
| **Totale** | **4.268.262.344 byte complessivi nei quattro componenti** | **22** | **1.831.918.140** |

Ogni parte è al massimo 90 MiB. Le dimensioni e gli SHA-256 sono in [backups/manifest.json](backups/manifest.json). Il trasferimento iniziale era stato verificato con ripristino integrale in una directory temporanea, inclusi i symlink. L'aggiornamento sostituisce soltanto l'archivio dei dati; i dati originali non sono stati sostituiti.

Non sono inclusi credenziali, `.git`, configurazioni private dell'esecutore o l'intero sistema operativo. La venv è conservata integralmente come **snapshot Linux x86_64, Python 3.12.14**; il suo interprete base è esterno alla venv e gli shebang contengono percorsi assoluti. Su macOS, Windows o percorsi diversi va ricreata dai lockfile. Non viene presentata come ambiente portabile.

## Riprendere nell'app Codex

1. Aprire/clonare `FrancescoM00/Humanization`, branch `main`.
2. Creare una nuova sessione e chiedere di leggere **`CODEX_HANDOFF.md`** e **`README.md`**.
3. Leggere prima i risultati esistenti; evitare di rilanciare inferenze soltanto per consultare il lavoro.
4. Recuperare i backup soltanto quando servono dati, modelli o cache locali.

La cronologia completa di ChatGPT non è contenuta nel repository. Il riepilogo documenta obiettivi, esperimenti, conclusioni, limiti e passi successivi.

## Recuperare i backup dal ramo dedicato

Servono Git LFS, Python 3.12 e `zstd`. Dalla radice del clone:

```bash
git lfs install
git fetch origin cloud-backup
git lfs fetch origin FETCH_HEAD --include='backups/**'
git restore --source=FETCH_HEAD --worktree -- backups/
```

Per vedere o ripristinare un componente:

```bash
python scripts/restore_backups.py --component models --list
python scripts/restore_backups.py --component data
python scripts/restore_backups.py --component models
```

Per estrarre tutti i componenti in una directory vuota:

```bash
python scripts/restore_backups.py --component all --root /percorso/Humanization-ripristinato
```

Il restorer rifiuta di sovrascrivere componenti esistenti senza `--overwrite`. Preservare le modifiche locali a `backups/` prima del comando `git restore`. Dettagli in [backups/README.md](backups/README.md).

L'autorizzazione dell'utente riguardava questo trasferimento completo; non autorizza automaticamente pubblicazioni future.
