# Trasferimento su GitHub completato

Repository: [FrancescoM00/Humanization](https://github.com/FrancescoM00/Humanization).

## Contenuti disponibili

- **`main`**: tutto il codice sperimentale, test, dipendenze fissate, report, CSV, grafici, ZIP delle versioni precedenti, [riepilogo per Codex](CODEX_HANDOFF.md), manifest dei backup e script di ripristino.
- **[`cloud-backup`](https://github.com/FrancescoM00/Humanization/tree/cloud-backup/backups)**: archivi completi dei dataset, dei modelli, della venv e delle cache, memorizzati tramite **Git LFS**. Il caricamento di tutti gli oggetti LFS e del ramo è riuscito.

Le parti pesanti non sono presenti nel working tree di `main`: aprire il codice e leggere i risultati non richiede di scaricarle. Il normale ZIP del codice non include i backup pesanti; usare Git LFS come descritto sotto.

| Componente | Contenuto originale | Parti | Byte compressi |
| --- | --- | ---: | ---: |
| Dataset | `data/`, escluso `data/models/`: corpora, campioni, coppie, cache e probabilità | 2 | 130.431.676 |
| Modelli | Tutto `data/models/`, inclusi MiniLM, RoBERTa, DeBERTa-v3 base e ONNX | 13 | 1.160.384.762 |
| Venv | Tutto `.venv/`, inclusi pacchetti, spaCy e CMUdict | 6 | 518.027.092 |
| Cache | `.cache/`, `.pytest_cache/` e `semantic_run.log` del progetto | 1 | 14.228 |
| **Totale** | **4.095.678.049 byte di dati, modelli e venv, più cache** | **22** | **1.808.857.758** |

Ogni parte è al massimo 90 MiB. Le dimensioni e gli SHA-256 sono in [backups/manifest.json](backups/manifest.json). Un ripristino integrale in una directory temporanea ha riprodotto gli inventari dei quattro componenti, inclusi i symlink; i dati originali non sono stati sostituiti.

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
