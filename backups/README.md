# Backup completo dell'ambiente sperimentale

`manifest.json` descrive quattro componenti indipendenti. Ogni archivio tar.zst è diviso in parti di massimo **90 MiB**, inferiori al limite GitHub di 100 MiB per file. Dimensioni e SHA-256 delle singole parti e dell'archivio concatenato sono registrati nel manifest. La compressione è Zstandard livello 3, al massimo tre thread.

| Componente | Contenuto ripristinato | Directory delle parti |
| --- | --- | --- |
| `data` | `data/`, esclusi i modelli | `datasets/` |
| `models` | `data/models/`, inclusi tutti i checkpoint e ONNX | `model_weights/` |
| `venv` | `.venv/`, pacchetti installati, spaCy e CMUdict | `python_environment/` |
| `cache` | `.cache/`, `.pytest_cache/`, `semantic_run.log` del progetto | `runtime_cache/` |

Codice, report, CSV, figure e pacchetti ZIP rimangono normali file del repository. Gli archivi non comprendono `.git`, configurazioni dell'esecutore o credenziali. I dati e i pesi originali non vengono modificati creando il backup.

## Recuperare il ramo dei backup

Il ramo `main` mantiene codice, risultati, questo documento, manifest e script di ripristino. Le parti pesanti sono conservate sul ramo `cloud-backup`. Dalla radice del clone, prima di ripristinare:

```bash
git fetch origin cloud-backup
git restore --source=FETCH_HEAD -- backups/
```

Se le parti sono distribuite con Git LFS, installare anche Git LFS sul computer e inizializzarlo prima del recupero. Se i file restano piccoli puntatori testuali anziché archivi reali:

```bash
git lfs install
git lfs pull origin cloud-backup --include='backups/**'
```

Lo script riconosce i puntatori LFS e mostra un messaggio esplicito invece di presentarli come archivi corrotti. Il manifest descrive i byte degli archivi reali.

Il secondo comando importa la directory dei backup nel working tree; **non esegue gli esperimenti e non pubblica modifiche**. Preservare eventuali modifiche locali personali nella directory `backups/` prima di usarlo. Il fetch del ramo può scaricare tutti gli archivi anche se poi si ripristina un solo componente.

## Elencare e ripristinare

Servono Python 3.12 e il comando `zstd`. La verifica dei file non dipende dai pacchetti della venv salvata. Per vedere i nomi dei file senza estrarli:

```bash
python scripts/restore_backups.py --component models --list
```

Per recuperare soltanto dati o modelli:

```bash
python scripts/restore_backups.py --component data
python scripts/restore_backups.py --component models
```

Per recuperare tutto in una directory vuota:

```bash
python scripts/restore_backups.py --component all --root /percorso/Humanization-ripristinato
```

Il programma rifiuta di sovrascrivere componenti già presenti. Per una sostituzione intenzionale aggiungere `--overwrite`: i componenti vengono prima verificati ed estratti in staging, poi sostituiti. Non usare l'opzione per una ripresa ordinaria senza aver conservato le modifiche personali. Il ripristino valida SHA-256, radici dei membri e tipi di file; non scrive attraverso symlink dell'archivio.

## Limite della venv salvata

L'ambiente salvato è **Linux x86_64, Python 3.12.14**. È uno snapshot completo dei pacchetti e dei corpora installati, ma **non è un ambiente portabile per macOS o Windows**. Non comprende l'interprete di sistema esterno: il symlink `.venv/bin/python3` punta al runtime cloud originale in `/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3`. Anche gli shebang dei comandi Python incorporano `/workspace/Humanization/.venv`.

Su un'altra macchina/percorso ricreare la venv con i lockfile e le istruzioni del README, riutilizzando dati e modelli ripristinati. Conservare lo snapshot venv è utile per l'inventario esatto e per recuperare dipendenze locali; estrarlo non garantisce che il suo interprete sia eseguibile.

## Ricreare gli archivi

```bash
python scripts/backup_workspace.py
```

Un componente esistente non viene sostituito senza `--overwrite`. È possibile selezionare `--component data|models|venv|cache`. Lo stream tar→zstd viene diviso direttamente nelle parti: non viene creato un secondo archivio enorme temporaneo. I file temporanei di una creazione riuscita vengono eliminati; se la creazione fallisce, gli elementi in `.build/` rimangono per la diagnosi. Il manifest propone anche gruppi di parti da massimo 450 MiB per caricamenti Git incrementali.
