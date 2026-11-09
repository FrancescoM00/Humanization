# Binoculars prima dello studio a 126 fonti

Il runner `scripts/run_zero_shot_detectors.py` aggiunge un detector zero-shot basato su due modelli causali, distinto dai tre classificatori RoBERTa. Non addestra un classificatore sui dati della tesi. La soglia è tuttavia calibrata sugli stessi umani del progetto: il metodo di scoring è zero-shot, la decisione è calibrata sul dominio di riferimento.

Fonti consultate: [repository ufficiale](https://github.com/ahans30/Binoculars), commit `c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8`; [detector.py](https://github.com/ahans30/Binoculars/blob/c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8/binoculars/detector.py), [metrics.py](https://github.com/ahans30/Binoculars/blob/c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8/binoculars/metrics.py); Hans et al., *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*, [arXiv:2401.12070](https://arxiv.org/abs/2401.12070).

## Modelli e configurazione

| Ruolo | Checkpoint pubblico | Revisione fissata |
| --- | --- | --- |
| Observer | `tiiuae/falcon-7b` | `ec89142b67d748a1865ea4451372db8313ada0d8` |
| Performer | `tiiuae/falcon-7b-instruct` | `8782b5c5d8c9290412416618f36a133653e85285` |

Entrambi i checkpoint risultano pubblici e con licenza Apache-2.0 al momento della verifica. Si caricano i file safetensors con l'implementazione Falcon nativa di Transformers, senza codice remoto. I due tokenizer devono avere identici vocabolari/indici e produrre gli stessi token per ogni testo. Non si applica un template chat.

Aggiungere al file di configurazione una lista separata:

```json
"zero_shot_detectors": [{
  "name": "binoculars_falcon7b",
  "kind": "binoculars",
  "observer_model_id": "tiiuae/falcon-7b",
  "observer_revision": "ec89142b67d748a1865ea4451372db8313ada0d8",
  "performer_model_id": "tiiuae/falcon-7b-instruct",
  "performer_revision": "8782b5c5d8c9290412416618f36a133653e85285",
  "precision": "float16",
  "max_tokens": 512,
  "positions_per_batch": 32
}]
```

## Score e calibrazione

Il numeratore è la media della negative log-likelihood assegnata dal performer ai token effettivamente osservati, con lo shift causale. Il denominatore è la media dell'entropia incrociata fra le distribuzioni dell'observer e del performer: aspettativa sotto l'observer del negativo del logaritmo della probabilità del performer. Il rapporto viene conservato in `score`: **valori più bassi indicano maggiore compatibilità AI**, non una probabilità di origine.

Si mantiene una convenzione del codice ufficiale: il numeratore usa i token causali da predire, mentre il denominatore comprende tutte le posizioni reali, inclusa l'ultima. Si usano maschere di attenzione; un token EOS realmente presente non viene eliminato soltanto perché ha lo stesso ID usato per il padding. Il runner lavora su un testo alla volta e accumula le statistiche in FP32, anche con pesi FP16.

La soglia pubblicata dal repository, selezionata usando BF16, **non viene riutilizzata**. Il runner sceglie il quantile inferiore dei 300 rapporti umani: AI soltanto se `score < threshold`, con pareggi umani. Il FPR empirico di calibrazione non supera il 5%; non è una garanzia per nuovi dati. Si riportano FPR test, TPR degli originali, ASR sul sottoinsieme baseline-AI, TPR prima/dopo e intervalli bootstrap per fonte, mantenendo fisse calibrazione e modelli.

I rapporti non vengono convertiti in probabilità o mediati con gli score RoBERTa. Il vecchio voto di maggioranza dei tre classificatori resta una sintesi separata: questo runner non crea automaticamente una nuova maggioranza e non presume indipendenza fra detector.

## Contesto e testi lunghi

Il limite configurato è di 512 token per finestra. Il codice ufficiale tronca a quel limite; qui il testo viene coperto interamente, senza tagli silenziosi. Finestre successive condividono un token di contesto: ogni transizione osservata viene valutata una sola volta, e la posizione di confine nel denominatore viene contata una volta sola. Il rapporto documentale è il rapporto delle due medie globali sui rispettivi token, non la media dei rapporti delle finestre. Il contesto viene reimpostato a ogni finestra e i relativi conteggi sono esportati.

Questa estensione, l'uso FP16 su T4, le statistiche FP32 e la calibrazione del progetto sono dichiarati dal protocollo `binoculars_fullcoverage_fp32_statistics_v1`. I risultati si riferiranno a questa condizione; non sono una replica numerica della soglia globale BF16 del lavoro originale.

## Avvio Kaggle

```bash
python scripts/run_zero_shot_detectors.py \
  --cohort /percorso/cohorts/study.jsonl \
  --rewrites /percorso/rewriting/rewrites.jsonl \
  --config /percorso/config.json \
  --output /percorso/zero_shot \
  --observer-device cuda:0 --performer-device cuda:1
```

Servono **due GPU distinte**, una per Falcon 7B non quantizzato. Il controllo preliminare richiede almeno 14 GiB liberi su ciascuna; è un controllo preliminare, non una garanzia contro OOM. Entrambi i modelli devono restare caricati nello stesso momento. La cache Hugging Face richiede spazio per due checkpoint 7B; liberare gli altri modelli GPU prima di questo blocco. Il runner usa FP16 sulle T4 e non installa il vecchio ambiente del repository upstream. Non cambia automaticamente precisione, modelli o quantizzazione per far passare il caricamento.

Gli output sono `zero_shot_scores.csv`, `zero_shot_thresholds.json`, `zero_shot_summary.csv`, `zero_shot_run.json` e la cache incrementale per testo. Identità e testi ripetuti riusano lo score esatto. Una ripresa recupera gli score completati; `--analysis-only` ricalcola solo le statistiche da un run concluso. Un errore di scoring interrompe il run e non diventa una classificazione umana. Usare un output nuovo se cambiano input o configurazione.

## Verifiche e limite attuale

I test locali verificano formula e orientamento, maschere, copertura delle finestre, quantile con pareggi e denominatore ASR con generazioni fallite. Non scaricano Falcon e non sostituiscono la prova reale di caricamento/inferenza su due T4. Finché quella prova non è stata eseguita e restituita, non ci sono risultati Binoculars da presentare.

Fast-DetectGPT è un'altra opzione zero-shot, con statistica e checkpoint propri. Aggiungerlo richiede una condizione separata, calibrazione e verifica numerica; non è implementato in questo blocco. La priorità operativa è completare e verificare Binoculars prima dello studio esteso.

## Download Xet: filesystem di sola lettura su Kaggle

Nel primo smoke v2 il Falcon observer è stato scaricato e caricato; il download del performer è fallito in `hf_hub_download → xet_get → download_files` con `Read-only file system (os error 30)`. Non è un errore CUDA/OOM e l'avviso `torch_dtype` è una deprecazione distinta. Dal solo traceback non si identifica il percorso interno di Xet che rifiuta la scrittura.

Il notebook aggiornato imposta `HF_HUB_CACHE`, `HF_XET_CACHE` e `TMPDIR` sotto `/kaggle/temp/humanization_runtime`, verifica che siano scrivibili e usa `HF_HUB_DISABLE_XET=1`. Le variabili devono essere presenti **prima degli import Hugging Face**: il subprocess della sezione 7 le rilegge a ogni avvio. Si conserva la cache dei file completi; si possono rimuovere soltanto i `.incomplete` del performer interrotto per evitare il riuso di un file parziale Xet durante il passaggio a HTTP.

La sessione attuale può impostare queste variabili in una cella nuova e poi rieseguire solo la sezione 7. Non cambiano checkpoint, precisione, formula del ratio o calibrazione. Il download HTTP standard deve ancora completarsi sul runtime dell'autore; non è già dichiarato risolto dal test locale del notebook.

### Cache originale confermata di sola lettura: ripresa dedicata

La successiva prova di scrittura dell'autore fallisce direttamente in `/kaggle/temp/humanization_runtime/huggingface/hub`: usare lo stesso percorso non risolve il problema, anche con Xet disabilitato. Il loader aggiornato cerca uno spazio scrivibile distinto, prima sotto `/kaggle/temp/humanization_downloads_http`, poi `/tmp/humanization_downloads_http`, infine `/kaggle/working/.humanization_downloads_http`.

Prima di scaricare controlla le snapshot alle revisioni fissate: una configurazione e **tutti** gli shard safetensors devono essere presenti. Una snapshot completa viene letta direttamente dalla cache precedente, anche se il filesystem è di sola lettura; i `.incomplete` non valgono come pesi completi. Non viene ricopiato il primo Falcon. I file mancanti vanno nella nuova cache, dopo una prova di scrittura e spazio libero. Non si cambiano permessi al mount e non si cancellano i vecchi pesi.

Il [pacchetto di ripresa](../kaggle/Binoculars_Resume.zip) contiene il runner e i suoi tre file di supporto. Può essere estratto sotto il RUN in `/kaggle/working`, così la correzione funziona anche se il checkout precedente non è modificabile. La stessa sezione viene eseguita indicando coorte, controlli, config e directory di output esistenti. Il codice corretto e il suo hash sono inclusi negli output.

Con due checkpoint mancanti il fallback su `/kaggle/working` non viene scelto, perché circa 28 GB di pesi superano la consueta quota di output di 20 GB. Quando il primo è già completo, il secondo richiede circa 14 GB. I controlli locali verificano la selezione del filesystem e il rifiuto delle snapshot parziali; la ripresa sul runtime Kaggle resta da confermare dall'esecuzione.
