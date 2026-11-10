# Kaggle: scelta della modalità NLI sul solo split di calibrazione

Questo passaggio usa **139 positivi presunti, 297 alterazioni e 137 identità**, non il test storico. [Specifica e regola di scelta](../NLI_CALIBRATION.md). Nessuna riscrittura o chiamata Groq viene eseguita.

## Nella sessione Kaggle

Usare il notebook corrente con GPU e Internet attivi. **Aggiungere una nuova cella alla fine**, senza importare un altro notebook o ripetere lo study, e incollare:

```python
from pathlib import Path
import runpy, urllib.request

launcher = Path("/kaggle/working/run_nli_calibration.py")
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/FrancescoM00/Humanization/"
    "main/kaggle/run_nli_calibration.py",
    launcher,
)
runpy.run_path(str(launcher), run_name="__main__")
```

Il launcher scarica il pacchetto congelato, usa una cartella nuova `/kaggle/working/nli_calibration_v1` e riusa la venv e DeBERTa-v3 base della sessione se disponibili. Non modifica la vecchia venv leggibile. Se la sessione precedente non è più disponibile, prepara un ambiente minimo che riusa PyTorch CUDA e scarica il solo checkpoint NLI fissato. Serve una GPU; non passa silenziosamente alla CPU.

## Tempo, ripresa e file da restituire

Sono previste **323.336 evidenze uniche**. La stima è circa **45–90 minuti**, oltre all'eventuale setup e download. La preparazione del piano precede il progresso delle inferenze e può rimanere senza nuovi messaggi per alcuni minuti. Lo scorer stampa poi il numero di confronti completati.

Terminato lo scoring, lo script confronta i 16 candidati e mostra la selezione oppure dichiara che nessuno supera i criteri. Scaricare e restituire **`nli_calibration_v1.zip`**, creato in `/kaggle/working`. Solo allora saranno disponibili i nuovi risultati da interpretare.

Lo ZIP è leggero: include input, codice del pacchetto, policy, metadati, CSV e analisi. Esclude i temporanei, la cache delle probabilità e il JSONL delle evidenze estese; questi ultimi rimangono nella cartella del run per riprendere e non vengono cancellati. Pesi e venv sono esterni alla cartella dei risultati.

Se l'esecuzione si interrompe, la stessa cella riusa il pacchetto già congelato e gli score completati. Non cancellare il RUN per riprovare. Per un esperimento deliberatamente nuovo usare un suffisso esplicito, non sovrascrivere gli input o la policy del run esistente.

Le [84 coppie reference da annotare](../exports/kaggle_v2_study/reference/README.md) sono un'attività distinta. Non aprire le chiavi e non usare le nuove etichette per ritoccare la scelta NLI.
