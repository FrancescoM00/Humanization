# Avviare il pilot Humanization su Kaggle

Il notebook passa dai controlli artificiali alle riscritture prodotte da un sistema reale. Esegue un riscrittore con prompt fisso e tre detector, poi misura conservazione semantica e cambiamento stilistico con il codice già validato. Genera anche schede cieche per la valutazione manuale della naturalezza.

**File da importare:** [Humanization_Pilot.ipynb](Humanization_Pilot.ipynb). [Download diretto](https://raw.githubusercontent.com/FrancescoM00/Humanization/main/kaggle/Humanization_Pilot.ipynb).

Per un download unico: [pacchetto ZIP con notebook e guida](Humanization_Kaggle.zip). Estrarre lo ZIP sul computer e importare in Kaggle soltanto il file `.ipynb`.

## 1. Aprire il notebook

1. Accedi a [Kaggle Code](https://www.kaggle.com/code) e crea un **New Notebook**.
2. Dal menu **File → Import Notebook**, carica `Humanization_Pilot.ipynb`. Il file deve mantenere l'estensione `.ipynb`, non `.txt`.
3. Nelle impostazioni del notebook imposta **Accelerator → GPU**. T4 o P100 sono le prime opzioni considerate; se compare T4 ×2 il preset usa una sola GPU, non presume il raddoppio automatico della memoria.
4. Attiva **Internet**: servono GitHub, Hugging Face e le dipendenze. I checkpoint scelti sono pubblicamente accessibili, senza richiedere un token personale nel preset.
5. Mantieni il notebook e i suoi output privati durante il pilot.

## 2. Prima esecuzione: tre testi

Nella prima cella di codice lascia:

```python
MODE = "smoke"
RUN_SUFFIX = ""
ACTIVE_SYSTEMS = ["identity", "qwen_humanize"]
```

Esegui le celle dall'alto verso il basso. La prima stampa versione PyTorch, nome della GPU, spazio disponibile e commit del codice. Se CUDA non è disponibile, correggi l'acceleratore prima di proseguire. L'immagine deve avere PyTorch CUDA almeno 2.6 per caricare i pesi `.bin` di RADAR.

Il primo avvio scarica modelli e dipendenze; il tempo effettivo dipende da GPU e rete. Il notebook usa una venv temporanea che riutilizza PyTorch CUDA di Kaggle. **Non eseguire `scripts/setup.sh` né installare `requirements-semantic.txt` su Kaggle:** quei percorsi ricostruiscono l'ambiente CPU del cloud.

La prova produce tre riscritture Qwen e tre identità, una fonte per dominio. I 300 umani di calibrazione sono presenti anche nello smoke: le percentuali di evasione su tre testi sono soltanto controlli tecnici.

## 3. Cosa viene eseguito

| Blocco | Attività | Output principale |
| --- | --- | --- |
| Dati | Ricostruzione dei passaggi storici e split congelato | `cohorts/*.jsonl`, manifest |
| Riscrittura | Identità e Qwen con prompt fisso | `rewriting/rewrites.jsonl`, `pairs.jsonl` |
| Detector | Tre modelli, soglie umane separate, voto di maggioranza | `detection/detection_summary.csv` |
| Stile | Feature, quattro delta della shortlist, distanza RMS e POS-JS | `style/style_pairs.csv` |
| Significato | ROUGE/BLEU, coseno, BERTScore e NLI nelle due direzioni | `semantic/scores/*.csv` |
| Confronto equo | Similarità allineate con media, q10 e minimo | `aligned_cosine.csv`, `aligned_bertscore.csv` |
| Suite continua | Distribuzioni e metadati delle coppie | `continuous_suite/*.csv` |
| Revisione | Moduli ciechi per significato e naturalezza | `style/*_review_form.csv` |

Qwen2.5-3B-Instruct è una **baseline di humanization via prompt**: il sistema è definito da checkpoint, istruzione fissa e decoding. Produce nuove riscritture; non è un servizio commerciale né un modello specificamente addestrato a umanizzare. Il suo ruolo è quello di riscrittore, distinto dal giudice più grande suggerito dal professore per la naturalezza. Le licenze sono descritte in [MODELS.md](MODELS.md).

I detector sono OpenAI RoBERTa-base, Fakespot RoBERTa-base e RADAR RoBERTa-large. Hanno training differenti ma appartengono alla stessa famiglia architetturale: questo primo pool non rappresenta tutti i detector possibili. OpenAI è una baseline storica GPT-2; Fakespot è eseguito nella variante con input grezzo, senza il preprocessing specifico della sua model card.

## 4. Controllare la prova

- In `rewrites.jsonl` devono comparire gli esiti di entrambe le condizioni. Un risultato `truncated`, `input_too_long` o `generation_error` resta registrato e non diventa un successo.
- Per `identity`, originale e modificato devono coincidere, RMS e delta stilometrici devono essere zero. Il coseno e BERTScore devono essere prossimi a uno, entro l'aritmetica del modello. **NLI sull'identità non è imposto a uno.**
- L'ASR dell'identità è zero quando esiste almeno un originale riconosciuto AI. Se il denominatore è zero, l'ASR rimane mancante.
- Leggi qualche output Qwen: controlla che non sia una risposta con preambolo, una copia sistematica o una generazione interrotta. Non selezionare soltanto le riscritture che ottengono buoni score.

## 5. Passare al pilot su 42 testi

Dopo aver controllato lo smoke, cambia soltanto:

```python
MODE = "pilot"
```

Riparti dalla prima cella. I risultati vengono salvati in una directory distinta. Il pilot comprende **42 testi macchina: 2 per ciascuno dei 3 domini × 7 generatori**, più 42 umani appaiati di test. I 300 umani di calibrazione sono fonti diverse; la separazione vale anche rispetto alle 126 fonti riservate per l'eventuale studio successivo.

Il notebook aggiornato usa un clone di codice distinto per ogni modalità, riutilizzando la stessa cache temporanea di modelli e venv. **Se stai continuando nel notebook iniziale con le patch locali dello smoke**, cambia anche la riga `REPO` nella prima cella:

```python
REPO = Path("/kaggle/temp/Humanization_pilot")
```

Questo permette di scaricare il codice corretto senza sovrascrivere le patch del vecchio clone. Lascia `TMP` invariato. Esegui poi le celle dall'inizio; non occorre ripetere lo smoke. Il [report del primo smoke completo](https://github.com/FrancescoM00/Humanization/blob/main/results_kaggle_smoke/report.md) documenta i risultati e i limiti emersi.

Il [pilot su 42 fonti è ora completato](https://github.com/FrancescoM00/Humanization/blob/main/results_kaggle_pilot/report.md). Prima di ampliare il campione, sono disponibili [schede di annotazione](https://github.com/FrancescoM00/Humanization/blob/main/exports/kaggle_pilot/README.md): 24 coppie per la semantica e 28 testi per la naturalezza. Il confronto successivo consigliato è con un secondo sistema sulle stesse fonti; la sezione DIPPER qui sotto permette di iniziare dal suo smoke separato.

I testi sono i passaggi iniziali di 90–180 parole del precedente controllo a lunghezza esattamente abbinata. **Non sono documenti completi** e le lunghezze delle riscritture non vengono forzate a coincidere. Questo mantiene il primo esperimento contenuto e chiarisce a quali testi si riferiscono i risultati.

Il preset `MODE="study"` prepara 126 fonti, ma l'estensione va decisa dopo aver letto il pilot. Mantieni gli stessi prompt, detector e regole durante il confronto.

## 6. Salvare e scaricare

Alla fine il notebook crea:

```text
/kaggle/working/humanization_smoke.zip
/kaggle/working/humanization_pilot.zip
```

Scarica lo ZIP della modalità eseguita dal link dell'ultima cella o dal pannello dei file di output. Salva anche una versione privata del notebook con gli output disponibili; le opzioni di salvataggio di Kaggle possono avviare un nuovo run, quindi controlla se stai salvando la sessione completata oppure richiedendo un'altra esecuzione.

Lo ZIP conserva coorti, configurazione, commit, riscritture, score, soglie e moduli manuali. Modelli e venv restano sotto `/kaggle/temp` e non entrano nello ZIP. Dopo la chiusura della sessione potranno dover essere riscaricati.

Per l'analisi successiva, carica nella chat lo ZIP dei risultati, oppure almeno `detection_summary.csv`, `detection_macro_summary.csv`, `rewrites.jsonl`, `style_pairs.csv` e i punteggi continui. Questi file permettono di leggere insieme evasione e perdita di contenuto.

## 7. DIPPER come secondo sistema

Il runner supporta anche DIPPER. Per una condizione distinta imposta:

```python
MODE = "smoke"
RUN_SUFFIX = "_dipper"
ACTIVE_SYSTEMS = ["identity", "dipper"]
```

Nel vecchio notebook sostituire anche la riga `REPO`, più in basso nella stessa cella, con una definizione che segue modalità e suffisso:

```python
REPO = Path("/kaggle/temp") / f"Humanization_{MODE}{RUN_SUFFIX}"
```

Eseguire le sezioni 1–3: devono risultare 300 umani di calibrazione e tre fonti test. Il notebook aggiornato controlla anche il backend NF4 nella sezione 2. Per il notebook già aperto, prima della sezione 4 aggiungere questo controllo:

```python
quantization_probe = r"""
import torch
import bitsandbytes as bnb
layer = bnb.nn.Linear4bit(
    64, 64, bias=False, compute_dtype=torch.float16,
    compress_statistics=True, quant_type="nf4"
).to("cuda:0").eval()
x = torch.ones((1, 64), device="cuda:0", dtype=torch.float16)
with torch.inference_mode():
    y = layer(x)
torch.cuda.synchronize()
assert bool(torch.isfinite(y).all())
print("Backend CUDA 4-bit: OK")
"""
subprocess.run([str(PYTHON), "-c", quantization_probe], check=True)
```

Il test usa una piccola matrice, senza scaricare il modello, e verifica il backend bitsandbytes nella venv effettiva. Non dimostra ancora che l'intero DIPPER sia caricabile o che generi correttamente: a questo serve lo smoke. Se il test fallisce, esaminare l'errore prima del grande download.

Se il controllo riesce, eseguire le sezioni 4–9 come per Qwen. Il file finale sarà **`humanization_smoke_dipper.zip`**. La condizione di generazione deve essere `dipper`, selezionata da `ACTIVE_SYSTEMS` attraverso gli argomenti `--only`; non serve modificare `enabled` nel config. La cartella di output con suffisso `_dipper` conserva i risultati Qwen.

Il preset è disabilitato nel config normale e viene attivato esplicitamente dalla selezione. Richiede circa **45 GB di download**; il controllo iniziale richiede circa 55 GiB liberi prima del primo caricamento e almeno 8 GiB di VRAM libera come controllo preliminare. La quantizzazione a 4 bit riduce la memoria del modello, non la dimensione degli shard da scaricare. Il controllo non garantisce assenza di OOM: se la sessione non ha risorse sufficienti, il preset si interrompe prima di promettere un risultato.

DIPPER lavora sull'intero passaggio breve senza prefisso di contesto: è una variante esplicita, non una replica della procedura ufficiale a finestre di tre frasi. DeBERTa NLI rimane base float32 anche quando il riscrittore DIPPER è quantizzato.

## 8. Limiti e problemi pratici

Il primo pilot completa l'applicazione degli assi evasione, significato e stile a riscritture prodotte dal sistema. **La naturalezza non è automaticamente validata:** i moduli manuali sono da compilare, e Flesch/PPL non equivalgono a naturalezza. Un giudice LLM separato richiede un protocollo e risorse propri.

La generazione e i detector salvano progressivamente. Rieseguire con gli stessi input riusa gli esiti disponibili; cambiare prompt, coorte, sistemi o codice richiede una directory nuova (`RUN_SUFFIX`). Un tentativo di generazione fallito resta tale nel run primario; un nuovo tentativo costituisce una nuova condizione documentata.

Se NLI esaurisce la memoria GPU, riduci `NLI_BATCH_SIZE` a 2 e riesegui il blocco. Per errori di download verifica Internet e il messaggio della cella. Il notebook interrompe gli errori di scoring: non li interpreta come classificazioni umane.

### Errore `ensurepip` nella creazione della venv

Alcune immagini Kaggle, inclusa quella Python 3.13 segnalata durante il primo avvio, non completano il bootstrap `ensurepip`. Il notebook aggiornato crea la venv con `--without-pip`, abilita esplicitamente i pacchetti di sistema e usa il pip dell'immagine per installare pip nella sola venv. Questo conserva PyTorch CUDA preinstallato. La correzione gestisce anche una venv parziale, che può già contenere `bin/python`.

Se hai già importato la versione iniziale, aggiungi una cella ed esegui:

```python
import sys, subprocess
from pathlib import Path

VENV = Path("/kaggle/temp/humanization_runtime/venv")
PYTHON = VENV / "bin/python"
subprocess.run([sys.executable, "-m", "venv", "--without-pip",
                "--system-site-packages", str(VENV)], check=True)
subprocess.run([sys.executable, "-m", "pip", "--python", str(PYTHON),
                "install", "--upgrade", "--ignore-installed", "pip"], check=True)
```

Poi riesegui la seconda cella originale. La sessione corrente può essere mantenuta; il controllo finale deve stampare `CUDA nella venv` con la versione CUDA di Kaggle.

### Messaggi di conflitto con TensorFlow, Gradio e pacchetti Google

Con `--system-site-packages`, pip vede anche i pacchetti preinstallati nell'immagine. Il progetto mantiene versioni compatibili con Transformers 4.57, incluse `huggingface-hub` 0.x e protobuf 5; altre applicazioni dell'immagine possono richiedere versioni diverse. Se pip termina e la cella stampa `Configurazione salvata`, il messaggio del resolver non è di per sé un fallimento dell'installazione.

Il pilot usa PyTorch; TensorFlow, Gradio, Diffusers e i servizi Google elencati non ne fanno parte. Non si devono aggiornare le dipendenze per soddisfare indiscriminatamente tutti questi pacchetti. Il notebook aggiornato imposta `USE_TF=0`, `USE_FLAX=0`, `USE_TORCH=1` nei processi figli e verifica gli import concreti di Qwen, RoBERTa, DeBERTa, T5, Sentence Transformers e BERTScore, oltre a spaCy, textstat e un piccolo prodotto matriciale CUDA FP16. Non scarica checkpoint per questa verifica.

Se usi la versione già aperta, imposta queste variabili in una cella prima di proseguire:

```python
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
```

Il controllo va eseguito con `PYTHON` della venv in un nuovo subprocess: gli import del kernel Kaggle possono riferirsi a versioni diverse. Un eventuale errore negli import necessari va esaminato prima delle generazioni. `pip check` sull'intera immagine ereditata non è il criterio di avvio della suite, perché include anche le applicazioni estranee al pilot.

### Errore `Expected a torch.device with a specified index ... cuda`

Il runner iniziale passava l'alias `cuda` a `torch.cuda.set_device`, che richiede un indice esplicito. Il codice aggiornato risolve l'alias nella GPU corrente e il notebook passa esplicitamente `cuda:0` ai quattro CLI che eseguono modelli.

Per continuare sul clone già aperto, sostituisci la cella della sezione 4 con:

```python
REWRITES = RUN / "rewriting_cuda0"
selected_args = [item for name in ACTIVE_SYSTEMS for item in ("--only", name)]
run("run_humanizers.py", "--cohort", COHORT, "--config", CONFIG,
    "--output", REWRITES, "--device", "cuda:0", *selected_args)
rewrites = read_jsonl(REWRITES / "rewrites.jsonl")
```

La directory nuova conserva il tentativo precedente e registra il dispositivo corretto. Cambiare dispositivo nella vecchia directory farebbe intervenire il controllo di compatibilità del manifest. Le identità vengono ricreate immediatamente, mentre Qwen riparte dal caricamento; le sezioni successive seguono la variabile `REWRITES`. Il rimedio non richiede di modificare i metadati dei risultati esistenti o reinstallare l'ambiente.

### BERTScore: tensori su `cuda:0` e `cpu`

BERTScore 0.3.13 può restituire gli embedding sulla GPU e i pesi IDF dei token sulla CPU. Il calcolo del matching deve spostare anche questi pesi sul dispositivo degli embedding. La correzione è applicata sia al percorso documentale sia al BERTScore allineato, lasciando invariati definizione della metrica e comportamento CPU.

Per la sessione già aperta, aggiungi una cella prima della sezione 7:

```python
snapshot = RUN / "patched_sources"
snapshot.mkdir(exist_ok=True)
for name, vector_name in {
    "metrics.py": "vector",
    "aligned_similarity.py": "vec",
}.items():
    path = REPO / "semantics" / name
    code = path.read_text()
    old = "weight = weights[i, :length].clone()"
    new = f"weight = weights[i, :length].to(device={vector_name}.device).clone()"
    assert old in code or new in code, f"Versione inattesa: {name}"
    code = code.replace(old, new)
    path.write_text(code)
    (snapshot / name).write_text(code)
    print(name, "corretto")
```

Poi riesegui la sezione 7. Gli stadi lessicali e gli embedding già completati vengono riusati, come i pesi scaricati. Le copie in `patched_sources` conservano nello ZIP i sorgenti effettivamente usati nel run; i metadati degli stadi registrano gli hash del codice. Non occorre rifare riscritture o detector.

Il warning sui pesi `pooler.dense.*` di RoBERTa è distinto: il pooler non è usato per gli embedding token di BERTScore e non richiede addestramento per questo esperimento.

Metodo e formule: [PROTOCOL.md](PROTOCOL.md). Modelli, revisioni, licenze e fonti: [MODELS.md](MODELS.md).

## Verifiche prima della consegna

Sono passati 110 test locali. La ricostruzione dei 1.800 passaggi storici ha verificato gli identificatori di testo e ha prodotto coorti identiche a quelle ricavate dai campioni già salvati. Su tre identità reali sono stati eseguiti stile, guardie, metriche lessicali, embedding, BERTScore, NLI, similarità allineate ed esportazione continua, usando i modelli semantici già disponibili su CPU. Queste sono prove di integrazione: nessuna riscrittura Qwen/DIPPER né inferenza di detector reali è stata eseguita in questa preparazione, e il percorso GPU va verificato su Kaggle con lo smoke.
