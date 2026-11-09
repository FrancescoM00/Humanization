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

Il preset è disabilitato nel config normale e viene attivato esplicitamente dalla selezione. Richiede circa **45 GB di download**; il controllo iniziale richiede circa 55 GiB liberi prima del primo caricamento e almeno 8 GiB di VRAM libera come controllo preliminare. La quantizzazione a 4 bit riduce la memoria del modello, non la dimensione degli shard da scaricare. Il controllo non garantisce assenza di OOM: se la sessione non ha risorse sufficienti, il preset si interrompe prima di promettere un risultato.

DIPPER lavora sull'intero passaggio breve senza prefisso di contesto: è una variante esplicita, non una replica della procedura ufficiale a finestre di tre frasi. DeBERTa NLI rimane base float32 anche quando il riscrittore DIPPER è quantizzato.

## 8. Limiti e problemi pratici

Il primo pilot completa l'applicazione degli assi evasione, significato e stile a riscritture prodotte dal sistema. **La naturalezza non è automaticamente validata:** i moduli manuali sono da compilare, e Flesch/PPL non equivalgono a naturalezza. Un giudice LLM separato richiede un protocollo e risorse propri.

La generazione e i detector salvano progressivamente. Rieseguire con gli stessi input riusa gli esiti disponibili; cambiare prompt, coorte, sistemi o codice richiede una directory nuova (`RUN_SUFFIX`). Un tentativo di generazione fallito resta tale nel run primario; un nuovo tentativo costituisce una nuova condizione documentata.

Se NLI esaurisce la memoria GPU, riduci `NLI_BATCH_SIZE` a 2 e riesegui il blocco. Per errori di download verifica Internet e il messaggio della cella. Il notebook interrompe gli errori di scoring: non li interpreta come classificazioni umane.

Metodo e formule: [PROTOCOL.md](PROTOCOL.md). Modelli, revisioni, licenze e fonti: [MODELS.md](MODELS.md).

## Verifiche prima della consegna

Sono passati 110 test locali. La ricostruzione dei 1.800 passaggi storici ha verificato gli identificatori di testo e ha prodotto coorti identiche a quelle ricavate dai campioni già salvati. Su tre identità reali sono stati eseguiti stile, guardie, metriche lessicali, embedding, BERTScore, NLI, similarità allineate ed esportazione continua, usando i modelli semantici già disponibili su CPU. Queste sono prove di integrazione: nessuna riscrittura Qwen/DIPPER né inferenza di detector reali è stata eseguita in questa preparazione, e il percorso GPU va verificato su Kaggle con lo smoke.
