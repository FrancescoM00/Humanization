# Kaggle: protocollo v2 e pannello da 126 fonti

Importare **Humanization_Study_v2.ipynb** dal [pacchetto ZIP](Humanization_Study_v2.zip). Il [protocollo](../STUDY_V2_PROTOCOL.md) fissa condizioni e coorte; il [report dello smoke v2](../results_kaggle_v2_smoke/report.md) descrive il collaudo completato. I notebook precedenti restano disponibili per riprodurre i pilot.

**Run ricevuto il 10 novembre 2026:** il [report delle 126 fonti](../results_kaggle_v2_study/report.md) contiene gli esiti di tutti i preset e lo scoring; i tre confronti Groq sono parziali. Non occorre rilanciare i modelli per leggere o annotare i risultati. Le istruzioni seguenti servono alla riproduzione.

## Riprodurre il pannello da 126 fonti

1. Nella sessione Kaggle che ha completato lo smoke, aggiornare il notebook con questo file. Conservare le cartelle e la cache esistenti: non arrestare o azzerare la sessione per liberare spazio.
2. Confermare **T4 ×2**, Internet e il Secret **GROQ_API_KEY** abilitato per il notebook. Non incollare la chiave nelle celle.
3. Nella prima cella impostare `MODE = "study"`. Lasciare `RUN_SUFFIX = "_protocol_v2"`, i cinque sistemi e gli altri preset invariati.
4. Eseguire le sezioni **1–3**. Nella sessione precedente, che ha il runtime di sola lettura, devono comparire `riuso locale: True` e `Riutilizzo venv senza modificarla`. Il codice viene clonato sotto `/kaggle/working/.humanization_code/`, separatamente dal vecchio checkout. La prova Groq sui sei confronti precedenti viene riusata quando il piano coincide esattamente; non occorre ripeterne i POST.
5. Eseguire le sezioni **4–13**. Sono attesi **300 umani di calibrazione**, **126 fonti test** e **630 esiti fra i cinque sistemi**, inclusa l'identità. I controlli nulli sono condizioni ulteriori sulle stesse fonti.
6. Scaricare e restituire `humanization_v2_study_protocol_v2.zip`. I risultati dello smoke rimangono in una cartella distinta.

Il notebook usa la coorte già congelata e conserva le consegne nel risultato primario. Non serve ripetere lo smoke nella sessione che lo ha già concluso. Il passaggio a 420 fonti resta rinviato.

Se esiste già un risultato study con un commit diverso, il controllo lo segnala: scegliere un suffisso nuovo per quel run; non sovrascrivere la cartella esistente. Una nuova sessione Kaggle può perdere le cache temporanee: in quel caso il notebook torna alla preparazione con download e va nuovamente verificata la disponibilità delle risorse prima dello study.

## Riutilizzo del runtime di sola lettura

I log dello smoke hanno mostrato `/kaggle/temp/humanization_runtime` e le directory temporanee standard non scrivibili. Il notebook aggiornato distingue gli asset leggibili dalle nuove scritture:

- Il vecchio runtime rimane una sorgente di venv, checkpoint, CSV e corpus NLTK. Una venv con hash delle dipendenze corrispondente e pip funzionante viene riutilizzata senza ricrearla, reinstallare pacchetti o riscriverne il file di stato.
- Clone, risultati e temporanei usano `/kaggle/working`. `TMPDIR`, `TMP` e `TEMP` puntano a `RUN / "_tmp"`; cache di configurazione e altre risorse piccole sono esterne alla cartella dei risultati.
- `REUSE_CACHED_MODELS = None` attiva automaticamente il riuso locale quando trova il vecchio runtime non scrivibile, la venv e una cache popolata. In questa modalità Hugging Face lavora offline: i riscrittori richiedono pesi e tokenizer alla revisione fissata, senza download impliciti. Binoculars ricerca anche i due checkpoint nelle cache usate per la ripresa dello smoke. Groq continua a usare la propria API, indipendente dal flag offline di Hugging Face.
- Se manca un checkpoint in modalità locale, fermarsi e leggere il messaggio. Impostare `REUSE_CACHED_MODELS = False` consente i download soltanto verso una cache scrivibile; richiede spazio sufficiente. Non trasformare una cache di sola lettura in destinazione dei download. Una sessione nuova con runtime temporaneo scrivibile è il percorso per ricostruire gli asset pesanti quando quelli precedenti non sono disponibili.
- `cmudict` viene prima cercato nei percorsi NLTK esistenti. CSV e riferimenti dei pilot vengono riusati se presenti; i file mancanti vengono scaricati in un percorso scrivibile.

La sezione 9 conserva i circa 1 GB di modelli semantici in `/kaggle/working/.humanization_semantic_models`, fuori dallo ZIP. Riusa quanto già presente e copia gli asset leggibili dal vecchio `TMP/models` quando necessario. Le revisioni previste vengono verificate dal downloader. Le directory di pesi e la venv non entrano nell'archivio finale.

Se la sezione 7 supera il controllo iniziale della cache ma fallisce poi con `http_get → temp_file.write → Read-only file system`, dopo aver completato l'observer, seguire la [ripresa con cache di scrittura esplicita](../docs/BINOCULARS.md#study-controllo-iniziale-riuscito-scrittura-http-poi-rifiutata). È il caso osservato nello study: una nuova cella forza il solo download mancante sotto `/kaggle/working`, poi si riesegue **soltanto la sezione 7**. Non reimportare il notebook né rifare le riscritture.

## Giudice e budget

Lo smoke ha confermato la disponibilità su Groq di `qwen/qwen3.8-27b` e l'opzione `reasoning_effort="none"`. Il modello non viene sostituito implicitamente. I sei confronti iniziali con le 24 valutazioni dell'assistente sono **concordanza AI–AI, non validazione umana**; il notebook ne confronta il piano completo prima di riusare i risultati.

Le tre nuove comparazioni study hanno al massimo **256 tentativi POST ciascuna, 768 in tutto**, inclusi i retry. La prova iniziale, se non già disponibile con piano identico, ha budget separato di 16. Vengono registrati i token; il costo monetario non viene inferito da prezzi sconosciuti.

Nello smoke quattro dei nove nuovi confronti hanno cambiato preferenza invertendo l'ordine, scegliendo sempre la posizione B. Le preferenze discordanti restano mancanti nel riepilogo primario: non sono una classifica affidabile di naturalezza. Il manifest va letto anche quando viene creato lo ZIP; rate limit e budget possono lasciare giudizi parziali. Le annotazioni umane rimangono necessarie.

La sezione 12 aggiornata usa `--failure-policy report`: coppie con generazioni fallite o vuote non vengono inviate al giudice, restano nel report e non sono rimpiazzate. Il denominatore selezionato resta distinto dalle coppie giudicabili. Se il notebook già avviato si ferma con `A failed generation cannot be silently omitted or judged as natural`, usare la [cella di ripresa della sola sezione 12](../docs/NATURALNESS_JUDGE.md#generazioni-fallite-nello-study-riprendere-la-sola-sezione-12), poi la sezione 13. Non rigenerare i testi né ricominciare il run.

## Condizioni e limiti

- **Detector:** Binoculars aggiunge un metodo zero-shot; i tre RoBERTa e il loro voto correlato restano separati. Le soglie sono calibrate sugli stessi 300 umani.
- **Controlli nulli:** prefisso con uguale conteggio alfabetico, rumore lessicale minimo e loro combinazione. I tagli possono terminare una frase; output più lunghi e sostituzioni impossibili sono marcati non eleggibili. Nessun padding e nessuna equivalenza semantica gold attribuita al rumore.
- **Humanizer:** baseline Qwen2.5-3B, 14B NF4 con lo stesso prompt e due preset DIPPER XXL. Il preset a tre frasi usa contesto cumulativo e differisce dal rilascio per segmentazione spaCy, generazione greedy e budget per finestra.
- **NLI:** DeBERTa-v3 base float32. Premesse singole, finestre fino a due/tre unità e documento intero restano modalità distinte, con copertura registrata. Il documento viene usato solo quando entra nel budget: nessun troncamento silenzioso o soglia semantica trasferita.
- **Annotazione:** reference scelto senza score e diagnostico arricchito, moduli per due annotatori e chiavi separate. Il campione scelto per score non stima ingenuamente l'accuratezza nella popolazione.

Binoculars usa due Falcon 7B FP16, uno per T4, con statistiche FP32. DIPPER richiede circa 45 GB di download; il 14B ha un proprio checkpoint. I subprocess rilasciano i modelli fra le fasi. Lo smoke ha verificato l'esecuzione reale di queste condizioni su T4 ×2, senza dimostrarne la qualità generale. Fast-DetectGPT e un humanizer commerciale restano rinviati.
