# Kaggle: protocollo rivisto prima del pannello da 126

Importare **Humanization_Study_v2.ipynb** dal [pacchetto ZIP](Humanization_Study_v2.zip). Il notebook precedente rimane disponibile per riprodurre i pilot. [Protocollo completo](../STUDY_V2_PROTOCOL.md), [prova NLI già eseguita](../results_nli_context_pilot/report.md).

## Prima prova nella sessione attuale

1. Estrarre lo ZIP sul computer, aprire un nuovo notebook Kaggle e scegliere **File → Import Notebook**.
2. Attivare **T4 ×2**, Internet e il Secret **GROQ_API_KEY** per questo notebook. La chiave non va inserita nelle celle.
3. Lasciare `MODE = "smoke"` e `RUN_SUFFIX = "_protocol_v2"`. I cinque sistemi sono identità, baseline Qwen2.5-3B con prompt fissato, 14B NF4 con lo stesso prompt e le due configurazioni DIPPER XXL.
4. Eseguire le sezioni **1–3**. La terza verifica il modello Groq `qwen/qwen3.8-27b` nel catalogo autenticato e prova le opzioni senza ragionamento. Non sono ancora confermate disponibili: se l'ID o `reasoning_effort="none"` non sono accettati, il notebook si arresta senza cambiare modello o rimuovere il vincolo. L'errore può essere condiviso, la chiave no.
5. La prova del giudice usa i risultati dei vecchi pilot: 12 risposte valide per sei confronti AB/BA, con massimo 16 tentativi POST. Le 24 valutazioni dell'assistente sono un proxy AI; il confronto fra riscrittori usa 12 di quei testi su sei fonti. Non è una validazione umana.
6. Eseguire le sezioni **4–13**: tre fonti, 15 esiti di riscrittura attesi, controlli nulli, detector, Binoculars, stile, semantica e nuovi giudizi. La sezione 7 deve verificare il caricamento effettivo di Falcon su entrambe le T4: i test locali non lo sostituiscono.
7. Scaricare `humanization_v2_smoke_protocol_v2.zip` e restituirlo per la verifica del nuovo percorso GPU e del giudice.

Se la sessione Kaggle precedente è ancora attiva, la cache sotto `/kaggle/temp/humanization_runtime` riusa i checkpoint già scaricati. Il nuovo clone e le cartelle di output sono distinti; non modificare i file dei pilot.

## Dopo il collaudo: 126 fonti

Impostare soltanto:

```python
MODE = "study"
```

Rieseguire dall'inizio con lo stesso preset. Sono attesi **300 umani di calibrazione**, **126 fonti test** e **630 esiti fra i cinque sistemi**, inclusa l'identità. I controlli nulli sono condizioni ulteriori e non aumentano le fonti indipendenti. Il pannello è verificato contro l'hash già congelato; le consegne restano nell'analisi primaria.

Il file finale è `humanization_v2_study_protocol_v2.zip`. Il manifest della naturalezza va letto anche quando viene creato lo ZIP: rate limit e budget possono lasciare giudizi parziali, che rimangono mancanti. Le tre comparazioni nello study hanno al massimo 256 tentativi POST ciascuna, 768 in tutto; la prova iniziale ha budget separato. I token del provider vengono registrati, il costo monetario non viene inventato.

## Cosa cambia

- **Detector:** Binoculars aggiunge un metodo zero-shot. I tre RoBERTa e il loro voto correlato restano riportati separatamente, con soglie sugli umani.
- **Lunghezza:** il prefisso nullo ha lo stesso conteggio alfabetico della riscrittura e può essere una frase incompleta dichiarata. Gli output più lunghi dell'originale sono non eleggibili per quel controllo; nessun padding.
- **Rumore:** una sostituzione lessicale minima, con edit espliciti e nessuna garanzia gold di equivalenza semantica; anche combinata con il prefisso.
- **Humanizer:** 14B NF4, oltre alla baseline 3B. DIPPER XXL conserva sia il preset sul passaggio intero sia quello nuovo a tre frasi con contesto cumulativo. Quest'ultimo differisce comunque dal rilascio per segmentazione spaCy, greedy e budget per finestra.
- **NLI:** tutte le modalità rimangono distinte, con assenza di troncamenti e copertura registrata. Il documento intero è un confronto supplementare, non una certificazione.
- **Annotazione:** reference senza score e diagnostico arricchito, moduli indipendenti per due lettori e chiavi separate. Il campione scelto per score non è trattato come casuale.

## Risorse e limiti verificati

Binoculars usa due Falcon 7B non quantizzati, uno per GPU. Richiede circa 14 GiB liberi su ciascuna T4 come controllo preliminare; non viene cambiato in 4 bit per farlo entrare. DIPPER scarica circa 45 GB; il nuovo 14B richiede il proprio checkpoint. I subprocess rilasciano i modelli tra una fase e l'altra.

Il codice è verificato localmente con test matematici e di integrazione; nessun risultato Binoculars, 14B, DIPPER contestuale o Groq viene dichiarato prima dello smoke effettivo su Kaggle. La prova NLI contestuale è già stata eseguita su CPU con il modello base locale. Fast-DetectGPT e un humanizer commerciale restano rinviati. Il pannello da 420 non va avviato in questa fase.
