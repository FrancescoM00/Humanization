# Integrità dello smoke Kaggle v2

Audit sui soli file ricevuti, senza nuove generazioni, inferenze neurali o chiamate API. Archivio `humanization_v2_smoke_protocol_v2.zip`, SHA-256 `e6ae84b78019542dafb50297768e4e41db7d57949a7201ac4529e68361f603d8`, 1.914.682 byte compressi, 139 elementi comprese le directory e 6.300.718 byte non compressi.

## Esito e copertura

**Lo ZIP documenta il completamento tecnico del nuovo percorso su tre fonti.** Non emergono file mancanti o incoerenze nei riepiloghi ricalcolati. La maggiore fragilità scientifica riguarda i giudizi di naturalezza rispetto all'ordine; il passaggio alle 126 richiede inoltre di conservare le correzioni operative per i filesystem di sola lettura.

- Runtime dichiarato: commit iniziale `c92e7c238401847ac257b298dae6d6a4f9875365`, PyTorch `2.11.0+cu130`, due Tesla T4. Le correzioni applicate durante il run sono riconoscibili dagli hash effettivi e dai sorgenti Binoculars inclusi.
- Configurazione SHA-256 `947f1aaee4739d3ed5929b05a08884027b4ce5438e4cf3c7c47d023ac3ef65cd`, uguale al config v2 pubblicato. Manifest runtime, riscrittura e detection rimandano agli stessi input.
- Pannelli smoke, pilot e study identici byte per byte a quelli già congelati. Lo smoke contiene 300 gruppi di calibrazione umana e tre gruppi test disgiunti: una fonte ChatGLM per dominio.
- Quindici esiti unici, tutti `ok`: tre identità e tre riscritture per ciascuna delle quattro condizioni. Gli originali coincidono con la coorte. Le tre riscritture della baseline 3B, quelle DIPPER sul passaggio intero e le identità conservano i testi dei precedenti smoke.
- `rewriting/pairs.jsonl`, `semantic/pairs.jsonl` e `semantic_context/pairs.jsonl` coincidono byte per byte. Il manifest registra una ripresa con nove esiti già completati; il file finale non contiene duplicati o esiti falliti.
- La configurazione DIPPER contestuale completa 3 finestre in Essay, 2 in Reuters e 3 in WritingPrompts; tutti i relativi esiti raggiungono EOS. I prompt per finestra registrano rispettivamente 107/178/191, 117/207 e 92/130/188 token, sotto il limite di 512.

## Detector e controlli nulli

Sono presenti 42 righe nel file congiunto di riscritture e controlli: 14 condizioni per ciascuna delle tre fonti. Ogni detector riceve 348 record: 300 umani di calibrazione, tre umani test, tre originali macchina e 42 esiti. I tre classificatori hanno quindi 1.044 score; Binoculars ne ha 348.

Rieseguendo soltanto i riepiloghi sugli score salvati:

- le 224 righe di `detection_summary.csv`, le 56 righe macro e tutte le soglie dei classificatori coincidono entro 10⁻¹⁰ nella lettura CSV;
- le 56 righe di `zero_shot_summary.csv` e la soglia Binoculars coincidono alla stessa tolleranza;
- le 40 righe di `null_control_comparison.csv`, con orientamento AI basso per Binoculars, coincidono con il ricalcolo.

Le soglie RoBERTa sono identiche a quelle dei pilot precedenti. Binoculars dichiara AI quando il ratio grezzo è **minore di 0,9046559856180912**, con 15 falsi positivi sui 300 umani di calibrazione. Questo 5% è empirico sulla calibrazione, non una garanzia di popolazione. Tutti e quattro i detector riconoscono i tre originali macchina e nessuno dei tre umani test; l'assenza di falsi positivi su tre testi non stima un FPR nullo.

La cache Binoculars conclusa contiene 342 testi distinti per 348 record, riusando i testi identici. È conservata anche una cache del precedente tentativo interrotto, con solo header e zero score: non entra nei riepiloghi conclusi. Il ratio non è convertito in probabilità e la maggioranza resta quella separata dei tre classificatori RoBERTa.

## Semantica, stile e provenienza del codice

Tutti i sei stadi semantici risultano `executed` su esattamente gli stessi 15 ID: lessicale, embedding, BERTScore, NLI, coseno allineato e BERTScore allineato. Gli hash delle coppie sono coerenti. Stile, guardie ed esportazione continua contengono anch'essi 15 righe con gli stessi ID. Le scale stilometriche sono identiche byte per byte a quelle del pilot precedente e non usano macchine o test per la stima.

Il percorso contestuale risulta eseguito su 15 coppie e 3.594 confronti unici, con DeBERTa-v3 base float32. Le 120 righe sono 15 coppie × due direzioni × quattro modalità; in questo smoke tutte hanno copertura completa. Questo non garantisce la stessa copertura sui 126 o sui documenti più lunghi.

Identità: RMS e variazioni stilometriche nulle, rapporto parole uno, coseno e BERTScore prossimi a uno entro il roundoff. NLI identità non è imposto a uno: medie forward/reverse circa 0,9879, 0,9915 e 0,9946, uguali nelle due direzioni.

Il modello NLI risulta caricato dalla cache scrivibile `/kaggle/working/.humanization_semantic_models/nli`; l'header della cache registra CUDA e float32. I pesi non sono nello ZIP, quindi l'audit non ne ricontrolla indipendentemente i byte.

Il sorgente Binoculars incluso in `binoculars_resume/humanizers/zero_shot.py` ha hash `63e900b9820e79269932ccc521886312ebc296d436fa7d02d6bfc4b3c8289805`, uguale al manifest e al codice pubblicato della correzione. Anche gli hash di riscrittura, metriche storiche, similarità allineate e NLI contestuale coincidono con i sorgenti corrispondenti. Il client Groq registra `4eaa28671a2c1fcb5e3cb1682fd7bab2d3de1742c04bfa05f418433b8867229b`, quello della correzione con requests; non si presume che il solo commit iniziale descriva ogni patch del run.

Questa verifica riguarda i sorgenti del run ricevuto, confrontati con la versione pubblicata prima dell'analisi (`02ab7f0`). I successivi aggiornamenti per riusare cache e venv nello study cambiano gli hash di riscrittura e Binoculars; non sostituiscono i sorgenti o gli output conservati nello ZIP originale.

## Groq: completamento e sensibilità all'ordine

Piani, fingerprint, corrispondenza fra fonti/sistemi, inversione esatta dei testi AB/BA, ID del modello, numero di tentativi e uso token sono stati verificati. I piani richiedono `reasoning_effort=none`, temperatura zero e risposte brevi. Le 30 risposte salvate hanno stato `ok` e modello `qwen/qwen3.8-27b`; il conteggio complessivo è 32 tentativi POST. I due tentativi aggiuntivi si trovano nel confronto DIPPER passaggio/contestuale; i file dei tentativi non permettono di identificare da soli il tipo di errore transitorio.

| Confronto | Risposte valide | Tentativi POST | Coppie stabili AB/BA |
| --- | ---: | ---: | ---: |
| Validazione esplorativa sui vecchi pilot | 12/12 | 12 | 4/6 |
| Baseline 3B contro 14B | 6/6 | 6 | 2/3 |
| Baseline 3B contro DIPPER contestuale | 6/6 | 6 | 2/3 |
| DIPPER passaggio contro contestuale | 6/6 | 8 | 1/3 |

Il riepilogo AB/BA e `rating_validation.json` sono stati ricalcolati dai giudizi salvati e coincidono. L'accordo 100% è condizionato ai quattro confronti stabili del preflight e usa il proxy dell'assistente, non giudizi umani. Il kappa è indefinito perché tutti e quattro preferiscono la stessa classe.

Nei tre nuovi confronti, **quattro coppie su nove cambiano preferenza canonica all'inversione**. In tutti e quattro i casi il giudice sceglie la posizione **B in entrambi gli ordini**; non sono passaggi da parità a vittoria. Tutte le 30 preferenze sono coerenti con i voti ordinali A/B restituiti: non emerge un errore del parser o una contraddizione fra voto e preferenza. Il pattern è compatibile con sensibilità alla posizione, pur su un campione troppo piccolo per attribuirle una causa generale. Sono 13 preferenze B e cinque A nelle 18 risposte nuove.

Lo stato `completed` certifica la copertura delle richieste, non affidabilità o validazione umana. I confronti instabili devono restare visibili e non essere trasformati in vittorie, parità o esempi eliminati senza denominatore.

## Annotazioni, credenziali e ripresa a 126

Il campione semantico di questo smoke contiene 12 riscritture non identità su tre fonti: tutte nel braccio reference, nessuna nel diagnostico. Il target diagnostico di 100 non è raggiungibile e il manifest lo dichiara. Le due schede dei valutatori hanno giudizi vuoti: non vengono creati esempi o etichette per arrivare a cento.

Il controllo dei file esportati non rileva stringhe con i pattern delle credenziali Groq o Hugging Face. Non è presente un dump dell'ambiente con la chiave; i piani e i giudizi non contengono header di autorizzazione. Questa è una verifica dei file forniti, non una certificazione su informazioni esterne allo ZIP.

Non emerge un blocco di scoring nell'archivio completato. Prima di avviare lo study nella stessa sessione bisogna però verificare che nuovo checkout/setup e directory di scrittura non ricadano nei percorsi `/kaggle/temp` già risultati di sola lettura. Il successo della ripresa manuale non dimostra che basti cambiare `MODE` e ricreare tutto nello stesso filesystem. Vanno mantenuti gli stessi pannelli e le correzioni operative, con una collocazione scrivibile o un riuso esplicito dei componenti leggibili.

L'estensione può misurare meglio rilevabilità e confronti nulli; il giudice richiede prudenza e annotazioni indipendenti. Tre fonti ChatGLM costituiscono un collaudo funzionale, non una stima delle prestazioni generali dei quattro riscrittori o del giudice di naturalezza.
