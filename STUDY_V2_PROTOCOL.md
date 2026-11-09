# Studio su 126 fonti: protocollo v2 fissato prima dell'estensione

Data: 9 novembre 2026. I pilot da 42 fonti sono risultati esplorativi già osservati. Questo protocollo definisce l'estensione e i controlli; **non dichiara che le nuove inferenze GPU o Groq siano già eseguite**. Il pannello da 126 include i 42 pilot e non è una replica indipendente. L'estensione a 420 è rinviata.

**Aggiornamento di esecuzione:** il successivo [smoke v2](results_kaggle_v2_smoke/report.md) ha completato le nuove condizioni GPU e Groq sulle tre fonti tecniche. Il pannello da 126 resta da eseguire. Le formulazioni di disponibilità nelle sezioni seguenti descrivono il momento in cui il protocollo è stato fissato; il report documenta gli esiti senza cambiare coorte o preset. La naturalezza risulta sensibile all'ordine: quattro dei nove confronti nuovi sono instabili AB/BA.

## Domande principali e conclusioni candidate

1. **La riscrittura con un LLM riduce o aumenta la rilevabilità?** Si riportano TPR sull'intera coorte e transizioni in entrambe le direzioni prima dell'ASR. Nei pilot, la baseline 3B porta Fakespot da 37 a 41 rilevazioni e RADAR da 28 a 35. È una candidata a risultato da verificare sul pannello esteso, non una conclusione generale su tutti gli LLM.
2. **L'evasione dipende dalla famiglia di detector?** Il forte ASR della baseline 3B sul detector storico OpenAI non viene usato come titolo del successo generale. Si confrontano separatamente checkpoint supervisionati e Binoculars zero-shot. I tre RoBERTa non sono voti indipendenti.
3. **Quali metriche rilevano gli errori reali senza penalizzare le buone fusioni?** Coseno, BERTScore e NLI sono valutati su annotazioni e protocolli dichiarati. Le loro medie grezze non sono percentuali confrontabili di significato preservato.

## Dati e consegne

Sono mantenute le **126 fonti già congelate**, sei per ciascuna delle 21 celle dominio/generatore, e gli stessi 300 umani di calibrazione. SHA del pannello: `84f9f6933b2ee72412d65eedbf8f1fbe70cf0139bcd3d6d395cb85250c0edb68`. `prepare_study_v2.py` verifica questi byte, i conteggi e la separazione dei gruppi; non ricampiona i dati.

Le 42 fonti del pilot sono già state osservate: per verificare le ipotesi nate dal pilot si riporteranno separatamente le **84 fonti aggiuntive**, oltre al confronto complessivo sulle 126. L'appartenenza deriva dagli ID dei pannelli già congelati, non dai risultati. L'aggregato su 126 non verrà presentato come conferma interamente indipendente.

**Criterio primario scritto prima della generazione: includere tutte le fonti congelate, anche consegne e input misti.** Gli errori di generazione restano registrati. Il tipo di input si annota sul solo originale, senza consultare riscritture o detector. `source_screening.csv` contiene flag candidati deterministici e campi manuali vuoti: i flag non sono etichette né esclusioni. Le due consegne StableLM già note restano nel primario.

L'analisi secondaria sulla sola prosa richiede una revisione del tipo per tutte le fonti interessate, con criteri `prose`, `instructions`, `mixed`, motivi e conteggi di esclusione. Finché non è disponibile, si può soltanto mantenere la sensibilità esplicita sulle due fonti note; non chiamare «clean» il resto del corpus. Nessun input viene rimosso perché fa peggiorare lo humanizer.

## Sistemi definiti dalla configurazione

| Nome operativo | Condizione |
| --- | --- |
| `identity` | Testo macchina invariato |
| `qwen_humanize` | **Baseline Qwen2.5-3B-Instruct**, prompt fissato, greedy FP16 |
| `qwen14b_humanize` | Qwen2.5-14B-Instruct a 4 bit NF4, stesso prompt e greedy |
| `dipper` | **DIPPER XXL NF4 sul passaggio intero senza prefisso**, lessico 60/ordine 0 |
| `dipper_context3` | DIPPER XXL NF4 su finestre di tre frasi con prefisso riscritto cumulativo, lessico 60/ordine 0 |

Il nome breve nelle tabelle rinvia sempre a queste condizioni. La variante contestuale usa la collocazione del prefisso e i codici inversi del README ufficiale; usa però spaCy, greedy e un budget di 512 nuovi token per finestra. Non è una replica esatta dell'esempio NLTK campionato del rilascio. Il budget complessivo cresce con le finestre e non è uguale alla condizione sul passaggio intero. Il prefisso non viene troncato silenziosamente: un input oltre budget è registrato come tale.

Il modello 14B è fissato alla revisione `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`. Il confronto 3B/14B cambia anche precisione e non isola causalmente la sola dimensione. Un tentativo per fonte, nessuna scelta del candidato migliore mediante detector. Il nuovo config è [kaggle/config-study-v2.json](kaggle/config-study-v2.json); quello v1 rimane conservato.

## Detector e condizioni nulle

Il pool viene ampliato con **Binoculars**, observer `tiiuae/falcon-7b` e performer `tiiuae/falcon-7b-instruct`, revisioni fissate. Su T4 usa due GPU distinte, pesi FP16 e statistiche FP32, senza quantizzazione. Il ratio grezzo ha orientamento AI basso. La soglia è il quantile inferiore dei 300 umani; non si importa la soglia pubblicata BF16. La prova di caricamento reale è parte dello smoke nuovo. Fast-DetectGPT è rinviato; non viene presentato come implementato o eseguito.

OpenAI RoBERTa-base resta una baseline storica addestrata su GPT-2. Fakespot raw e RADAR sono riportati individualmente. Il vecchio voto a tre rimane una sintesi correlata separata; non si presenta una nuova maggioranza come indipendente e non si mediano i ratio Binoculars con probabilità dei classificatori.

La diversità del metodo e della famiglia di modelli non garantisce, da sola, indipendenza statistica fra gli errori dei detector.

Per ogni riscrittura sono generati:

- **Prefisso dell'originale con lo stesso numero di token alfabetici spaCy** della riscrittura. Il taglio può interrompere una frase ed è dichiarato: serve a controllare il conteggio, non a conservare il significato.
- **Rumore lessicale minimo**: al massimo una sostituzione da un dizionario fissato, evitando entità, negazioni, modali e numeri. Può essere semanticamente imperfetto; vengono registrati edit e casi senza edit.
- Rumore minimo applicato anche al prefisso, per osservare le due componenti insieme.

Per output più lunghi dell'originale non si inventa padding: il controllo conserva l'originale e viene marcato **non eleggibile** per il confronto esatto di lunghezza. Anche i riferimenti falliti sono marcati, senza lunghezze fittizie. Il confronto controlli/humanizer usa soltanto le fonti eleggibili condivise e gli originali baseline-AI comuni, affiancando sempre conteggi, transizioni e copertura.

Il controllo non isola ogni effetto causale della lunghezza: tagliare cambia anche contenuto e posizione. Quantifica un riferimento nullo esplicito rispetto al quale leggere l'evasione.

## Semantica e annotazioni

DeBERTa-v3 base resta float32. Alla pipeline storica si affiancano premesse singole esaustive, finestre fino a due e tre unità e documento esatto soltanto quando entra nel tokenizer. La [prova su 19 coppie](results_nli_context_pilot/report.md) mostra miglioramenti sulle fusioni ma anche problemi sui documenti interi e aumenti di supporto su errori. Nessuna soglia o scelta di accuratezza viene ricavata da quel pannello mirato.

Le annotazioni sono suddivise in due bracci, disgiunti per fonte:

- **Reference**: una fonte per cella scelta tramite hash senza score, con tutte le riscritture disponibili; si registrano probabilità d'inclusione e numerosità.
- **Diagnostico**: target di almeno 100 coppie, arricchito per minimo NLI basso e coseno alto con NLI basso. Si conservano motivi e strati di selezione; non è un campione dal quale stimare ingenuamente accuratezza o AUC di popolazione.

Con quattro riscrittori, il reference comprende fino a 84 coppie da 21 fonti; il diagnostico mira a 100 coppie ulteriori, mantenendo interi i gruppi. Due annotatori ricevono moduli indipendenti senza chiavi o score; i loro giudizi non aumentano il numero di fonti. Si riporta accordo e kappa solo dopo aver ricevuto etichette, con disaccordi e regola di eventuale adjudication separati. Il kappa 0,30 citato nel feedback non è ricostruito come nuovo risultato in assenza dei relativi file.

Sul pilot esistono soltanto 84 coppie reali non identità: il sampler prepara il censimento in due bracci da 42 e dichiara che il target diagnostico di 100 non è raggiunto. Non duplica coppie per colmare il numero.

## Naturalezza

Giudice richiesto: **`qwen/qwen3.8-27b` su Groq**, confronto fra riscritture della stessa fonte, ordini AB e BA, risposte JSON brevi, `reasoning_effort="none"`. ID e opzioni devono essere confermati dal catalogo autenticato e dalla prima richiesta: non si cambia modello o si nasconde soltanto il ragionamento per far passare la chiamata. La disponibilità non è ancora verificata nel cloud; la chiave è nei Secrets Kaggle secondo l'autore e non viene richiesta in chat.

Le 24 schede sono state annotate dall'assistente senza aprire le chiavi per questa attività: [voti e limiti](results_naturalness_assistant24/report.md). Sono un proxy AI, non gold umana. Il primo confronto fra i due riscrittori sulle sei fonti usa 12 di quei testi e produce **concordanza AI–AI su sei preferenze**, non 24 confronti umani indipendenti. I controlli umani/originali restano disponibili negli altri 12 voti.

Il notebook esegue prima un piano di validazione da 12 risposte valide, con massimo 16 tentativi POST; per lo study sono fissate tre comparazioni fra sistemi, ciascuna con massimo 256 tentativi e 252 risposte attese se tutte le 126 coppie sono valide. Totale massimo 768 tentativi nello study, più la prova iniziale. Retry e riprese contano nel budget; quote e risposte mancanti rimangono visibili. La disponibilità del modello o una quota non sufficiente non viene trasformata in giudizi fittizi. Prezzi non verificati non vengono tradotti in costi monetari inventati.

Precisazione operativa del 10 novembre, dopo il blocco della sezione 12: le generazioni fallite o vuote non ricevono un voto di naturalezza e non sono rimpiazzate da altre fonti. Con `--failure-policy report` si giudicano le coppie valide mantenendo il campione selezionato prima del controllo degli stati; fallimenti, coppie giudicabili e preferenze stabili hanno conteggi e coperture distinti sul denominatore originario. La naturalezza è quindi condizionata alla riuscita di entrambi gli output, mentre i fallimenti restano un risultato separato. Modello, rubrica, AB/BA e budget non cambiano.

## Ordine operativo

1. Nuovo notebook `Humanization_Study_v2.ipynb`, `MODE="smoke"`, T4 ×2 e Internet; attach del Secret Groq.
2. Verifica catalogo/opzioni Groq e concordanza preliminare; caricamento dei nuovi modelli e Binoculars sui tre input tecnici, controlli nulli e tutti gli stadi.
3. Dopo il collaudo, `MODE="study"` con la stessa configurazione. Nessun tuning dopo aver visto i 126 risultati.
4. Restituire lo ZIP, leggere copertura e fallimenti prima delle conclusioni, raccogliere annotazioni separate e applicare i confronti appaiati.

Quarantadue o 126 fonti non garantiscono da sole un intervallo di ±10 punti: l'ASR ha un denominatore specifico per detector, inferiore o uguale alle fonti disponibili. Gli intervalli vengono calcolati sui dati effettivi, con la limitazione della calibrazione mantenuta fissa.
