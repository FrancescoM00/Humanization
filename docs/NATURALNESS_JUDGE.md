# Giudice opzionale di naturalezza su Groq

Il giudice confronta due riscritture della **stessa fonte**, presentate prima A/B e poi B/A. Valuta formulazioni idiomatiche, registro, ritmo e coerenza linguistica. Non valuta origine umana/AI, evasione, verità o fedeltà a un originale che non vede. Non si attribuisce a questo esperimento un risultato prima di ricevere le risposte e le annotazioni.

## Modello e assenza di ragionamento

Il modello richiesto è esattamente `qwen/qwen3.8-27b`. L'accesso pubblico a `https://console.groq.com/docs/models` è stato tentato dal cloud ma negato dal proxy (403); il modello **non è stato confermato disponibile**. Non è stata eseguita alcuna generazione Groq. Lo script non sostituisce il modello con un altro.

Da Kaggle, con Internet attivo e `GROQ_API_KEY` configurata come secret, si può verificare il catalogo autenticato:

```bash
python scripts/judge_naturalness.py --check-model --model qwen/qwen3.8-27b
```

La chiave viene letta dall'ambiente, non scritta nei file o nei log. Il client usa soltanto `https://api.groq.com/openai/v1` e non segue redirect. Non incollare la chiave in chat o nel notebook pubblico.

Ogni richiesta include esplicitamente **`reasoning_effort: "none"`**, temperatura zero e output JSON. Questo parametro deve essere accettato dal modello selezionato; se non lo supporta, il run si arresta. Nascondere il ragionamento con `reasoning_format=hidden` non equivale a disabilitarlo e non è usato. La presenza nel catalogo conferma l'ID, non la compatibilità delle opzioni: questa resta da verificare sulla prima richiesta reale. Le risposte con campi di ragionamento non vuoti, modello differente o output troncato sono respinte. Non si elimina il parametro per ottenere una risposta.

## Preparazione senza chiamate API

Usare i JSONL completi di riscrittura dei due sistemi. Nel pilot Qwen storico il file è `rewriting_cuda0/rewrites.jsonl`; nei nuovi run può essere `rewriting/rewrites.jsonl`. Indicare i percorsi effettivi, senza rilanciare i modelli:

```bash
python scripts/judge_naturalness.py \
  --rewrites RUN_QWEN/rewriting_cuda0/rewrites.jsonl \
  --rewrites RUN_DIPPER/rewriting/rewrites.jsonl \
  --humanizer-a qwen_humanize --humanizer-b dipper \
  --source-key exports/kaggle_dipper_pilot/naturalness_paired_24_key.csv \
  --model qwen/qwen3.8-27b --reasoning-effort none \
  --max-pairs 3 --max-requests 6 \
  --output /kaggle/working/naturalness_smoke
```

Senza `--execute`, il comando salva soltanto il piano: non legge la chiave e non chiama alcuna API. La selezione usa gli ID delle fonti e il seed; non usa i punteggi o la qualità degli output. La chiave opzionale restringe il confronto alle sei fonti dei 24 testi di naturalezza. Le tre fonti dello smoke producono sei richieste pianificate, una per ciascun ordine.

I passaggi non vengono troncati. Il limite di ingresso è esplicito in caratteri; se superato lo script si interrompe. Ogni risposta è limitata a 200 token al massimo e deve contenere soltanto:

```json
{"preference": "A", "naturalness_A": 4, "naturalness_B": 3}
```

Le valutazioni 1–5 sono una rubrica esplorativa, non una scala già validata. Il prompt tratta i passaggi come materiale da valutare, non come istruzioni.

## Esecuzione e budget

Rieseguire lo stesso comando aggiungendo **`--execute`** solo quando modello, secret e piano sono pronti. La richiesta dell'utente autorizza questo esperimento; non è necessario un secondo flag di approvazione.

`--max-requests 6` limita **tutti i tentativi POST cumulativi del run**, compresi retry e tentativi precedenti a un'interruzione. I completamenti già salvati vengono riusati. Una risposta non valida viene conservata come errore, senza estrazioni ripetute finché vince un sistema. I retry 429/5xx rispettano `Retry-After`; un'attesa superiore a 30 secondi lascia il run parziale invece di ignorare il limite. Il budget può quindi esaurirsi prima di sei risposte valide. Per aumentarlo occorre dichiarare un nuovo valore esplicito.

Il limite del numero di richieste e dei token di completamento controlla il consumo; il costo in dollari non viene inventato in assenza di un prezzo verificato. Vengono registrati i token restituiti dal provider. Il controllo `/models` è separato dalle richieste di generazione.

Output:

- `judge_plan.json`: modello, prompt, testi, entrambi gli ordini e fingerprint delle richieste; nessuna chiave.
- `api_attempts.jsonl`: conteggio cumulativo dei tentativi prima di ogni POST.
- `judgments.jsonl`: esiti validi o risposte non valide, provenienza `LLM`, modello e uso token.
- `pairwise_summary.json`: preferenza riferita ai sistemi originali, con disaccordo tra ordini esplicito.
- `judge_run.json`: completamento, budget, opzioni e controlli del modello.

Una directory esistente rifiuta un piano differente. Cambiare modello, prompt, input o rubrica richiede una nuova directory.

## Validazione con le schede di naturalezza

Il form `exports/kaggle_dipper_pilot/naturalness_paired_24.csv` contiene 24 testi su sei fonti: umano, originale macchina e due riscritture. La validazione del confronto fra i due riscrittori usa **due voti per fonte**, quindi 12 dei 24 testi se tutte le sei fonti sono giudicate. Gli altri testi restano controlli e non si inventa una validazione su 24 confronti indipendenti.

I voti devono essere compilati senza consultare il giudice. Se annota un assistente, dichiarare `assistant_proxy`: quel riferimento non è una validazione umana. Le schede umane vuote restano vuote e producono `n_compared=0`, non un accordo presunto.

Per confrontare i giudizi già salvati, aggiungere al comando di preparazione, **senza `--execute`**:

```text
--ratings PERCORSO_CSV_COMPILATO
--rating-key exports/kaggle_dipper_pilot/naturalness_paired_24_key.csv
--label-source human
```

Per voti dell'assistente usare invece `--label-source assistant_proxy`. Lo script controlla gli ID e, quando presente, l'hash del testo del form. Per ciascuna fonte deriva la preferenza dai due voti 1–5, includendo le parità. Riporta accordo, Cohen kappa quando definito, numero di fonti con entrambi i voti, copertura e coerenza A/B–B/A. Le coppie incomplete o incoerenti nell'ordine non diventano automaticamente parità: restano escluse dall'accordo e visibili nel denominatore di copertura.

Con sole sei fonti, kappa e accordo sono esplorativi. Valori degeneri hanno kappa non definito, non zero. Non esiste in questo nuovo esperimento un kappa 0,30 già verificato. Le preferenze derivate da voti ordinali non sono identiche a un'annotazione pairwise diretta; questa differenza va dichiarata.

Per l'estensione si mantiene la rubrica prima di osservare gli esiti. Nessun punteggio del giudice sostituisce automaticamente la valutazione indipendente di naturalezza.
