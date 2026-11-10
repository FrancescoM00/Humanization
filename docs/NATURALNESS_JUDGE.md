# Giudice opzionale di naturalezza su Groq

Il giudice confronta due riscritture della **stessa fonte**, presentate prima A/B e poi B/A. Valuta formulazioni idiomatiche, registro, ritmo e coerenza linguistica. Non valuta origine umana/AI, evasione, verità o fedeltà a un originale che non vede. Non si attribuisce a questo esperimento un risultato prima di ricevere le risposte e le annotazioni.

## Modello e assenza di ragionamento

Il modello richiesto è esattamente `qwen/qwen3.8-27b`. Il successivo [smoke v2](../results_kaggle_v2_smoke/report.md) ha confermato modello e `reasoning_effort=none` su Kaggle: trenta risposte valide includendo la prova iniziale. Quattro dei nove confronti nuovi sono però instabili all'ordine. Il successivo [study sulle 126 fonti](../results_kaggle_v2_study/naturalness.md) contiene 364 risposte valide nuove, ma tutti i confronti sono parziali e 89 coppie complete discordano all’ordine. Non è validazione umana e non sostiene una graduatoria generale. Lo script non sostituisce il modello con un altro.

Da Kaggle, con Internet attivo e `GROQ_API_KEY` configurata come secret, si può verificare il catalogo autenticato:

```bash
python scripts/judge_naturalness.py --check-model --model qwen/qwen3.8-27b
```

La chiave viene letta dall'ambiente, non scritta nei file o nei log. Il client usa soltanto `https://api.groq.com/openai/v1` e non segue redirect. Non incollare la chiave in chat o nel notebook pubblico.

### HTTP 403 sul catalogo

Un 403 da `GET /models` avviene prima di inviare ID del modello e opzioni di generazione: non dimostra che `reasoning_effort=none` sia incompatibile o che il modello richiesto sia assente. Servono codice e motivo del rifiuto per distinguere autorizzazione, account e filtro di rete/client.

Il client aggiornato usa `requests` (già nelle dipendenze Kaggle), rimuove soltanto gli spazi esterni della chiave e conserva nei messaggi una versione limitata e redatta del motivo restituito dal provider. Non cambia host, non segue redirect e non aggira restrizioni dell'account o dell'IP. Le eccezioni distinguono il catalogo dalle richieste di completamento; i manifest registrano anche il motivo sanitizzato dell'ultimo errore e l'hash del client effettivo.

Per aggiornare una sessione già alla sezione 3 è sufficiente sostituire il solo `humanizers/naturalness.py` con la versione pubblicata e rieseguire la sezione 3. Non occorre ripetere setup o generazioni. Se il 403 rimane, condividere la nuova riga di errore redatta, senza la chiave. Il cambio di trasporto è una correzione e una diagnosi: finché il catalogo non risponde non viene dichiarato risolto il problema.

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
- `judge_eligibility.json`: fonti selezionate, coppie giudicabili e generazioni fallite, con stati e hash degli input.

Una directory esistente rifiuta un piano differente. Cambiare modello, prompt, input o rubrica richiede una nuova directory.

## Generazioni fallite nello study: riprendere la sola sezione 12

Il 10 novembre il run sulle 126 fonti si è fermato in `load_pairs`, prima delle chiamate di quel confronto, con `A failed generation cannot be silently omitted or judged as natural`. Il traceback segnala almeno un output non `ok` o vuoto, ma non identifica quanti, quali o la causa dei fallimenti. Non attribuirli automaticamente a memoria, budget o modello.

Il comportamento predefinito del comando resta rigoroso. La nuova opzione esplicita **`--failure-policy report`** permette di giudicare soltanto le coppie con entrambi gli output validi, mantenendo le altre come non valutabili e senza rimpiazzarle. La selezione per ID e il limite del campione sono applicati **prima** del controllo degli stati. Non si rigenera il testo e non si trasforma il fallimento in pareggio, sconfitta o basso punteggio di naturalezza. Per detector/ASR resta in vigore la politica dei fallimenti già prevista.

`judge_eligibility.json` conserva tutti gli ID selezionati, gli stati dei due sistemi e i motivi di ineleggibilità. `pairwise_summary.json` comprende anche fonti non giudicate e distingue fallimento della generazione, API incompleta, disaccordo all'ordine e preferenza stabile. Il manifest riporta coppie selezionate, giudicabili, fallite e copertura sul denominatore originario. `completed` riguarda le richieste pianificate sulle coppie valide, non significa copertura di tutte le 126 fonti. Senza coppie valide lo stato è `not_judged_no_eligible_pairs` e non viene contattata l'API.

Il piano, il modello, la rubrica AB/BA e il budget restano gli stessi; gli esiti già salvati vengono riusati. I conteggi di fallimento e la naturalezza condizionata a entrambi gli output riusciti vanno presentati insieme, perché un sistema con più fallimenti non deve apparire migliore grazie alla sola selezione dei sopravvissuti.

Per la sessione già avviata, **sostituire la cella della sezione 12** con questa ripresa. Il piccolo bundle contiene script e modulo aggiornati, estratti nel RUN per conservarne i sorgenti. Non richiede reimport del notebook o aggiornamento del checkout esistente:

```python
import io, json, subprocess, urllib.request, zipfile

fix = RUN / "naturalness_resume"
url = "https://raw.githubusercontent.com/FrancescoM00/Humanization/main/kaggle/Naturalness_Resume.zip"
with urllib.request.urlopen(url, timeout=60) as response:
    with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
        archive.extractall(fix)

rows = read_jsonl(REWRITES / "rewrites.jsonl")
print("Output non giudicabili:", [(r["humanizer"], r["source_id"], r.get("status"))
    for r in rows if r.get("status") != "ok" or not str(r.get("modified") or "").strip()])

comparisons = [("qwen_humanize", "qwen14b_humanize"),
               ("qwen_humanize", "dipper_context3"), ("dipper", "dipper_context3")]
for a, b in comparisons:
    if not {a, b}.issubset(ACTIVE_SYSTEMS):
        continue
    destination = RUN / "naturalness" / (a + "__" + b)
    subprocess.run([str(PYTHON), str(fix / "scripts/judge_naturalness.py"),
        "--rewrites", str(REWRITES / "rewrites.jsonl"),
        "--humanizer-a", a, "--humanizer-b", b, "--model", JUDGE_MODEL,
        "--max-pairs", str(3 if MODE == "smoke" else 126),
        "--max-requests", str(16 if MODE == "smoke" else 256),
        "--failure-policy", "report", "--execute", "--output", str(destination)],
        cwd=REPO, check=True)
    manifest = json.loads((destination / "judge_run.json").read_text())
    print(a, b, manifest["status"], "giudicabili:", manifest["n_eligible_pairs"],
          "/", manifest["n_selected_sources"],
          "coppie fallite:", manifest["n_generation_failure_pairs"])
```

La chiave rimane quella già caricata dalla sezione 3. Terminata la ripresa, eseguire la sezione 13 per creare lo ZIP completo, anche se qualche giudizio API è parziale: i mancanti rimangono espliciti. Il bundle non altera riscritture, detector o score semantici/stilometrici.

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
