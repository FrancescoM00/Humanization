# Modelli del pilot Kaggle

Revisioni e model card verificate il 9 novembre 2026. Nessun checkpoint di humanizer o detector è stato eseguito nel cloud durante la preparazione; la prova GPU si svolge nel notebook Kaggle.

## Scelta consigliata

Pilot riproducibile: Qwen2.5-3B-Instruct come riscrittore istruito via prompt + tre detector eseguiti separatamente (OpenAI RoBERTa-base, Fakespot RoBERTa-base, RADAR RoBERTa-large). Non presentare Qwen come servizio commerciale di humanization o come modello specificamente addestrato per evadere detector. Si tratta di una baseline di humanization via prompt. DIPPER è il confronto specifico di paraphrasing per robustezza ai detector, da rendere opzionale per il grande download.

I tre detector sono checkpoint con procedure/corpora di addestramento diversi, ma restano della famiglia RoBERTa; non costituiscono un campione rappresentativo di tutte le famiglie di detector. Tenerne separati i risultati anche quando si aggiunge un voto di maggioranza.

## Riscrittore principale

- ID esatto: `Qwen/Qwen2.5-3B-Instruct`.
- Revisione: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Scheda: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/README.md
- Licenza: **Qwen Research License**, non Apache. Consente ricerca/valutazione non commerciale; la licenza dice esplicitamente `FOR NON-COMMERCIAL PURPOSES ONLY`.
- Testo licenza: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE
- Nongated, 3.085.938.688 parametri BF16. Due shard safetensors da 3.968.658.944 e 2.203.268.048 byte; somma 6.171.926.992 byte.
- Modello standard `Qwen2ForCausalLM`, `AutoModelForCausalLM`, tokenizer con chat template.
- Su T4/P100 usare float16 esplicito: BF16 non è una scelta sicura su queste GPU. Il fatto che i soli pesi FP16 occupino circa 5,75 GiB non è una misura del picco VRAM completo.
- Batch 1 e input/output brevi sono adatti come primo tentativo su 16 GB; la compatibilità non equivale a una verifica di inferenza già effettuata in questo cloud CPU.
- Non usare questo 3B come giudice di naturalezza sostitutivo del giudice grande richiesto dal professore: i ruoli sono diversi.

Alternativa con Apache-2.0: `Qwen/Qwen3-4B-Instruct-2507`, revisione `cdbee75f17c01a7cc42f958dc650907174af0554`, nongated, 4.022.468.096 parametri BF16 e circa 8,045 GB di shard. Non necessaria per avviare il pilot con il 3B.

## Detector 1: baseline storica GPT-2

- ID canonico: `openai-community/roberta-base-openai-detector`. L'alias storico `roberta-base-openai-detector` compare nella model card, ma utilizzare l'ID con namespace per non dipendere da redirect.
- Revisione: `6cba99c003b711c7fe94f8a3aa2be35a792cb6fa`.
- Scheda: https://huggingface.co/openai-community/roberta-base-openai-detector/blob/6cba99c003b711c7fe94f8a3aa2be35a792cb6fa/README.md
- Config: https://huggingface.co/openai-community/roberta-base-openai-detector/blob/6cba99c003b711c7fe94f8a3aa2be35a792cb6fa/config.json
- Licenza MIT, nongated, `RobertaForSequenceClassification`, circa 125M parametri, safetensors disponibile.
- **Probabilità AI = softmax(logits)[0]**: `id2label={0:'Fake',1:'Real'}`.
- Limite principale: addestrato a distinguere output GPT-2 1.5B da WebText. È una baseline storica, non una prova delle prestazioni di un detector moderno su ChatGPT/Qwen.
- Budget: 512 token totali, config `max_position_embeddings=514` tipico RoBERTa (i 514 non sono un invito a tokenizzare 514 token utili).

## Detector 2: Fakespot

- ID: `fakespot-ai/roberta-base-ai-text-detection-v1`.
- Revisione: `f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db`.
- Scheda: https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/blob/f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db/README.md
- Config: https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/blob/f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db/config.json
- Licenza Apache-2.0, nongated, `RobertaForSequenceClassification`, 124.647.170 parametri F32. `model.safetensors`: 498.612.824 byte.
- **Probabilità AI = softmax(logits)[1]**: `id2label={0:'Human',1:'AI'}`.
- Model card raccomanda `clean_text` del repo; codice: https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/blob/f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db/utils.py
- Tale funzione rimuove code block/inline code, markdown, blocchi citati, tabelle e tag HTML; decodifica entità HTML e normalizza whitespace. Se si sceglie input grezzo per il pilot, dichiararlo esplicitamente come **raw-input variant** e applicare la medesima scelta a calibrazione, originali, umani e riscritture. Non dire che il preprocessing originale è stato eseguito.
- Le eventuali trasformazioni specifiche del detector non devono cambiare i testi passati alla stilometria o alle metriche semantiche.
- Budget 512 token totali (max_position_embeddings514).

## Detector 3: RADAR

- ID: `TrustSafeAI/RADAR-Vicuna-7B`.
- Revisione: `4ff1f23a69a36aa1df47b0933be6279f1b896c9b`.
- Scheda: https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B/blob/4ff1f23a69a36aa1df47b0933be6279f1b896c9b/README.md
- Config: https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B/blob/4ff1f23a69a36aa1df47b0933be6279f1b896c9b/config.json
- Nongated. La card dichiara licenza **non commerciale ereditata da Vicuna-7B-v1.1**. Non chiamarlo Apache/MIT o genericamente open-source senza distinguere open weights da licenze permissive.
- Il nome `Vicuna-7B` identifica il generatore del training: il detector è **RoBERTa-large (~355M)**, non un LLM di 7B.
- Addestramento avversariale tra detector e paraphraser su originali OpenWebText, generazioni Vicuna e parafrasi.
- Pesi disponibili solo in `pytorch_model.bin`: 1.421.627.665 byte. Caricamento standard `AutoModelForSequenceClassification`.
- **Probabilità AI = softmax(logits)[0]**. Il config non documenta label semantiche; la fonte decisiva è la demo ufficiale degli autori, che mappa `F.log_softmax(...logits,-1)[:,0].exp()` a `AI-generated`:
  https://huggingface.co/spaces/TrustSafeAI/RADAR-AI-Text-Detector/blob/0b17bd20035a44d5bd4df265c03d088ea9389960/app.py
- Budget 512 token totali.
- Con Transformers4.57, il caricamento `.bin` richiede Torch>=2.6 per la guardia `torch.load`. Verificare Torch GPU preinstallato su Kaggle; non installare il lockfile CPU del progetto.

## DIPPER opzionale

- ID: `kalpeshk2011/dipper-paraphraser-xxl`.
- Revisione: `c1fbf7a958a2aab022e9e6f81f7a3139f9e6ee3c`.
- Scheda e codice d'esempio ufficiale: https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl/blob/c1fbf7a958a2aab022e9e6f81f7a3139f9e6ee3c/README.md
- Apache-2.0, nongated, T5-XXL con 11.266.928.640 parametri **F32** nei safetensors. I soli dati dei pesi occupano circa45,07GB. Sono presenti cinque shard safetensors e cinque `.bin`: selezionare un solo formato, non scaricare entrambi.
- Quantizzare4bit può ridurre la VRAM di caricamento ma **non riduce il download degli shard originali né lo spazio della cache**. Non presumere che il runtime Kaggle abbia spazio senza `shutil.disk_usage` e senza distinguere cache temporanea da output persistenti.
- La capacità di stare in16GB non è stata verificata con inferenza: dipende da bitsandbytes, dispositivo, sequenze, picchi di caricamento. Rendere opt-in.
- Il repo DIPPER non contiene il tokenizer. Tokenizer ufficiale: `google/t5-v1_1-xxl`, revisione `3db67ab1af984cf10548a73467f0e5bca2aaaeb2`, Apache-2.0, nongated; scaricare solo file tokenizer, non i suoi pesi.
- Tokenizerconfig: https://huggingface.co/google/t5-v1_1-xxl/blob/3db67ab1af984cf10548a73467f0e5bca2aaaeb2/tokenizer_config.json (`model_max_length=512`).
- Diversità lessicale e ordine ammesse:0,20,40,60,80,100. Codici invertiti: `lex_code=100-lex_diversity`, `order_code=100-order_diversity`.
- Formato esatto: `lexical = {lex_code}, order = {order_code}` + eventuale ` {prefix}` + ` <sent> {curr_sent_window} </sent>`.
- Codice ufficiale raggruppa tre frasi per finestra (`sent_interval=3`) con `nltk.sent_tokenize`; dopo ciascuna generazione aggiunge l'output al prefix come contesto per la finestra successiva.
- Applicazione a un singolo passaggio intero senza prefix è una variante ammissibile esplicita per testi brevi; non dichiarare che replica esattamente la procedura a tre frasi.
- In tutti i casi controllare budget e fine generazione; non troncare silenziosamente né input né riscritture.

## Alternative verificate ma non scelte

- `Hello-SimpleAI/chatgpt-detector-roberta`, rev`d2b342c61775d5dd0221808a79983ed3b86ffd86`: Human0/ChatGPT1; 125M RoBERTa base, training HC3 all answers full+sentences senza held-out per1epoch. Nessuna licenza esplicita nella modelcard; `.bin` soltanto. Non necessario al pool3.
- `desklib/ai-text-detector-v1.01`, rev`5fdea974cd4287c61674951ec78803aa274e2fb7`: MIT,nongated,DeBERTa-v3-large434M, safetensors1.736.100.972byte. **Custom** `DesklibAIDetectionModel` con meanpool e un logit sigmoid(AI); `AutoModelForSequenceClassification` non ricostruisce la testa corretta. La card fornisce definizione di classe. Evitare nel primo pilot per semplificare adattatori. `desklib/ai-text-detector-v1.2` non è l'ID corretto (401 su APIpubblica).
- `PirateXX/AI-Content-Detector`: card licenseartistic2.0, label0Fake1Real, ma documentazione training troppo scarna; non preferibile ai tre scelti.

## Metadati HTTP riproducibili

Per dimensioni e revisioni usare `https://huggingface.co/api/models/{repo_id}?blobs=true`; le info `siblings[].size` evitano il download dei modelli. Le revisioni sopra sono fissate rispetto ai risultati della lettura pubblica attuale; le stesse revisioni devono comparire in config e manifest del pilot.
