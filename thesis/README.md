# Sezione sperimentale e sintesi per il professore

Aggiornamento: **10 novembre 2026**. I documenti distinguono esperimenti conclusi, limiti e attività in corso. Le **84 coppie reference sono state inviate ai colleghi**; le annotazioni umane sono ancora in raccolta e non vengono anticipati risultati.

## Documenti da scaricare

| Documento | PDF | LaTeX autonomo | Testo modificabile |
|---|---|---|---|
| Capitolo sperimentale | [PDF](sezione_sperimentale.pdf) | [LaTeX completo](sezione_sperimentale_completa.tex) | [Markdown](sezione_sperimentale.md) |
| Sintesi per l'incontro | [PDF di due pagine](sintesi_incontro_professore.pdf) | [LaTeX](sintesi_incontro_professore.tex) | [Markdown](sintesi_incontro_professore.md) |

**[Archivio con entrambi i documenti e i sorgenti](sezione_sperimentale.zip)**. È disponibile anche lo [ZIP della sola sintesi](sintesi_incontro_professore.zip).

I file LaTeX autonomi contengono già il loro preambolo: si compilano con **XeLaTeX**, anche su Overleaf scegliendo questo compilatore nelle impostazioni. Per scaricare da GitHub, aprire il file e usare **Download raw file**, oppure il collegamento `raw.githubusercontent.com`.

## Contenuto

Il capitolo parte dall'obiettivo e dalle quattro dimensioni di valutazione. Spiega dati, controllo della lunghezza, feature e classificatore stilometrico; perturbazioni semantiche, controlli positivi, allineamento e aggregatori; pilot e studio Kaggle sulle 126 fonti; detector e controlli nulli; calibrazione della modalità NLI; limiti del giudice di naturalezza; organizzazione del codice e attività ancora da completare.

La sintesi riporta obiettivo, dati, modelli, risultati centrali e domande per il professore. Non sostituisce il capitolo né i report numerici di dettaglio.

La stesura usa i risultati già presenti nel repository, fino alla calibrazione NLI. Non sono state avviate nuove inferenze per scriverla. Le fonti bibliografiche aggiunte sono indicate in note con autore o collegamento; non vengono inventati numeri corrispondenti alla bibliografia del PDF teorico.

## Inserimento nella tesi

Il PDF teorico fornito termina la discussione con la sezione 12 e contiene poi la sezione 13 Bibliografia. Inserire il **frammento** LaTeX dopo la sezione 12 e prima della bibliografia:

```latex
\input{thesis/sezione_sperimentale.tex}
```

Il frammento [sezione_sperimentale.tex](sezione_sperimentale.tex) non contiene preambolo né contatori: segue quelli del documento che lo include. La nuova parte diventa la sezione 13 e la bibliografia passa alla 14. L'anteprima autonoma riparte dalla sezione 13, equazione 42 e tabella 7, con paginazione propria.

Il preambolo deve includere `amsmath`, `amssymb`, `longtable`, `booktabs`, `array`, `calc`, `hyperref` e `xurl`. Le macro minime `\tightlist` e `\passthrough` sono definite in [anteprima.tex](anteprima.tex). Se il documento principale non le definisce già, copiarle nel suo preambolo. Il testo è UTF-8; i file autonomi usano Latin Modern e XeLaTeX.

## Rigenerare PDF e sorgenti

Servono Pandoc, Python 3 e XeLaTeX, con i pacchetti indicati nei due preamboli:

```bash
bash thesis/compila.sh
```

Modificare i due Markdown e poi eseguire il comando: aggiorna i frammenti LaTeX, i documenti autonomi, entrambi i PDF e gli archivi ZIP. I due file autonomi non dipendono da altri file del progetto. I file temporanei di compilazione sono ignorati da Git. Gli strumenti di impaginazione non sono dipendenze delle pipeline sperimentali Python.
