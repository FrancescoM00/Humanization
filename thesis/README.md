# Sezione sperimentale della tesi

- **[PDF da leggere](sezione_sperimentale.pdf)**: anteprima autonoma con indice, formule e tabelle.
- **[LaTeX completo](sezione_sperimentale_completa.tex)**: un solo file autonomo, da compilare con **XeLaTeX**; in Overleaf selezionare questo compilatore nelle impostazioni del progetto.
- **[Sorgente LaTeX integrabile](sezione_sperimentale.tex)**: frammento da inserire nel documento della tesi.
- **[Testo Markdown](sezione_sperimentale.md)**: sorgente editoriale leggibile anche su GitHub.
- **[Scarica tutto in ZIP](sezione_sperimentale.zip)**: PDF, sorgenti e istruzioni in un unico archivio.

Il testo riprende tono, notazione e struttura del PDF teorico fornito, senza ripubblicarlo. Descrive gli esperimenti effettivamente svolti fino al controllo delle 34 coppie numeriche e lo stato del codice dopo il riordino della repository. I dati sono quelli dei report e dei CSV conservati nel progetto; per questa stesura non sono state eseguite nuove inferenze.

La provenienza è stata aggiornata dopo il [confronto dei tre CSV forniti dall'autore come scaricati da Google Drive](https://github.com/FrancescoM00/Humanization/blob/main/provenance/mgtbench_drive_comparison.md): sono identici byte per byte a quelli già usati. Il PDF, il Markdown e i sorgenti LaTeX riportano ora questa verifica; i risultati numerici restano gli stessi.

## Inserimento nel documento originale

Il PDF teorico termina la discussione con la sezione 12 e contiene poi la sezione 13 Bibliografia. Inserire il frammento **dopo la sezione 12 e prima della bibliografia**:

```latex
\input{thesis/sezione_sperimentale.tex}
```

La nuova parte diventerà la sezione 13 e la bibliografia passerà alla 14. Le formule e le tabelle sono numerate automaticamente. Il frammento non imposta contatori: segue quelli del documento in cui viene incluso. L'anteprima autonoma imposta invece i contatori alla sezione 12, equazione 41 e tabella 6, come nel PDF fornito; la sua paginazione è autonoma.

Il preambolo della tesi deve disporre dei pacchetti matematici e tabellari usati: `amsmath`, `amssymb`, `longtable`, `booktabs`, `array`, `calc`, `hyperref` e `xurl`. Le macro Pandoc minime `\tightlist` e `\passthrough`, se richieste, sono definite in [anteprima.tex](anteprima.tex). Il frammento è UTF-8 e i segni meno sono esportati come comandi matematici compatibili anche con pdfLaTeX. L'anteprima usa XeLaTeX e Latin Modern, senza vincolare il motore del documento originale.

I riferimenti numerici `[1]`–`[48]` presenti nel testo riprendono le voci pertinenti della bibliografia del PDF teorico. Non essendo stato fornito il sorgente `.bib`, sono conservati come numeri, senza inventare chiavi BibTeX. In un documento che usa `\cite`, si possono ricollegare alle chiavi originali. Le schede dei nuovi checkpoint e la repository sono indicate in note distinte, senza attribuire loro numeri bibliografici inesistenti.

## Rigenerare l'anteprima

Servono Pandoc, Python 3 e XeLaTeX, con i pacchetti elencati in `anteprima.tex`:

```bash
bash thesis/compila.sh
```

Modificare `sezione_sperimentale.md` e rigenerare: il comando aggiorna il frammento `.tex`, il documento autonomo `sezione_sperimentale_completa.tex`, il PDF e l'archivio ZIP. Il file autonomo incorpora il contenuto e il preambolo di `anteprima.tex`, senza dipendenze da altri file del progetto. I file di compilazione temporanei sono ignorati da Git. Gli strumenti di impaginazione non sono dipendenze delle pipeline sperimentali Python.
