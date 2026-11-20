---
title: "Naturalezza delle riscritture"
subtitle: "Dal bias di posizione ai voti con ordini bilanciati — 20 novembre 2026"
lang: it-IT
fontsize: 10pt
papersize: a4
geometry: margin=2cm
colorlinks: true
---

**Stato attuale.** Abbiamo sostituito i confronti a due testi, fortemente sensibili alla posizione, con voti sulle quattro riscritture presentate in ordini bilanciati e ripetuti. Il giudice Qwen assegna mediamente voti più alti ai due riscrittori Qwen rispetto a DIPPER. Resta però un effetto della posizione e manca il confronto esteso con Gemma: i risultati sono esplorativi, non una graduatoria validata da persone.

## 1. Che cosa vogliamo misurare

La naturalezza riguarda quanto un testo risulta scorrevole, idiomatico e appropriato al genere, considerando formulazione, registro, ritmo delle frasi e coerenza linguistica. Non coincide con la conservazione dei fatti, con l'origine umana del testo o con l'evasione dei detector.

Per ogni testo macchina di partenza dello studio MGTBench abbiamo quattro riscritture: **Qwen 3B, Qwen 14B NF4, DIPPER intero e DIPPER a gruppi di tre frasi**. Quest'ultimo riscrive gruppi successivi con contesto cumulativo; non applica tre volte DIPPER allo stesso testo. Il giudice è un altro modello, **Qwen 27B**, identificato dal servizio Groq come `qwen/qwen3.8-27b`, con temperatura zero e `reasoning_effort=none`.

## 2. Perché abbiamo cambiato il primo protocollo

Inizialmente mostravamo due riscritture, A e B, e chiedevamo quale fosse più naturale. Ripetevamo il confronto scambiandone le posizioni. Una preferenza stabile dovrebbe seguire il testo: se il testo migliore passa da B ad A, anche la risposta dovrebbe cambiare da B ad A.

Nella prova iniziale **4 confronti nuovi su 9 erano instabili**, e in tutti e quattro il giudice sceglieva B in entrambi gli ordini. Complessivamente sceglieva B 13 volte contro 5 scelte A. Il fenomeno si confermava nell'estensione, rimasta parziale: su 364 risposte valide, B era scelto 263 volte (72,3%); **85 delle 89 coppie discordanti sceglievano B sia prima sia dopo lo scambio**. Le coppie appartenevano a tre confronti fra sistemi e non erano altrettanti originali indipendenti.

Questo comportamento segnala una sensibilità sistematica alla presentazione. I soli confronti rimasti stabili, selezionati sugli esiti del giudice e con copertura parziale, non permettevano una graduatoria generale affidabile.

## 3. Il nuovo protocollo con voti e permutazioni

Seguendo la proposta del professore, mostriamo insieme le **quattro riscritture dello stesso originale**, anonime, e chiediamo un voto da **1 a 5 per ciascuna**: da prosa molto innaturale a prosa costantemente fluida e idiomatica. Sono ammessi voti uguali; non si impone una graduatoria o un totale fisso di punti. Il giudice conosce il genere, ma non vede il testo originale e non deve valutarne la fedeltà.

Per ogni originale usiamo **quattro ordini distinti bilanciati**, ciascuno ripetuto **tre volte**. Ogni metodo compare quindi tre volte in ciascuna delle quattro posizioni. Il disegno, detto Williams, bilancia anche le adiacenze fra metodi: utilizza quattro delle 24 permutazioni possibili, non tutte.

I conteggi da distinguere sono: **12 chiamate per originale**, ciascuna con quattro voti, quindi **48 voti per originale**. Le ripetizioni dello stesso ordine verificano la variabilità a presentazione fissa; i diversi ordini misurano la sensibilità alla posizione. Le chiamate ripetute sullo stesso originale non sono osservazioni indipendenti.

## 4. Campione analizzato e risultati Qwen

Le quattro riscritture sono valide per 123 dei 126 originali dello studio. Le quote del servizio hanno richiesto più riprese del run. Il checkpoint disponibile contiene 747 risposte valide, ma soltanto **61 originali hanno tutte le 12 chiamate previste**: 17 Essay, 20 Reuters e 24 WritingPrompts. L'analisi primaria usa questi 61 pannelli completi: **732 chiamate e 2.928 voti**. Le 15 risposte valide appartenenti a due pannelli parziali sono conservate ma escluse dai confronti principali. Non sono quindi risultati su tutti i 123 originali eleggibili.

Si calcola prima la **media dei 12 voti** di ciascun originale e metodo, poi la media fra i 61 originali. Gli intervalli al 95% usano 2.000 ricampionamenti appaiati per gruppo di fonte, non per singolo voto.

| Riscrittore | Voto medio [IC 95%], scala 1–5 |
|---|---:|
| Qwen 3B | 4,07 [3,95; 4,19] |
| Qwen 14B NF4 | 3,99 [3,82; 4,13] |
| DIPPER intero | 3,29 [3,12; 3,46] |
| DIPPER tre frasi | 3,34 [3,17; 3,51] |

Questo giudice attribuisce voti superiori alle riscritture Qwen. Le differenze fra 3B e 14B e fra le due configurazioni DIPPER sono piccole e i relativi intervalli appaiati comprendono zero: non emerge un ordine netto entro ciascuna coppia. Gli intervalli sono esplorativi e puntuali, senza correzione per confronti multipli. Il giudice appartiene alla stessa famiglia di due riscrittori: una possibile preferenza di famiglia resta da verificare.

## 5. Bias residuo e ripetibilità

**Il disegno bilancia l'esposizione alle posizioni, ma resta un effetto residuo misurabile.** Il testo in prima posizione riceve mediamente **0,235 punti in meno** rispetto alla media delle presentazioni dello stesso testo, con intervallo [−0,299; −0,173]. Le altre posizioni hanno scostamenti medi positivi di circa 0,075–0,083 punti.

A ordine fisso, i tre voti ripetuti sono identici in **957 dei 976 gruppi originale × metodo × ordine, cioè nel 98,1%**. Per la stessa riscrittura, la differenza fra la media più alta e quella più bassa dei quattro ordini è però in media **0,71 punti**. Una risposta molto ripetibile può quindi restare sensibile alla presentazione. Quel 98,1% non è un'accuratezza e non misura accordo con persone. Il controbilanciamento evita che un metodo occupi sistematicamente una posizione favorevole; non garantisce la neutralità del giudice.

## 6. Che cosa aspettiamo da Gemma

Il secondo giudice è **Gemma 4 31B**, identificato da Google come `gemma-4-31b-it`, di una famiglia diversa da Qwen. Il pilota tecnico su **10 originali, 120 risposte valide e 480 voti** è concluso. Il confronto esteso con il riferimento Qwen è ancora in attesa: il piano prevede 732 chiamate sulle stesse 61 fonti, con gli stessi testi, ordini e ripetizioni. I pannelli parziali saranno esclusi dal confronto principale, usando soltanto gli originali completi per entrambi i giudici.

Confronteremo soprattutto **le differenze fra riscrittori e gli effetti della posizione**, perché due giudici possono usare diversamente la scala 1–5. Il punto è verificare se le preferenze osservate con Qwen si mantengono con un'altra famiglia. Anche un accordo fra Qwen e Gemma resterà concordanza fra modelli: non sostituirà giudizi umani indipendenti e non certificherà la conservazione del significato.

*Fonti del riepilogo: audit AB/BA dello smoke e dello studio; tabella Qwen dei 61 pannelli completi; archivio Qwen di riferimento del 18 novembre; ZIP del pilota Gemma ricevuto. Sono riassunti risultati già disponibili, senza nuove chiamate ai giudici.*
