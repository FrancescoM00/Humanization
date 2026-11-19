"""Confronta i detector prima e dopo la riscrittura usando score già salvati."""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .detection import text_records
from .detector_performance_metrics import summarize_paired_metrics


INPUT_MEMBERS = (
    "config.json", "cohorts/study.jsonl", "null_controls/rewrites.jsonl",
    "detection/detector_scores.csv", "zero_shot/zero_shot_scores.csv",
    "detector_thresholds_combined.json",
)
PRIMARY_SYSTEMS = ("qwen_humanize", "qwen14b_humanize", "dipper", "dipper_context3")
IDENTITY = "null_identity"
PAIR_COLUMNS = [
    "source_id", "source_group_id", "domain", "generator", "detector", "humanizer",
    "human_score", "original_score", "rewrite_score", "human_is_ai", "original_is_ai", "rewrite_is_ai",
    "human_raw_score", "original_raw_score", "rewrite_raw_score", "raw_threshold", "score_orientation",
]


def load_saved_detection(path):
    """Legge i soli file necessari, senza estrarre archivi o caricare modelli."""
    path = Path(path)
    if path.is_dir():
        raw = {name: (path / name).read_bytes() for name in INPUT_MEMBERS}
    else:
        with zipfile.ZipFile(path) as archive:
            if any(archive.namelist().count(name) != 1 for name in INPUT_MEMBERS):
                raise ValueError("Lo ZIP deve contenere una sola copia di ogni input richiesto")
            if sum(archive.getinfo(name).file_size for name in INPUT_MEMBERS) > 128 * 2**20:
                raise ValueError("Gli input superano la dimensione prevista")
            raw = {name: archive.read(name) for name in INPUT_MEMBERS}
    jsonl = lambda name: [json.loads(line) for line in raw[name].splitlines() if line.strip()]
    frame = pd.concat([
        pd.read_csv(io.BytesIO(raw[name]), keep_default_na=False, float_precision="round_trip")
        for name in ("detection/detector_scores.csv", "zero_shot/zero_shot_scores.csv")
    ], ignore_index=True)
    return dict(config=json.loads(raw["config.json"]), cohort=jsonl("cohorts/study.jsonl"),
                rewrites=jsonl("null_controls/rewrites.jsonl"), scores=frame,
                thresholds=json.loads(raw["detector_thresholds_combined.json"]),
                input_sha256={name: hashlib.sha256(content).hexdigest() for name, content in raw.items()})


def _orientation(name, entry, classifier_names, zero_shot_names):
    decision = entry.get("decision", "")
    if name in classifier_names and "score > threshold" in decision:
        return 1
    if (name in zero_shot_names and "< threshold" in decision
            and entry.get("orientation", entry.get("score_orientation")) == "lower_is_more_ai_like"):
        return -1
    raise ValueError(f"{name}: orientamento o decisione congelata non riconosciuti")


def prepare_score_views(data):
    """Controlla integrità, separazione calibrazione/test e soglie congelate."""
    cohort, rewrites, frame = data["cohort"], data["rewrites"], data["scores"].copy()
    records = text_records(cohort, rewrites)
    expected = {(str(r["source_id"]), r["role"], r.get("humanizer") or ""): r for r in records}
    tests = [row for row in cohort if row["split"] == "test"]
    calibration = [row for row in cohort if row["split"] == "calibration"]
    hashes = {split: {hashlib.sha256(row["human"].encode()).hexdigest()
                      for row in cohort if row["split"] == split} for split in ("calibration", "test")}
    if hashes["calibration"] & hashes["test"]:
        raise ValueError("Un testo umano compare sia nella calibrazione sia nel test")
    # Una coppia è un'unità del pannello; duplicati testuali non diventano nuove osservazioni.
    if len(hashes["test"]) != len(tests):
        raise ValueError("Il pannello contiene umani di test duplicati: correggere i gruppi prima dell'analisi")
    classifier_names = [d["name"] for d in data["config"]["detectors"] if d.get("enabled", True)]
    zero_names = [d["name"] for d in data["config"].get("zero_shot_detectors", []) if d.get("enabled", True)]
    names = classifier_names + zero_names
    if len(set(names)) != len(names) or set(frame.detector) != set(names):
        raise ValueError("Gli score non corrispondono ai detector configurati")
    if any(name not in data["thresholds"] for name in names):
        raise ValueError("Manca una soglia già calibrata; questa analisi non ricalibra")
    views, directions = {}, {}
    for name in names:
        part = frame[frame.detector == name].copy()
        rows = part.to_dict("records")
        keys = [(str(r["source_id"]), r["role"], r.get("humanizer") or "") for r in rows]
        if len(set(keys)) != len(keys) or set(keys) != set(expected):
            raise ValueError(f"{name}: score mancanti, duplicati o estranei al pannello")
        for key, row in zip(keys, rows):
            reference = expected[key]
            for field in ("source_group_id", "domain", "generator", "split", "rewrite_id"):
                if str(row.get(field) or "") != str(reference.get(field) or ""):
                    raise ValueError(f"{name}: metadati dello score diversi dagli input")
            if row.get("text_sha256") != hashlib.sha256(reference["text"].encode()).hexdigest():
                raise ValueError(f"{name}: hash del testo diverso dallo score salvato")
        values = part.score.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Gli score devono essere finiti: uno score mancante non è evasione")
        entry = data["thresholds"][name]
        threshold = entry.get("threshold")
        if type(threshold) not in {float, int} or not math.isfinite(threshold):
            raise ValueError("La soglia congelata non è finita")
        if entry.get("calibration_n") != len(calibration):
            raise ValueError("Il numero di umani di calibrazione differisce dalla soglia salvata")
        direction = _orientation(name, entry, classifier_names, zero_names)
        if direction == 1 and ((values < 0).any() or (values > 1).any()):
            raise ValueError("Lo score del classificatore deve essere compreso fra zero e uno")
        part["is_ai"] = values > threshold if direction == 1 else values < threshold
        observed_calibration = float(part.loc[part.role.eq("calibration_human"), "is_ai"].mean())
        if not np.isclose(observed_calibration, entry["calibration_fpr"], atol=1e-12, rtol=0):
            raise ValueError("La soglia o gli score non riproducono l'FPR di calibrazione salvato")
        part["oriented_score"] = direction * values
        part["raw_threshold"] = threshold
        part["score_orientation"] = "higher_is_more_ai_like" if direction == 1 else "lower_is_more_ai_like"
        views[name] = {(str(r["source_id"]), r["role"], r.get("humanizer") or ""): r
                       for r in part.to_dict("records")}
        directions[name] = "higher_is_more_ai_like" if direction == 1 else "lower_is_more_ai_like"
    if "majority_vote" in data["thresholds"]:
        if len(classifier_names) != 3:
            raise ValueError("Il voto storico richiede esattamente i tre classificatori originali")
        vote = {}
        for key in expected:
            row = dict(views[classifier_names[0]][key])
            decision = sum(bool(views[name][key]["is_ai"]) for name in classifier_names) >= 2
            row.update(detector="majority_vote", is_ai=decision, score=float(decision), oriented_score=float(decision),
                       raw_threshold=data["thresholds"]["majority_vote"]["threshold"],
                       score_orientation="binary_three_classifier_vote_no_continuous_score")
            vote[key] = row
        views["majority_vote"] = vote
        directions["majority_vote"] = "binary_three_classifier_vote_no_continuous_score"
        entry = data["thresholds"]["majority_vote"]
        observed = np.mean([r["is_ai"] for r in vote.values() if r["role"] == "calibration_human"])
        if (entry.get("calibration_n") != len(calibration)
                or not np.isclose(observed, entry.get("calibration_fpr"), atol=1e-12, rtol=0)):
            raise ValueError("Il voto di maggioranza non riproduce la calibrazione salvata")
    return tests, views, directions


def _pairs(tests, lookup, system, available):
    rows = []
    for source in sorted(tests, key=lambda row: str(row["id"])):
        identifier = str(source["id"])
        if identifier not in available:
            continue
        human, original, rewrite = [lookup[identifier, role, method]
                                    for role, method in [("human", ""), ("machine", ""), ("rewrite", system)]]
        rows.append(dict(source_id=identifier, source_group_id=source["source_group_id"],
            domain=source["domain"], generator=source["generator"], detector=human["detector"], humanizer=system,
            human_score=human["oriented_score"], original_score=original["oriented_score"],
            rewrite_score=rewrite["oriented_score"], human_is_ai=bool(human["is_ai"]),
            original_is_ai=bool(original["is_ai"]), rewrite_is_ai=bool(rewrite["is_ai"]),
            human_raw_score=human["score"], original_raw_score=original["score"],
            rewrite_raw_score=rewrite["score"], raw_threshold=human["raw_threshold"],
            score_orientation=human["score_orientation"]))
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def evaluate_saved_detection(data, *, bootstrap=5000, seed=42, systems=None):
    """Valuta ogni coppia sullo stesso sottoinsieme e aggiunge il test comune."""
    tests, views, directions = prepare_score_views(data)
    available_systems = {row["humanizer"] for row in data["rewrites"]}
    systems = list(systems) if systems is not None else [*PRIMARY_SYSTEMS, IDENTITY]
    if not systems or len(set(systems)) != len(systems) or not set(systems) <= available_systems:
        raise ValueError("Richiedere condizioni distinte e presenti negli score")
    ok = {system: {str(r["source_id"]) for r in data["rewrites"]
                   if r["humanizer"] == system and r["status"] == "ok"} for system in available_systems}
    if not set(PRIMARY_SYSTEMS) <= available_systems:
        raise ValueError("Il confronto comune richiede tutti i quattro riscrittori primari")
    common = set.intersection(*(ok[name] for name in PRIMARY_SYSTEMS))
    summaries, baseline, coverage, all_pairs = [], [], [], []
    strata = [("overall", "all", tests)]
    for field in ("domain", "generator"):
        strata.extend((field, value, [r for r in tests if r[field] == value])
                      for value in sorted({r[field] for r in tests}))
    for name, lookup in views.items():
        continuous = name != "majority_vote"
        entry = data["thresholds"][name]
        for stratum, value, selected in strata:
            n = len(selected)
            # Lo stadio 'prima' usa sempre soltanto umani e originali del test.
            original_pairs = []
            for source in selected:
                identifier = str(source["id"])
                human, machine = lookup[identifier, "human", ""], lookup[identifier, "machine", ""]
                original_pairs.append(dict(source_id=identifier, source_group_id=source["source_group_id"],
                    human_score=human["oriented_score"], original_score=machine["oriented_score"],
                    rewrite_score=machine["oriented_score"], human_is_ai=bool(human["is_ai"]),
                    original_is_ai=bool(machine["is_ai"]), rewrite_is_ai=bool(machine["is_ai"])))
            base = summarize_paired_metrics(pd.DataFrame(original_pairs), bootstrap=bootstrap,
                                            seed=seed, auroc=continuous)
            baseline.append(dict(detector=name, stratum=stratum, stratum_value=value,
                n_pairs=n, threshold=entry["threshold"], score_orientation=directions[name],
                calibration_n=entry["calibration_n"], calibration_fpr=entry["calibration_fpr"],
                target_fpr=entry.get("target_fpr"), auroc_available=continuous,
                auroc_unavailable_reason=base["auroc_unavailable_reason"],
                **{key: item for key, item in base.items() if key.endswith("_before") or "_before_ci95_" in key}))
            for system in systems:
                own = _pairs(selected, lookup, system, ok[system])
                known_tp = int(own.rewrite_is_ai.sum())
                tn_all = int(base["tn_before"])
                missing = n-len(own)
                coverage.append(dict(detector=name, humanizer=system, stratum=stratum, stratum_value=value,
                    n_test_sources=n, n_available=len(own), n_failed=missing, generation_coverage=len(own)/n,
                    accuracy_before_full=base["accuracy_before"],
                    accuracy_after_full_lower=(tn_all+known_tp)/(2*n),
                    accuracy_after_full_upper=(tn_all+known_tp+missing)/(2*n),
                    delta_accuracy_full_lower=(tn_all+known_tp)/(2*n)-base["accuracy_before"],
                    delta_accuracy_full_upper=(tn_all+known_tp+missing)/(2*n)-base["accuracy_before"],
                    missing_score_policy="No score imputation; full accuracy bounds only; observed AUROC uses complete pairs"))
                for scope, subset in [("paired_available", own),
                                      ("common_success", _pairs(selected, lookup, system, ok[system] & common))]:
                    digest = hashlib.sha256(json.dumps([seed, name, scope, stratum, value]).encode()).hexdigest()
                    metrics = summarize_paired_metrics(subset, bootstrap=bootstrap,
                                                       seed=int(digest[:8], 16), auroc=continuous)
                    summaries.append(dict(detector=name, humanizer=system, scope=scope,
                        stratum=stratum, stratum_value=value, n_test_sources=n,
                        n_failed_for_system=missing, n_available_for_system=len(own),
                        n_excluded_from_scope=n-len(subset), evaluation_coverage=len(subset)/n,
                        generation_coverage=len(own)/n,
                        threshold=entry["threshold"], calibration_n=entry["calibration_n"],
                        calibration_fpr=entry["calibration_fpr"], target_fpr=entry.get("target_fpr"),
                        score_orientation=directions[name], **metrics))
                if stratum == "overall":
                    all_pairs.append(own.assign(in_common_success=own.source_id.isin(common)))
    nonempty_pairs = [part for part in all_pairs if not part.empty]
    pairs = (pd.concat(nonempty_pairs, ignore_index=True) if nonempty_pairs
             else pd.DataFrame(columns=PAIR_COLUMNS + ["in_common_success"]))
    return dict(metrics=pd.DataFrame(summaries), baseline=pd.DataFrame(baseline),
                coverage=pd.DataFrame(coverage), pairs=pairs,
                common_source_ids=sorted(common), systems=systems, directions=directions)
