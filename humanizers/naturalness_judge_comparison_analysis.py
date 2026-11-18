"""Confronta due giudici sugli stessi pannelli completi, senza chiamare servizi."""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .naturalness_panel_analysis import (
    _bootstrap_means, _cluster_draws, _interval, _json_value, _read_ratings,
    analyze_panel,
)


SLOT = ["source_id", "order_id", "repeat_index"]
PARAMETERS = [
    "protocol", "systems", "repetitions", "seed", "n_sources", "n_orders_per_source",
    "expected_completion_requests", "max_completion_tokens", "max_input_characters",
    "input_character_count", "temperature", "score_type", "ordering", "identical_repetitions",
]
SHIFT_COLUMNS = [
    "system", "n_sources", "n_source_groups", "reference_mean", "candidate_mean",
    "mean_shift_candidate_minus_reference", "ci95_low", "ci95_high",
]
CONTRAST_COLUMNS = [
    "system_a", "system_b", "n_sources", "n_source_groups", "reference_a_minus_b",
    "candidate_a_minus_b", "contrast_shift_candidate_minus_reference", "ci95_low", "ci95_high",
]
AGREEMENT_COLUMNS = [
    "scope", "stratum", "n_sources", "n_source_groups", "n_paired_ratings",
    "exact_agreement", "mean_absolute_difference", "spearman", "quadratic_weighted_kappa",
]


def _plan_slots(plan):
    """Identifica gli slot senza usare gli ID, che dipendono dal modello."""
    requests = plan.get("requests")
    if not isinstance(requests, list) or plan.get("expected_completion_requests") != len(requests):
        raise ValueError("Il numero delle richieste non coincide con il piano")
    if not isinstance(plan.get("model"), str) or not plan["model"].strip():
        raise ValueError("Ogni piano deve fissare il modello del giudice")
    slots, ids = {}, set()
    for item in requests:
        key = tuple(item[name] for name in SLOT)
        identifier = item.get("request_id")
        if key in slots or not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("Il piano contiene slot o ID duplicati/non validi")
        if item.get("payload", {}).get("model") != plan["model"]:
            raise ValueError("Il modello della richiesta differisce dal piano")
        slots[key] = item
        ids.add(identifier)
    if len({key[0] for key in slots}) != plan.get("n_sources"):
        raise ValueError("Il numero di fonti non coincide con il piano")
    return slots


def _validate_comparable(reference, candidate):
    left, right = _plan_slots(reference), _plan_slots(candidate)
    if left.keys() != right.keys():
        raise ValueError("I giudici devono ricevere esattamente gli stessi slot")
    for key in PARAMETERS:
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"Parametro scientifico diverso tra i giudici: {key}")
    for slot, request in left.items():
        normalized = []
        for value in (request, right[slot]):
            value = deepcopy(value)
            value.pop("request_id", None)
            value["payload"].pop("model", None)
            value["payload"].pop("reasoning_effort", None)
            normalized.append(value)
        if normalized[0] != normalized[1]:
            raise ValueError("Testi, ordine, metadati o parametri diversi nello stesso slot")


def _checked_ratings(plan, rows):
    for row in rows:
        if row.get("requested_model") != plan["model"] or row.get("label_source") != "LLM":
            raise ValueError("La provenienza della risposta non coincide con il piano")
        if row.get("status") not in {"ok", "invalid_response"}:
            raise ValueError("Stato della risposta non riconosciuto")
        if row["status"] == "ok" and row.get("response_model") != plan["model"]:
            raise ValueError("Il modello effettivo della risposta differisce dal piano")
    return _read_ratings(plan, rows)


def _filtered(plan, rows, common):
    result = deepcopy(plan)
    result["requests"] = [item for item in result["requests"] if item["source_id"] in common]
    result["n_sources"] = len(common)
    result["expected_completion_requests"] = len(result["requests"])
    return result, [row for row in rows if row["source_id"] in common]


def _agreement_values(reference, candidate):
    """Gli indici sugli slot sono descrittivi: le ripetizioni non sono nuove fonti."""
    x, y = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if not len(x):
        return dict(exact_agreement=None, mean_absolute_difference=None,
                    spearman=None, quadratic_weighted_kappa=None)
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    correlation = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) and np.std(ry) else None
    observed = np.zeros((5, 5))
    np.add.at(observed, (x.astype(int)-1, y.astype(int)-1), 1)
    observed /= len(x)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    weights = (np.arange(5)[:, None]-np.arange(5)[None, :])**2 / 16
    denominator = float((weights*expected).sum())
    kappa = 1-float((weights*observed).sum())/denominator if denominator else None
    return dict(exact_agreement=float(np.mean(x == y)),
                mean_absolute_difference=float(np.mean(np.abs(x-y))),
                spearman=correlation, quadratic_weighted_kappa=kappa)


def _comparison_tables(reference_scores, candidate_scores, systems, bootstrap, seed):
    columns = ["source_id", "source_group_id", "domain", "generator", "original_word_count",
               "system", "n_ratings_reference", "mean_rating_reference",
               "n_ratings_candidate", "mean_rating_candidate", "shift_candidate_minus_reference"]
    if reference_scores.empty:
        return (pd.DataFrame(columns=columns), pd.DataFrame(columns=SHIFT_COLUMNS),
                pd.DataFrame(columns=CONTRAST_COLUMNS))
    meta = ["source_id", "source_group_id", "domain", "generator", "original_word_count", "system"]
    sources = reference_scores[meta+["n_ratings", "mean_rating"]].merge(
        candidate_scores[meta+["n_ratings", "mean_rating"]], on=meta,
        suffixes=("_reference", "_candidate"), validate="one_to_one")
    sources["shift_candidate_minus_reference"] = sources.mean_rating_candidate-sources.mean_rating_reference
    ref = sources.pivot(index="source_id", columns="system", values="mean_rating_reference").reindex(columns=systems)
    cand = sources.pivot(index="source_id", columns="system", values="mean_rating_candidate").reindex(columns=systems)
    groups = sources.drop_duplicates("source_id").set_index("source_id").loc[ref.index, "source_group_id"]
    names, draws = _cluster_draws(groups, bootstrap, seed)
    shifts = cand.to_numpy()-ref.to_numpy()
    boots = _bootstrap_means(shifts, groups, names, draws)
    system_rows = []
    for index, system in enumerate(systems):
        low, high = _interval(boots[:, index], len(names))
        system_rows.append(dict(system=system, n_sources=len(ref), n_source_groups=len(names),
                                reference_mean=float(ref.iloc[:, index].mean()),
                                candidate_mean=float(cand.iloc[:, index].mean()),
                                mean_shift_candidate_minus_reference=float(shifts[:, index].mean()),
                                ci95_low=low, ci95_high=high))
    contrasts = []
    for a, b in combinations(range(len(systems)), 2):
        low, high = _interval(boots[:, a]-boots[:, b], len(names))
        contrasts.append(dict(system_a=systems[a], system_b=systems[b], n_sources=len(ref),
                              n_source_groups=len(names),
                              reference_a_minus_b=float((ref.iloc[:, a]-ref.iloc[:, b]).mean()),
                              candidate_a_minus_b=float((cand.iloc[:, a]-cand.iloc[:, b]).mean()),
                              contrast_shift_candidate_minus_reference=float((shifts[:, a]-shifts[:, b]).mean()),
                              ci95_low=low, ci95_high=high))
    return (sources.reindex(columns=columns), pd.DataFrame(system_rows, columns=SHIFT_COLUMNS),
            pd.DataFrame(contrasts, columns=CONTRAST_COLUMNS))


def _plots(system_shifts, positions, output):
    names = ["judge_system_means.png", "judge_position_effects.png"]
    if system_shifts.empty:
        for name in names:
            (output/name).unlink(missing_ok=True)
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    index = np.arange(len(system_shifts))
    ax.bar(index-.18, system_shifts.reference_mean, width=.36, label="Giudice di riferimento")
    ax.bar(index+.18, system_shifts.candidate_mean, width=.36, label="Secondo giudice")
    ax.set_xticks(index, system_shifts.system, rotation=12, ha="right", fontsize=8)
    ax.set_ylim(1, 5)
    ax.set_ylabel("Voto medio sugli stessi originali completi")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output/names[0], dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for role, frame in positions[positions.scope == "overall"].groupby("judge_role", sort=False):
        frame = frame.sort_values("position")
        ax.plot(frame.position, frame.mean_centered_rating, marker="o", label=role)
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xticks(sorted(positions.position.unique()))
    ax.set_xlabel("Posizione del testo nella richiesta")
    ax.set_ylabel("Voto centrato entro originale e metodo")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output/names[1], dpi=170)
    plt.close(fig)
    return names


def analyze_judge_comparison(reference_plan, reference_rows, candidate_plan, candidate_rows,
                             output, *, bootstrap=2000, seed=42):
    """Confronta i giudici solo sull'intersezione degli originali completi.

    I voti incompleti rimangono nei file grezzi. Gli intervalli ricampionano
    insieme i gruppi di fonte; gli indici ordinali sugli slot sono descrittivi.
    """
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 2:
        raise ValueError("Servono almeno due ricampionamenti bootstrap")
    reference_rows, candidate_rows = list(reference_rows), list(candidate_rows)
    _validate_comparable(reference_plan, candidate_plan)
    left = _checked_ratings(reference_plan, reference_rows)
    right = _checked_ratings(candidate_plan, candidate_rows)
    common = left[5] & right[5]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for role, rows, values in [("reference", reference_rows, left), ("candidate", candidate_rows, right)]:
        (output/f"{role}_judgments.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        values[0].to_csv(output/f"raw_{role}_ratings.csv", index=False)
    summaries = {}
    for role, plan, rows in [("reference", reference_plan, reference_rows),
                             ("candidate", candidate_plan, candidate_rows)]:
        matched_plan, matched_rows = _filtered(plan, rows, common)
        summaries[role] = analyze_panel(matched_plan, matched_rows, output/f"matched_{role}",
                                        bootstrap=bootstrap, seed=seed)
    combined = {}
    for filename in ["system_summary.csv", "paired_differences.csv", "position_effects.csv"]:
        parts = []
        for role, plan in [("reference", reference_plan), ("candidate", candidate_plan)]:
            table = pd.read_csv(output/f"matched_{role}"/filename)
            table.insert(0, "judge_model", plan["model"])
            table.insert(0, "judge_role", role)
            parts.append(table)
        combined[filename.replace(".csv", "_by_judge.csv")] = pd.concat(parts, ignore_index=True)
    ref_scores = pd.read_csv(output/"matched_reference"/"source_system_scores.csv")
    cand_scores = pd.read_csv(output/"matched_candidate"/"source_system_scores.csv")
    source_scores, shifts, contrasts = _comparison_tables(
        ref_scores, cand_scores, reference_plan["systems"], bootstrap, seed)
    keys = SLOT+["system", "source_group_id", "domain", "generator", "original_word_count", "position"]
    ref_ratings = left[0][left[0].source_id.isin(common)]
    cand_ratings = right[0][right[0].source_id.isin(common)]
    paired = ref_ratings[keys+["rating"]].merge(cand_ratings[keys+["rating"]],
                on=keys, suffixes=("_reference", "_candidate"), validate="one_to_one")
    agreement_rows = []
    scopes = [("overall", "all", paired)] if len(paired) else []
    scopes += [("system", str(system), frame) for system, frame in paired.groupby("system", sort=True)]
    for scope, stratum, frame in scopes:
        agreement_rows.append(dict(scope=scope, stratum=stratum, n_sources=frame.source_id.nunique(),
            n_source_groups=frame.source_group_id.nunique(), n_paired_ratings=len(frame),
            **_agreement_values(frame.rating_reference, frame.rating_candidate)))
    agreement = pd.DataFrame(agreement_rows, columns=AGREEMENT_COLUMNS)
    all_sources = sorted(left[1])
    coverage = pd.DataFrame([
        dict(source_id=source, source_group_id=left[1][source]["source_group_id"],
             domain=left[1][source]["domain"], complete_reference=source in left[5],
             complete_candidate=source in right[5], included_in_comparison=source in common,
             reference_valid_requests=left[3].get(source, 0),
             candidate_valid_requests=right[3].get(source, 0),
             planned_requests=left[2][source]) for source in all_sources
    ], columns=["source_id", "source_group_id", "domain", "complete_reference", "complete_candidate",
                "included_in_comparison", "reference_valid_requests", "candidate_valid_requests", "planned_requests"])
    tables = dict(combined, **{
        "source_system_comparison.csv": source_scores, "system_shifts.csv": shifts,
        "humanizer_contrast_shifts.csv": contrasts, "ordinal_agreement.csv": agreement,
        "matched_ratings.csv": paired, "comparison_coverage.csv": coverage,
    })
    for name, frame in tables.items():
        frame.to_csv(output/name, index=False)
    plots = _plots(shifts, combined["position_effects_by_judge.csv"], output)
    overall = agreement[agreement.scope == "overall"]
    summary = dict(
        protocol="naturalness_cross_judge_matched_sources_v1", status="partial" if len(common)<len(all_sources) else "complete",
        label_source="LLM_not_human", human_annotation_produced=False,
        reference_model=reference_plan["model"], candidate_model=candidate_plan["model"],
        n_planned_sources=len(all_sources), n_complete_reference=len(left[5]),
        n_complete_candidate=len(right[5]), n_common_complete_sources=len(common),
        common_complete_source_ids=sorted(common),
        n_common_source_groups=int(coverage.loc[coverage.included_in_comparison, "source_group_id"].nunique()),
        n_paired_ratings=len(paired), common_complete_coverage=len(common)/len(all_sources) if all_sources else None,
        primary_estimator="source-system means on the same complete original texts for both judges",
        incomplete_panel_policy="Excluded from primary comparisons; retained in raw ratings and judgment files",
        confidence_intervals="Exploratory pointwise percentile 95%; paired bootstrap of source_group_id clusters; no multiplicity correction",
        bootstrap_draws=bootstrap, bootstrap_seed=seed,
        ordinal_agreement_scope="Descriptive paired slot × system ratings; repeated calls are not independent samples",
        ordinal_agreement=overall.iloc[0].to_dict() if len(overall) else _agreement_values([], []),
        judge_scope="Agreement between LLM judges is not human validation or proof of naturalness",
        family_scope="A different model family reduces one direct family overlap but does not guarantee neutrality",
        matched_analyses=summaries, tables=list(tables), figures=plots,
    )
    summary = _json_value(summary)
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                allow_nan=False)+"\n", encoding="utf-8")
    return summary
