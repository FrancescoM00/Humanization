"""Rianalizza lunghezza e rumore con campioni comuni e confronti appaiati."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .detector_performance import PRIMARY_SYSTEMS, load_saved_detection, prepare_score_views
from .null_control_statistics import holm_adjust, mcnemar_exact, wilson_interval


CONTROLS = ("length", "lexical_noise", "length_and_lexical_noise")
SCOPES = ("eligible_per_humanizer", "common_eligible", "all_success_per_humanizer",
          "common_success_for_control", "common_all_controls")


def load_null_study(path):
    """Recupera anche il pilot congelato, per identificare le 84 fonti nuove."""
    data = load_saved_detection(path)
    member = "cohorts/pilot.jsonl"
    if Path(path).is_dir():
        raw = (Path(path) / member).read_bytes()
    else:
        with zipfile.ZipFile(path) as archive:
            if archive.namelist().count(member) != 1:
                raise ValueError("Manca una copia univoca del pilot originale")
            raw = archive.read(member)
    pilot = [json.loads(line) for line in raw.splitlines() if line.strip()]
    data["pilot_source_ids"] = {str(row["id"]) for row in pilot if row["split"] == "test"}
    data["input_sha256"][member] = hashlib.sha256(raw).hexdigest()
    return data


def _condition(kind, system):
    if kind == "lexical_noise":
        return "null_lexical_noise"
    return ("null_length__" if kind == "length" else "null_length_noise__") + system


def _eligibility(data, tests):
    """Distingue mancata generazione, allungamento e assenza di parole modificabili."""
    lookup = {(r["humanizer"], str(r["source_id"])): r for r in data["rewrites"]}
    rows = []
    for source in tests:
        identifier = str(source["id"])
        for system in PRIMARY_SYSTEMS:
            reference = lookup[system, identifier]
            length = lookup[_condition("length", system), identifier]["metadata"]
            valid = reference["status"] == "ok"
            original_n, target = length.get("original_n_words"), length.get("target_n_words")
            longer = bool(valid and target is not None and target > original_n)
            for kind in CONTROLS:
                control = lookup[_condition(kind, system), identifier]
                metadata = control["metadata"]
                strict = bool(valid and control["status"] == "ok" and metadata["comparison_eligible"])
                noise_ok = kind == "length" or bool(metadata.get("noise_applied"))
                capped = bool(valid and control["status"] == "ok" and noise_ok
                              and target is not None and target > 0)
                if not valid:
                    reason = "reference_" + reference["status"]
                elif control["status"] != "ok":
                    reason = "control_" + control["status"]
                elif kind != "lexical_noise" and longer:
                    reason = "rewrite_longer_than_original"
                elif not noise_ok:
                    reason = "no_allowed_lexical_replacement"
                elif not strict:
                    reason = "other_ineligible"
                else:
                    reason = "included"
                rows.append(dict(source_id=identifier, source_group_id=source["source_group_id"],
                    domain=source["domain"], generator=source["generator"], humanizer=system,
                    control_kind=kind, condition=control["humanizer"], reference_status=reference["status"],
                    control_status=control["status"], original_n_words=original_n,
                    reference_n_words=target, control_n_words=metadata.get("modified_n_words"),
                    rewrite_longer=longer, strict_eligible=strict, capped_eligible=capped,
                    noise_applied=bool(metadata.get("noise_applied")), strict_exclusion_reason=reason,
                    hard_cut=bool(metadata.get("hard_cut")),
                    capped_control_not_length_matched=bool(kind != "lexical_noise" and longer)))
    return pd.DataFrame(rows)


def _summarize_pairs(part):
    """L'ASR usa soltanto gli originali rilevati; il TPR usa tutte le fonti selezionate."""
    n = len(part)
    baseline = part.original_is_ai.to_numpy(dtype=bool)
    control = part.control_is_ai.to_numpy(dtype=bool)
    reference = part.reference_is_ai.to_numpy(dtype=bool)
    a, b = baseline & ~control, baseline & ~reference
    denominator = int(baseline.sum())
    control_only = int((a & ~b).sum())
    reference_only = int((b & ~a).sum())
    out = dict(eligible_sources=n, eligible_original_detected=denominator,
               control_only_successes=control_only, reference_only_successes=reference_only,
               mcnemar_p=mcnemar_exact(control_only, reference_only), testable=denominator > 0,
               n_expanded_reference=int(part.rewrite_longer.sum()),
               n_capped_controls_not_length_matched=int(part.capped_control_not_length_matched.sum()))
    for label, successes, detected in (("control", int(a.sum()), control),
                                       ("reference", int(b.sum()), reference)):
        low, high = wilson_interval(successes, denominator)
        out.update({label + "_successes": successes,
                    label + "_asr": successes / denominator if denominator else np.nan,
                    label + "_asr_ci95_low": low, label + "_asr_ci95_high": high,
                    "eligible_" + label + "_detected": int(detected.sum()),
                    "eligible_" + label + "_tpr": float(detected.mean()) if n else np.nan})
    out["eligible_original_tpr"] = float(baseline.mean()) if n else np.nan
    out["delta_asr_reference_minus_control"] = out["reference_asr"] - out["control_asr"]
    out["reference_minus_control_tpr"] = out["eligible_reference_tpr"] - out["eligible_control_tpr"]
    return out


def _correct_multiplicity(table):
    """Le famiglie includono sempre tutti i detector e tutti i quattro riscrittori."""
    table = table.copy()
    for fields, column in [(["panel", "scope", "control_kind"], "mcnemar_p_holm_20"),
                           (["panel", "scope"], "mcnemar_p_holm_60")]:
        for _, index in table.groupby(fields, sort=False).groups.items():
            table.loc[index, column] = holm_adjust(table.loc[index, "mcnemar_p"].tolist())
            table.loc[index, column + "_family_size"] = len(index)
    table["greater_than_control_holm20"] = (table.delta_asr_reference_minus_control.gt(0)
                                            & table.mcnemar_p_holm_20.lt(.05) & table.testable)
    table["greater_than_control_holm60"] = (table.delta_asr_reference_minus_control.gt(0)
                                            & table.mcnemar_p_holm_60.lt(.05) & table.testable)
    return table


def analyze_null_controls(data):
    """Confronta i tre controlli senza ricalibrare soglie o eseguire detector."""
    tests, views, directions = prepare_score_views(data)
    if len({r["source_group_id"] for r in tests}) != len(tests):
        raise ValueError("Wilson e McNemar richiedono qui una sola osservazione per gruppo di fonte")
    test_ids = {str(row["id"]) for row in tests}
    pilot_ids = set(data["pilot_source_ids"])
    if not pilot_ids or not pilot_ids < test_ids:
        raise ValueError("Il pilot deve essere un sottoinsieme proprio e non vuoto del test")
    panels = {"all126": test_ids, "unseen84": test_ids - pilot_ids}
    eligible = _eligibility(data, tests)
    raw_pairs = []
    for detector, lookup in views.items():
        for row in eligible[eligible.capped_eligible].to_dict("records"):
            source = row["source_id"]
            original = lookup[source, "machine", ""]
            control = lookup[source, "rewrite", row["condition"]]
            reference = lookup[source, "rewrite", row["humanizer"]]
            raw_pairs.append(dict(**row, detector=detector,
                original_raw_score=original["score"], control_raw_score=control["score"],
                reference_raw_score=reference["score"], original_is_ai=bool(original["is_ai"]),
                control_is_ai=bool(control["is_ai"]), reference_is_ai=bool(reference["is_ai"]),
                threshold=data["thresholds"][detector]["threshold"], score_orientation=directions[detector]))
    pairs = pd.DataFrame(raw_pairs)
    rows, memberships, coverage = [], [], []
    for panel, identifiers in panels.items():
        selected = eligible[eligible.source_id.isin(identifiers)]
        strict_sets, capped_sets = {}, {}
        for kind in CONTROLS:
            for system in PRIMARY_SYSTEMS:
                part = selected[selected.control_kind.eq(kind) & selected.humanizer.eq(system)]
                strict_sets[kind, system] = set(part.loc[part.strict_eligible, "source_id"])
                capped_sets[kind, system] = set(part.loc[part.capped_eligible, "source_id"])
        common_all = set.intersection(*strict_sets.values())
        for scope in SCOPES:
            for kind in CONTROLS:
                strict_common = set.intersection(*(strict_sets[kind, s] for s in PRIMARY_SYSTEMS))
                capped_common = set.intersection(*(capped_sets[kind, s] for s in PRIMARY_SYSTEMS))
                for system in PRIMARY_SYSTEMS:
                    sources = {"eligible_per_humanizer": strict_sets[kind, system],
                               "common_eligible": strict_common,
                               "all_success_per_humanizer": capped_sets[kind, system],
                               "common_success_for_control": capped_common,
                               "common_all_controls": common_all}[scope]
                    own = selected[selected.control_kind.eq(kind) & selected.humanizer.eq(system)]
                    info = dict(panel=panel, scope=scope, control_kind=kind, humanizer=system,
                        total_sources=len(identifiers), eligible_sources=len(sources),
                        excluded_sources=len(identifiers)-len(sources),
                        excluded_failed_reference=int(own.reference_status.ne("ok").sum()),
                        own_strict_eligible_sources=len(strict_sets[kind, system]),
                        own_capped_eligible_sources=len(capped_sets[kind, system]),
                        own_expanded_references=int(own.rewrite_longer.sum()),
                        own_excluded_no_noise=int((own.reference_status.eq("ok") & ~own.noise_applied).sum())
                            if kind != "length" else 0,
                        excluded_to_common_panel=len((capped_sets if scope == "common_success_for_control" else strict_sets)[kind, system] - sources)
                            if scope in {"common_success_for_control", "common_eligible", "common_all_controls"} else 0)
                    coverage.append(info)
                    memberships.extend(dict(panel=panel, scope=scope, control_kind=kind,
                                            humanizer=system, source_id=s) for s in sorted(sources))
                    for detector in views:
                        part = pairs[pairs.detector.eq(detector) & pairs.humanizer.eq(system)
                                     & pairs.control_kind.eq(kind) & pairs.source_id.isin(sources)]
                        if len(part) != len(sources):
                            raise ValueError("Manca una coppia richiesta dal pannello")
                        counts = _summarize_pairs(part)
                        rows.append(dict(**info, detector=detector, condition=_condition(kind, system),
                            **{k:v for k,v in counts.items() if k != "eligible_sources"},
                            threshold=data["thresholds"][detector]["threshold"],
                            score_orientation=directions[detector],
                            same_sources_all_humanizers=scope in {"common_eligible", "common_success_for_control", "common_all_controls"},
                            same_sources_all_controls=scope == "common_all_controls"))
    summary = _correct_multiplicity(pd.DataFrame(rows))
    gates = []
    for (panel, scope, detector, system), part in summary.groupby(["panel", "scope", "detector", "humanizer"], sort=False):
        if len(part) != 3:
            raise ValueError("La verifica congiunta richiede tutti i tre controlli")
        gates.append(dict(panel=panel, scope=scope, detector=detector, humanizer=system,
            min_sources=int(part.eligible_sources.min()), max_sources=int(part.eligible_sources.max()),
            same_sources_all_controls=scope == "common_all_controls",
            positive_delta_all_three=bool(part.delta_asr_reference_minus_control.gt(0).all()),
            greater_than_all_three_holm60=bool(part.greater_than_control_holm60.all()),
            max_mcnemar_p_holm60=float(part.mcnemar_p_holm_60.max())))
    eligible["historical_panel"] = np.where(eligible.source_id.isin(pilot_ids), "pilot42", "unseen84")
    return dict(summary=summary, eligibility=eligible, coverage=pd.DataFrame(coverage),
                pairs=pairs, memberships=pd.DataFrame(memberships), three_controls=pd.DataFrame(gates),
                panel_sizes={p:len(s) for p,s in panels.items()})
