"""Aggrega i rating anonimi per fonte e misura sensibilità all'ordine e stabilità."""
from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


RATING_COLUMNS = [
    "request_id", "source_id", "source_group_id", "domain", "generator",
    "original_word_count", "order_id", "repeat_index", "system", "position",
    "rating", "complete_panel", "label_source",
]
SOURCE_COLUMNS = [
    "source_id", "source_group_id", "domain", "generator", "original_word_count",
    "length_bin", "eligible_generation", "planned_requests", "valid_requests",
    "invalid_requests", "missing_requests", "complete_panel", "status",
]
SCORE_COLUMNS = [
    "source_id", "source_group_id", "domain", "generator", "original_word_count",
    "system", "n_ratings", "mean_rating", "median_rating", "sd_rating",
    "min_rating", "max_rating", "ceiling_rate",
    *[f"rating_{score}_count" for score in range(1, 6)],
    *[f"rating_{score}_fraction" for score in range(1, 6)],
]
SYSTEM_COLUMNS = [
    "system", "n_sources", "n_source_groups", "n_ratings", "mean_rating",
    "ci95_low", "ci95_high", "median_source_mean", "median_rating",
    "mean_source_median", "mean_source_sd", "ceiling_rate",
    *[f"rating_{score}_fraction" for score in range(1, 6)],
]
PAIR_COLUMNS = [
    "system_a", "system_b", "n_sources", "n_source_groups", "mean_difference",
    "ci95_low", "ci95_high", "median_source_difference", "source_win_fraction",
    "source_tie_fraction", "source_loss_fraction",
]
REPEAT_COLUMNS = [
    "source_id", "source_group_id", "domain", "generator", "system", "order_id",
    "planned_repetitions", "valid_repetitions", "mean_rating", "median_rating",
    "sd_rating", "rating_range", "exact_agreement", "complete_panel",
]
ORDER_COLUMNS = [
    "source_id", "source_group_id", "domain", "generator", "system",
    "n_orders", "n_ratings", "order_mean_sd", "order_mean_range",
    "within_order_sd_rms", "within_order_exact_fraction", "complete_panel",
]
POSITION_COLUMNS = [
    "scope", "stratum", "position", "n_sources", "n_source_groups",
    "n_ratings", "mean_centered_rating", "ci95_low", "ci95_high", "mean_raw_rating",
]
COVERAGE_COLUMNS = [
    "scope", "stratum", "n_selected_sources", "n_eligible_sources",
    "n_generation_ineligible_sources", "n_complete_sources", "n_partial_sources",
    "n_not_started_sources", "eligible_fraction", "complete_fraction_selected",
    "complete_fraction_eligible", "planned_requests", "valid_requests",
    "invalid_requests", "missing_requests",
]


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _length_bin(value):
    if value is None or pd.isna(value):
        return "unknown"
    return "<150" if value < 150 else "150–299" if value < 300 else ">=300"


def _metadata(item):
    source = str(item["source_id"])
    group = item.get("source_group_id")
    if group is None or not str(group).strip():
        raise ValueError("Every planned source requires its source_group_id")
    count = item.get("original_word_count")
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("original_word_count must be a nonnegative integer")
    return dict(source_id=source, source_group_id=str(group),
                domain=item.get("domain") or "unknown", generator=item.get("generator") or "unknown",
                original_word_count=count)


def _cluster_draws(groups, bootstrap, seed):
    names = sorted({str(group) for group in groups})
    if not names:
        return names, np.empty((bootstrap, 0), dtype=np.int64)
    draws = np.random.default_rng(seed).multinomial(
        len(names), np.full(len(names), 1 / len(names)), size=bootstrap)
    return names, draws


def _bootstrap_means(values, groups, group_names, draws):
    """Ricampiona gruppi interi e conserva il peso di ogni fonte inclusa nel gruppo."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    index = {name: position for position, name in enumerate(group_names)}
    positions = np.array([index[str(group)] for group in groups], dtype=int)
    sums = np.zeros((len(group_names), values.shape[1]), dtype=float)
    np.add.at(sums, positions, values)
    counts = np.bincount(positions, minlength=len(group_names))
    return (draws @ sums) / (draws @ counts)[:, None]


def _interval(values, n_groups):
    if n_groups < 2 or not len(values):
        return None, None
    return tuple(float(value) for value in np.quantile(values, [.025, .975]))


def _read_ratings(plan, judgments):
    systems = list(plan.get("systems", []))
    requests = list(plan.get("requests", []))
    lookup = {item["request_id"]: item for item in requests}
    if len(lookup) != len(requests):
        raise ValueError("The plan has duplicate request IDs")
    if len(systems) != len(set(systems)):
        raise ValueError("The plan has duplicate systems")
    source_meta, expected, order_counts, position_counts, order_sequences, repetitions = {}, {}, {}, {}, {}, set()
    for item in requests:
        meta = _metadata(item)
        source = meta["source_id"]
        if source in source_meta and source_meta[source] != meta:
            raise ValueError("A source has inconsistent metadata in the plan")
        source_meta[source] = meta
        if len(item["system_order"]) != len(systems) or set(item["system_order"]) != set(systems):
            raise ValueError("Each planned request must contain all selected systems once")
        expected[source] = expected.get(source, 0) + 1
        key = source, str(item["order_id"])
        sequence = tuple(item["system_order"])
        if key in order_sequences and order_sequences[key] != sequence:
            raise ValueError("A repeated presentation order changed its system sequence")
        order_sequences[key] = sequence
        repetition = source, str(item["order_id"]), item["repeat_index"]
        if repetition in repetitions:
            raise ValueError("The plan repeats an order/repetition identifier")
        repetitions.add(repetition)
        order_counts[key] = order_counts.get(key, 0) + 1
        counts = position_counts.setdefault(source, {})
        for position, system in enumerate(sequence, start=1):
            counts[system, position] = counts.get((system, position), 0) + 1
    for counts in position_counts.values():
        if len(counts) != len(systems)**2 or len(set(counts.values())) != 1:
            raise ValueError("Each source plan must balance every system across all positions")
    seen, valid, invalid, ratings = set(), {}, {}, []
    for row in judgments:
        request_id = row.get("request_id")
        if request_id not in lookup or request_id in seen:
            raise ValueError("Judgments must have unique IDs belonging to the frozen plan")
        seen.add(request_id)
        item = lookup[request_id]
        for key in ("source_id", "source_group_id", "domain", "generator", "order_id",
                    "repeat_index", "system_order", "original_word_count"):
            if row.get(key) != item.get(key):
                raise ValueError(f"Stored judgment metadata differ from the plan: {key}")
        source = str(item["source_id"])
        if row.get("status") != "ok":
            invalid[source] = invalid.get(source, 0) + 1
            continue
        result = row.get("judgment", {})
        ids = [f"T{position}" for position in range(1, len(systems) + 1)]
        if not isinstance(result, dict) or set(result) != {"ratings"}:
            raise ValueError("A successful judgment requires the exact rating object")
        scores = result["ratings"]
        if not isinstance(scores, dict) or set(scores) != set(ids):
            raise ValueError("A successful judgment must rate every anonymous passage exactly once")
        if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5
               for value in scores.values()):
            raise ValueError("Ratings must be integers between 1 and 5")
        valid[source] = valid.get(source, 0) + 1
        meta = source_meta[source]
        for position, system in enumerate(item["system_order"], start=1):
            ratings.append(dict(meta, request_id=request_id, order_id=str(item["order_id"]),
                                repeat_index=item["repeat_index"], system=system,
                                position=position, rating=scores[f"T{position}"], label_source="LLM"))
    complete = {source for source, count in expected.items() if valid.get(source, 0) == count}
    for row in ratings:
        row["complete_panel"] = row["source_id"] in complete
    return (pd.DataFrame(ratings, columns=RATING_COLUMNS), source_meta, expected,
            valid, invalid, complete, order_counts)


def _source_coverage(eligibility, source_meta, expected, valid, invalid, complete):
    selected = list(eligibility.get("selected_source_ids", source_meta))
    selected = [str(source) for source in selected]
    if len(selected) != len(set(selected)) or not set(source_meta) <= set(selected):
        raise ValueError("The eligibility record does not cover the planned sources exactly once")
    failures = {str(row["source_id"]): row for row in eligibility.get("failures", [])}
    if set(failures) & set(source_meta):
        raise ValueError("A source cannot be both planned and generation-ineligible")
    result = []
    for source in selected:
        item = source_meta.get(source, failures.get(source, {}))
        meta = dict(source_id=source, source_group_id=str(item.get("source_group_id") or source),
                    domain=item.get("domain") or "unknown", generator=item.get("generator") or "unknown",
                    original_word_count=item.get("original_word_count"))
        planned = expected.get(source, 0)
        n_valid, n_invalid = valid.get(source, 0), invalid.get(source, 0)
        eligible = source in source_meta
        status = ("complete" if source in complete else "partial" if n_valid or n_invalid
                  else "not_started" if eligible else "generation_ineligible")
        result.append(dict(meta, length_bin=_length_bin(meta["original_word_count"]),
                           eligible_generation=eligible, planned_requests=planned,
                           valid_requests=n_valid, invalid_requests=n_invalid,
                           missing_requests=planned - n_valid - n_invalid,
                           complete_panel=source in complete, status=status))
    return pd.DataFrame(result, columns=SOURCE_COLUMNS)


def _coverage_table(sources):
    scopes = [("overall", "all", sources)]
    for column in ("domain", "length_bin"):
        scopes.extend((column, str(name), part) for name, part in sources.groupby(column, sort=True))
    rows = []
    for scope, stratum, part in scopes:
        n = len(part)
        eligible = int(part.eligible_generation.sum())
        complete = int(part.complete_panel.sum())
        rows.append(dict(scope=scope, stratum=stratum, n_selected_sources=n,
                         n_eligible_sources=eligible, n_generation_ineligible_sources=n-eligible,
                         n_complete_sources=complete, n_partial_sources=int(part.status.eq("partial").sum()),
                         n_not_started_sources=int(part.status.eq("not_started").sum()),
                         eligible_fraction=eligible/n if n else None,
                         complete_fraction_selected=complete/n if n else None,
                         complete_fraction_eligible=complete/eligible if eligible else None,
                         **{name: int(part[name].sum()) for name in
                            ("planned_requests", "valid_requests", "invalid_requests", "missing_requests")}))
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def _source_scores(primary):
    rows = []
    keys = ["source_id", "source_group_id", "domain", "generator", "system"]
    for key, part in primary.groupby(keys, sort=True):
        values = part.rating.to_numpy(dtype=float)
        row = dict(zip(keys, key), original_word_count=part.original_word_count.iloc[0],
                   n_ratings=len(values), mean_rating=float(values.mean()),
                   median_rating=float(np.median(values)),
                   sd_rating=float(values.std(ddof=1)) if len(values) > 1 else None,
                   min_rating=int(values.min()), max_rating=int(values.max()),
                   ceiling_rate=float(np.mean(values == 5)))
        for score in range(1, 6):
            row[f"rating_{score}_count"] = int(np.sum(values == score))
            row[f"rating_{score}_fraction"] = float(np.mean(values == score))
        rows.append(row)
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def _system_tables(scores, primary, systems, bootstrap, seed):
    if not len(scores):
        return pd.DataFrame(columns=SYSTEM_COLUMNS), pd.DataFrame(columns=PAIR_COLUMNS)
    means = scores.pivot(index="source_id", columns="system", values="mean_rating").reindex(columns=systems)
    if means.isna().any().any():
        raise ValueError("Primary comparisons require all systems on the same complete sources")
    meta = scores.drop_duplicates("source_id").set_index("source_id").loc[means.index]
    group_names, draws = _cluster_draws(meta.source_group_id, bootstrap, seed)
    boots = _bootstrap_means(means.to_numpy(), meta.source_group_id, group_names, draws)
    rows = []
    for position, system in enumerate(systems):
        part = scores[scores.system == system]
        raw = primary[primary.system == system]
        low, high = _interval(boots[:, position], len(group_names))
        rows.append(dict(system=system, n_sources=len(part), n_source_groups=len(group_names),
                         n_ratings=len(raw), mean_rating=float(part.mean_rating.mean()),
                         ci95_low=low, ci95_high=high,
                         median_source_mean=float(part.mean_rating.median()),
                         median_rating=float(raw.rating.median()),
                         mean_source_median=float(part.median_rating.mean()),
                         mean_source_sd=float(part.sd_rating.mean()),
                         ceiling_rate=float(part.ceiling_rate.mean()),
                         **{f"rating_{score}_fraction": float(part[f"rating_{score}_fraction"].mean())
                            for score in range(1, 6)}))
    pairs = []
    for left, right in combinations(range(len(systems)), 2):
        delta = means.iloc[:, left].to_numpy() - means.iloc[:, right].to_numpy()
        low, high = _interval(boots[:, left] - boots[:, right], len(group_names))
        pairs.append(dict(system_a=systems[left], system_b=systems[right], n_sources=len(delta),
                          n_source_groups=len(group_names), mean_difference=float(delta.mean()),
                          ci95_low=low, ci95_high=high, median_source_difference=float(np.median(delta)),
                          source_win_fraction=float(np.mean(delta > 0)),
                          source_tie_fraction=float(np.mean(delta == 0)),
                          source_loss_fraction=float(np.mean(delta < 0))))
    return pd.DataFrame(rows, columns=SYSTEM_COLUMNS), pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def _repeatability(ratings, planned_orders):
    rows = []
    keys = ["source_id", "source_group_id", "domain", "generator", "system", "order_id"]
    for key, part in ratings.groupby(keys, sort=True):
        values = part.rating.to_numpy(dtype=float)
        spread = float(values.max() - values.min())
        rows.append(dict(zip(keys, key), planned_repetitions=planned_orders[key[0], key[-1]],
                         valid_repetitions=len(values), mean_rating=float(values.mean()),
                         median_rating=float(np.median(values)),
                         sd_rating=float(values.std(ddof=1)) if len(values) > 1 else None,
                         rating_range=spread, exact_agreement=spread == 0 if len(values) > 1 else None,
                         complete_panel=bool(part.complete_panel.iloc[0])))
    return pd.DataFrame(rows, columns=REPEAT_COLUMNS)


def _order_sensitivity(ratings, repeatability):
    rows = []
    keys = ["source_id", "source_group_id", "domain", "generator", "system"]
    for key, part in ratings.groupby(keys, sort=True):
        means = part.groupby("order_id").rating.mean()
        residual = part.rating.to_numpy() - part.order_id.map(means).to_numpy()
        df = len(part) - len(means)
        exact = repeatability[(repeatability.source_id == key[0]) & (repeatability.system == key[-1])]
        exact = exact[exact.valid_repetitions >= 2]
        rows.append(dict(zip(keys, key), n_orders=len(means), n_ratings=len(part),
                         order_mean_sd=float(means.std(ddof=1)) if len(means) > 1 else None,
                         order_mean_range=float(means.max() - means.min()),
                         within_order_sd_rms=float(np.sqrt(np.sum(residual**2)/df)) if df else None,
                         within_order_exact_fraction=float(exact.exact_agreement.astype(float).mean())
                         if len(exact) else None, complete_panel=bool(part.complete_panel.iloc[0])))
    return pd.DataFrame(rows, columns=ORDER_COLUMNS)


def _position_effects(primary, bootstrap, seed):
    if not len(primary):
        return pd.DataFrame(columns=POSITION_COLUMNS)
    part = primary.copy()
    part["centered_rating"] = part.rating - part.groupby(["source_id", "system"]).rating.transform("mean")
    scopes = [("overall", "all", part)]
    scopes.extend(("domain", str(domain), frame) for domain, frame in part.groupby("domain", sort=True))
    rows = []
    for index, (scope, stratum, frame) in enumerate(scopes):
        source_positions = frame.groupby(["source_id", "position"]).centered_rating.mean().unstack("position")
        if source_positions.isna().any().any():
            raise ValueError("Complete panels must cover every presentation position")
        meta = frame.drop_duplicates("source_id").set_index("source_id").loc[source_positions.index]
        group_names, draws = _cluster_draws(meta.source_group_id, bootstrap, seed+1001+index)
        boots = _bootstrap_means(source_positions.to_numpy(), meta.source_group_id, group_names, draws)
        for column, position in enumerate(source_positions.columns):
            raw = frame[frame.position == position]
            low, high = _interval(boots[:, column], len(group_names))
            rows.append(dict(scope=scope, stratum=stratum, position=int(position),
                             n_sources=len(source_positions), n_source_groups=len(group_names),
                             n_ratings=len(raw), mean_centered_rating=float(source_positions[position].mean()),
                             ci95_low=low, ci95_high=high, mean_raw_rating=float(raw.rating.mean())))
    return pd.DataFrame(rows, columns=POSITION_COLUMNS)


def _plot_tables(system_summary, position_effects, output):
    if not len(system_summary):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    bottom = np.zeros(len(system_summary))
    for score, color in zip(range(1, 6), plt.cm.viridis(np.linspace(.1, .9, 5))):
        values = system_summary[f"rating_{score}_fraction"].to_numpy()
        ax.bar(np.arange(len(values)), values, bottom=bottom, color=color, label=str(score))
        bottom += values
    ax.set_xticks(np.arange(len(system_summary)), system_summary.system, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("Quota di rating")
    ax.set_ylim(0, 1)
    ax.legend(title="Voto", ncol=5, frameon=False)
    ax.set_title("Naturalezza: distribuzione dei voti sui pannelli completi")
    fig.tight_layout()
    path = output / "rating_distribution.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    files.append(path.name)
    frame = position_effects[position_effects.scope == "overall"].sort_values("position")
    if len(frame):
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        ax.axhline(0, color="gray", lw=.8)
        ax.scatter(frame.position, frame.mean_centered_rating, color="#187b80")
        finite = frame.dropna(subset=["ci95_low", "ci95_high"])
        if len(finite):
            ax.errorbar(finite.position, finite.mean_centered_rating,
                        yerr=[np.maximum(0, finite.mean_centered_rating-finite.ci95_low),
                              np.maximum(0, finite.ci95_high-finite.mean_centered_rating)],
                        fmt="none", capsize=4, color="#187b80")
        ax.set_xticks(frame.position)
        ax.set_xlabel("Posizione del testo nella richiesta")
        ax.set_ylabel("Voto centrato entro fonte e metodo")
        ax.set_title("Sensibilità alla posizione; IC 95% per gruppo di fonte")
        fig.tight_layout()
        path = output / "position_effects.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        files.append(path.name)
    return files


def analyze_panel(plan, judgments, output, *, eligibility=None, bootstrap=2000, seed=42):
    """Usa come unità indipendente la fonte, con bootstrap appaiato dei suoi gruppi."""
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 2:
        raise ValueError("Use at least two bootstrap draws")
    eligibility = eligibility or {}
    systems = list(plan.get("systems") or eligibility.get("systems") or [])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ratings, meta, expected, valid, invalid, complete, orders = _read_ratings(plan, judgments)
    sources = _source_coverage(eligibility, meta, expected, valid, invalid, complete)
    coverage = _coverage_table(sources)
    primary = ratings[ratings.complete_panel.eq(True)].copy()
    scores = _source_scores(primary)
    system_summary, pairs = _system_tables(scores, primary, systems, bootstrap, seed)
    repeatability = _repeatability(ratings, orders)
    sensitivity = _order_sensitivity(ratings, repeatability)
    positions = _position_effects(primary, bootstrap, seed)
    tables = {
        "ratings.csv": ratings, "source_coverage.csv": sources, "coverage.csv": coverage,
        "source_system_scores.csv": scores, "system_summary.csv": system_summary,
        "paired_differences.csv": pairs, "repeatability.csv": repeatability,
        "order_sensitivity.csv": sensitivity, "position_effects.csv": positions,
    }
    for name, table in tables.items():
        table.to_csv(output/name, index=False)
    plots = _plot_tables(system_summary, positions, output)
    valid_count = sum(valid.values())
    invalid_count = sum(invalid.values())
    eligible_count = len(meta)
    complete_count = len(complete)
    primary_repeatability = repeatability[repeatability.complete_panel.eq(True)]
    primary_sensitivity = sensitivity[sensitivity.complete_panel.eq(True)]
    overall_positions = positions[positions.scope == "overall"]
    n_orders = int(plan.get("n_orders_per_source", 0))
    summary = dict(
        protocol=plan.get("protocol"), model=plan.get("model"), label_source="LLM_not_human",
        human_annotation_produced=False,
        status="complete" if eligible_count and complete_count == eligible_count else
               "no_eligible_panels" if not eligible_count else "partial" if valid_count or invalid_count else "planned",
        n_systems=len(systems), systems=systems,
        n_selected_sources=len(sources), n_eligible_sources=eligible_count,
        n_generation_ineligible_sources=len(sources)-eligible_count,
        n_complete_sources=complete_count,
        n_partial_sources=int(sources.status.eq("partial").sum()),
        n_complete_source_groups=int(sources.loc[sources.complete_panel.eq(True), "source_group_id"].nunique()),
        n_expected_requests=len(plan.get("requests", [])), n_valid_requests=valid_count,
        n_invalid_requests=invalid_count,
        n_missing_requests=len(plan.get("requests", []))-valid_count-invalid_count,
        n_primary_ratings=len(primary), n_all_valid_ratings=len(ratings),
        eligible_generation_coverage=eligible_count/len(sources) if len(sources) else None,
        complete_panel_coverage_eligible=complete_count/eligible_count if eligible_count else None,
        complete_panel_coverage_selected=complete_count/len(sources) if len(sources) else None,
        repetitions=plan.get("repetitions"), distinct_orders_per_source=n_orders,
        possible_orders=math.factorial(len(systems)) if systems else 0,
        bootstrap_draws=bootstrap, bootstrap_seed=seed,
        primary_estimator="mean within complete source × system, then mean across sources",
        confidence_intervals="pointwise percentile 95%; paired resampling of source_group_id clusters",
        ordinal_companions="rating median, source median and full 1–5 distribution; no forced ranking",
        incomplete_panel_policy="All valid ratings retained; primary comparisons require every planned request valid",
        repeatability_scope="At temperature 0, repeated identical requests measure residual service variability only",
        position_scope="Source-system-centered residuals; counterbalanced subset of presentation orders, not proof of absent bias",
        judge_scope="LLM naturalness ratings; not human validation, semantic fidelity or detector evasion",
        method_family_scope="A Qwen judge may favor outputs from its own family; no independent-family validation inferred",
        within_order_exact_agreement_fraction=float(primary_repeatability.exact_agreement.dropna().astype(float).mean())
            if len(primary_repeatability.exact_agreement.dropna()) else None,
        mean_within_order_sd=float(primary_sensitivity.within_order_sd_rms.mean())
            if len(primary_sensitivity) else None,
        mean_order_mean_range=float(primary_sensitivity.order_mean_range.mean())
            if len(primary_sensitivity) else None,
        maximum_absolute_position_effect=float(overall_positions.mean_centered_rating.abs().max())
            if len(overall_positions) else None,
        tables=list(tables), figures=plots,
    )
    summary = _json_value(summary)
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                allow_nan=False)+"\n", encoding="utf-8")
    return summary
