"""Controlla denominatori, soglie fissate e ricampionamento comune nell'esperimento 3."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from humanizers.semantic_evasion_analysis import analyze


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    value = json.loads((ROOT / "results_semantic_cost_evasion/config_exp3.json").read_text())
    value.update(bootstrap_replicates=120, bootstrap_seed=42)
    return value


def sample(config, *, identical=False, zero_evasion=False):
    rows = []
    means = [.8, .9, .7, .79, .95, .85]
    minimum = [.5, .8, .3, .7, .6, .9]
    for di, detector in enumerate(config["detectors"]):
        for mi, method in enumerate(config["methods"]):
            for source in range(6):
                original = source < 4
                evaded = bool(original and not zero_evasion and (source < (2 if identical else 2 + mi % 2)))
                rows.append(dict(source_id=f"s{source}", source_group_id=f"g{source}", domain="Essay",
                    method=method, detector=detector, orig_detected=original,
                    rew_detected=bool(original and not evaded), evaded=evaded,
                    nli_cons_mean=means[source], nli_min=minimum[source],
                    bs_resc_mean=means[source]-.1, len_ratio=1.1 if source == 0 else .9,
                    score_reduction=(source+1)/10))
    return pd.DataFrame(rows)


def test_fixed_denominator_and_inclusive_frozen_requirements(config):
    result = analyze(sample(config), config)
    main = result["main_table"]
    assert len(main) == 8 and main.n_original_detected.eq(4).all()
    assert main.simple_k.isin([2, 3]).all()
    assert main.moderate_k.eq(2).all() and main.stringent_k.eq(2).all()
    assert (main.stringent_asr <= main.moderate_asr).all()
    assert (main.moderate_asr <= main.simple_asr).all()
    assert main.evaded_expanded.eq(1).all()
    np.testing.assert_allclose(main.moderate_retained_fraction, main.moderate_k / main.simple_k)
    assert (main.stringent_retained_fraction <= main.moderate_retained_fraction).all()
    # Il punteggio esattamente 0,8 e il minimo esattamente 0,5 vengono mantenuti.
    assert main.moderate_asr.eq(.5).all()


def test_curves_are_monotone_and_use_the_same_denominator(config):
    result = analyze(sample(config), config)
    curves = result["curves"]
    assert len(curves) == 8 * 3 * 101
    assert curves.n_original_detected.eq(4).all()
    for _, own in curves.groupby(["detector", "method", "curve"]):
        assert np.all(np.diff(own.sort_values("tau").asr) <= 0)
    basic = curves[curves.curve.eq("nli_mean") & curves.tau.eq(0)]
    np.testing.assert_array_equal(basic.asr, basic.simple_asr)
    assert set(result["sensitivity"].tau) == {.7, .8, .9}


def test_same_source_draws_preserve_identical_methods_and_detectors(config):
    result = analyze(sample(config, identical=True), config)
    curves = result["curves"]
    for _, own in curves.groupby(["curve", "tau"]):
        assert own.bootstrap_ci95_low.nunique(dropna=False) == 1
        assert own.bootstrap_ci95_high.nunique(dropna=False) == 1
    repeated = analyze(sample(config, identical=True).sample(frac=1, random_state=7), config)
    pd.testing.assert_frame_equal(result["curves"], repeated["curves"])


def test_zero_events_have_wilson_uncertainty_and_empty_group_is_not_imputed(config):
    result = analyze(sample(config, zero_evasion=True), config)
    main = result["main_table"]
    assert main.simple_k.eq(0).all() and main.simple_wilson95_high.gt(0).all()
    assert main.moderate_retained_fraction.isna().all()
    groups = result["evaded_vs_not"]
    assert groups.n_evaded.eq(0).all()
    assert groups.evaded_median.isna().all() and groups.difference_evaded_minus_not.isna().all()
    assert groups.bootstrap_n_valid.eq(0).all()
    assert groups.interval_status.eq("empty_evaded_group").all()


def test_non_evaded_group_excludes_originals_that_were_not_detected(config):
    result = analyze(sample(config), config)
    groups = result["evaded_vs_not"]
    assert (groups.n_evaded + groups.n_not_evaded).eq(4).all()
    assert groups.n_not_evaded.isin([1, 2]).all()
    assert len(groups) == 16


def test_mcnemar_family_contains_all_twelve_prespecified_tests(config):
    result = analyze(sample(config, identical=True), config)
    comparisons = result["method_comparisons"]
    assert len(comparisons) == 12
    assert comparisons.holm_family_size.eq(12).all()
    assert comparisons.a_only.eq(0).all() and comparisons.b_only.eq(0).all()
    assert comparisons.mcnemar_p.eq(1).all() and comparisons.mcnemar_p_holm.eq(1).all()


def test_negative_bertscore_is_not_clipped_into_the_tau_zero_curve(config):
    data = sample(config, identical=True)
    data.loc[data.source_id.eq("s0"), "bs_resc_mean"] = -.1
    curve = analyze(data, config)["curves"]
    point = curve[curve.curve.eq("bertscore_rescaled_mean") & curve.tau.eq(0)]
    assert point.k.eq(1).all()


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong_evasion", "wrong_baseline"])
def test_inconsistent_pairing_is_rejected(config, change):
    data = sample(config)
    if change == "duplicate":
        data = pd.concat([data, data.iloc[[0]]])
    elif change == "missing":
        data = data.iloc[1:]
    elif change == "wrong_evasion":
        data.loc[0, "evaded"] = False
    else:
        data.loc[0, "orig_detected"] = False
        data.loc[0, "evaded"] = False
    with pytest.raises(ValueError):
        analyze(data, config)
