"""Controlla gli intervalli e i test usati per confrontare i controlli nulli."""
import math

import numpy as np
import pytest

from humanizers.null_control_statistics import holm_adjust, mcnemar_exact, wilson_interval


def test_wilson_zero_successes_has_nonzero_bilateral_uncertainty():
    low, high = wilson_interval(0, 95)
    assert low == 0
    assert high == pytest.approx(.038864853539472964)
    # Il 3,1% è un limite esatto unilaterale, non quello bilaterale al 95%.
    assert high > .0311


def test_wilson_all_successes_is_symmetric_and_empty_is_undefined():
    low, high = wilson_interval(95, 95)
    assert low == pytest.approx(1 - .038864853539472964)
    assert high == 1
    assert all(math.isnan(value) for value in wilson_interval(0, 0))
    assert wilson_interval(np.int64(0), np.int64(95))[0] == 0


def test_wilson_known_interior_value_and_confidence():
    assert wilson_interval(5, 10) == pytest.approx((.236593090512564, .763406909487436))
    low95, high95 = wilson_interval(5, 10)
    low99, high99 = wilson_interval(5, 10, .99)
    assert low99 < low95 < high95 < high99


@pytest.mark.parametrize("successes,total", [(True, 1), (1, True), (1.0, 2), (-1, 2),
                                               (1, -2), (2, 1), (1, 0), (0, "2")])
def test_wilson_rejects_invalid_counts(successes, total):
    with pytest.raises(ValueError):
        wilson_interval(successes, total)


@pytest.mark.parametrize("confidence", [True, 0, 1, -1, 2, math.nan, math.inf, ".95"])
def test_wilson_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        wilson_interval(0, 10, confidence)


def test_exact_mcnemar_known_discordant_pairs():
    assert mcnemar_exact(0, 38) == pytest.approx(2 / 2**38, abs=1e-23)
    assert mcnemar_exact(1, 48) == pytest.approx(100 / 2**49, abs=1e-24)
    assert mcnemar_exact(48, 1) == mcnemar_exact(1, 48)
    assert mcnemar_exact(8, 8) == 1
    assert mcnemar_exact(0, 0) == 1
    assert mcnemar_exact(np.int64(0), np.int64(38)) == mcnemar_exact(0, 38)


@pytest.mark.parametrize("left,right", [(True, 1), (1, False), (1.0, 2), (-1, 2), (1, "2")])
def test_exact_mcnemar_rejects_invalid_counts(left, right):
    with pytest.raises(ValueError):
        mcnemar_exact(left, right)


def test_holm_known_family_preserves_original_order_and_inputs():
    values = [.03, .001, .04, .01]
    assert holm_adjust(values) == pytest.approx([.06, .004, .06, .03])
    assert values == [.03, .001, .04, .01]
    assert holm_adjust([]) == []
    assert holm_adjust([.4]) == [.4]


def test_holm_ties_monotonicity_and_unit_bounds():
    assert holm_adjust([.01, .01, .02, .9]) == pytest.approx([.04, .04, .04, .9])
    assert holm_adjust([0, 1, 1]) == [0, 1, 1]
    raw = [.5, .04, .001, .2, .01]
    adjusted = holm_adjust(raw)
    order = sorted(range(len(raw)), key=raw.__getitem__)
    assert [adjusted[i] for i in order] == sorted(adjusted)
    assert all(p <= adj <= 1 for p, adj in zip(raw, adjusted))


@pytest.mark.parametrize("values", [[math.nan], [math.inf], [-.1], [1.1], [True], [".05"]])
def test_holm_rejects_invalid_p_values(values):
    with pytest.raises(ValueError):
        holm_adjust(values)
