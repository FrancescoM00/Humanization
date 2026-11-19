"""Casi semantici concreti per le guardie deterministiche delle riscritture."""
import json

import pytest
import spacy

from semantics.humanizer_guards import (
    ALERT_COLUMNS, compare_inventories, definitions, extract_inventory, score_pairs,
)


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


def compare(nlp, original, modified):
    return compare_inventories(extract_inventory(nlp(original)), extract_inventory(nlp(modified)))


def test_identity_has_no_alert_and_inventories_are_json_serializable(nlp):
    text = "In May 2020, London might need up to 100 buses, perhaps more."
    inventory = extract_inventory(nlp(text))
    json.dumps(inventory, allow_nan=False)
    result = compare_inventories(inventory, inventory)
    assert not any(result[name] for name in ALERT_COLUMNS)
    assert result["word_ratio"] == 1


@pytest.mark.parametrize("before,after", [
    ("We counted twenty items.", "We counted 20 items."),
    ("We spent 1,000 dollars.", "We spent one thousand dollars."),
    ("The rate is twenty percent.", "The rate is 20%."),
])
def test_number_formatting_does_not_change_values(nlp, before, after):
    assert not compare(nlp, before, after)["alert_number"]


def test_changed_number_is_visible_even_if_the_number_count_is_unchanged(nlp):
    row = compare(nlp, "The rate is 20%.", "The rate is 30%.")
    assert row["guard_number_signed_count_delta"] == 0
    assert row["alert_number"]
    assert row["guard_number_removed"] == {"percent:20": 1}
    assert row["guard_number_added"] == {"percent:30": 1}


@pytest.mark.parametrize("before,after", [
    ("We met on May 5, 2020.", "We met on 5 May 2020."),
    ("We met on 2020-05-05.", "We met on May 5, 2020."),
    ("We met on May 5th, 2020.", "We met on 5 May 2020."),
    ("We met on 5May2020.", "We met on May5,2020."),
])
def test_dates_preserve_explicit_calendar_values_across_formats(nlp, before, after):
    row = compare(nlp, before, after)
    assert not row["alert_date"]
    assert not row["alert_number"]
    assert row["guard_date_original_inventory"] == {"calendar:2020-05-05": 1}


def test_changed_and_removed_dates_are_visible(nlp):
    row = compare(nlp, "We met on May 5, 2020.", "We met on May 6, 2020.")
    assert row["alert_date"]
    assert row["guard_date_lost"] and row["guard_date_introduced"]
    assert compare(nlp, "We met on May 5, 2020.", "We met.")["alert_date"]


def test_entity_repetition_does_not_define_preservation(nlp):
    row = compare(nlp, "Alice visited London. Alice liked London.", "Alice visited London.")
    assert not row["alert_entity"]


def test_replaced_entity_produces_directional_evidence(nlp):
    row = compare(nlp, "Alice visited London.", "Alice visited Paris.")
    assert row["alert_entity"]
    assert "london" in row["guard_entity_removed"]
    assert "paris" in row["guard_entity_added"]


def test_contracted_negation_is_preserved_but_added_negation_is_flagged(nlp):
    assert not compare(nlp, "They did not arrive.", "They didn't arrive.")["alert_negation"]
    row = compare(nlp, "They arrived.", "They did not arrive.")
    assert row["alert_negation"]
    assert row["guard_negation_signed_count_delta"] == 1


def test_up_to_over_changes_bound_while_preserving_number(nlp):
    row = compare(nlp, "The height is up to 100 feet.", "The height is over 100 feet.")
    assert row["alert_quantifier"]
    assert not row["alert_number"]
    assert row["guard_quantifier_removed"] == {"upper_inclusive|number:100": 1}
    assert row["guard_quantifier_added"] == {"lower_exclusive|number:100": 1}


def test_equivalent_bound_phrases_do_not_duplicate_negations_or_quantifiers(nlp):
    row = compare(nlp, "They need no more than 100 buses.", "They need at most 100 buses.")
    assert not row["alert_quantifier"]
    assert not row["alert_negation"]
    assert row["guard_quantifier_original_count"] == 1


def test_spatial_over_without_a_number_does_not_define_a_numeric_bound(nlp):
    row = extract_inventory(nlp("The bird flew over the river."))
    assert not row["quantifier"]


def test_some_to_all_changes_quantifier(nlp):
    assert compare(nlp, "Some people arrived.", "All people arrived.")["alert_quantifier"]
    assert not compare(nlp, "All students arrived.", "Every student arrived.")["alert_quantifier"]


def test_modal_synonyms_are_preserved_and_removed_hedges_are_flagged(nlp):
    assert not compare(nlp, "They may arrive.", "They might arrive.")["alert_hedge_removal"]
    row = compare(nlp, "In May, they might possibly arrive.", "In May, they arrive.")
    assert row["alert_hedge_removal"]
    assert row["guard_hedge_signed_count_delta"] == -2
    assert row["guard_hedge_original_count"] == 2
    assert not row["alert_date"]


def test_pair_scoring_returns_evidence_lengths_and_unique_ids(nlp):
    pairs = [{"id": "x", "original": "They might arrive.", "modified": "They arrive."}]
    frame = score_pairs(pairs, nlp=nlp)
    assert frame.loc[0, "alert_any"]
    assert frame.loc[0, "original_n_words"] == 3
    assert frame.loc[0, "modified_n_words"] == 2
    assert frame.loc[0, "word_ratio"] == pytest.approx(2 / 3)
    assert all(name in frame for name in ALERT_COLUMNS)
    with pytest.raises(ValueError, match="univoci"):
        score_pairs(pairs * 2, nlp=nlp)
    assert not definitions()["warnings_are_confirmed_semantic_errors"]


def test_ner_cannot_be_silently_disabled():
    with pytest.raises(ValueError, match="NER"):
        extract_inventory(spacy.blank("en")("London might change."))
