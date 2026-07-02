#!/usr/bin/env python3

from pathlib import Path
import sys
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "poster_a0"))

from make_assets import (  # noqa: E402
    add_current_result_flags,
    explicit_pass_column,
    explicit_true_column,
    explicit_true_value,
    production_accuracy_status,
    stale_operator_mask,
)


def test_current_result_requires_current_operator():
    df = pd.DataFrame([
        {
            "metadata_status": "ok",
            "operator_status": "old_absorbing_operator_unverified",
            "raw_pass10": True,
        },
        {
            "metadata_status": "ok",
            "operator_status": "complex_operator",
            "raw_pass10": True,
        },
    ])

    flagged = add_current_result_flags(df)
    pass10 = flagged["raw_pass10"] & flagged["current_ok"]

    assert flagged.loc[0, "metadata_ok"]
    assert not flagged.loc[0, "operator_ok"]
    assert not flagged.loc[0, "current_ok"]
    assert not pass10.loc[0]

    assert flagged.loc[1, "operator_ok"]
    assert flagged.loc[1, "current_ok"]
    assert pass10.loc[1]


def test_missing_operator_status_is_not_current():
    df = pd.DataFrame([{"metadata_status": "ok"}])

    flagged = add_current_result_flags(df)

    assert flagged.loc[0, "metadata_ok"]
    assert flagged.loc[0, "operator_status"] == "missing"
    assert not flagged.loc[0, "operator_ok"]
    assert not flagged.loc[0, "current_ok"]


def test_stale_operator_mask_counts_all_unverified_statuses():
    df = pd.DataFrame([
        {"operator_status": "complex_operator"},
        {"operator_status": "not_required"},
        {"operator_status": "old_absorbing_operator_unverified"},
        {"operator_status": "missing"},
        {"operator_status": ""},
    ])

    assert stale_operator_mask(df).tolist() == [False, False, True, True, True]
    assert stale_operator_mask(pd.DataFrame([{"metadata_status": "ok"}])).tolist() == [True]


def test_explicit_pass_column_does_not_fallback_to_status():
    missing = pd.DataFrame([{"status": "PASS"}])
    explicit = pd.DataFrame([
        {"raw_pass10": "true"},
        {"raw_pass10": "pass"},
        {"raw_pass10": "false"},
    ])

    assert explicit_pass_column(missing, "raw_pass10").tolist() == [False]
    assert explicit_pass_column(explicit, "raw_pass10").tolist() == [True, True, False]


def test_explicit_true_column_treats_false_strings_as_false():
    df = pd.DataFrame([
        {"flag": "True"},
        {"flag": "False"},
        {"flag": ""},
        {"flag": "pass"},
    ])

    assert explicit_true_column(df, "flag").tolist() == [True, False, False, True]
    assert explicit_true_column(df, "missing").tolist() == [False, False, False, False]


def test_explicit_true_value_treats_false_strings_as_false():
    assert explicit_true_value(True) is True
    assert explicit_true_value(False) is False
    assert explicit_true_value("True") is True
    assert explicit_true_value("False") is False
    assert explicit_true_value("yes") is True
    assert explicit_true_value("0") is False


def test_production_accuracy_status_requires_accuracy_row():
    missing_status, missing_pass = production_accuracy_status(None)
    stale_status, stale_pass = production_accuracy_status(pd.Series({"status": "STALE", "pass10": "False"}))
    pass_status, pass_pass = production_accuracy_status(pd.Series({"status": "PASS", "pass10": "true"}))

    assert missing_status == "missing_accuracy"
    assert missing_pass is False
    assert stale_status == "STALE"
    assert stale_pass is False
    assert pass_status == "PASS"
    assert pass_pass is True


def test_make_assets_has_no_truthy_string_bool_casts():
    source = (ROOT / "poster_a0" / "make_assets.py").read_text()

    forbidden = [
        r"astype\s*\(\s*bool\s*\)",
        r"bool\s*\(\s*v\s*\)",
        r"map\s*\(\s*lambda\s+v\s*:\s*bool\s*\(",
        r"(?:==|!=)\s*[\"'](?:True|False)[\"']",
    ]
    matches = []
    for pattern in forbidden:
        matches.extend(re.findall(pattern, source))

    assert not matches, matches


if __name__ == "__main__":
    test_current_result_requires_current_operator()
    test_missing_operator_status_is_not_current()
    test_stale_operator_mask_counts_all_unverified_statuses()
    test_explicit_pass_column_does_not_fallback_to_status()
    test_explicit_true_column_treats_false_strings_as_false()
    test_explicit_true_value_treats_false_strings_as_false()
    test_production_accuracy_status_requires_accuracy_row()
    test_make_assets_has_no_truthy_string_bool_casts()
