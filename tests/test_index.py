"""Tests for the archive index and its timestamp arithmetic.

The archive spans May-Sep of a single year, so the day-of-year arithmetic in
:mod:`radar_nwp_bench.data.index` only has to be exact inside that window.
These tests pin the behaviour that the rest of the pipeline relies on:
``MMDD``+``HHMM`` packing, minute arithmetic that rolls over hours, days and
months, and the filename scanner that maps a directory tree onto timestamps.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from radar_nwp_bench.data.index import (
    NWP_STEP_MIN,
    RADAR_STEP_MIN,
    CaseIndex,
    add_minutes,
    decode_stamp,
    discover_cases,
    encode_stamp,
    floor_hour,
    minutes_between,
    scan_case,
)


# ---------------------------------------------------------------------------
# encode_stamp / decode_stamp
# ---------------------------------------------------------------------------
def test_encode_stamp_packs_month_day_and_hour_min():
    assert encode_stamp(517, 1518) == 5171518
    assert encode_stamp(1231, 2359) == 12312359


def test_decode_stamp_is_the_inverse_of_encode_stamp():
    for month_day, hour_min in ((517, 1518), (601, 40), (1231, 2359), (101, 0)):
        stamp = encode_stamp(month_day, hour_min)
        assert decode_stamp(stamp) == (month_day, hour_min)


def test_encode_decode_roundtrip_over_a_whole_day():
    for hour in range(24):
        for minute in (0, 1, 30, 59):
            stamp = encode_stamp(517, hour * 100 + minute)
            assert decode_stamp(stamp) == (517, hour * 100 + minute)


# ---------------------------------------------------------------------------
# add_minutes
# ---------------------------------------------------------------------------
def test_add_minutes_within_the_hour():
    stamp = encode_stamp(517, 1518)
    assert add_minutes(stamp, 0) == stamp
    assert add_minutes(stamp, 12) == encode_stamp(517, 1530)
    assert add_minutes(stamp, 41) == encode_stamp(517, 1559)


def test_add_minutes_rolls_over_the_hour():
    stamp = encode_stamp(517, 1558)
    assert add_minutes(stamp, 5) == encode_stamp(517, 1603)
    assert add_minutes(stamp, 120) == encode_stamp(517, 1758)


def test_add_minutes_rolls_over_the_day():
    # 23:30 + 60 min is 00:30 of the next day.
    stamp = encode_stamp(517, 2330)
    assert add_minutes(stamp, 60) == encode_stamp(518, 30)
    assert decode_stamp(add_minutes(stamp, 60)) == (518, 30)


def test_add_minutes_rolls_over_the_month():
    # 05-31 23:40 + 60 min is 06-01 00:40.
    stamp = encode_stamp(531, 2340)
    assert add_minutes(stamp, 60) == encode_stamp(601, 40)
    assert decode_stamp(add_minutes(stamp, 60)) == (601, 40)


def test_add_minutes_rolls_over_the_month_across_a_year_boundary():
    # 09-30 23:30 + 60 min is 10-01 00:30.
    stamp = encode_stamp(930, 2330)
    assert add_minutes(stamp, 60) == encode_stamp(1001, 30)


def test_add_minutes_is_negative_safe():
    stamp = encode_stamp(517, 1530)
    assert add_minutes(stamp, -30) == encode_stamp(517, 1500)
    # Going back over midnight lands on the previous day.
    assert add_minutes(encode_stamp(518, 30), -60) == encode_stamp(517, 2330)


def test_add_minutes_by_a_full_day_returns_the_same_clock_time():
    stamp = encode_stamp(517, 1518)
    assert add_minutes(stamp, 24 * 60) == encode_stamp(518, 1518)


# ---------------------------------------------------------------------------
# minutes_between
# ---------------------------------------------------------------------------
def test_minutes_between_is_signed_and_forward_positive():
    a = encode_stamp(517, 1600)
    b = encode_stamp(517, 1500)
    assert minutes_between(a, b) == 60
    assert minutes_between(b, a) == -60


def test_minutes_between_zero_for_identical_stamps():
    stamp = encode_stamp(517, 1518)
    assert minutes_between(stamp, stamp) == 0


def test_minutes_between_spans_a_day_boundary():
    a = encode_stamp(518, 30)
    b = encode_stamp(517, 2330)
    assert minutes_between(a, b) == 60
    assert minutes_between(b, a) == -60


def test_minutes_between_spans_a_month_boundary():
    a = encode_stamp(601, 40)
    b = encode_stamp(531, 2340)
    assert minutes_between(a, b) == 60


def test_minutes_between_is_consistent_with_add_minutes():
    base = encode_stamp(517, 1518)
    for delta in (1, 6, 59, 60, 61, 600, 1440):
        assert minutes_between(add_minutes(base, delta), base) == delta


# ---------------------------------------------------------------------------
# floor_hour
# ---------------------------------------------------------------------------
def test_floor_hour_truncates_the_minutes():
    assert floor_hour(encode_stamp(517, 1518)) == encode_stamp(517, 1500)
    assert floor_hour(encode_stamp(517, 1559)) == encode_stamp(517, 1500)
    assert floor_hour(encode_stamp(517, 1500)) == encode_stamp(517, 1500)


def test_floor_hour_keeps_the_date():
    assert floor_hour(encode_stamp(1231, 2359)) == encode_stamp(1231, 2300)


def test_floor_hour_is_idempotent():
    stamp = encode_stamp(517, 1518)
    assert floor_hour(floor_hour(stamp)) == floor_hour(stamp)


# ---------------------------------------------------------------------------
# scan_case
# ---------------------------------------------------------------------------
def _write_case(root: Path, case_id: str = "CP_01_05171518_00001") -> Path:
    """Create a tiny synthetic case tree and return its path."""
    case = root / case_id
    for var in ("CAP20", "CR", "HBR"):
        var_dir = case / "RADAR" / var
        var_dir.mkdir(parents=True, exist_ok=True)
        for stamp in ("0517-1518", "0517-1524", "0517-1530"):
            np.save(
                var_dir / f"CP_RADA_00001_{stamp}_SA_CR.npy",
                np.zeros((2, 2), dtype=np.int16),
            )
    label_dir = case / "LABEL" / "RA"
    label_dir.mkdir(parents=True, exist_ok=True)
    for stamp in ("0517-1518", "0517-1524", "0517-1530"):
        np.save(label_dir / f"CP_Label_RA_00001_{stamp}.npy", np.zeros((2, 2), dtype=np.int16))
    for var in ("CAPE", "Q1000"):
        var_dir = case / "NWP" / var
        var_dir.mkdir(parents=True, exist_ok=True)
        for stamp in ("0517-1500", "0517-1600"):
            np.save(
                var_dir / f"CP_RRA_00001_{stamp}_{var}.npy",
                np.zeros((2, 2), dtype=np.float16),
            )
    return case


def test_scan_case_finds_radar_timestamps():
    with tempfile.TemporaryDirectory() as tmp:
        case = _write_case(Path(tmp))
        index = scan_case(case)

        assert isinstance(index, CaseIndex)
        assert index.case_id == "CP_01_05171518_00001"
        assert index.radar_times == [
            encode_stamp(517, 1518),
            encode_stamp(517, 1524),
            encode_stamp(517, 1530),
        ]


def test_scan_case_finds_label_timestamps_with_the_timestamp_at_the_end_of_the_name():
    with tempfile.TemporaryDirectory() as tmp:
        case = _write_case(Path(tmp))
        index = scan_case(case)

        assert index.label_times == [
            encode_stamp(517, 1518),
            encode_stamp(517, 1524),
            encode_stamp(517, 1530),
        ]
        # The LABEL filename form is CP_Label_RA_00001_MMDD-HHMM.npy, so the
        # stamp sits at the very end of the file name.
        label_file = index.label_file(encode_stamp(517, 1518))
        assert label_file is not None
        assert label_file.name == "CP_Label_RA_00001_0517-1518.npy"


def test_scan_case_finds_nwp_timestamps():
    with tempfile.TemporaryDirectory() as tmp:
        case = _write_case(Path(tmp))
        index = scan_case(case)

        assert index.nwp_times == [encode_stamp(517, 1500), encode_stamp(517, 1600)]


def test_scan_case_records_the_radar_file_per_variable():
    with tempfile.TemporaryDirectory() as tmp:
        case = _write_case(Path(tmp))
        index = scan_case(case)

        stamp = encode_stamp(517, 1524)
        for var in ("CAP20", "CR", "HBR"):
            path = index.radar_file(var, stamp)
            assert path is not None
            assert path.parent.name == var
        assert index.radar_file("CAP20", encode_stamp(517, 9999)) is None


def test_scan_case_records_the_nwp_file_per_variable():
    with tempfile.TemporaryDirectory() as tmp:
        case = _write_case(Path(tmp))
        index = scan_case(case)

        stamp = encode_stamp(517, 1600)
        for var in ("CAPE", "Q1000"):
            path = index.nwp_file(var, stamp)
            assert path is not None
            assert path.parent.name == var
        assert index.nwp_file("CAPE", encode_stamp(517, 1700)) is None


def test_scan_case_nwp_hour_for_returns_the_containing_hour():
    with tempfile.TemporaryDirectory() as tmp:
        case = _write_case(Path(tmp))
        index = scan_case(case)

        assert index.nwp_hour_for(encode_stamp(517, 1518)) == encode_stamp(517, 1500)
        assert index.nwp_hour_for(encode_stamp(517, 1600)) == encode_stamp(517, 1600)
        # 17:00 is not archived.
        assert index.nwp_hour_for(encode_stamp(517, 1700)) is None


def test_scan_case_radar_times_are_the_intersection_over_variables():
    """A frame missing from one variable must not appear in radar_times."""
    with tempfile.TemporaryDirectory() as tmp:
        case = _write_case(Path(tmp))
        # Drop one CAP20 frame; CAP20 no longer covers the full timeline.
        missing = case / "RADAR" / "CAP20" / "CP_RADA_00001_0517-1524_SA_CR.npy"
        missing.unlink()
        index = scan_case(case)

        assert encode_stamp(517, 1524) not in index.radar_times
        assert index.radar_times == [
            encode_stamp(517, 1518),
            encode_stamp(517, 1530),
        ]


def test_scan_case_on_a_directory_without_data_is_empty():
    with tempfile.TemporaryDirectory() as tmp:
        index = scan_case(Path(tmp))
        assert index.radar == {}
        assert index.label == {}
        assert index.nwp == {}
        assert index.radar_times == []
        assert index.label_times == []
        assert index.nwp_times == []


def test_case_index_summary_reports_the_counts():
    with tempfile.TemporaryDirectory() as tmp:
        case = _write_case(Path(tmp))
        summary = scan_case(case).summary()

        assert summary["case_id"] == "CP_01_05171518_00001"
        assert summary["n_radar_vars"] == 3
        assert summary["n_nwp_vars"] == 2
        assert summary["n_radar_frames"] == 3
        assert summary["n_label_frames"] == 3
        assert summary["n_nwp_frames"] == 2
        assert summary["radar_first"] == encode_stamp(517, 1518)
        assert summary["radar_last"] == encode_stamp(517, 1530)


# ---------------------------------------------------------------------------
# discover_cases
# ---------------------------------------------------------------------------
def test_discover_cases_lists_case_directories():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_case(root, "CP_01_05171518_00001")
        _write_case(root, "CP_01_05172106_00001")

        cases = discover_cases(root)
        assert [p.name for p in cases] == [
            "CP_01_05171518_00001",
            "CP_01_05172106_00001",
        ]


def test_discover_cases_raises_for_a_missing_root():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            discover_cases(Path(tmp) / "does_not_exist")
        except FileNotFoundError:
            return
        raise AssertionError("discover_cases should raise for a missing root")


def test_discover_cases_raises_when_there_is_nothing_to_find():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            discover_cases(Path(tmp))
        except FileNotFoundError:
            return
        raise AssertionError("discover_cases should raise for an empty root")


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
def test_step_constants_match_the_archive_cadence():
    assert RADAR_STEP_MIN == 6
    assert NWP_STEP_MIN == 60


# ---------------------------------------------------------------------------
# case-level split
# ---------------------------------------------------------------------------
def test_split_cases_partitions_every_case_exactly_once():
    from radar_nwp_bench.data.index import split_cases

    ids = [f"CP_{i:02d}_0001" for i in range(62)]
    parts = split_cases(ids)
    assert set(parts) == {"train", "val", "test"}
    union = parts["train"] + parts["val"] + parts["test"]
    assert sorted(union) == sorted(ids)
    assert len(union) == len(set(union)), "a case landed in two splits"


def test_split_cases_has_no_overlap_between_splits():
    from radar_nwp_bench.data.index import split_cases

    ids = [f"CP_{i:02d}_0001" for i in range(62)]
    parts = split_cases(ids)
    assert not set(parts["train"]) & set(parts["val"])
    assert not set(parts["train"]) & set(parts["test"])
    assert not set(parts["val"]) & set(parts["test"])


def test_split_cases_is_deterministic_regardless_of_input_order():
    from radar_nwp_bench.data.index import split_cases

    ids = [f"CP_{i:02d}_0001" for i in range(20)]
    assert split_cases(ids) == split_cases(list(reversed(ids)))


def test_split_cases_gives_every_split_at_least_one_case():
    from radar_nwp_bench.data.index import split_cases

    parts = split_cases([f"c{i}" for i in range(10)])
    assert all(len(v) > 0 for v in parts.values())
