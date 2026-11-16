"""Tests for the NWP temporal-alignment policy.

The alignment policy is the defect this benchmark exists to fix: a 10-frame
radar window spans only 54 minutes, so looking NWP up "in the past" returns
one static slice.  These tests pin the four policies of
:mod:`radar_nwp_bench.data.alignment`.
"""
from __future__ import annotations

from radar_nwp_bench.data.alignment import (
    NWPAlignment,
    NWPWindow,
    select_nwp_window,
)
from radar_nwp_bench.data.index import (
    add_minutes,
    encode_stamp,
    floor_hour,
    minutes_between,
)


#: A radar timeline of 61 six-minute frames, 05-17 15:00 .. 05-17 21:00.
RADAR_TIMES: list[int] = [
    add_minutes(encode_stamp(517, 1500), 6 * k) for k in range(61)
]
#: The NWP archive: whole hours from 05-17 15:00 to 05-17 20:00.
NWP_TIMES: list[int] = [
    encode_stamp(517, 1500 + 100 * k) for k in range(6)
]

OUT_LEN = 10
#: 10 frames * 6 min = 60 min, i.e. the forecast window is one hour long.
HORIZON_MIN = OUT_LEN * 6


def test_alignment_enum_members():
    assert NWPAlignment.BRACKET.value == "bracket"
    assert NWPAlignment.FORECAST.value == "forecast"
    assert NWPAlignment.ANALYSIS_PAIR.value == "analysis_pair"
    assert NWPAlignment.FULL_SERIES.value == "full_series"


# ---------------------------------------------------------------------------
# FORECAST
# ---------------------------------------------------------------------------
def test_forecast_second_slice_is_valid_after_the_issue_time():
    issue = encode_stamp(517, 1600)
    window = select_nwp_window(
        NWPAlignment.FORECAST, issue, RADAR_TIMES, NWP_TIMES, OUT_LEN
    )
    assert window is not None
    assert isinstance(window, NWPWindow)

    forecast_valid_time = window.valid_times[1]
    assert forecast_valid_time > issue
    assert forecast_valid_time == floor_hour(add_minutes(issue, HORIZON_MIN))


def test_forecast_window_has_a_forecast_slice():
    issue = encode_stamp(517, 1600)
    window = select_nwp_window(
        NWPAlignment.FORECAST, issue, RADAR_TIMES, NWP_TIMES, OUT_LEN
    )
    assert window is not None
    assert window.has_forecast is True
    assert window.lead_minutes[1] > 0


def test_forecast_first_slice_is_the_issue_hour():
    issue = encode_stamp(517, 1612)
    window = select_nwp_window(
        NWPAlignment.FORECAST, issue, RADAR_TIMES, NWP_TIMES, OUT_LEN
    )
    assert window is not None
    assert window.valid_times[0] == floor_hour(issue)
    assert window.lead_minutes[0] <= 0


def test_forecast_returns_none_when_the_forecast_hour_is_not_archived():
    """The NWP archive stops at 20:00, so an issue at 20:30 cannot forecast."""
    issue = encode_stamp(517, 2030)
    assert issue in RADAR_TIMES  # a real radar stamp, just past the archive
    window = select_nwp_window(
        NWPAlignment.FORECAST, issue, RADAR_TIMES, NWP_TIMES, OUT_LEN
    )
    assert window is None


def test_forecast_returns_none_when_the_issue_hour_is_missing():
    """An issue hour that is not in the NWP archive cannot be aligned."""
    issue = encode_stamp(517, 1600)
    nwp_without_issue_hour = [t for t in NWP_TIMES if t != floor_hour(issue)]
    window = select_nwp_window(
        NWPAlignment.FORECAST, issue, RADAR_TIMES, nwp_without_issue_hour, OUT_LEN
    )
    assert window is None


# ---------------------------------------------------------------------------
# BRACKET
# ---------------------------------------------------------------------------
def test_bracket_slices_are_inside_the_radar_window():
    issue = RADAR_TIMES[10]
    window = select_nwp_window(
        NWPAlignment.BRACKET, issue, RADAR_TIMES, NWP_TIMES, OUT_LEN
    )
    assert window is not None

    first, second = window.valid_times
    assert first == floor_hour(RADAR_TIMES[0])
    assert second == floor_hour(issue)
    # Both slices sit inside the radar window that they bracket.
    assert RADAR_TIMES[0] <= first <= issue
    assert RADAR_TIMES[0] <= second <= issue


def test_bracket_slices_are_ordered():
    issue = RADAR_TIMES[10]
    window = select_nwp_window(
        NWPAlignment.BRACKET, issue, RADAR_TIMES, NWP_TIMES, OUT_LEN
    )
    assert window is not None
    assert window.valid_times[0] <= window.valid_times[1]
    assert window.lead_minutes[0] <= window.lead_minutes[1]


def test_bracket_returns_none_when_a_slice_hour_is_missing():
    issue = RADAR_TIMES[10]
    nwp_missing_first = [t for t in NWP_TIMES if t != floor_hour(RADAR_TIMES[0])]
    assert (
        select_nwp_window(
            NWPAlignment.BRACKET, issue, RADAR_TIMES, nwp_missing_first, OUT_LEN
        )
        is None
    )


# ---------------------------------------------------------------------------
# FULL_SERIES
# ---------------------------------------------------------------------------
def test_full_series_returns_the_slices_inside_the_radar_window():
    issue = RADAR_TIMES[10]
    window = select_nwp_window(
        NWPAlignment.FULL_SERIES, issue, RADAR_TIMES, NWP_TIMES, OUT_LEN
    )
    assert window is not None

    for valid_time in window.valid_times:
        assert RADAR_TIMES[0] <= valid_time <= issue


def test_full_series_returns_every_archived_hour_inside_the_window():
    issue = RADAR_TIMES[10]
    window = select_nwp_window(
        NWPAlignment.FULL_SERIES, issue, RADAR_TIMES, NWP_TIMES, OUT_LEN
    )
    assert window is not None

    expected = [t for t in NWP_TIMES if RADAR_TIMES[0] <= t <= issue]
    assert list(window.valid_times) == expected
    assert len(window.valid_times) > 1


def test_full_series_lead_minutes_are_signed_relative_to_the_issue():
    issue = RADAR_TIMES[10]
    window = select_nwp_window(
        NWPAlignment.FULL_SERIES, issue, RADAR_TIMES, NWP_TIMES, OUT_LEN
    )
    assert window is not None
    for lead, valid_time in zip(window.lead_minutes, window.valid_times):
        assert lead == minutes_between(valid_time, issue)


def test_full_series_returns_none_when_no_hour_falls_inside_the_window():
    """A radar window that starts after the last archived hour selects nothing."""
    late_radar = [
        add_minutes(encode_stamp(517, 1506), 6 * k) for k in range(4)
    ]
    issue = late_radar[0]
    window = select_nwp_window(
        NWPAlignment.FULL_SERIES, issue, late_radar, NWP_TIMES, OUT_LEN
    )
    assert window is None


# ---------------------------------------------------------------------------
# ANALYSIS_PAIR
# ---------------------------------------------------------------------------
def test_analysis_pair_uses_the_previous_hour_and_the_issue_hour():
    issue = encode_stamp(517, 1600)
    window = select_nwp_window(
        NWPAlignment.ANALYSIS_PAIR, issue, RADAR_TIMES, NWP_TIMES, OUT_LEN
    )
    assert window is not None
    assert window.valid_times == (
        add_minutes(floor_hour(issue), -60),
        floor_hour(issue),
    )
    assert window.has_forecast is False


# ---------------------------------------------------------------------------
# general behaviour
# ---------------------------------------------------------------------------
def test_every_policy_returns_none_for_an_empty_nwp_archive():
    issue = encode_stamp(517, 1600)
    for alignment in NWPAlignment:
        assert (
            select_nwp_window(alignment, issue, RADAR_TIMES, [], OUT_LEN) is None
        ), alignment


def test_nwp_window_has_forecast_is_false_for_past_only_windows():
    window = NWPWindow(valid_times=(100, 200), lead_minutes=(-10, 0))
    assert window.has_forecast is False

    window = NWPWindow(valid_times=(100, 200), lead_minutes=(-10, 5))
    assert window.has_forecast is True
