"""NWP temporal alignment policy.

This is the defect the original dataloader had, stated precisely:

    The radar history spans 6-minute steps and the NWP series spans whole hours.
    A radar window of 10 frames covers only 54 minutes, so *every* radar frame
    inside it can fall in the same hour. Looking NWP up "in the past" therefore
    returns one static slice, repeated across the whole window, and the model
    receives no mode guidance for the T+6 ... T+60 minutes it must predict.

The fix is to align NWP by **valid time** and to hand the model two slices:

    ``history``  the slice valid at the hour containing the *first* radar frame
    ``forecast`` the slice valid at the hour covering the *end* of the forecast
                 window (issue time + forecast horizon)

so the pair brackets the forecast window and carries the mode's evolution
across it. See docs/DATA_AUDIT.md §3 for the evidence that NWP filenames encode
valid times rather than issue times.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .index import (
    NWP_STEP_MIN,
    RADAR_STEP_MIN,
    add_minutes,
    floor_hour,
    minutes_between,
)


class NWPAlignment(str, Enum):
    """How the two NWP slices are chosen for a sample."""

    #: Slice valid at the hour of the first radar frame, and slice valid at the
    #: hour of the last radar frame. The two bracket the radar window.
    BRACKET = "bracket"

    #: Slice valid at the hour containing the issue time, and the slice one hour
    #: later -- i.e. the forecast hour that covers the end of the target window.
    #: This is the mode that restores the mode's dynamical guidance.
    FORECAST = "forecast"

    #: Slice valid at the hour of the issue time and the slice one hour earlier.
    ANALYSIS_PAIR = "analysis_pair"

    #: All NWP slices inside the radar window, ordered by valid time.
    FULL_SERIES = "full_series"


@dataclass(frozen=True)
class NWPWindow:
    """The NWP slices selected for one sample."""

    valid_times: tuple[int, ...]
    #: For each selected slice, its offset from the issue time in minutes.
    #: Negative = past (analysis), positive = future (forecast).
    lead_minutes: tuple[int, ...]

    @property
    def has_forecast(self) -> bool:
        """True when at least one slice is valid after the issue time."""
        return any(l > 0 for l in self.lead_minutes)


def select_nwp_window(
    alignment: NWPAlignment,
    issue_time: int,
    radar_times: list[int],
    nwp_times: list[int],
    out_len: int,
) -> NWPWindow | None:
    """Choose the NWP slices for a sample.

    Args:
        alignment: policy to apply.
        issue_time: timestamp of the last observed radar frame (the issue time).
        radar_times: all radar timestamps of the case, ascending.
        nwp_times: all NWP timestamps of the case, ascending (whole hours).
        out_len: number of forecast frames (6 minutes each).

    Returns:
        The selected window, or ``None`` when the case cannot satisfy the policy
        (e.g. the forecast hour is outside the archived NWP range).
    """
    if not nwp_times:
        return None
    nwp_set = set(nwp_times)
    horizon_min = out_len * RADAR_STEP_MIN
    issue_hour = floor_hour(issue_time)

    if alignment is NWPAlignment.FULL_SERIES:
        inside = [t for t in nwp_times if radar_times[0] <= t <= issue_time]
        if not inside:
            return None
        return NWPWindow(
            valid_times=tuple(inside),
            lead_minutes=tuple(_lead(t, issue_time) for t in inside),
        )

    if alignment is NWPAlignment.BRACKET:
        first_hour = floor_hour(radar_times[0])
        last_hour = floor_hour(issue_time)
        if first_hour not in nwp_set or last_hour not in nwp_set:
            return None
        return NWPWindow(
            valid_times=(first_hour, last_hour),
            lead_minutes=(_lead(first_hour, issue_time), _lead(last_hour, issue_time)),
        )

    if alignment is NWPAlignment.FORECAST:
        forecast_hour = floor_hour(add_minutes(issue_time, horizon_min))
        if issue_hour not in nwp_set or forecast_hour not in nwp_set:
            return None
        return NWPWindow(
            valid_times=(issue_hour, forecast_hour),
            lead_minutes=(_lead(issue_hour, issue_time), _lead(forecast_hour, issue_time)),
        )

    if alignment is NWPAlignment.ANALYSIS_PAIR:
        prev_hour = add_minutes(issue_hour, -NWP_STEP_MIN)
        if prev_hour not in nwp_set or issue_hour not in nwp_set:
            return None
        return NWPWindow(
            valid_times=(prev_hour, issue_hour),
            lead_minutes=(_lead(prev_hour, issue_time), _lead(issue_hour, issue_time)),
        )

    raise ValueError(f"unknown alignment: {alignment}")


def _lead(valid_time: int, issue_time: int) -> int:
    """Signed offset of a valid time from the issue time, in minutes."""
    return minutes_between(valid_time, issue_time)
