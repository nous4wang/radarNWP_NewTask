"""Scanning the on-disk archive and building the sample index.

The archive is a directory of small ``.npy`` files; the loader must not depend on
their order. Everything here is keyed on the ``MMDD-HHMM`` timestamp embedded in
each filename, and missing frames are tolerated (4 of the 62 BJ cases have
RADAR/LABEL frame counts that disagree).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

RADAR_VARS: tuple[str, ...] = (
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70",
    "CR", "ET", "HBR", "VIL",
)

NWP_VARS: tuple[str, ...] = (
    "CAPE", "DVG200", "DVG850", "DVG925", "HT0", "HT10", "HTw0", "KI",
    "LCL", "LFC", "LI300", "LI500", "muLCL", "PE", "PWAT", "Q1000",
    "Q700", "Q850", "RH1000", "RH500", "RH700", "TdSfc850", "TTdMax74",
    "TTdMean74", "WS500", "WS700", "WS925",
)

RADAR_STEP_MIN = 6
NWP_STEP_MIN = 60

#: The cases whose radar series contains a skipped 6-minute step.
#:
#: These are held out of the standard train/val/test partition and form the
#: ``robustness_missing`` split, so that the fault-tolerance behaviour can be
#: measured on cases the model never saw during training. The ids are the
#: directory names in the archive, which carry a ``_00001`` suffix.
#:
#: The list was verified against the archive rather than assumed: a case
#: belongs here when its radar timestamps skip a 6-minute step, i.e. when
#: the series has a hole *inside* it. Counting a difference between the
#: radar and label frame counts is not the right test -- that difference
#: counts frames outside the history window as well, and it is 1 for many
#: cases whose series is internally continuous.
IRREGULAR_CASES: frozenset[str] = frozenset({
    "CP_01_05301413_00001",
    "CP_01_06220410_00001",
    "CP_01_06230359_00001",
    "CP_01_06251506_00001",
    "CP_01_06252006_00001",
    "CP_01_06280022_00001",
    "CP_01_06300430_00001",
    "CP_01_07011914_00001",
    "CP_01_07061806_00001",
    "CP_01_07120834_00001",
    "CP_01_07161413_00001",
    "CP_01_07291341_00001",
    "CP_01_08020529_00001",
    "CP_01_08051911_00001",
    "CP_01_08061235_00001",
    "CP_01_08092015_00001",
    "CP_01_08111925_00001",
    "CP_01_08120710_00001",
    "CP_01_08140134_00001",
    "CP_01_08161336_00001",
    "CP_01_09042035_00001",
    "CP_01_09071830_00001",
    "CP_01_09260116_00001",
})

#: Filename timestamp, e.g. ``CP_RADA_00001_0517-1518_SA_CR.npy`` -> ``(517, 1518)``.
#: The timestamp is followed by an underscore (RADAR/NWP) or by the extension
#: (LABEL), so both terminators are accepted.
_STAMP_RE = re.compile(r"_(\d{4})-(\d{4})(?:_|\.|$)")
#: Variable tag at the end of a RADAR filename; ``CAP020`` maps back to ``CAP20``.
_RADAR_TAG_RE = re.compile(r"_SA_([A-Z0-9]+)\.npy$")
_NWP_TAG_RE = re.compile(r"_([A-Za-z0-9]+)\.npy$")


def encode_stamp(month_day: int, hour_min: int) -> int:
    """Pack ``MMDD`` + ``HHMM`` into a single sortable integer."""
    return month_day * 10000 + hour_min


def decode_stamp(stamp: int) -> tuple[int, int]:
    """Inverse of :func:`encode_stamp`."""
    return stamp // 10000, stamp % 10000


def _stamp_to_minutes(stamp: int) -> int:
    """Convert an ``MMDDHHMM`` stamp to minutes since an arbitrary epoch.

    The month/day component is folded into a monotonically increasing day count
    so that differences are exact even across a day boundary.
    """
    month_day, hour_min = decode_stamp(stamp)
    month, day = month_day // 100, month_day % 100
    # A day-of-year style count; the archive only spans May-Sep so a simple
    # cumulative table is enough and stays exact.
    day_of_year = _CUMULATIVE_DAYS[month - 1] + day - 1
    return day_of_year * 1440 + (hour_min // 100) * 60 + (hour_min % 100)


def _minutes_to_stamp(total_minutes: int) -> int:
    """Inverse of :func:`_stamp_to_minutes`."""
    day_of_year, rem = divmod(total_minutes, 1440)
    hour, minute = divmod(rem, 60)
    month = _MONTH_OF_DAY[day_of_year]
    day = day_of_year - _CUMULATIVE_DAYS[month - 1] + 1
    return encode_stamp(month * 100 + day, hour * 100 + minute)


#: Days before each month in a non-leap year (index 0 == January), with a
#: trailing sentinel so the December entry can be built with the same rule.
_CUMULATIVE_DAYS = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
#: month (1-12) for each day-of-year index in a non-leap year.
_MONTH_OF_DAY = tuple(
    m for m in range(1, 13)
    for _ in range(_CUMULATIVE_DAYS[m - 1], _CUMULATIVE_DAYS[m])
)


def floor_hour(stamp: int) -> int:
    """Round a timestamp down to the containing whole hour."""
    month_day, hour_min = decode_stamp(stamp)
    return encode_stamp(month_day, (hour_min // 100) * 100)


def add_minutes(stamp: int, minutes: int) -> int:
    """Shift a timestamp by ``minutes``, rolling over hours and days."""
    return _minutes_to_stamp(_stamp_to_minutes(stamp) + minutes)


def minutes_between(a: int, b: int) -> int:
    """Signed difference ``a - b`` in minutes."""
    return _stamp_to_minutes(a) - _stamp_to_minutes(b)


@dataclass
class CaseIndex:
    """Everything known about one ``CP_*`` case, without reading array payloads."""

    case_id: str
    root: Path
    radar: dict[str, dict[int, Path]] = field(default_factory=dict)
    label: dict[int, Path] = field(default_factory=dict)
    nwp: dict[str, dict[int, Path]] = field(default_factory=dict)

    # -- time axes ----------------------------------------------------------
    @property
    def radar_times(self) -> list[int]:
        """Timestamps where *every* RADAR variable has a frame."""
        if not self.radar:
            return []
        common = set(self.radar[next(iter(self.radar))])
        for var in self.radar:
            common &= set(self.radar[var])
        return sorted(common)

    @property
    def label_times(self) -> list[int]:
        return sorted(self.label)

    @property
    def nwp_times(self) -> list[int]:
        """Timestamps where *every* NWP variable has a frame."""
        if not self.nwp:
            return []
        common = set(self.nwp[next(iter(self.nwp))])
        for var in self.nwp:
            common &= set(self.nwp[var])
        return sorted(common)

    def radar_file(self, var: str, stamp: int) -> Path | None:
        return self.radar.get(var, {}).get(stamp)

    def label_file(self, stamp: int) -> Path | None:
        return self.label.get(stamp)

    def nwp_file(self, var: str, stamp: int) -> Path | None:
        return self.nwp.get(var, {}).get(stamp)

    def nwp_hour_for(self, stamp: int) -> int | None:
        """The NWP slice valid at (the hour containing) ``stamp``."""
        hour = floor_hour(stamp)
        return hour if hour in set(self.nwp_times) else None

    def summary(self) -> dict:
        return {
            "case_id": self.case_id,
            "n_radar_vars": len(self.radar),
            "n_nwp_vars": len(self.nwp),
            "n_radar_frames": len(self.radar_times),
            "n_label_frames": len(self.label_times),
            "n_nwp_frames": len(self.nwp_times),
            "radar_first": self.radar_times[0] if self.radar_times else None,
            "radar_last": self.radar_times[-1] if self.radar_times else None,
        }


def scan_case(case_dir: str | Path) -> CaseIndex:
    """Index one ``CP_*`` directory by timestamp."""
    case_dir = Path(case_dir)
    index = CaseIndex(case_id=case_dir.name, root=case_dir)

    radar_root = case_dir / "RADAR"
    if radar_root.is_dir():
        for var_dir in sorted(radar_root.iterdir()):
            if not var_dir.is_dir():
                continue
            var = var_dir.name
            frames: dict[int, Path] = {}
            for f in sorted(var_dir.glob("*.npy")):
                m = _STAMP_RE.search(f.name)
                if not m:
                    continue
                frames[encode_stamp(int(m.group(1)), int(m.group(2)))] = f
            if frames:
                index.radar[var] = frames

    label_root = case_dir / "LABEL" / "RA"
    if label_root.is_dir():
        for f in sorted(label_root.glob("*.npy")):
            m = _STAMP_RE.search(f.name)
            if not m:
                continue
            index.label[encode_stamp(int(m.group(1)), int(m.group(2)))] = f

    nwp_root = case_dir / "NWP"
    if nwp_root.is_dir():
        for var_dir in sorted(nwp_root.iterdir()):
            if not var_dir.is_dir():
                continue
            frames = {}
            for f in sorted(var_dir.glob("*.npy")):
                m = _STAMP_RE.search(f.name)
                if not m:
                    continue
                frames[encode_stamp(int(m.group(1)), int(m.group(2)))] = f
            if frames:
                index.nwp[var_dir.name] = frames

    return index


def discover_cases(data_root: str | Path) -> list[Path]:
    """List the ``CP_*`` case directories under ``data_root``."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    cases = sorted(p for p in root.glob("CP_*") if p.is_dir())
    if not cases:
        cases = sorted(p for p in root.iterdir() if p.is_dir() and (p / "RADAR").is_dir())
    if not cases:
        raise FileNotFoundError(f"no CP_* cases under {root}")
    return cases


def scan_all(data_root: str | Path) -> list[CaseIndex]:
    """Index every case under ``data_root``."""
    return [scan_case(c) for c in discover_cases(data_root)]


def list_case_ids(data_root: str | Path, packed: bool = False) -> list[str]:
    """List the case ids available under ``data_root``.

    Works for both layouts: the raw archive (``CP_*`` directories) and the
    packed containers (``<case>.npz`` files).
    """
    root = Path(data_root)
    if packed:
        return sorted(p.stem for p in root.glob("*.npz"))
    return sorted(p.name for p in discover_cases(root))


def split_cases(
    case_ids: list[str],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    exclude_irregular: bool = True,
) -> dict[str, list[str]]:
    """Split cases into train / val / test, deterministically.

    The split is **by case**, never by sample: consecutive samples inside one
    case overlap by 21 frames (11 history + 10 target), so a random sample
    split would leak the label of a training window into the validation set
    and report a meaningless score.

    Cases are ordered by name and cut at fixed fractions, so the split is
    reproducible across machines and does not depend on the filesystem order.

    The cases with a hole in their radar series are removed **here**, not by
    the caller. They form the ``robustness_missing`` split, and a model that
    trained on them would have seen the very fault it is later measured on.
    Doing the subtraction inside this function means every caller gets the
    same partition; doing it outside means every caller has to remember to.

    Args:
        case_ids: all case ids, in any order.
        train_frac: fraction of cases used for training.
        val_frac: fraction of cases used for validation.
        exclude_irregular: drop the cases whose radar series has a hole.
            Set it to ``False`` only when the caller wants the irregular
            cases back (the robustness split itself).

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}``.
    """
    ordered = sorted(case_ids)
    if exclude_irregular:
        irregular = [c for c in ordered if c in IRREGULAR_CASES]
        ordered = [c for c in ordered if c not in IRREGULAR_CASES]
    else:
        irregular = []
    n = len(ordered)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    parts = {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }
    if irregular:
        # The irregular cases are reported alongside the partition, so a
        # caller can build the robustness split from the same list.
        parts["robustness_missing"] = irregular
    return parts
