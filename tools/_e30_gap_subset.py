#!/usr/bin/env python3
"""E30 / §4.32: split the three counterfactual arms into gapped/clean subsets
and check the pre-registered anchors before any bootstrap is trusted.

Runs on the REMOTE host (the per-window tables are ~318 kB each and the SSH
channel truncates at ~256 kB).  It does no inference: it only reads the four
tables produced by ``scripts/evaluate.py --missing-frame-fill ...`` plus the
E28 ``t7_on`` / ``t7_off`` tables.

Outputs, next to the inputs:

  e30_<arm>_gap149.windows.json   paired-bootstrap input, 149 gapped windows
  e30_<arm>_clean813.windows.json paired-bootstrap input, 813 clean windows
  e30_anchor.json                 anchor-by-anchor verdicts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TIERS = ("0.1", "0.5", "2.0", "4.0")
ARMS = ("linear", "prev", "zero")
COUNTS = ("hits", "misses", "false_alarms", "correct_negatives")


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def rows(table: dict[str, Any]) -> list[dict]:
    return table["windows"]


def keys(table: dict[str, Any]) -> list[str]:
    return table["window_keys"]


def by_key(table: dict[str, Any]) -> dict[str, dict]:
    return dict(zip(keys(table), rows(table)))


def row_equal(a: dict, b: dict) -> bool:
    return all(
        all(a[t][c] == b[t][c] for c in COUNTS)
        for t in TIERS
    )


def write_table(path: Path, sel_keys: list[str], sel_rows: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "n_windows": len(sel_rows),
                "thresholds": list(TIERS),
                "window_keys": sel_keys,
                "windows": sel_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="runs", help="directory holding e30_*.windows.json")
    ap.add_argument("--t7-on", default="runs/t7_on.windows.json")
    ap.add_argument("--t7-off", default="runs/t7_off.windows.json")
    ap.add_argument("--t7-on-eval", default="runs/t7_on.eval.json")
    ap.add_argument("--prefix", default="e30")
    args = ap.parse_args()

    d = Path(args.dir)
    tables = {
        arm: load(d / f"{args.prefix}_{arm}.windows.json") for arm in ARMS
    }
    t7_on = load(Path(args.t7_on))
    t7_off = load(Path(args.t7_off))
    t7_eval = load(Path(args.t7_on_eval))

    checks: dict[str, Any] = {}
    n_all = len(rows(tables["linear"]))
    checks["n_all"] = n_all
    checks["n_all_is_962"] = n_all == 962
    checks["all_arms_same_keys"] = all(
        keys(tables[arm]) == keys(tables["linear"]) for arm in ARMS
    )
    checks["all_arms_same_window_count"] = all(
        len(rows(tables[arm])) == n_all for arm in ARMS
    )

    # The dataset's SampleSpec is the source of truth for which windows
    # actually contain a rebuilt frame.  It is recorded by evaluate.py.
    steps = tables["linear"].get("missing_frame_steps")
    if steps is None or len(steps) != n_all:
        raise SystemExit(
            "e30_linear.windows.json has no valid missing_frame_steps; "
            "re-run scripts/evaluate.py with the E30 build"
        )
    gap_idx = [i for i, s in enumerate(steps) if len(s) > 0]
    clean_idx = [i for i, s in enumerate(steps) if len(s) == 0]
    checks["n_gapped"] = len(gap_idx)
    checks["n_clean"] = len(clean_idx)
    checks["counts_813_149"] = (len(clean_idx) == 813 and len(gap_idx) == 149)

    # Adjacent holes make the "previous frame" approximation imperfect;
    # a hole at step 0 has no preceding frame at all.  Both are disclosed.
    adjacent = [i for i, s in enumerate(steps)
                if any(b - a == 1 for a, b in zip(s, s[1:]))]
    at_zero = [i for i, s in enumerate(steps) if 0 in s]
    checks["gapped_windows_with_adjacent_holes"] = len(adjacent)
    checks["gapped_windows_with_hole_at_step0"] = len(at_zero)
    checks["hole_step_histogram"] = {
        str(step): sum(1 for s in steps for x in s if x == step)
        for step in range(11)
    }

    # Anchor 2: the default arm must reproduce E28's interpolation-on arm.
    on_by_key = by_key(t7_on)
    lin_by_key = by_key(tables["linear"])
    common = [k for k in keys(tables["linear"]) if k in on_by_key]
    checks["t7_on_key_coverage"] = len(common)
    checks["linear_matches_t7_on_rows"] = (
        len(common) == n_all
        and all(row_equal(lin_by_key[k], on_by_key[k]) for k in common)
    )
    scores = t7_eval["thresholds"]
    linear_scores = load(d / f"{args.prefix}_linear.eval.json")["thresholds"]
    checks["linear_vs_t7_on_csi_abs_diff"] = {
        t: abs(float(linear_scores[t]["csi"]) - float(scores[t]["csi"]))
        for t in TIERS
    }
    checks["linear_reproduces_t7_on_scores"] = all(
        v <= 1e-6 for v in checks["linear_vs_t7_on_csi_abs_diff"].values()
    )

    # Anchor 3: on clean windows all three arms must be bit-identical.
    checks["clean813_bitwise_identical"] = all(
        row_equal(rows(tables[arm])[i], rows(tables["linear"])[i])
        for arm in ("prev", "zero") for i in clean_idx
    )

    # Anchor 4: the 149 gapped keys must equal E28's dropped-window set.
    off_keys = set(keys(t7_off))
    dropped = set(keys(t7_on)) - off_keys
    gap_keys = {keys(tables["linear"])[i] for i in gap_idx}
    checks["t7_dropped_windows"] = len(dropped)
    checks["gap149_keys_equal_t7_dropped"] = gap_keys == dropped
    checks["off_is_subset_of_on"] = off_keys <= set(keys(t7_on))

    anchors = [
        "n_all_is_962", "all_arms_same_keys", "all_arms_same_window_count",
        "counts_813_149", "linear_matches_t7_on_rows",
        "linear_reproduces_t7_on_scores", "clean813_bitwise_identical",
        "gap149_keys_equal_t7_dropped", "off_is_subset_of_on",
    ]
    checks["ANCHORS_ALL_PASS"] = all(checks[k] for k in anchors)

    # Paired-bootstrap inputs.  The bootstrap resamples WINDOWS, so the three
    # arms must be sliced with the very same index list (single-variable
    # comparison on the same 149 / 813 windows).
    for arm in ARMS:
        t = tables[arm]
        write_table(
            d / f"{args.prefix}_{arm}_gap149.windows.json",
            [keys(t)[i] for i in gap_idx],
            [rows(t)[i] for i in gap_idx],
        )
        write_table(
            d / f"{args.prefix}_{arm}_clean813.windows.json",
            [keys(t)[i] for i in clean_idx],
            [rows(t)[i] for i in clean_idx],
        )

    (d / "e30_anchor.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print("E30_ANCHORS_ALL_PASS:", checks["ANCHORS_ALL_PASS"])
    return 0 if checks["ANCHORS_ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
