#!/usr/bin/env python3
"""Align the two fault-tolerance arms to their COMMON windows (E28 / §4.30).

``--no-interpolate`` drops every window whose radar history needs a rebuilt
frame: §4.16.4 reported 962 windows with interpolation and 813 without, and
already warned that those are not a strict paired subset.  A paired bootstrap
on the raw tables is therefore invalid (and ``_bootstrap_qpe_vs_zr.py`` will
refuse it on the window-count mismatch).

``scripts/evaluate.py --per-window-out`` now emits a ``window_keys`` list of
``case:issue_time`` identifiers.  This tool restricts the interpolation-on
table to the keys retained by the interpolation-off table, preserving the
off-arm order, and reports how many rows of the aligned pair are bit-identical.
Zero-training-compute; it only reads JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TIERS = ("0.1", "0.5", "2.0", "4.0")
COUNTS = ("hits", "misses", "false_alarms", "correct_negatives")


def load(path: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not d.get("window_keys") or len(d["window_keys"]) != len(d["windows"]):
        raise SystemExit(
            f"{path}: missing or misaligned window_keys; re-run evaluate.py "
            f"with the version that emits them")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--on", required=True, help="interpolation-on window table")
    ap.add_argument("--off", required=True, help="interpolation-off window table")
    ap.add_argument("--out", required=True, help="aligned on-arm table to write")
    args = ap.parse_args()

    a, b = load(args.on), load(args.off)
    keys_on, keys_off = a["window_keys"], b["window_keys"]
    if len(set(keys_on)) != len(keys_on) or len(set(keys_off)) != len(keys_off):
        raise SystemExit("duplicate window keys: alignment is ambiguous")
    pos = {k: i for i, k in enumerate(keys_on)}
    missing = [k for k in keys_off if k not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} off-arm windows missing from on arm, "
                         f"e.g. {missing[:3]}")

    idx = [pos[k] for k in keys_off]
    aligned = [a["windows"][i] for i in idx]
    identical = 0
    for ra, rb in zip(aligned, b["windows"]):
        if all(ra[t][c] == rb[t][c] for t in TIERS for c in COUNTS):
            identical += 1

    out = {
        "n_windows": len(aligned),
        "thresholds": a["thresholds"],
        "window_keys": keys_off,
        "windows": aligned,
        "alignment": {
            "on_windows": len(keys_on),
            "off_windows": len(keys_off),
            "dropped_by_no_interpolate": len(keys_on) - len(keys_off),
            "off_is_subset_of_on": True,
            "rows_bit_identical_on_common_windows": identical,
            "rows_compared": len(aligned),
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"on={len(keys_on)} off={len(keys_off)} "
          f"dropped={len(keys_on) - len(keys_off)} "
          f"bit_identical_rows={identical}/{len(aligned)} -> {args.out}")
    return 0


if __name__ == "__main__":
    main()
