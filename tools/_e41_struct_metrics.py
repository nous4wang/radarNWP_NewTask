#!/usr/bin/env python3
"""E41 / MASTER 4.43 -- structure metrics on the PREDICTED field (S1/S3).

E41 section 5, frozen 2026-11-09 09:29:

* S1 (primary) requires four structure quantities on the forecast field:
  ``BIAS@2.0``, ``tail_max_over_p99``, ``zero_frac`` and ``kernel_px_gt_p99``,
  and it says explicitly that they must be computed with **the same
  implementation the section 2 probe used**
  (``tools/_struct_probe_one_case.py::metrics``), "not a second, looser
  statistic".  That is why this file imports ``metrics`` from that module
  instead of copying it: a copy would be a second implementation.

* The section 2 probe measures one field per case (a radar frame and the
  label frame of the same instant).  Here the object is the forecast:
  ``scripts/evaluate.py --dump-fields`` writes the stitched whole-field
  forecast and truth for every scored window, float32, shape
  ``(n_win, out_len, grid_h, grid_w)``.  The forecast is what the model
  produced; the truth is written next to it so the structure of the
  forecast can be compared with the structure of the label on the same
  pixels.

* The structure of a forecast is a property of a FIELD, not of a single
  frame.  The probe's ``metrics`` takes one 2-D field.  So each
  ``(window, lead)`` frame of the forecast is measured on its own, and the
  reported value is the MEDIAN over the frames -- the same aggregation the
  probe used for its 62 cases (it reported medians over cases).  The
  median is also the robust choice: one bad frame cannot move it.

* The predicted field is in millimetres, the label is in millimetres, and
  the probe's quantities are unit-free by construction (quantile ratios,
  pixel counts, and a z-scored total variation), so no scaling is applied.
  The one exception is ``zero_frac``: the probe defines it as the fraction
  of pixels at the field MINIMUM (``x <= x.min() + 1e-9``), which for the
  label means "the fraction of exactly-zero pixels" because the label has
  a hard zero floor.  The forecast does not have a hard zero floor, so the
  same expression would measure the fraction of pixels at the smallest
  forecast value -- a different thing under a different name.  The
  forecast's dry fraction is therefore reported as the fraction of pixels
  below the 0.1 mm business threshold, and the SAME expression is also
  applied to the truth field so that the forecast and the truth are
  compared like with like.  Both numbers are written to the output; the
  probe's original expression is kept under its own name.

Nothing here trains, evaluates or touches the test truth of a model that
was not scored by ``scripts/evaluate.py`` on the same windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

# The section 2 probe's implementation, imported rather than copied.
from _struct_probe_one_case import metrics as probe_metrics  # noqa: E402
from _struct_probe_one_case import radial_grid  # noqa: E402

#: The section 2 probe's own definition of a dry pixel, kept under its own
#: name because it is NOT the same thing on a forecast (see the docstring).
PROBE_ZERO_FRAC = "probe_zero_frac"


def _median(values: list[float]) -> float | None:
    arr = np.asarray(
        [v for v in values if v is not None and not np.isnan(v)],
        dtype=np.float64,
    )
    if arr.size == 0:
        return None
    return float(np.nanmedian(arr))


def structure_of_field(field: np.ndarray, grid: np.ndarray) -> dict:
    """Run the section 2 metrics on one 2-D field.

    The probe's ``metrics`` returns ``zero_frac`` as the fraction of
    pixels at the field minimum.  On the label that is the dry fraction;
    on a forecast it is not, so the forecast's dry fraction is computed
    here at the 0.1 mm threshold and the probe's own number is kept
    separately.
    """
    out = probe_metrics(field, grid)
    dry = float(np.mean(field < 0.1))
    out["dry_frac_lt_0p1"] = round(dry, 6)
    out[PROBE_ZERO_FRAC] = out.pop("zero_frac")
    return out


def aggregate(fields: np.ndarray, grid: np.ndarray, max_frames: int | None,
              sample: int = 0) -> dict:
    """Median over the frames of one field stack.

    Args:
        fields: ``(n_frames, H, W)`` float32 array.
        grid: the radial frequency grid from the section 2 probe.
        max_frames: if given, only the first ``max_frames`` frames are used.
        sample: which frames to keep when ``max_frames`` truncates --
            ``0`` takes every ``sample``-th frame, which spreads the
            sample over the lead times instead of taking a prefix.

    Returns:
        A dict with the median of every structure quantity.
    """
    n = fields.shape[0]
    if max_frames is not None and n > max_frames:
        idx = np.arange(0, n, max(1, sample))[:max_frames]
        fields = fields[idx]
    rows = [structure_of_field(np.asarray(fields[i], dtype=np.float64), grid)
            for i in range(fields.shape[0])]
    keys = rows[0].keys()
    return {k: _median([r[k] for r in rows]) for k in keys}


def load_stack(path: Path) -> np.ndarray:
    """Load a ``.pred.npy`` / ``.truth.npy`` stack as float32 in RAM."""
    arr = np.load(path, mmap_mode="r")
    return np.asarray(arr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", required=True,
                    help="directory holding <tag>.pred.npy / <tag>.truth.npy")
    ap.add_argument("--tags", nargs="+", required=True,
                    help="tags to measure, e.g. B0_s0_ep00 A1_s0_ep00")
    ap.add_argument("--out", required=True, help="where to write the JSON")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="measure only this many frames per window")
    ap.add_argument("--frame-sample", type=int, default=0,
                    help="stride used when --max-frames truncates")
    args = ap.parse_args()

    fields_dir = Path(args.fields)
    out: dict = {
        "n_windows": None,
        "out_len": None,
        "grid": None,
        "aggregation": "median over (window, lead) frames",
        "probe_implementation": "tools/_struct_probe_one_case.py::metrics",
        "tags": {},
    }

    for tag in args.tags:
        pred_path = fields_dir / f"{tag}.pred.npy"
        truth_path = fields_dir / f"{tag}.truth.npy"
        if not pred_path.exists():
            out["tags"][tag] = {"error": f"missing {pred_path.name}"}
            continue
        pred = load_stack(pred_path)
        truth = load_stack(truth_path) if truth_path.exists() else None
        n_win, out_len, gh, gw = pred.shape
        out["n_windows"] = n_win
        out["out_len"] = out_len
        out["grid"] = [gh, gw]
        grid = radial_grid((gh, gw))

        # Per-frame medians.  The forecast is measured frame by frame; the
        # truth is measured on the same frames so that the comparison is
        # like with like.
        pred_rows = []
        truth_rows = []
        for w in range(n_win):
            pf = np.asarray(pred[w], dtype=np.float64)
            pred_rows.append(aggregate(pf, grid, args.max_frames,
                                        args.frame_sample))
            if truth is not None:
                tf = np.asarray(truth[w], dtype=np.float64)
                truth_rows.append(aggregate(tf, grid, args.max_frames,
                                            args.frame_sample))
        entry: dict = {
            "pred": {
                k: _median([r[k] for r in pred_rows])
                for k in pred_rows[0]
            },
            "n_windows_measured": len(pred_rows),
        }
        if truth_rows:
            entry["truth"] = {
                k: _median([r[k] for r in truth_rows])
                for k in truth_rows[0]
            }
        out["tags"][tag] = entry
        print(f"measured {tag}: {len(pred_rows)} windows", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"structure metrics written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
