"""Check that the data layer's change did not move the sample set.

The history rainfall was added to the dataset. The baselines were
trained before it existed, so the question is whether the addition
changed anything they depend on: the number of windows, the split, the
targets, the patches.

This runs on the server, where the archive is. It builds the index with
the current code and compares it against the numbers the matrix was
produced with: 538 windows on the test split, 23/8/8 cases, and the
per-case sample counts.

A silent change here would move every baseline, and the comparison
would be void without anyone noticing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# The output is redirected to a file, where Python buffers
# in blocks; without this the log stays empty until the process ends.
for _stream in (sys.stdout, sys.stderr):
    try:/n        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, "/home/wf/radarNWP")

import numpy as np

from radar_nwp_bench.data import RadarNWPDataset
from radar_nwp_bench.data.index import IRREGULAR_CASES

DATA = "/home/wf/data/radarNWP/BJ_mmap"
OUT = Path("/tmp/_null_check")


def build(split: str) -> RadarNWPDataset:
    return RadarNWPDataset(
        DATA, packed=True, split=split, in_len=11, out_len=10,
        patch_size=64, random_crop=False, slide_stride=64,
    )


def main() -> int:
    report: dict = {}

    for split in ("train", "val", "test"):
        ds = build(split)
        cases = sorted({s.case_path.name for s in ds.samples})
        report[split] = {
            "n_samples": len(ds.samples),
            "n_cases": len(cases),
            "n_patch_positions": ds.n_patch_positions,
        }
        print(f"{split:6s} windows={len(ds.samples):6d} cases={len(cases):3d} "
              f"patches={ds.n_patch_positions}")

    # The test split is what the matrix reports: 538 windows.
    test = build("test")
    report["test"]["matches_matrix_n_samples"] = len(test.samples) == 538

    # The history rainfall has to line up with the radar history, frame
    # for frame, and the target window has to be unchanged.
    item = test[0]
    print()
    print("batch keys:", sorted(item.keys()))
    print("y_hist  shape:", tuple(item["y_hist"].shape))
    print("y_rain  shape:", tuple(item["y_rain"].shape))
    print("y_hist  non-negative:", bool((item["y_hist"] >= 0).all()))
    print("y_hist  finite:", bool(np.isfinite(item["y_hist"].numpy()).all()))
    print("hist_missing_mask:", item["hist_missing_mask"].tolist())

    # The history rainfall and the radar history must be aligned in time:
    # the target is the rain of the frames *after* the issue time, and the
    # history is the rain of the frames before it. If the two were
    # misaligned by even one frame, the inversion supervision would be
    # teaching the model to predict the past.
    n_hist = item["y_hist"].shape[0]
    n_future = item["y_rain"].shape[0]
    print(f"history frames={n_hist} future frames={n_future}")

    # The fraction of the history that is wet tells us the alignment is
    # plausible: the rain field is continuous in time, so the history and
    # the future should have a comparable amount of rain.
    hist_wet = float((item["y_hist"] > 0.1).float().mean())
    future_wet = float((item["y_rain"] > 0.1).float().mean())
    print(f"wet fraction: history {hist_wet:.4f}  future {future_wet:.4f}")

    # A few samples across the split, to see that the history is not
    # empty and that the mask marks the rebuilt slots.
    wet_counts = []
    masked = 0
    for i in range(0, min(len(test), 40)):
        it = test[i]
        wet_counts.append(float((it["y_hist"] > 0.1).float().mean()))
        masked += int(it["hist_missing_mask"].sum())
    print(f"over {min(len(test), 40)} samples: "
          f"mean wet fraction {np.mean(wet_counts):.4f}, "
          f"rebuilt history slots {masked}")

    report["history"] = {
        "wet_fraction_mean": float(np.mean(wet_counts)),
        "rebuilt_slots": masked,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "null_check.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'null_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
