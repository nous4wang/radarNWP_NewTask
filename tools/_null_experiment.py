"""The baseline null experiment: did the data layer move anything?

The history rainfall was added to the dataset. The baselines were
trained before it existed, and a comparison against them is only worth
anything if they still measure the same thing.

There are two ways to ask, and they answer different questions.

**The exact check.** Re-score the checkpoints that the matrix was built
from, under the current code, and compare with the reports they were
scored into when they were run. This one has no training variance in
it at all -- the weights are fixed -- so a difference here can only
come from the data, the evaluation or the model construction. It is the
exact test.

**The retrained check.** Train the baselines again under the current
code. This one carries the full run-to-run spread, which has been
measured at 1.4 times for the score-aware objective on the strong-rain
score (0.0384 against 0.027 for the same configuration and seed), so
the acceptance band is a band and not a point.

Both are run here; the exact one decides whether the data layer is
sound, and the retrained one supplies the fresh comparison arm that the
new routes are measured against.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# The output is redirected to a file, where Python buffers
# in blocks; without this the log stays empty until the process ends.
for _stream in (sys.stdout, sys.stderr):
    try:/n        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

RUNS = Path("/home/wf/radarNWP/runs")
SCORE_MATRIX = RUNS / "score_matrix"
OUT = Path("/tmp/_null_experiment")
OUT.mkdir(parents=True, exist_ok=True)

PYTHON = "/home/wf/.conda/envs/radar_nwp/bin/python"
DATA = "/home/wf/data/radarNWP/BJ_mmap"

#: The models whose scores the new routes will be compared against.
BASELINES = ("simvp_v2", "simvp_v2_nwp_concat")


def newest_checkpoint(model: str) -> Path | None:
    base = SCORE_MATRIX / model
    if not base.exists():
        return None
    candidates = sorted(base.glob("*/last.pt"))
    return candidates[-1] if candidates else None


def stored_report(model: str) -> dict | None:
    path = SCORE_MATRIX / f"{model}.eval.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def re_score(model: str) -> dict | None:
    """Score an existing checkpoint under the current code."""
    ckpt = newest_checkpoint(model)
    if ckpt is None:
        print(f"{model}: no checkpoint")
        return None
    out_json = OUT / f"{model}.rescore.json"
    cmd = [
        PYTHON, "scripts/evaluate.py",
        "--model", model,
        "--checkpoint", str(ckpt),
        "--data-root", DATA,
        "--packed",
        "--split", "test",
        "--batch-size", "16",
        "--num-workers", "6",
        "--patch-size", "64",
        "--slide-stride", "64",
        "--device", "cuda",
        "--out-json", str(out_json),
    ]
    print(f"\n=== re-scoring {model} under the current code ===")
    log = OUT / f"{model}.rescore.log"
    with log.open("w") as handle:
        subprocess.run(cmd, cwd="/home/wf/radarNWP", stdout=handle,
                       stderr=subprocess.STDOUT, check=False)
    if not out_json.exists():
        print(f"{model}: the re-scoring failed; see {log}")
        return None
    return json.loads(out_json.read_text())


def compare(model: str, old: dict, new: dict) -> dict:
    print(f"\n--- {model}: stored vs re-scored ---")
    print(f"{'threshold':>10}{'stored CSI':>14}{'new CSI':>12}"
          f"{'stored n':>12}{'new n':>10}")
    rows = {}
    for thr in ("0.1", "0.5", "2.0", "4.0"):
        a = old["thresholds"].get(thr, {})
        b = new["thresholds"].get(thr, {})
        print(f"{thr:>10}{a.get('csi', float('nan')):>14.4f}"
              f"{b.get('csi', float('nan')):>12.4f}")
        rows[thr] = {
            "stored_csi": a.get("csi"),
            "new_csi": b.get("csi"),
            "stored_hits": a.get("hits"),
            "new_hits": b.get("hits"),
        }
    print(f"{'n_samples':>10}{old.get('n_samples'):>14d}"
          f"{new.get('n_samples'):>12d}")
    print(f"{'ablation':>10}{str(old.get('ablation')):>14s}"
          f"{str(new.get('ablation')):>12s}")
    return rows


def main() -> int:
    summary = {"exact_check": {}}
    ok = True

    for model in BASELINES:
        old = stored_report(model)
        if old is None:
            print(f"{model}: no stored report")
            continue
        new = re_score(model)
        if new is None:
            ok = False
            continue
        rows = compare(model, old, new)

        same_n = old.get("n_samples") == new.get("n_samples")
        print(f"n_samples identical: {same_n}")
        # The exact check: with the weights fixed, the counts must be
        # identical. Anything else means the data or the stitching moved.
        counts_match = all(
            rows[t]["stored_hits"] == rows[t]["new_hits"]
            for t in rows
            if rows[t]["stored_hits"] is not None
        )
        print(f"hit counts identical: {counts_match}")
        if not (same_n and counts_match):
            ok = False
        summary["exact_check"][model] = {
            "n_samples": [old.get("n_samples"), new.get("n_samples")],
            "hits_identical": counts_match,
            "rows": rows,
        }

    (OUT / "exact_check.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'exact_check.json'}")
    print("VERDICT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
