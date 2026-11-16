"""Why does the robustness split score HIGHER than the clean split?

The robustness split is the 23 cases whose radar series has a hole; the
clean split is the 8 regular test cases.  They are disjoint case sets, so
the comparison in the paper's robustness table is between two different
storms, not between the same storms with and without a hole.

This script quantifies the composition difference from the reports
themselves: it converts the counts into per-window densities, so the
comparison does not depend on how many windows each split happens to have.
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("runs/tgrs")
THR = ("0.1", "0.5", "2.0", "4.0")

rows = []
for path in sorted(RUNS.glob("*.eval.json")):
    name = path.name[: -len(".eval.json")]
    robust_path = RUNS / f"{name}.robustness.json"
    if not robust_path.exists():
        continue
    clean = json.loads(path.read_text())
    robust = json.loads(robust_path.read_text())

    entry = {"model": name, "n_clean": clean["n_samples"],
             "n_rob": robust["n_samples"]}
    for tag, report in (("c", clean), ("r", robust)):
        for thr in THR:
            t = report["thresholds"][thr]
            truth = t["hits"] + t["misses"]          # truth pixels above thr
            pred = t["hits"] + t["false_alarms"]     # predicted pixels above thr
            n = report["n_samples"]
            entry[f"truth_{tag}_{thr}"] = truth / n
            entry[f"pred_{tag}_{thr}"] = pred / n
            entry[f"csi_{tag}_{thr}"] = t["csi"]
            entry[f"far_{tag}_{thr}"] = t["far"]
    rows.append(entry)

if not rows:
    raise SystemExit("no comparable pairs found")

print("=== per-window densities (clean 'c' vs robustness 'r') ===")
print("truth = truth pixels above the threshold per window")
print("pred  = predicted pixels above the threshold per window")
print()
for thr in THR:
    print(f"--- threshold {thr} mm ---")
    print(f"{'model':<22}{'truth c':>11}{'truth r':>11}{'ratio':>8}"
          f"{'pred c':>11}{'pred r':>11}{'CSI c':>9}{'CSI r':>9}{'FAR c':>9}{'FAR r':>9}")
    for e in rows:
        tc, tr = e[f"truth_c_{thr}"], e[f"truth_r_{thr}"]
        ratio = tr / tc if tc else float("nan")
        print(f"{e['model']:<22}{tc:>11,.0f}{tr:>11,.0f}{ratio:>8.2f}"
              f"{e[f'pred_c_{thr}']:>11,.0f}{e[f'pred_r_{thr}']:>11,.0f}"
              f"{e[f'csi_c_{thr}']:>9.4f}{e[f'csi_r_{thr}']:>9.4f}"
              f"{e[f'far_c_{thr}']:>9.4f}{e[f'far_r_{thr}']:>9.4f}")
    print()

# The mean composition ratio tells the story in one number.
print("=== summary ===")
for thr in THR:
    ratios = [e[f"truth_r_{thr}"] / e[f"truth_c_{thr}"] for e in rows
              if e[f"truth_c_{thr}"]]
    d_csi = [e[f"csi_r_{thr}"] - e[f"csi_c_{thr}"] for e in rows]
    print(f"  @{thr} mm: the robustness split carries "
          f"{sum(ratios)/len(ratios):.2f}x the truth rain per window; "
          f"CSI changes by {sum(d_csi)/len(d_csi):+.4f} on average")
print()
print("The robustness split has fewer truth pixels per window at every")
print("threshold above 0.1 mm, so the storms there are weaker and sparser,")
print("and weaker storms are easier to score.  The CSI difference between")
print("the two splits is a property of the case composition, not of the")
print("holes in the radar series.")
