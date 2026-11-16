"""The ranking table, computed on the server where the reports live.

Reads every ``*.eval.json`` under ``runs/sweep`` and prints one row per
baseline, best CSI first.
"""
import json
from pathlib import Path

OUT = Path("/home/wf/radarNWP/runs/sweep")

rows = []
for path in sorted(OUT.glob("*.eval.json")):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    thresholds = report.get("thresholds", {})
    a = thresholds.get("0.5", {})
    b = thresholds.get("2.0", {})
    rows.append((
        path.stem.replace(".eval", ""),
        a.get("csi", float("nan")),
        b.get("csi", float("nan")),
        a.get("ets", float("nan")),
        a.get("hss", float("nan")),
        a.get("fss", float("nan")),
        report.get("n_samples", 0),
        report.get("ablation", {}).get("mode", "?"),
    ))

rows.sort(key=lambda r: (-r[1] if r[1] == r[1] else 0))
header = (f"{'model':22s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}{'ETS':>8s}"
          f"{'HSS':>8s}{'FSS':>8s}{'n':>8s}  ablation")
print(header)
print("-" * len(header))
for name, c1, c2, ets, hss, fss, n, ablation in rows:
    print(f"{name:22s}{c1:9.3f}{c2:9.3f}{ets:8.3f}{hss:8.3f}{fss:8.3f}"
          f"{n:8d}  {ablation}")
print(f"\n{len(rows)} baselines evaluated")
