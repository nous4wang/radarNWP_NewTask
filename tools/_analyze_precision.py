"""Is the score-aware gain a real skill gain, or a rain-spray artifact?"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).parent / "_evals"
ALL18 = ["simvp_v2", "simvp_v2_nwp_concat", "simvp_v2_quantum", "simvp_v2_nwp_xattn",
         "simvp_v2_nwp_delta", "simvp_early_concat", "simvp_radar_only", "convlstm",
         "predrnn", "earthformer", "metnet3", "diffcast", "cascast_nwp", "mocast",
         "pw_foucast", "fusioncast", "mmf_rnn", "furecast"]


def load(d, m):
    p = BASE / d / f"{m}.eval.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def g(r, t, k):
    if r is None:
        return float("nan")
    return r["thresholds"].get(t, {}).get(k, float("nan"))


def prec(r, t):
    h, f = g(r, t, "hits"), g(r, t, "false_alarms")
    return h / (h + f) if (h + f) else float("nan")


new = {m: load("score_matrix", m) for m in ALL18}
old = {m: load("tgrs", m) for m in ALL18}
py = load("tgrs", "pysteps_adv")

print("=== precision at the heavy tier: score-aware ===")
print(f"{'model':22s}{'prec@2.0':>10s}{'prec@4.0':>10s}{'stdprec@2.0':>13s}{'FA@2.0':>10s}{'hits@2.0':>11s}{'miss@2.0':>12s}")
for m in sorted(ALL18, key=lambda m: -g(new[m], "2.0", "csi")):
    n = new[m]
    print(f"{m:22s}{prec(n,'2.0'):10.4f}{prec(n,'4.0'):10.4f}{prec(old[m],'2.0'):13.4f}"
          f"{g(n,'2.0','false_alarms'):10.0f}{g(n,'2.0','hits'):11.0f}{g(n,'2.0','misses'):12.0f}")
print(f"{'pysteps_adv':22s}{prec(py,'2.0'):10.4f}{prec(py,'4.0'):10.4f}{'':13s}"
      f"{g(py,'2.0','false_alarms'):10.0f}{g(py,'2.0','hits'):11.0f}{g(py,'2.0','misses'):12.0f}")

print()
print("=== how much of the truth at 2 mm is still being missed? ===")
print(f"{'model':22s}{'miss@2.0':>13s}{'POD@2.0':>9s}")
for m in sorted(ALL18, key=lambda m: -g(new[m], "2.0", "csi")):
    n = new[m]
    print(f"{m:22s}{g(n,'2.0','misses'):13.0f}{g(n,'2.0','pod'):9.4f}")
print(f"{'pysteps_adv':22s}{g(py,'2.0','misses'):13.0f}{g(py,'2.0','pod'):9.4f}")

print()
print("=== the extreme case: fusioncast and diffcast were near-zero under the standard loss ===")
for m in ("fusioncast", "diffcast", "mmf_rnn", "mocast"):
    print(f"{m}: std CSI@2.0={g(old[m],'2.0','csi'):.4f} std FA@2.0={g(old[m],'2.0','false_alarms'):.0f} "
          f"std hits={g(old[m],'2.0','hits'):.0f}")
    print(f"{' ' * len(m)}  sa  CSI@2.0={g(new[m],'2.0','csi'):.4f} sa  FA@2.0={g(new[m],'2.0','false_alarms'):.0f} "
          f"sa  hits={g(new[m],'2.0','hits'):.0f}")

print()
print("=== the tier table for the paper (score-aware, all 18) ===")
hdr = f"{'model':22s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'BIAS@0.5':>10s}{'hits@2.0':>11s}{'prec@2.0':>10s}"
print(hdr)
print("-" * len(hdr))
for m in sorted(ALL18, key=lambda m: -g(new[m], "2.0", "csi")):
    n = new[m]
    print(f"{m:22s}{g(n,'0.1','csi'):9.4f}{g(n,'0.5','csi'):9.4f}{g(n,'2.0','csi'):9.4f}{g(n,'4.0','csi'):9.4f}"
          f"{g(n,'0.5','far'):9.4f}{g(n,'0.5','bias'):10.4f}{g(n,'2.0','hits'):11.0f}{prec(n,'2.0'):10.4f}")
print(f"{'pysteps_adv':22s}{g(py,'0.1','csi'):9.4f}{g(py,'0.5','csi'):9.4f}{g(py,'2.0','csi'):9.4f}"
      f"{g(py,'4.0','csi'):9.4f}{g(py,'0.5','far'):9.4f}{g(py,'0.5','bias'):10.4f}"
      f"{g(py,'2.0','hits'):11.0f}{prec(py,'2.0'):10.4f}")

print()
print("=== the standard-loss table for the paper (17 models with a matched baseline) ===")
print(hdr)
print("-" * len(hdr))
for m in sorted([m for m in ALL18 if old[m]], key=lambda m: -g(old[m], "2.0", "csi")):
    o = old[m]
    print(f"{m:22s}{g(o,'0.1','csi'):9.4f}{g(o,'0.5','csi'):9.4f}{g(o,'2.0','csi'):9.4f}{g(o,'4.0','csi'):9.4f}"
          f"{g(o,'0.5','far'):9.4f}{g(o,'0.5','bias'):10.4f}{g(o,'2.0','hits'):11.0f}{prec(o,'2.0'):10.4f}")
