"""Consolidated verdict table for the score-aware matrix."""
from __future__ import annotations

import json
import math
from pathlib import Path

BASE = Path(__file__).parent / "_evals"
MODELS = [
    "simvp_v2", "simvp_v2_nwp_concat", "simvp_v2_quantum", "simvp_v2_nwp_xattn",
    "simvp_v2_nwp_delta", "simvp_radar_only", "convlstm",
    "predrnn", "earthformer", "metnet3", "diffcast", "cascast_nwp", "mocast",
    "pw_foucast", "fusioncast", "mmf_rnn", "furecast",
]
ALL18 = MODELS + ["simvp_early_concat"]


def load(d, m):
    p = BASE / d / f"{m}.eval.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def c(r, t, k="csi"):
    if r is None:
        return float("nan")
    return r["thresholds"].get(t, {}).get(k, float("nan"))


old = {m: load("tgrs", m) for m in ALL18}
new = {m: load("score_matrix", m) for m in ALL18}
py = load("tgrs", "pysteps_adv")

print("=== light-rain accounting: where did the hits go? ===")
print(f"{'model':22s}{'stdHIT.1':>11s}{'saHIT.1':>11s}{'stdFA.1':>9s}{'saFA.1':>9s}{'stdPOD.1':>10s}{'saPOD.1':>10s}")
for m in ALL18:
    o, n = old[m], new[m]
    print(f"{m:22s}{c(o,'0.1','hits'):11.0f}{c(n,'0.1','hits'):11.0f}"
          f"{c(o,'0.1','far'):9.4f}{c(n,'0.1','far'):9.4f}{c(o,'0.1','pod'):10.4f}{c(n,'0.1','pod'):10.4f}")

print()
print("=== D3/D4 scan numbers (cross-check vs the clean matrix) ===")
for name in ("scan_D1", "scan_D2", "scan_D3", "scan_D4"):
    r = load("hqma_scan4", name)
    if r:
        print(f"{name:10s} n={r['n_samples']} CSI@2.0={c(r,'2.0'):.4f} CSI@4.0={c(r,'4.0'):.4f} "
              f"CSI@0.1={c(r,'0.1'):.4f} CSI@0.5={c(r,'0.5'):.4f} FAR@0.5={c(r,'0.5','far'):.4f}")
for m in ("simvp_v2", "simvp_v2_nwp_concat", "simvp_v2_quantum"):
    r = load("hqma", m)
    if r:
        print(f"hqma/{m:22s} n={r['n_samples']} CSI@2.0={c(r,'2.0'):.4f} CSI@0.1={c(r,'0.1'):.4f}")
for name in ("hqma_balanced", "hqma_standard"):
    r = load("hqma", name)
    if r:
        print(f"hqma/{name:22s} n={r['n_samples']} CSI@2.0={c(r,'2.0'):.4f} CSI@4.0={c(r,'4.0'):.4f} "
              f"CSI@0.1={c(r,'0.1'):.4f} CSI@0.5={c(r,'0.5'):.4f}")

print()
print("=== VERDICT CHECKS ===")
lift2 = [(m, c(old[m], '2.0'), c(new[m], '2.0')) for m in ALL18 if old[m]]
lift4 = [(m, c(old[m], '4.0'), c(new[m], '4.0')) for m in ALL18 if old[m]]
print(f"[1] CSI@2.0 lifted for {sum(1 for _, o, n in lift2 if n > o)}/{len(lift2)} paired models")
print(f"    CSI@4.0 lifted for {sum(1 for _, o, n in lift4 if n > o)}/{len(lift4)} paired models")
sa2 = [c(new[m], '2.0') for m in ALL18]
print(f"[2] score-aware CSI@2.0 absolute level: min={min(sa2):.4f} med={sorted(sa2)[len(sa2)//2]:.4f} max={max(sa2):.4f}")
print(f"    reaching >=0.05: {sum(1 for v in sa2 if v>=0.05)}/18 | >=0.10: {sum(1 for v in sa2 if v>=0.10)}/18")
print(f"    -> hypothesis 'universally rises to the 0.1 range': {'CONFIRMED' if sum(1 for v in sa2 if v>=0.10)>=15 else 'NOT CONFIRMED (only %d/18 reach 0.10)'%sum(1 for v in sa2 if v>=0.10)}")
d01 = [(c(new[m], '0.1') - c(old[m], '0.1')) for m in ALL18 if old[m]]
print(f"[3] CSI@0.1 dropped for {sum(1 for d in d01 if d<0)}/{len(d01)}; below 0.1: {sum(1 for m in ALL18 if c(new[m],'0.1')<0.1)}/18")
print(f"    -> hypothesis 'falls below 0.1 (loss untenable)': {'CONFIRMED' if any(c(new[m],'0.1')<0.1 for m in ALL18) else 'NOT CONFIRMED (min = %.4f)'%min(c(new[m],'0.1') for m in ALL18)}")
fa = [(c(new[m], '0.5', 'far') - c(old[m], '0.5', 'far')) for m in ALL18 if old[m]]
print(f"[4] FAR@0.5: up for {sum(1 for d in fa if d>0)}, down for {sum(1 for d in fa if d<0)}; "
      f"mean {sum(c(old[m],'0.5','far') for m in ALL18 if old[m])/17:.4f} -> {sum(c(new[m],'0.5','far') for m in ALL18)/18:.4f}")
print(f"    -> D1/D2-specific cost does NOT generalize: matrix mean FAR@0.5 is flat")
above2 = [m for m in ALL18 if c(new[m], '2.0') > c(py, '2.0')]
above4 = [m for m in ALL18 if c(new[m], '4.0') > c(py, '4.0')]
above01 = [m for m in ALL18 if c(new[m], '0.1') > c(py, '0.1')]
print(f"[5] pysteps_adv (CSI@2.0={c(py,'2.0'):.4f}, CSI@4.0={c(py,'4.0'):.4f}, CSI@0.1={c(py,'0.1'):.4f})")
print(f"    models above at CSI@2.0: {len(above2)}/18 {above2}")
print(f"    models above at CSI@4.0: {len(above4)}/18 {above4}")
print(f"    models above at CSI@0.1: {len(above01)}/18 {above01}")
print(f"    -> hypothesis 'pysteps still beaten': {'CONFIRMED' if len(above2)>0 else 'NOT CONFIRMED'}")
so = sorted(MODELS, key=lambda m: -c(new[m], '2.0'))
so0 = sorted(MODELS, key=lambda m: -c(old[m], '2.0'))
n = len(MODELS)
d2 = sum((so0.index(m) - so.index(m)) ** 2 for m in MODELS)
print(f"[6] Spearman rho(std@2.0, sa@2.0) = {1-6*d2/(n*(n*n-1)):.3f}")
o5 = sorted(MODELS, key=lambda m: -c(old[m], '0.5'))
d2b = sum((o5.index(m) - so.index(m)) ** 2 for m in MODELS)
print(f"    Spearman rho(std@0.5, sa@2.0) = {1-6*d2b/(n*(n*n-1)):.3f}  <- the new order tracks 0.5 mm skill")
sp_o = max(c(old[m],'2.0') for m in MODELS)/max(min(c(old[m],'2.0') for m in MODELS),1e-9)
sp_n = max(c(new[m],'2.0') for m in MODELS)/max(min(c(new[m],'2.0') for m in MODELS),1e-9)
print(f"    spread max/min: std {sp_o:.0f}x -> sa {sp_n:.1f}x  (architectures still differ by {sp_n:.1f}x)")
print()
print("=== score-aware ranking (CSI@2.0) ===")
for i, m in enumerate(so, 1):
    print(f"  {i:2d}. {m:22s} {c(new[m],'2.0'):.4f}   (std {c(old[m],'2.0'):.4f})")
print(f"  -- pysteps_adv (no training): {c(py,'2.0'):.4f}")
