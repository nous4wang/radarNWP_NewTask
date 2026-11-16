"""Is the score-aware ordering a re-measurement of the same ranking, or a new one?"""
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
]  # simvp_early_concat dropped: no comparable standard baseline (sweep, n=794, no norm-v2)


def load(d, m):
    p = BASE / d / f"{m}.eval.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def c(r, t):
    return r["thresholds"].get(t, {}).get("csi", float("nan"))


def rho(a, b):
    n = len(a)
    d2 = sum((a.index(m) - b.index(m)) ** 2 for m in a)
    return 1 - 6 * d2 / (n * (n * n - 1))


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else float("nan")


old = {m: load("tgrs", m) for m in MODELS}
new = {m: load("score_matrix", m) for m in MODELS}
py = load("tgrs", "pysteps_adv")

print(f"models in the paired analysis: {len(MODELS)} (simvp_early_concat excluded)")
print()
print("=== rank agreement between standard-loss and score-aware CSI@2.0 ===")
o2 = sorted(MODELS, key=lambda m: -c(old[m], "2.0"))
n2 = sorted(MODELS, key=lambda m: -c(new[m], "2.0"))
print(f"Spearman rho (CSI@2.0)          = {rho(o2, n2):.3f}")

print()
print("=== does the score-aware CSI@2.0 order agree with the standard-loss CSI@0.5 order? ===")
o5 = sorted(MODELS, key=lambda m: -c(old[m], "0.5"))
n5 = sorted(MODELS, key=lambda m: -c(new[m], "0.5"))
print(f"Spearman rho std@0.5  vs sa@0.5  = {rho(o5, n5):.3f}")
print(f"Spearman rho std@0.5  vs sa@2.0  = {rho(o5, n2):.3f}")
print(f"Spearman rho std@2.0  vs sa@0.5  = {rho(o2, n5):.3f}")
print(f"Spearman rho std@0.1  vs sa@2.0  = {rho(sorted(MODELS, key=lambda m: -c(old[m],'0.1')), n2):.3f}")

print()
print("=== Pearson on the numbers themselves ===")
for t in ("0.1", "0.5", "2.0", "4.0"):
    a = [c(old[m], t) for m in MODELS]
    b = [c(new[m], t) for m in MODELS]
    print(f"  CSI@{t}: r = {pearson(a, b):+.3f}")

print()
print("=== how much of the score-aware CSI@2.0 is explained by the standard-loss CSI@0.5? ===")
xs = [c(old[m], "0.5") for m in MODELS]
ys = [c(new[m], "2.0") for m in MODELS]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
alpha = my - beta * mx
ss_res = sum((y - (alpha + beta * x)) ** 2 for x, y in zip(xs, ys))
ss_tot = sum((y - my) ** 2 for y in ys)
print(f"  sa@2.0 = {alpha:.4f} + {beta:.4f} * std@0.5    R^2 = {1 - ss_res/ss_tot:.3f}")
for m in MODELS:
    pred = alpha + beta * c(old[m], "0.5")
    print(f"    {m:22s} actual={c(new[m],'2.0'):.4f} predicted={pred:.4f} "
          f"resid={c(new[m],'2.0')-pred:+.4f}")

print()
print("=== score-aware CSI@2.0 vs score-aware CSI@0.5 (internal consistency) ===")
print(f"  Spearman = {rho(n5, n2):.3f}")

print()
print("=== the 'lift' distribution: is it uniform, or do a few models do all the lifting? ===")
lifts = sorted(((c(new[m], "2.0") - c(old[m], "2.0"), m, c(old[m], "2.0"), c(new[m], "2.0")) for m in MODELS),
               reverse=True)
for d, m, o, n in lifts:
    print(f"  {m:22s} {o:.4f} -> {n:.4f}   (+{d:.4f})")
print(f"  total lift = {sum(d for d, _, _, _ in lifts):.4f}")
print(f"  top-3 models contribute {sum(d for d, _, _, _ in lifts[:3])/sum(d for d,_,_,_ in lifts)*100:.1f}% of it")

print()
print("=== pysteps_adv, final position ===")
print(f"  pysteps_adv CSI@2.0 = {c(py,'2.0'):.4f}  CSI@4.0 = {c(py,'4.0'):.4f}  CSI@0.1 = {c(py,'0.1'):.4f}  CSI@0.5 = {c(py,'0.5'):.4f}")
best = max(MODELS, key=lambda m: c(new[m], "2.0"))
print(f"  best score-aware CSI@2.0: {best} = {c(new[best],'2.0'):.4f}  "
      f"({c(new[best],'2.0')/c(py,'2.0')*100:.0f}% of pysteps)")
print(f"  median score-aware CSI@2.0: {sorted(c(new[m],'2.0') for m in MODELS)[len(MODELS)//2]:.4f}")
