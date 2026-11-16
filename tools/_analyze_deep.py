"""Deeper read: recall/precision at 2 mm, the physics baseline, and NWP value."""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).parent / "_evals"
MODELS = [
    "simvp_v2", "simvp_v2_nwp_concat", "simvp_v2_quantum", "simvp_v2_nwp_xattn",
    "simvp_v2_nwp_delta", "simvp_early_concat", "simvp_radar_only", "convlstm",
    "predrnn", "earthformer", "metnet3", "diffcast", "cascast_nwp", "mocast",
    "pw_foucast", "fusioncast", "mmf_rnn", "furecast",
]
PAIRED = [m for m in MODELS if m != "simvp_early_concat"]  # baseline is n=794


def read(d: str, m: str):
    p = BASE / d / f"{m}.eval.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None


def g(rep, tier, key):
    if rep is None:
        return float("nan")
    return rep.get("thresholds", {}).get(tier, {}).get(key, float("nan"))


new = {m: read("score_matrix", m) for m in MODELS}
old = {m: read("tgrs", m) for m in MODELS}
py = read("tgrs", "pysteps_adv")

print("=== 1. the physics baseline against the whole score-aware matrix ===")
print(f"pysteps_adv: CSI@2.0={g(py,'2.0','csi'):.4f} CSI@4.0={g(py,'4.0','csi'):.4f} "
      f"CSI@0.5={g(py,'0.5','csi'):.4f} CSI@0.1={g(py,'0.1','csi'):.4f}")
print()
hdr = (f"{'model':22s}{'CSI@2.0':>9s}{'POD@2.0':>9s}{'prec@2.0':>10s}"
       f"{'hits@2.0':>12s}{'miss@2.0':>12s}{'FA@2.0':>12s}{'CSI@4.0':>9s}")
print(hdr)
print("-" * len(hdr))
rows = []
for m in MODELS:
    n = new[m]
    t = n["thresholds"]
    h = t["2.0"]["hits"]; ms = t["2.0"]["misses"]; fa = t["2.0"]["false_alarms"]
    prec = h / (h + fa) if (h + fa) else float("nan")
    rows.append((m, t["2.0"]["csi"], t["2.0"]["pod"], prec, h, ms, fa, t["4.0"]["csi"]))
for r in sorted(rows, key=lambda x: -x[1]):
    print(f"{r[0]:22s}{r[1]:9.4f}{r[2]:9.4f}{r[3]:10.4f}{r[4]:12.0f}{r[5]:12.0f}{r[6]:12.0f}{r[7]:9.4f}")
tp = py["thresholds"]["2.0"]
prec_py = tp["hits"] / (tp["hits"] + tp["false_alarms"])
print(f"{'pysteps_adv':22s}{tp['csi']:9.4f}{tp['pod']:9.4f}{prec_py:10.4f}"
      f"{tp['hits']:12.0f}{tp['misses']:12.0f}{tp['false_alarms']:12.0f}"
      f"{py['thresholds']['4.0']['csi']:9.4f}")

print()
print("=== 2. how much of the truth at 2 mm does each model actually reach? ===")
truth2 = g(new["simvp_v2"], "2.0", "hits") + g(new["simvp_v2"], "2.0", "misses")
print(f"(truth pixels >= 2 mm, from one report: {truth2:.0f})")
best = max(rows, key=lambda r: r[2])
worst = min(rows, key=lambda r: r[2])
print(f"POD@2.0 range across score-aware matrix: {worst[2]:.4f} ({worst[0]}) .. {best[2]:.4f} ({best[0]})")
print(f"pysteps_adv POD@2.0 = {tp['pod']:.4f}  -> "
      f"models with higher POD than the physics baseline: "
      f"{sum(1 for r in rows if r[2] > tp['pod'])}/18")

print()
print("=== 3. precision at 2 mm: is the score-aware gain precision or recall? ===")
print(f"{'model':22s}{'std prec@2.0':>15s}{'sa prec@2.0':>14s}{'std POD@2.0':>13s}{'sa POD@2.0':>12s}")
for m in PAIRED:
    o, n = old[m], new[m]
    print(f"{m:22s}{g(o,'2.0','hits')/(g(o,'2.0','hits')+g(o,'2.0','false_alarms')):15.4f}"
          f"{g(n,'2.0','hits')/(g(n,'2.0','hits')+g(n,'2.0','false_alarms')):14.4f}"
          f"{g(o,'2.0','pod'):13.4f}{g(n,'2.0','pod'):12.4f}")
print(f"{'pysteps_adv':22s}{prec_py:15.4f}{'-':>14s}{tp['pod']:13.4f}{'-':>12s}")
sa_prec = [g(new[m],'2.0','hits')/(g(new[m],'2.0','hits')+g(new[m],'2.0','false_alarms')) for m in MODELS]
print(f"\nscore-aware prec@2.0: min={min(sa_prec):.4f} median={sorted(sa_prec)[9]:.4f} max={max(sa_prec):.4f}")
print(f"models with prec@2.0 above pysteps_adv ({prec_py:.4f}): "
      f"{sum(1 for p in sa_prec if p > prec_py)}/18")

print()
print("=== 4. what does NWP buy under the score-aware objective? ===")
pairs = [("simvp_radar_only", "simvp_v2_nwp_concat", "M1 naive concat"),
         ("simvp_radar_only", "simvp_v2_nwp_xattn", "cross-attention"),
         ("simvp_radar_only", "simvp_v2_nwp_delta", "delta"),
         ("simvp_v2", "simvp_v2_quantum", "quantum plugin"),
         ("simvp_v2", "simvp_early_concat", "early concat")]
for base, arm, name in pairs:
    b, a = new[base], new[arm]
    print(f"{name:22s} vs {base:20s}  CSI@2.0 {b['thresholds']['2.0']['csi']:.4f} -> "
          f"{a['thresholds']['2.0']['csi']:.4f}   CSI@0.1 {b['thresholds']['0.1']['csi']:.4f} -> "
          f"{a['thresholds']['0.1']['csi']:.4f}")

print()
print("=== 5. dry reference (a zero prediction must score 0) ===")
for m in ("simvp_v2", "mmf_rnn", "mocast"):
    print(f"{m:12s} new={new[m].get('dry_reference_csi')}  old={old[m].get('dry_reference_csi') if old[m] else None}")
