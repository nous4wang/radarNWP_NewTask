"""Local analysis of the score-aware matrix against the standard-loss matrix."""
from __future__ import annotations

import json
import math
from pathlib import Path

BASE = Path(__file__).parent / "_evals"
MODELS = [
    "simvp_v2", "simvp_v2_nwp_concat", "simvp_v2_quantum", "simvp_v2_nwp_xattn",
    "simvp_v2_nwp_delta", "simvp_early_concat", "simvp_radar_only", "convlstm",
    "predrnn", "earthformer", "metnet3", "diffcast", "cascast_nwp", "mocast",
    "pw_foucast", "fusioncast", "mmf_rnn", "furecast",
]
TIERS = ("0.1", "0.5", "2.0", "4.0")


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


def csi(rep, t):
    return g(rep, t, "csi")


def far(rep, t):
    return g(rep, t, "far")


def pod(rep, t):
    return g(rep, t, "pod")


def hits(rep, t):
    return g(rep, t, "hits")


new = {m: read("score_matrix", m) for m in MODELS}
old = {m: read("tgrs", m) or read("sweep", m) for m in MODELS}
old_src = {m: ("tgrs" if read("tgrs", m) else "sweep") for m in MODELS}

print("=== provenance ===")
for m in MODELS:
    n, o = new[m], old[m]
    print(f"{m:22s} new_n={n.get('n_samples') if n else None} "
          f"old_n={o.get('n_samples') if o else None} old_src={old_src[m]} "
          f"new_ablation={n.get('ablation') if n else None}")

print()
print("=== A. HEAVY RAIN: does the score-aware objective lift CSI@2.0 ? ===")
hdr = f"{'model':22s}{'std@2.0':>9s}{'sa@2.0':>9s}{'x':>7s}{'std@4.0':>9s}{'sa@4.0':>9s}{'x':>7s}{'stdFA.5':>9s}{'saFA.5':>9s}"
print(hdr)
print("-" * len(hdr))
ratios2, ratios4, sa2_list = [], [], []
for m in MODELS:
    o, n = old[m], new[m]
    o2, n2 = csi(o, "2.0"), csi(n, "2.0")
    o4, n4 = csi(o, "4.0"), csi(n, "4.0")
    r2 = n2 / o2 if o2 and o2 > 0 else float("nan")
    r4 = n4 / o4 if o4 and o4 > 0 else float("nan")
    ratios2.append(r2)
    ratios4.append(r4)
    sa2_list.append(n2)
    print(f"{m:22s}{o2:9.4f}{n2:9.4f}{r2:7.1f}{o4:9.4f}{n4:9.4f}{r4:7.1f}"
          f"{far(o,'0.5'):9.4f}{far(n,'0.5'):9.4f}")

finite = [r for r in ratios2 if not math.isnan(r)]
print(f"\nCSI@2.0 ratio: min={min(finite):.1f}x  median={sorted(finite)[len(finite)//2]:.1f}x  max={max(finite):.0f}x")
f4 = [r for r in ratios4 if not math.isnan(r)]
print(f"CSI@4.0 ratio: min={min(f4):.1f}x  median={sorted(f4)[len(f4)//2]:.1f}x  max={max(f4):.0f}x")
print(f"models with CSI@2.0 lifted (sa>std): {sum(1 for m in MODELS if csi(new[m],'2.0')>csi(old[m],'2.0'))}/18")
print(f"models with CSI@4.0 lifted: {sum(1 for m in MODELS if csi(new[m],'4.0')>csi(old[m],'4.0'))}/18")
print(f"score-aware CSI@2.0 range: {min(sa2_list):.4f} .. {max(sa2_list):.4f}  "
      f"(standard range {min(csi(old[m],'2.0') for m in MODELS):.4f} .. {max(csi(old[m],'2.0') for m in MODELS):.4f})")
print(f"score-aware CSI@2.0 >= 0.05 : {sum(1 for v in sa2_list if v>=0.05)}/18")
print(f"score-aware CSI@2.0 >= 0.10 : {sum(1 for v in sa2_list if v>=0.10)}/18")

print()
print("=== B. LIGHT RAIN: what the objective cost ===")
hdr2 = f"{'model':22s}{'std@0.1':>9s}{'sa@0.1':>9s}{'delta':>9s}{'stdPOD.1':>10s}{'saPOD.1':>10s}"
print(hdr2)
print("-" * len(hdr2))
d01 = []
for m in MODELS:
    o, n = old[m], new[m]
    a, b = csi(o, "0.1"), csi(n, "0.1")
    d01.append((b - a, m, a, b))
    print(f"{m:22s}{a:9.4f}{b:9.4f}{b-a:+9.4f}{pod(o,'0.1'):10.4f}{pod(n,'0.1'):10.4f}")
d01.sort()
print(f"\nCSI@0.1 below 0.1: {sum(1 for d_, m, a, b in d01 if b < 0.1)}/{len(d01)}")
print(f"CSI@0.1 dropped:    {sum(1 for d_, m, a, b in d01 if d_ < 0)}/{len(d01)}")
print(f"CSI@0.1 rose:       {sum(1 for d_, m, a, b in d01 if d_ > 0)}/{len(d01)}")
print(f"min sa@0.1 = {min(b for _,_,_,b in d01):.4f}, max sa@0.1 = {max(b for _,_,_,b in d01):.4f}")

print()
print("=== C. hits@2.0 and hits@4.0 (raw counts) ===")
for m in MODELS:
    print(f"{m:22s} h2 {hits(old[m],'2.0'):12.0f} -> {hits(new[m],'2.0'):12.0f}   "
          f"h4 {hits(old[m],'4.0'):10.0f} -> {hits(new[m],'4.0'):10.0f}")

print()
print("=== D. pysteps_adv reference ===")
py = read("tgrs", "pysteps_adv")
print(f"pysteps_adv: CSI@0.1={csi(py,'0.1'):.4f} CSI@0.5={csi(py,'0.5'):.4f} "
      f"CSI@2.0={csi(py,'2.0'):.4f} CSI@4.0={csi(py,'4.0'):.4f} FAR@0.5={far(py,'0.5'):.4f}")
for tier in ("0.1", "0.5", "2.0", "4.0"):
    above = [m for m in MODELS if csi(new[m], tier) > csi(py, tier)]
    print(f"  models above pysteps_adv at CSI@{tier}: {len(above)}/18"
          + (f"  -> {', '.join(above)}" if above else ""))

print()
print("=== E. ranking by CSI@2.0 ===")
old_rank = sorted([(m, csi(old[m], "2.0")) for m in MODELS], key=lambda t: -t[1])
new_rank = sorted([(m, csi(new[m], "2.0")) for m in MODELS], key=lambda t: -t[1])
print(f"{'rank':>5s}{'standard loss':>34s}{'score-aware':>34s}")
for i in range(18):
    print(f"{i+1:5d}{old_rank[i][0]+' '+format(old_rank[i][1],'.4f'):>34s}"
          f"{new_rank[i][0]+' '+format(new_rank[i][1],'.4f'):>34s}")
oo = [m for m, _ in old_rank]
no = [m for m, _ in new_rank]
n = 18
d2 = sum((oo.index(m) - no.index(m)) ** 2 for m in MODELS)
rho = 1 - 6 * d2 / (n * (n * n - 1))
print(f"\nSpearman rho = {rho:.3f}")
top_old = {m for m, _ in old_rank[:5]}
top_new = {m for m, _ in new_rank[:5]}
print(f"top-5 overlap: {len(top_old & top_new)}/5  -> {sorted(top_old & top_new)}")
spread_old = max(v for _, v in old_rank) / max(min(v for _, v in old_rank), 1e-9)
spread_new = max(v for _, v in new_rank) / max(min(v for _, v in new_rank), 1e-9)
print(f"spread (max/min) standard={spread_old:.1f}x  score-aware={spread_new:.1f}x")

print()
print("=== F. FAR@0.5 cost ===")
fa = [(far(new[m], '0.5'), far(old[m], '0.5'), m) for m in MODELS]
for v, o, m in sorted(fa, reverse=True):
    print(f"{m:22s} std={o:.4f} -> sa={v:.4f}  ({v-o:+.4f})")
print(f"mean std FAR@0.5 = {sum(o for _,o,_ in fa)/18:.4f}   mean sa = {sum(v for v,_,_ in fa)/18:.4f}")

print()
print("=== G. CSI@0.5 and bias ===")
for m in MODELS:
    print(f"{m:22s} CSI@0.5 {csi(old[m],'0.5'):.4f} -> {csi(new[m],'0.5'):.4f}   "
          f"BIAS@0.5 {g(old[m],'0.5','bias'):.4f} -> {g(new[m],'0.5','bias'):.4f}")

print()
print("=== H. the D1/D2 record (half-trained ckpts) vs the clean matrix ===")
for name, d in (("scan_D1", "hqma_scan4"), ("scan_D2", "hqma_scan4"),
                ("simvp_v2", "hqma"), ("simvp_v2_nwp_concat", "hqma"),
                ("hqma_balanced", "hqma"), ("hqma_standard", "hqma")):
    r = read(d, name)
    if r:
        print(f"{name:24s} n={r.get('n_samples')} CSI@2.0={csi(r,'2.0'):.4f} "
              f"CSI@4.0={csi(r,'4.0'):.4f} CSI@0.1={csi(r,'0.1'):.4f} FAR@0.5={far(r,'0.5'):.4f}")
print(f"{'score_matrix/simvp_v2':24s} n={new['simvp_v2'].get('n_samples')} "
      f"CSI@2.0={csi(new['simvp_v2'],'2.0'):.4f} CSI@4.0={csi(new['simvp_v2'],'4.0'):.4f} "
      f"CSI@0.1={csi(new['simvp_v2'],'0.1'):.4f} FAR@0.5={far(new['simvp_v2'],'0.5'):.4f}")
print(f"{'score_matrix/concat':24s} n={new['simvp_v2_nwp_concat'].get('n_samples')} "
      f"CSI@2.0={csi(new['simvp_v2_nwp_concat'],'2.0'):.4f} CSI@4.0={csi(new['simvp_v2_nwp_concat'],'4.0'):.4f} "
      f"CSI@0.1={csi(new['simvp_v2_nwp_concat'],'0.1'):.4f} FAR@0.5={far(new['simvp_v2_nwp_concat'],'0.5'):.4f}")

print()
print("=== I. dry reference (sanity) ===")
for m in ("simvp_v2", "mmf_rnn"):
    print(f"{m}: new dry_ref={new[m].get('dry_reference_csi')}  old dry_ref={old[m].get('dry_reference_csi')}")
