"""Build the HQMA scan-2 results table from the downloaded evaluation reports.

The table answers one question: does the field-wide tier loss raise the rain
rate without giving the false alarms away?  The soft-CSI row is the bar.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / ".workbuddy" / "scan2"

ORDER = [
    ("backbone_simvp_v2.eval.json", "simvp_v2 (backbone)"),
    ("m1_concat.eval.json", "M1 concat"),
    ("hqma_standard.eval.json", "hqma_standard (new head, standard loss)"),
    ("softcsi_hqma_balanced.eval.json", "hqma_balanced (soft-CSI)  <-- BAR"),
    ("scan_B1.eval.json", "scan_B1 (field-wide, alpha 0.1)"),
    ("scan_B2.eval.json", "scan_B2 (field-wide, alpha 0.3)"),
    ("scan_B3.eval.json", "scan_B3 (field-wide, alpha 0.3, per-pixel)"),
    ("scan_B4.eval.json", "scan_B4 (field-wide, alpha 0.5)"),
    ("scan_B5.eval.json", "scan_B5 (5 bins, alignment control)"),
]

THRESH = ["0.1", "0.5", "2.0", "4.0"]


def load(name: str) -> dict | None:
    path = BASE / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for fname, label in ORDER:
        d = load(fname)
        if d is None:
            print(f"MISSING {fname}")
            continue
        t = d["thresholds"]
        rows.append((label, d, t))

    # ---------------------------------------------------------------- header
    print("=" * 108)
    print("HQMA scan 2 -- field-wide tier loss + business thresholds (clean split, n=538)")
    print("=" * 108)
    print()
    print("### 全场表 (all thresholds)")
    hdr = (f"{'run':<34s} {'CSI@0.1':>8s} {'CSI@0.5':>8s} {'CSI@2.0':>8s} "
           f"{'CSI@4.0':>8s} {'FAR@0.5':>8s} {'POD@0.5':>8s} {'BIAS@0.5':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for label, d, t in rows:
        a = t["0.5"]
        print(f"{label:<34s} {t['0.1']['csi']:>8.3f} {a['csi']:>8.3f} "
              f"{t['2.0']['csi']:>8.3f} {t['4.0']['csi']:>8.3f} "
              f"{a['far']:>8.3f} {a['pod']:>8.3f} {a['bias']:>8.3f}")

    # ------------------------------------------------------- hit breakdown
    print()
    print("### 命中分解 @0.5 mm")
    hdr2 = (f"{'run':<34s} {'hits':>12s} {'FA':>12s} {'misses':>12s} "
            f"{'POD':>7s} {'FAR':>7s} {'BIAS':>7s}")
    print(hdr2)
    print("-" * len(hdr2))
    for label, d, t in rows:
        a = t["0.5"]
        print(f"{label:<34s} {int(a['hits']):>12,} {int(a['false_alarms']):>12,} "
              f"{int(a['misses']):>12,} {a['pod']:>7.3f} {a['far']:>7.3f} {a['bias']:>7.3f}")

    print()
    print("### 命中分解 @0.1 mm (weak rain, the primary criterion)")
    hdr3 = (f"{'run':<34s} {'hits':>12s} {'FA':>12s} {'misses':>12s} "
            f"{'POD':>7s} {'FAR':>7s} {'BIAS':>7s} {'CSI':>7s}")
    print(hdr3)
    print("-" * len(hdr3))
    for label, d, t in rows:
        a = t["0.1"]
        print(f"{label:<34s} {int(a['hits']):>12,} {int(a['false_alarms']):>12,} "
              f"{int(a['misses']):>12,} {a['pod']:>7.3f} {a['far']:>7.3f} "
              f"{a['bias']:>7.3f} {a['csi']:>7.3f}")

    print()
    print("### 命中分解 @2.0 mm (heavy rain)")
    hdr4 = (f"{'run':<34s} {'hits':>12s} {'FA':>12s} {'misses':>12s} "
            f"{'POD':>7s} {'FAR':>7s} {'BIAS':>7s} {'CSI':>7s}")
    print(hdr4)
    print("-" * len(hdr4))
    for label, d, t in rows:
        a = t["2.0"]
        print(f"{label:<34s} {int(a['hits']):>12,} {int(a['false_alarms']):>12,} "
              f"{int(a['misses']):>12,} {a['pod']:>7.3f} {a['far']:>7.3f} "
              f"{a['bias']:>7.3f} {a['csi']:>7.3f}")

    # ----------------------------------------------------------- lead times
    print()
    print("### 逐时效 CSI (lead_csi), thresholds present per lead")
    leads = sorted(rows[0][1]["lead_csi"].keys(), key=int)
    keys = sorted({k for _, d, _ in rows for k in d["lead_csi"]}, key=int)
    print(f"{'run':<34s} " + " ".join(f"{k:>7s}" for k in keys))
    for label, d, t in rows:
        cells = []
        for k in keys:
            v = d["lead_csi"].get(k)
            cells.append(f"{v:>7.3f}" if isinstance(v, (int, float)) else f"{'-':>7s}")
        print(f"{label:<34s} " + " ".join(cells))

    print()
    print("### 逐时效 CSI@0.5 明细 (lead_csi['0.5'] if nested)")
    for label, d, t in rows:
        lc = d["lead_csi"]
        sample = lc[keys[0]] if keys else None
        print(f"  {label:<34s} sample lead0 -> {sample}")

    # ------------------------------------------------------ dry reference
    print()
    print("### dry_reference_csi (预测场全干时的基线)")
    for label, d, t in rows:
        dr = d.get("dry_reference_csi", {})
        print(f"{label:<34s} " + " ".join(f"{k}={dr.get(k, float('nan')):.3f}"
                                          for k in THRESH))

    # --------------------------------------------------------- metadata
    print()
    print("### 元数据")
    for label, d, t in rows:
        ab = d.get("ablation", {})
        print(f"{label:<34s} n={d.get('n_samples')} mode={ab.get('mode')} "
              f"ckpt={ab.get('checkpoint')}")
        if ab.get("description"):
            print(f"{'':34s} desc={ab['description']}")


if __name__ == "__main__":
    main()
