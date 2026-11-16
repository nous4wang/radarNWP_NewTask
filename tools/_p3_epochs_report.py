"""Read the route P3 peak re-measurement.

Two things are read here and they are read differently.

The first is the lag curve at the peak epoch, which is the quantity the
sweep is for: how much of the benefit survives when the observation is
late.  The first run of P3 measured it at the last epoch, where the
arm-to-arm differences have already decayed, and it came out
non-monotone -- 12 minutes late scored below no observation at all,
which no mechanism explains.  This reads the same curve at the peak.

The second is the reproducibility check.  The first run used
``scripts/train.py``; this one used ``train_keep_epochs.py``, a wrapper
that patches ``torch.save``.  If the wrapper changed anything, the
epoch-19 scores would move, and the peak-epoch scores would not be
comparable with the original.  They have to come back identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TIERS = ["0.1", "0.5", "2.0", "4.0"]
ARMS = [("assim_lag0", "no delay"),
        ("assim_lag2", "12 min late"),
        ("assim_lag4", "24 min late"),
        ("assim_lag11", "no observation")]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csi(path: Path) -> dict:
    t = load(path)["thresholds"]
    return {th: t[th]["csi"] for th in TIERS}


def main() -> None:
    new_dir = Path(sys.argv[1])
    old_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    print("=" * 96)
    print("1. the lag curve at the peak epoch")
    print("=" * 96)
    header = (f"{'arm':16s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}"
              f"{'CSI@4.0':>9s}{'FAR@0.5':>9s}{'BIAS@0.5':>9s}")
    print(header)
    print("-" * len(header))
    curve = []
    for name, label in ARMS:
        path = new_dir / f"{name}_ep00.eval.json"
        if not path.exists():
            print(f"{label:16s}(not finished)")
            continue
        rep = load(path)
        t = rep["thresholds"]
        g = lambda th, key: t.get(th, {}).get(key, float("nan"))
        print(f"{label:16s}"
              + "".join(f"{g(th, 'csi'):9.4f}" for th in TIERS)
              + f"{g('0.5', 'far'):9.4f}{g('0.5', 'bias'):9.4f}")
        curve.append((int(name.split("lag")[-1]), label,
                      t["2.0"]["csi"], t["0.5"]["csi"]))

    print()
    print("  the lag curve (CSI@2.0 against the delay):")
    for lag, label, c2, c05 in sorted(curve):
        print(f"    lag {lag:>2} ({lag * 6:>2} min late)  "
              f"CSI@2.0 = {c2:.4f}  CSI@0.5 = {c05:.4f}")
    if len(curve) >= 2:
        full = min(curve, key=lambda c: c[0])
        none = max(curve, key=lambda c: c[0])
        print(f"    the observation is worth {full[2] / none[2]:.2f}x at "
              f"CSI@2.0 ({full[2]:.4f} with it, {none[2]:.4f} without)")
        mid = [c for c in curve if c[0] not in (full[0], none[0])]
        for lag, label, c2, c05 in sorted(mid):
            print(f"    a delay of {lag * 6} min leaves {c2:.4f}, "
                  f"{'below' if c2 < none[2] else 'above'} the "
                  f"no-observation arm")

    print()
    print("=" * 96)
    print("2. the reproducibility check (epoch 19 against the original)")
    print("=" * 96)
    if old_dir is None:
        print("  no original run given")
        return
    all_same = True
    for name, label in ARMS:
        new = new_dir / f"{name}_ep19.eval.json"
        old = old_dir / f"{name}.eval.json"
        if not new.exists() or not old.exists():
            print(f"  {label:16s}(missing: new={new.exists()} "
                  f"old={old.exists()})")
            all_same = False
            continue
        a, b = csi(new), csi(old)
        same = all(abs(a[th] - b[th]) < 1e-12 for th in TIERS)
        all_same = all_same and same
        print(f"  {label:16s} new CSI@2.0 = {a['2.0']:.10f}   "
              f"original = {b['2.0']:.10f}   "
              f"{'identical' if same else 'DIFFERENT'}")
    print()
    if all_same:
        print("  PASS: the epoch-19 scores are identical to the original")
        print("        run, so the peak-epoch scores are comparable with it")
        print("        and the wrapper changed nothing.")
    else:
        print("  FAIL: the epoch-19 scores moved. The peak-epoch numbers")
        print("        are NOT comparable with the original run.")


if __name__ == "__main__":
    main()
