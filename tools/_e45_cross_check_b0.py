"""E45 round-20: cross-experiment reproducibility check for the control arm B0_s0.

Handover §3 states: B0 of E45 is "与 E43/E44 的 B0_s0 逐位相同（可复现）".
This script re-verifies that claim directly from the local artifacts of
E43 / E44 / E45.

Structure-first: the key layout of the older artifacts is NOT assumed.  We
walk every nested container, collect every dict that carries a `csi_at_2.0`
field, and select the ones whose surrounding path mentions both the arm
token `B0` and the seed token `s0`.  Nothing is hard-coded except the
selection rule itself.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "tools", "_evals")

FAILS: list[str] = []
LINES: list[str] = []


def walk(node, path=()):
    """Yield (path, dict) for every dict in the JSON tree."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, path + ("[%d]" % i,))


def path_str(path) -> str:
    return "/".join(path)


def pick_b0_s0(path, d):
    """A dict that looks like a B0 seed-0 score record."""
    if "csi_at_2.0" not in d:
        return False
    p = path_str(path).lower()
    arm_hit = any(tok in p for tok in ("b0",)) or str(d.get("arm", "")).lower() == "b0"
    seed_hit = any(tok in p for tok in ("s0", "seed0", "_0", "seed_0")) or str(d.get("seed", "")) in ("0",)
    return arm_hit and seed_hit


def collect(fname):
    p = os.path.join(EV, fname)
    if not os.path.exists(p):
        LINES.append("  %-26s MISSING" % fname)
        return None
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    hits = []
    for path, d in walk(obj):
        if pick_b0_s0(path, d):
            hits.append((path_str(path), d))
    return hits


def main() -> int:
    print("=" * 78)
    print("E45 round-20 cross-experiment B0_s0 reproducibility check")
    print("=" * 78)

    results = {}
    for fname in ("e43_scores_summary.json", "e44_scores_summary.json",
                  "e43_struct_metrics.json", "e44_struct_metrics.json",
                  "e44_diag_probe.json"):
        hits = collect(fname)
        if hits is None:
            continue
        print("\n[%s] -> %d candidate B0_s0 record(s)" % (fname, len(hits)))
        for pth, d in hits[:6]:
            keys = {k: d[k] for k in list(d)[:8]}
            print("    path=%s" % pth)
            print("      %s" % json.dumps(keys, ensure_ascii=False)[:300])
        results[fname] = hits

    # E45 side: authoritative value from scores_all.json
    with open(os.path.join(EV, "e45", "scores_all.json"), "r", encoding="utf-8") as f:
        e45_all = json.load(f)
    e45_b0_s0 = e45_all.get("B0_s0_ep00")
    if e45_b0_s0 is None:
        FAILS.append("E45 scores_all.json lacks B0_s0_ep00")
        print("\nVERDICT: FAIL")
        return 1
    e45_csi = e45_b0_s0.get("csi_at_2.0")
    print("\nE45 B0_s0_ep00 csi_at_2.0 = %r  (bias=%r)" %
          (e45_csi, e45_b0_s0.get("bias_at_2.0")))

    # compare with every candidate whose csi we found
    compared = 0
    for fname, hits in results.items():
        for pth, d in hits:
            v = d.get("csi_at_2.0")
            if not isinstance(v, (int, float)):
                continue
            compared += 1
            same = abs(v - e45_csi) < 1e-12
            print("  compare %-26s %-40s %r  -> %s" %
                  (fname, pth[:40], v, "IDENTICAL" if same else "DIFFERENT"))
            if not same:
                FAILS.append("B0_s0 differs in %s at %s: %r vs %r" % (fname, pth, v, e45_csi))

    print("\ncomparisons made = %d" % compared)
    if compared == 0:
        FAILS.append("no B0_s0 candidate found in E43/E44 artifacts (extraction failed, "
                     "not 'different')")

    if FAILS:
        print("FAILURES:")
        for f in FAILS:
            print("  - " + f)
        print("VERDICT: FAIL")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
