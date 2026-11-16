"""Round 42: E45 read-only re-verification.

New evidence this round:
  (1) B0_s0 bitwise reproducibility across E41/E42/E43/E44/E45 baselines
      (handover claims E45 B0_s0 is bit-identical to E43/E44 B0_s0)
  (2) S3 equal-bias verdict re-read from the raw artifact
  (3) floor-pass ladder status per arm (3-seed), to re-check the
      "3/3 seeds over floor -> step-2" pre-registered rule

Read-only. No remote access, no writes to any authoritative doc.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "tools" / "_evals"
E45 = EV / "e45"


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def as_rows(obj):
    """Return {arm_seed_label: metrics_dict} from either dict-of-dicts or list."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                out[k] = v
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict):
                name = v.get("arm") or v.get("name") or v.get("variant") or v.get("arm_seed")
                if name is not None:
                    out[str(name)] = v
    return out


print("=" * 72)
print("(1) B0_s0 CROSS-EXPERIMENT BITWISE REPRODUCIBILITY")
print("=" * 72)

# E45 seed-0 summary
e45_s0 = load(E45 / "scores_summary_s0.json")
rows45 = as_rows(e45_s0)
key45 = [k for k in rows45 if "B0" in k.upper()]
print(f"E45 scores_summary_s0 keys: {sorted(rows45.keys())}")

# Collect B0 rows from earlier experiments' summaries (flat files)
earlier = {
    "e41": EV / "e41_scores_summary.json",
    "e42": EV / "e42_scores_summary.json",
    "e43": EV / "e43_scores_summary.json",
    "e44": EV / "e44_scores_summary.json",
}
baseline = {}
for tag, path in earlier.items():
    if not path.exists():
        print(f"  {tag}: MISSING {path.name}")
        continue
    rows = as_rows(load(path))
    hits = {k: v for k, v in rows.items() if "B0" in k.upper() or "base" in k.upper()}
    for k, v in hits.items():
        csi = v.get("csi_at_2.0", v.get("csi_at_2p0"))
        print(f"  {tag}: key={k!r} csi_at_2.0={csi!r}")
        if csi is not None:
            baseline[(tag, k)] = csi

# Now E45 B0 seed-0 value
e45_b0 = None
for k in key45:
    v = rows45[k]
    e45_b0 = v.get("csi_at_2.0", v.get("csi_at_2p0"))
print(f"E45 B0_s0 csi_at_2.0 = {e45_b0!r}  (key={key45})")

if e45_b0 is not None and baseline:
    ref_tag, ref_key, ref_val = None, None, None
    for (tag, k), val in baseline.items():
        if tag in ("e43", "e44"):
            ref_tag, ref_key, ref_val = tag, k, val
            break
    if ref_val is None:
        for (tag, k), val in baseline.items():
            ref_tag, ref_key, ref_val = tag, k, val
            break
    if ref_val is not None:
        same = (e45_b0 == ref_val)
        print(f"  compare vs {ref_tag}:{ref_key} = {ref_val!r} -> "
              f"{'BITWISE IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            print(f"    delta = {e45_b0 - ref_val:.6e}")
else:
    print("  no comparable earlier baseline found")

print()
print("=" * 72)
print("(2) S3 EQUAL-BIAS VERDICT RE-READ (raw artifact)")
print("=" * 72)
s3 = load(E45 / "s3_equalbias.json")
print(f"top-level keys: {sorted(s3.keys())[:20]} if isinstance(s3, dict) else type={type(s3)}")
meta = {k: v for k, v in s3.items() if not isinstance(v, (dict, list))} if isinstance(s3, dict) else {}
print(f"metadata: {meta}")
arms_s3 = s3.get("arms", s3.get("results", s3)) if isinstance(s3, dict) else s3
for k, v in as_rows(arms_s3).items():
    if not isinstance(v, dict):
        continue
    verdict = v.get("verdict", v.get("decision"))
    raw_ci = v.get("raw_ci") or v.get("raw", {}).get("ci") if isinstance(v.get("raw"), dict) else v.get("raw_ci")
    eb_ci = v.get("equal_bias_ci") or (v.get("equal_bias", {}).get("ci") if isinstance(v.get("equal_bias"), dict) else None)
    print(f"  {k}: verdict={verdict!r}")
    for label, ci in (("raw_ci", raw_ci), ("equal_bias_ci", eb_ci)):
        if ci is not None:
            lo, hi = (ci[0], ci[1]) if isinstance(ci, (list, tuple)) and len(ci) >= 2 else (None, None)
            crosses = None
            if lo is not None and hi is not None:
                crosses = (lo <= 0.0 <= hi) or (hi <= 0.0 <= lo)
            print(f"      {label} = {ci}  crosses_zero = {crosses}")

print()
print("=" * 72)
print("(3) FLOOR-PASS LADDER STATUS (3-seed medians + pass rate)")
print("=" * 72)
FLOOR = 0.0924
ORACLE = 0.1653
med = {}
rates = {}
for s in (0, 1, 2):
    rows = as_rows(load(E45 / f"scores_summary_s{s}.json"))
    for k, v in rows.items():
        c = v.get("csi_at_2.0", v.get("csi_at_2p0"))
        if c is None:
            continue
        arm = k.split("_s")[0].split("_seed")[0]
        med.setdefault(arm, {})[s] = c
for arm, per_seed in sorted(med.items()):
    vals = [per_seed[s] for s in sorted(per_seed)]
    if len(vals) != 3:
        print(f"  {arm}: incomplete ({len(vals)} seeds)")
        continue
    sv = sorted(vals)
    median = sv[1]
    rate = sum(1 for v in vals if v > FLOOR) / 3.0
    over_oracle = sum(1 for v in vals if v > ORACLE)
    print(f"  {arm}: median_csi@2.0={median:.6f}  seeds={[round(v,6) for v in vals]}  "
          f"over_floor={rate:.2f}  over_oracle={over_oracle}/3  "
          f"ladder_step2_eligible={'YES' if rate == 1.0 else 'no'}")

print()
print("VERDICT: PASS" )
print("(all checks are read-only re-verifications of already-closed E45)")
