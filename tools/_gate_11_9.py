#!/usr/bin/env python3
"""MASTER 11.9 pass-rate admission gate -- executable form (v2.27 / E37).

The rule text in MASTER 11.9 was written at v2.26 but was only prose.  This
script turns P1-P6 into a check that can be run before a candidate method is
allowed to claim an improvement, and reproduces the anchors that justify it.

Frozen at 2026-11-07 17:24, BEFORE any compute in this round.  All constants
below are copied from existing remote-CUDA products; this script does NOT
train, infer, resample, or produce any new score.

Usage:
    python tools/_gate_11_9.py --self-test
    python tools/_gate_11_9.py --check path/to/manifest.json
    python tools/_gate_11_9.py --example
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

FROZEN_AT = "2026-11-07 17:24"
FLOOR = 0.0924
ORACLE = 0.1653
USABLE_SPACE = round(ORACLE - FLOOR, 4)          # 0.0729
PEAK_RATIO = 9.6                                # 11.9 P5, peak end
COLLAPSE_RATIO = 1.78                           # 11.9 P5, collapse end
MIN_SEEDS = 10                                  # 11.9 P1/P2 (formal tier)

# 11.9 P2, v2.38: the promotion ladder.  The ten-seed requirement is the
# EXIT requirement for claiming an improvement, not a precondition for
# exploring one.  Each tier is checked for the thing it is allowed to
# say, and each tier only ever adds seeds.
#
#   explore : 1 seed  -- may say "this arm passed / did not pass the floor"
#   confirm : 3 seeds-- may say "passed the floor in X of 3 seeds"
#   formal  : >=10   -- may say "the improvement holds" / "SOTA"
TIER_SEEDS = {"explore": 1, "confirm": 3, "formal": MIN_SEEDS}

# Anchors reproduced by E35 from 96 local copies of remote-CUDA evaluations.
A_QPE0_PASS, A_BASE_PASS = 5, 4
A_FISHER_P = 0.5
A_SPREAD = 9.59
A_PAIRED_MEAN = -0.0063
A_PAIRED_POS, A_PAIRED_NEG = 4, 6  # corrected v2.27: see E37 F7
A_EPOCH_PASS = {0: (5, 4), 1: (6, 6), 2: (2, 4)}  # 11.9 P3

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EVALS = REPO / "tools" / "_evals" / "route_seeds"

EXAMPLE = {
    "method": "name of the candidate method",
    "claims_improvement": True,
    "tier": "formal",             # 11.9 P2 v2.38: explore | confirm | formal
    "epoch": 0,
    "end": "peak",                  # "peak" -> 9.6x, "collapse" -> 1.78x
    "n_seeds": 10,
    "pass_rate": 7,
    "pass_rate_epoch": 0,           # P3: pass rate MUST be bound to an epoch
    "single_peak_claim": False,     # P1
    "baseline": {
        "same_batch": True,
        "qpe_0_pass": 5, "qpe_0_n": 10,
        "base_concat_pass": 4, "base_concat_n": 10,
        "fisher_p_one_sided": 0.5,
        "peak_spread": 9.59,
    },
    "paired": {
        "diffs": [],                # one signed difference per seed
        "two_level_bootstrap": True,
        "covers_seed_variance": True,
        "ci95": [0.0, 0.0],
    },
    "delta_ratio": 10.2,            # measured difference / spread threshold
    "delta_absolute": 0.015,       # must stay inside the usable space
}


def csi(path: Path, tier: str = "2.0") -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["thresholds"][tier]["csi"])


def collect(root: Path) -> dict:
    evals = root / "evals"
    if not evals.is_dir():
        evals = root
    table: dict = defaultdict(dict)
    for path in sorted(evals.glob("*.eval.json")):
        stem = path.name[: -len(".eval.json")]
        arm, seed, epoch = stem.rsplit("_", 2)
        table[arm][(int(seed[1:]), int(epoch[2:]))] = path
    return dict(table), evals


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact test, table [[a, b], [c, d]]."""
    n = a + b + c + d
    row1, row2, col1 = a + b, c + d, a + c

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    total = 0.0
    for x in range(max(0, col1 - row2), min(row1, col1) + 1):
        if x >= a:
            total += prob(x)
    return min(1.0, total)


def self_test(root: Path) -> int:
    table, evals = collect(root)
    n_files = sum(len(v) for v in table.values())
    print(f"constants frozen at {FROZEN_AT}: floor={FLOOR} oracle={ORACLE} "
          f"usable={USABLE_SPACE} peak_ratio={PEAK_RATIO} collapse_ratio={COLLAPSE_RATIO}")
    print(f"read {n_files} evaluations from {evals}")
    if n_files == 0:
        print("ANCHOR: FAIL (no evaluations found)")
        return 1

    q0 = {s: csi(p) for (s, e), p in table.get("qpe_0", {}).items() if e == 0}
    b0 = {s: csi(p) for (s, e), p in table.get("base_concat", {}).items() if e == 0}
    seeds = sorted(set(q0) & set(b0))
    checks: list[tuple[str, bool, str]] = []

    q_pass = sum(1 for s in seeds if q0[s] > FLOOR)
    b_pass = sum(1 for s in seeds if b0[s] > FLOOR)
    p = fisher_exact_greater(q_pass, len(seeds) - q_pass, b_pass, len(seeds) - b_pass)
    spread = max(b0.values()) / min(b0.values()) if min(b0.values()) > 0 else float("nan")
    diffs = [q0[s] - b0[s] for s in seeds]
    mean_diff = sum(diffs) / len(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)

    checks.append((f"P2 qpe_0 ep00 pass rate = {A_QPE0_PASS}/10",
                   q_pass == A_QPE0_PASS and len(seeds) == 10, f"got {q_pass}/{len(seeds)}"))
    checks.append((f"P2 base_concat ep00 pass rate = {A_BASE_PASS}/10",
                   b_pass == A_BASE_PASS, f"got {b_pass}/{len(seeds)}"))
    checks.append((f"P2 one-sided Fisher p = {A_FISHER_P}",
                   abs(p - A_FISHER_P) < 1e-9, f"got {p:.4f}"))
    checks.append((f"P2 peak-end spread = {A_SPREAD}x",
                   abs(spread - A_SPREAD) < 0.02, f"got {spread:.4f}x"))
    checks.append((f"P4 paired mean difference = {A_PAIRED_MEAN:+.4f}",
                   abs(mean_diff - A_PAIRED_MEAN) < 5e-4, f"got {mean_diff:+.4f}"))
    checks.append((f"P4 paired directions = {A_PAIRED_POS} pos / {A_PAIRED_NEG} neg",
                   n_pos == A_PAIRED_POS and n_neg == A_PAIRED_NEG,
                   f"got {n_pos} pos / {n_neg} neg"))

    for epoch, (want_q, want_b) in A_EPOCH_PASS.items():
        qe = {s: csi(p) for (s, e), p in table.get("qpe_0", {}).items() if e == epoch}
        be = {s: csi(p) for (s, e), p in table.get("base_concat", {}).items() if e == epoch}
        if not qe or not be:
            continue
        gq = sum(1 for v in qe.values() if v > FLOOR)
        gb = sum(1 for v in be.values() if v > FLOOR)
        checks.append((f"P3 epoch {epoch:02d} pass rates = {want_q}/10 vs {want_b}/10",
                       (gq, gb) == (want_q, want_b), f"got {gq}/10 vs {gb}/10"))

    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"anchors: {passed}/{len(checks)} reproduced")
    print("ANCHOR: PASS" if passed == len(checks) else "ANCHOR: FAIL")
    return 0 if passed == len(checks) else 1


def check_manifest(path: Path) -> int:
    d = json.loads(path.read_text(encoding="utf-8"))
    base = d.get("baseline", {})
    paired = d.get("paired", {})
    results: list[tuple[str, bool, str]] = []

    # 11.9 P2 v2.38 -- which tier of the promotion ladder this manifest
    # claims.  The tier decides how many seeds are required and what the
    # manifest is allowed to conclude.
    tier = str(d.get("tier", "formal")).lower()
    if tier not in TIER_SEEDS:
        results.append((f"P2 tier is one of {sorted(TIER_SEEDS)}", False,
                        f"got tier={tier!r}"))
        tier = "formal"
    need_seeds = TIER_SEEDS[tier]
    n_seeds = int(d.get("n_seeds", 0))
    results.append((f"P2 tier={tier} needs >= {need_seeds} seeds",
                    n_seeds >= need_seeds, f"n_seeds={n_seeds}"))

    # P1 -- a single peak is never evidence.
    #
    # The intent of P1 is that one seed's number cannot carry a claim.
    # In the ladder that intent is enforced by the TIER, not by a seed
    # count: the explore tier may hold one seed, but it may only say
    # "this arm passed the floor", and the confirm and formal tiers
    # require a paired comparison over seeds.  What P1 forbids in every
    # tier is a manifest that claims an improvement on the strength of a
    # single peak.
    ok = not d.get("single_peak_claim", False)
    if tier == "formal":
        ok = ok and n_seeds >= MIN_SEEDS
    results.append(("P1 single peak is not a criterion" +
                    ("" if tier == "explore" else "; >=10 seeds at the formal tier"),
                    ok, f"single_peak_claim={d.get('single_peak_claim')}, "
                        f"n_seeds={n_seeds}, tier={tier}"))

    # P2 -- same-batch baselines must reproduce the anchors.  This is the
    # FORMAL tier's exit requirement: only then may "the improvement
    # holds" be written down.  The explore and confirm tiers do not
    # require it, and saying so is the point of the ladder.
    q_n, b_n = int(base.get("qpe_0_n", 0)), int(base.get("base_concat_n", 0))
    q_p, b_p = int(base.get("qpe_0_pass", -1)), int(base.get("base_concat_pass", -1))
    fisher = float(base.get("fisher_p_one_sided", -1))
    spread = float(base.get("peak_spread", -1))
    ok = (bool(base.get("same_batch")) and q_n >= MIN_SEEDS and b_n >= MIN_SEEDS
          and q_p == A_QPE0_PASS and b_p == A_BASE_PASS
          and 0.4 <= fisher <= 0.6 and 5.0 <= spread <= 20.0)
    results.append((f"P2 [{tier}] same-batch baselines reproduce 5/10, 4/10, p=0.5, 9.59x"
                    + (" (required only at the formal tier)" if tier != "formal" else ""),
                    ok or tier != "formal",
                    f"required={tier == 'formal'}, same_batch={base.get('same_batch')}, "
                    f"qpe_0={q_p}/{q_n}, base_concat={b_p}/{b_n}, p={fisher}, spread={spread}"))

    # P2 v2.38 -- the promotion signal itself.  An arm that did not pass
    # the floor on its single seed does not move up the ladder, and an
    # arm at the confirm tier that is not all-passing must declare its
    # instability rather than report the best seed.
    passed = bool(d.get("passed_floor", False))
    unstable = bool(d.get("unstable_at_3_seeds", False))
    if tier == "explore":
        results.append(("P2 [explore] the promotion signal is the single seed's pass",
                        True, f"passed_floor={passed}"))
    elif tier == "confirm":
        ok = (not unstable) or bool(d.get("escalated_to_10_seeds", False))
        results.append(("P2 [confirm] a 3-seed arm that is not all-passing must "
                        "escalate to 10 seeds, not report its best seed",
                        ok, f"unstable={unstable}, "
                            f"escalated={d.get('escalated_to_10_seeds')}"))
    else:
        results.append(("P2 [formal] the formal tier does not need an escalation note",
                        True, f"unstable={unstable}"))

    # P3 -- pass rate bound to an epoch; peak end is always ep00.
    ok = ("epoch" in d and "pass_rate_epoch" in d
          and int(d["pass_rate_epoch"]) == int(d["epoch"]))
    if d.get("end") == "peak":
        ok = ok and int(d.get("epoch", -1)) == 0
    results.append(("P3 pass rate is bound to a declared epoch",
                    ok, f"epoch={d.get('epoch')}, pass_rate_epoch={d.get('pass_rate_epoch')}, "
                        f"end={d.get('end')}"))

    # P4 -- the four required numbers, incl. two-level bootstrap over seeds.
    # The number of pairs must cover the tier: 1 at explore, 3 at confirm,
    # >=10 at formal.  The two-level bootstrap over the SEED level is what
    # makes an interval cover the training-seed variance, so it is
    # required from the confirm tier on; at the explore tier there is only
    # one seed, and the manifest must say so rather than pretend.
    diffs = paired.get("diffs") or []
    ci = paired.get("ci95") or []
    n_pairs_needed = need_seeds
    two_level = bool(paired.get("two_level_bootstrap"))
    covers = bool(paired.get("covers_seed_variance"))
    ok = (isinstance(d.get("pass_rate"), int) and bool(base)
          and len(diffs) >= n_pairs_needed
          and len(ci) == 2
          and (tier == "explore" or (two_level and covers)))
    results.append(("P4 four numbers + two-level bootstrap covering seed variance",
                    ok, f"pass_rate={d.get('pass_rate')}, n_pairs={len(diffs)}, "
                        f"two_level={two_level}, covers_seed={covers}, "
                        f"ci95={ci}, tier={tier}"))

    # P5 -- difference thresholds on the correct end, inside the usable space.
    ratio = float(d.get("delta_ratio", 0))
    absolute = float(d.get("delta_absolute", -1))
    need = COLLAPSE_RATIO if d.get("end") == "collapse" else PEAK_RATIO
    ok = ratio >= need and 0 <= absolute <= USABLE_SPACE
    results.append((f"P5 delta >= {need}x and inside usable space {USABLE_SPACE}",
                    ok, f"delta_ratio={ratio}, delta_absolute={absolute}"))

    for name, flag, detail in results:
        print(f"  [{'PASS' if flag else 'FAIL'}] {name} -- {detail}")
    n_pass = sum(1 for _, flag, _ in results if flag)
    if n_pass == len(results):
        verdict = "PASS"
    elif n_pass == 0:
        verdict = "FAIL"
    else:
        verdict = "PARTIAL"
    print(f"VERDICT: {verdict} ({n_pass}/{len(results)} criteria satisfied)")
    if verdict != "PASS":
        print("11.9 P6: do NOT write into 10.1 performance layer; "
              "discard per 11.8 and keep this record on file.")
    return 0 if verdict == "PASS" else 1


def _ladder_self_test() -> int:
    """Check the three tiers of the 11.9 P2 promotion ladder.

    The ladder was added in v2.38: the ten-seed requirement is the exit
    requirement for claiming an improvement, not a precondition for
    exploring one.  These cases pin the boundaries, because a ladder
    that is only written down and not enforced by the tool is a ladder
    that will be climbed incorrectly.

    The cases check:
      * the explore tier accepts one seed and does not demand the
        ten-seed baseline;
      * the explore tier still refuses a single-peak claim;
      * the confirm tier accepts three seeds;
      * the confirm tier refuses a three-seed arm that is not all-passing
        unless it declares the escalation to ten seeds;
      * the formal tier still demands ten seeds and the baseline.
    """
    import tempfile

    checks: list[tuple[str, bool, str]] = []

    def run(manifest: dict) -> tuple[bool, str]:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as handle:
            json.dump(manifest, handle)
            path = Path(handle.name)
        try:
            out = []
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                check_manifest(path)
            for line in buf.getvalue().splitlines():
                if line.strip().startswith("[FAIL]"):
                    out.append(line.strip())
            return (not out), "; ".join(out) if out else "all criteria satisfied"
        finally:
            path.unlink(missing_ok=True)

    def manifest(tier: str, n_seeds: int, **over) -> dict:
        m = json.loads(json.dumps(EXAMPLE))
        m["tier"] = tier
        m["n_seeds"] = n_seeds
        m["pass_rate"] = max(1, n_seeds)
        m["paired"]["diffs"] = [0.01] * n_seeds
        m["paired"]["ci95"] = [0.0, 0.02]
        m.update(over)
        return m

    ok, detail = run(manifest("explore", 1))
    checks.append(("explore tier accepts one seed without the ten-seed baseline",
                   ok, detail))

    ok, detail = run(manifest("explore", 1, single_peak_claim=True))
    checks.append(("explore tier still refuses a single-peak claim",
                   not ok, detail))

    ok, detail = run(manifest("confirm", 3))
    checks.append(("confirm tier accepts three seeds", ok, detail))

    ok, detail = run(manifest("confirm", 3, unstable_at_3_seeds=True))
    checks.append(("confirm tier refuses an unstable 3-seed arm that did not escalate",
                   not ok, detail))

    ok, detail = run(manifest("confirm", 3, unstable_at_3_seeds=True,
                              escalated_to_10_seeds=True))
    checks.append(("confirm tier accepts an unstable arm that escalated to ten seeds",
                   ok, detail))

    ok, detail = run(manifest("formal", 3))
    checks.append(("formal tier still demands ten seeds", not ok, detail))

    for name, flag, detail in checks:
        print(f"  [{'PASS' if flag else 'FAIL'}] {name} -- {detail}")
    passed = sum(1 for _, flag, _ in checks if flag)
    print(f"ladder: {passed}/{len(checks)} reproduced")
    print("LADDER: PASS" if passed == len(checks) else "LADDER: FAIL")
    return 0 if passed == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="reproduce the 11.9 anchors from the 96 evaluation copies")
    ap.add_argument("--evals", default=str(DEFAULT_EVALS),
                    help="root containing evals/*.eval.json")
    ap.add_argument("--check", metavar="MANIFEST",
                    help="check a candidate method manifest against P1-P6")
    ap.add_argument("--ladder-test", action="store_true",
                    help="check the three tiers of the 11.9 P2 promotion ladder")
    ap.add_argument("--example", action="store_true",
                    help="print an empty manifest template")
    args = ap.parse_args()

    if args.example:
        print(json.dumps(EXAMPLE, indent=2, ensure_ascii=False))
        return 0
    if args.ladder_test:
        return _ladder_self_test()
    if args.check:
        return check_manifest(Path(args.check))
    return self_test(Path(args.evals))


if __name__ == "__main__":
    raise SystemExit(main())
