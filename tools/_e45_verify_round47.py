"""E45 round-47 verification (READ-ONLY, local only).

New axis: storm-event (per-case) aggregation.
Rounds 40-46 judged E45 on: S3 per-seed decomposition (40), maxratio gate (41),
cross-experiment reproducibility (42), derived-boolean branch decomposition (43),
multi-threshold profile (44), S1 struct axis (45), window-level axis + null model (46).

Never judged: the *storm-event level*. 538 windows come from 8 storm cases
(round 46 LOCO).  The window axis resamples windows (pseudo-replication: seed is
the real unit).  The *case* is the natural replication unit for "does the arm
improve skill for a storm event", and a case-level bootstrap is the first CI in
E45 that is not window-level pseudo-replication.

Also audits the arithmetic of the published headline table (median + pass rate),
re-derived from the per-seed artifacts.

Rules honoured: read-only; report medians + pass rates; paired caliber and
difference caliber side by side; single-point scores never used as a verdict;
post-hoc diagnostic only -> does NOT change any verdict, no new criterion.
"""

import json
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "tools" / "_evals" / "e45" / "runs" / "e45" / "evals"
LOCAL = ROOT / "tools" / "_evals" / "e45"
OUT = LOCAL / "round47_case_level.json"

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]
CTRL = "B0"
FLOOR = 0.0924
THR = "2.0"
N_BOOT = 10000
RNG_SEED = 20261113

checks = []


def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def med(xs):
    """Median with the even-length branch written correctly (round-46 lesson)."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def as_num(v):
    """Published artifacts store some fields as '2/3' strings -> normalise."""
    if isinstance(v, str):
        s = v.strip()
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b)
        return float(s)
    return float(v)


def case_of(key):
    # key format: CP_01_08121100_00001:8121006  ->  case / patch index : valid time
    m = re.match(r"^(CP_\d+_\d+)_(\d+):", key)
    if m:
        return m.group(1)
    return key.split("_000")[0]


def load_cell(arm, seed):
    p = EVALS / f"{arm}_s{seed}_ep00.windows.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return d


def per_case_counts(cell, thr):
    """storm case -> summed contingency counts at one threshold."""
    keys = cell["window_keys"]
    wins = cell["windows"]
    acc = {}
    for k, w in zip(keys, wins):
        c = acc.setdefault(case_of(k), {"hits": 0.0, "misses": 0.0, "false_alarms": 0.0,
                                        "correct_negatives": 0.0})
        t = w[thr]
        c["hits"] += t["hits"]
        c["misses"] += t["misses"]
        c["false_alarms"] += t["false_alarms"]
        c["correct_negatives"] += t["correct_negatives"]
    return acc


def csi_of(c):
    den = c["hits"] + c["misses"] + c["false_alarms"]
    return c["hits"] / den if den > 0 else float("nan")


def bias_of(c):
    den = c["hits"] + c["misses"]
    return (c["hits"] + c["false_alarms"]) / den if den > 0 else float("nan")


def pod_of(c):
    den = c["hits"] + c["misses"]
    return c["hits"] / den if den > 0 else float("nan")


def boot_ci(deltas, rng, n_boot=N_BOOT):
    """Paired bootstrap over the resampling unit = storm case (n=8)."""
    n = len(deltas)
    stats = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        stats.append(med(sample))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[min(n_boot - 1, int(0.975 * n_boot))]
    return lo, hi


def main():
    rng = random.Random(RNG_SEED)
    result = {"threshold_mm": float(THR), "floor": FLOOR, "n_boot": N_BOOT,
              "rng_seed": RNG_SEED, "axis": "storm-case (per-event) aggregation"}

    # ---------- load all 21 cells ----------
    cells = {}
    ref_keys = None
    for arm in ARMS:
        for seed in SEEDS:
            c = load_cell(arm, seed)
            cells[(arm, seed)] = c
            if ref_keys is None:
                ref_keys = c["window_keys"]
            check(f"window_keys identical {arm}_s{seed}", c["window_keys"] == ref_keys)
    check("n_cells_loaded", len(cells) == 21, f"{len(cells)}")

    # ---------- per-case tables ----------
    case_tables = {}
    for (arm, seed), c in cells.items():
        acc = per_case_counts(c, THR)
        case_tables[(arm, seed)] = {
            "cases": sorted(acc.keys()),
            "csi": {k: csi_of(v) for k, v in acc.items()},
            "bias": {k: bias_of(v) for k, v in acc.items()},
            "pod": {k: pod_of(v) for k, v in acc.items()},
            "counts": acc,
        }
    CASES = case_tables[(CTRL, 0)]["cases"]
    check("n_storm_cases", len(CASES) == 8, str(len(CASES)))
    for (arm, seed), t in case_tables.items():
        check(f"same_case_set {arm}_s{seed}", t["cases"] == CASES)

    # ---------- global recompute vs published per-seed scores ----------
    pub_seed = {}
    for seed in SEEDS:
        pub_seed[seed] = json.loads((LOCAL / f"scores_summary_s{seed}.json").read_text(encoding="utf-8"))
    for arm in ARMS:
        for seed in SEEDS:
            acc = per_case_counts(cells[(arm, seed)], THR)
            g = {"hits": 0.0, "misses": 0.0, "false_alarms": 0.0, "correct_negatives": 0.0}
            for v in acc.values():
                for k in g:
                    g[k] += v[k]
            recomputed = csi_of(g)
            published = pub_seed[seed][f"{arm}_s{seed}_ep00"]["csi_at_2.0"]
            check(f"global_csi_matches_published {arm}_s{seed}",
                  abs(recomputed - published) <= 1e-12,
                  f"recomputed={recomputed!r} published={published!r}")

    # ---------- published headline table arithmetic audit ----------
    summary = json.loads((LOCAL / "scores_summary.json").read_text(encoding="utf-8"))
    audit = {}
    for arm in ARMS:
        csis = [pub_seed[s][f"{arm}_s{s}_ep00"]["csi_at_2.0"] for s in SEEDS]
        biass = [pub_seed[s][f"{arm}_s{s}_ep00"]["bias_at_2.0"] for s in SEEDS]
        rec = {
            "median_csi_at_2.0": med(csis),
            "median_bias_at_2.0": med(biass),
            "pass_rate_over_floor_strict": sum(1 for x in csis if x > FLOOR) / len(csis),
            "pass_rate_over_floor_weak": sum(1 for x in csis if x >= FLOOR) / len(csis),
            "seed_spread_max_over_min": (max(csis) / min(csis)) if min(csis) > 0 else float("nan"),
            "n_seeds": len(csis),
        }
        audit[arm] = rec
        for field in ("median_csi_at_2.0", "median_bias_at_2.0", "pass_rate_over_floor",
                      "seed_spread_csi_at_2.0", "n_seeds"):
            pub = summary[arm][field]
            if field == "pass_rate_over_floor":
                got = rec["pass_rate_over_floor_strict"]
            elif field == "seed_spread_csi_at_2.0":
                got = rec["seed_spread_max_over_min"]
            else:
                got = rec[field]
            check(f"summary_audit {arm}.{field}", abs(as_num(pub) - as_num(got)) <= 1e-9,
                  f"published={pub!r} rederived={got!r}")
    result["published_table_audit"] = audit

    # ---------- case-level paired comparison vs control ----------
    def paired_stats(arm, seed, ctrl_seed):
        t = case_tables[(arm, seed)]
        c = case_tables[(CTRL, ctrl_seed)]
        d = [t["csi"][k] - c["csi"][k] for k in CASES]
        lo, hi = boot_ci(d, rng)
        return {
            "per_case_delta": {k: t["csi"][k] - c["csi"][k] for k in CASES},
            "paired_median_delta": med(d),
            "difference_caliber": med([t["csi"][k] for k in CASES]) - med([c["csi"][k] for k in CASES]),
            "n_cases_positive": sum(1 for x in d if x > 0),
            "n_cases": len(d),
            "ci_lo": lo, "ci_hi": hi, "crosses_zero": (lo <= 0.0 <= hi),
            "arm_case_median_csi": med([t["csi"][k] for k in CASES]),
            "ctrl_case_median_csi": med([c["csi"][k] for k in CASES]),
            "arm_case_median_bias": med([t["bias"][k] for k in CASES]),
            "ctrl_case_median_bias": med([c["bias"][k] for k in CASES]),
        }

    null_pairs = [(0, 1), (0, 2), (1, 2)]
    null = {}
    for a, b in null_pairs:
        null[f"B0_s{a}_vs_B0_s{b}"] = paired_stats(CTRL, a, b)
    null_floor = max(abs(v["paired_median_delta"]) for v in null.values())
    result["null_model"] = null
    result["null_floor_paired_median_abs"] = null_floor

    arms_res = {}
    for arm in ARMS:
        if arm == CTRL:
            continue
        per_seed = {}
        for seed in SEEDS:
            per_seed[f"s{seed}"] = paired_stats(arm, seed, seed)
        pmeds = [per_seed[f"s{s}"]["paired_median_delta"] for s in SEEDS]
        dmeds = [per_seed[f"s{s}"]["difference_caliber"] for s in SEEDS]
        arms_res[arm] = {
            "per_seed": per_seed,
            "paired_median_median_over_seeds": med(pmeds),
            "difference_caliber_median_over_seeds": med(dmeds),
            "pass_rate_seeds_paired_median_positive": sum(1 for x in pmeds if x > 0) / len(pmeds),
            "pass_rate_seeds_ci_excludes_zero": sum(1 for s in SEEDS
                                                    if not per_seed[f"s{s}"]["crosses_zero"]) / len(SEEDS),
            "pass_rate_cases_positive_median_over_seeds": med(
                [per_seed[f"s{s}"]["n_cases_positive"] / len(CASES) for s in SEEDS]),
            "within_null_floor": abs(med(pmeds)) <= null_floor,
        }
    result["arms"] = arms_res

    # ---------- a couple of hard invariants ----------
    check("null_floor_positive", null_floor > 0, f"{null_floor!r}")
    check("all_arms_computed", len(arms_res) == 6, str(len(arms_res)))
    for arm, r in arms_res.items():
        check(f"{arm}_has_3_seeds", len(r["per_seed"]) == 3)

    result["checks"] = checks
    n_pass = sum(1 for c in checks if c["ok"])
    result["n_pass"] = n_pass
    result["n_total"] = len(checks)
    result["verdict"] = "PASS" if n_pass == len(checks) else "FAIL"

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- report ----------
    print(f"=== E45 round 47: storm-case axis | VERDICT: {result['verdict']} "
          f"({n_pass}/{len(checks)}) ===")
    print(f"storm cases = {len(CASES)}; threshold = {THR} mm; floor = {FLOOR}; "
          f"bootstrap = {N_BOOT} (resampling unit = storm case)")
    print(f"null model (control arm, different seed): paired-median |dCSI| per pair = "
          f"{ {k: round(v['paired_median_delta'], 6) for k, v in null.items()} }")
    print(f"null floor (max |paired median|) = {null_floor:.6f}")
    print()
    print("--- published headline table audit (re-derived from per-seed artifacts) ---")
    for arm in ARMS:
        a = audit[arm]
        print(f"  {arm:>4}: median CSI@2.0={a['median_csi_at_2.0']:.6f} "
              f"median BIAS@2.0={a['median_bias_at_2.0']:.6f} "
              f"pass_floor={a['pass_rate_over_floor_strict']:.2f} "
              f"spread(max/min)={a['seed_spread_max_over_min']:.4f}")
    print()
    print("--- case-level paired comparison (arm vs B0, same seed), thr=2.0 mm ---")
    hdr = f"  {'arm':>4} {'paired med':>11} {'diff calib':>11} {'cases>0':>9} {'CI(seed0)':>22} {'seeds pos':>10} {'vs null floor':>14}"
    print(hdr)
    for arm, r in arms_res.items():
        s0 = r["per_seed"]["s0"]
        ci = f"[{s0['ci_lo']:+.4f},{s0['ci_hi']:+.4f}]"
        print(f"  {arm:>4} {r['paired_median_median_over_seeds']:>+11.6f} "
              f"{r['difference_caliber_median_over_seeds']:>+11.6f} "
              f"{r['pass_rate_cases_positive_median_over_seeds']:>9.2f} {ci:>22} "
              f"{r['pass_rate_seeds_paired_median_positive']:>10.2f} "
              f"{str(r['within_null_floor']):>14}")
    print()
    print("--- per-seed detail (paired median dCSI, case-level) ---")
    for arm, r in arms_res.items():
        row = "  ".join(f"s{s}={r['per_seed'][f's{s}']['paired_median_delta']:+.6f}"
                        f"(ci{'' if r['per_seed'][f's{s}']['crosses_zero'] else '*'})"
                        for s in SEEDS)
        print(f"  {arm:>4}: {row}")
    print()
    print("--- null model detail ---")
    for k, v in null.items():
        print(f"  {k}: paired_med={v['paired_median_delta']:+.6f} diff_calib={v['difference_caliber']:+.6f} "
              f"cases>0={v['n_cases_positive']}/{v['n_cases']} CI=[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] "
              f"crosses_zero={v['crosses_zero']}")
    print()
    print(f"output -> {OUT}")
    print("NOTE: post-hoc diagnostic axis; S3 guardrail stays thr=2.0 mm; seed remains the "
          "cross-seed unit -> does NOT change any verdict.")


if __name__ == "__main__":
    main()
