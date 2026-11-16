#!/usr/bin/env python3
"""E45 round 91 -- is the S3 verdict fragile to the bootstrap RNG seed?

The E45 guard rail (S3) is a two-level bootstrap: the training seed is
resampled with replacement, and inside a drawn seed the evaluation
windows are resampled with replacement.  The published verdict rests on
ONE random seed (20261109, n_boot=10000).  Every earlier round checked
the STORED endpoints -- that they contain zero, that the stored flags
agree with the stored numbers -- but none of them re-ran the resampling
with a different seed.  A negative verdict ("six arms discarded") has to
be robust, so this round asks the question directly:

    for each arm and each comparison (raw, equal-bias), does the
    "the interval contains zero" answer change when the bootstrap's
    random seed changes?

Two halves:

* the RAW half needs only the window tables, which are on this machine,
  so it is done here.  It uses the guard rail's OWN implementation
  (``tools/_e41_s3_equalbias.py``), imported rather than reimplemented,
  so a difference cannot come from a reimplementation.  The registered
  seed must reproduce the stored endpoints first.
* the EQUAL-BIAS half needs the dumped fields, which live on the remote;
  it is done by ``tools/_e45_s3_seedrobust.py`` in a Slurm job and
  merged in here from ``tools/_evals/e45/s3_seedrobust.json`` when that
  file is present.

Only the arms whose equal-bias POINT estimate is positive can flip:
the pre-registered rule keeps an arm only when the equalised interval
excludes zero AND the point is positive.  The other arms have a
negative equalised point and are discarded whatever the interval does.

Read-only.  Writes tools/_evals/e45/round91_status_audit.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVDIR = ROOT / "tools/_evals" / "e45" / "evals_raw" / "evals"
S3PATH = ROOT / "tools/_evals" / "e45" / "s3_equalbias.json"
SUMPATH = ROOT / "tools/_evals" / "e45" / "scores_summary.json"
REMOTE_RES = ROOT / "tools/_evals" / "e45" / "s3_seedrobust.json"
OUT = ROOT / "tools/_evals" / "e45" / "round91_status_audit.json"

THR_KEY = "2.0"
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ALL_ARMS = ["B0"] + ARMS          # the seven arms of the headline table
SEEDS = [0, 1, 2]
FLOOR = 0.0924
ORACLE = 0.1653

# The registered seed plus nineteen others.  If the answer is the same
# for all twenty, the verdict does not depend on the draw.
ALT_SEEDS = [
    20261109, 1, 2, 3, 5, 7, 11, 13, 101, 1009,
    4242, 2026, 20261110, 20261111, 20261112, 20270101,
    123456789, 314159, 555555, 999983,
]

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    checks.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    return bool(ok)


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load_guard_rail():
    """Import the guard rail's own implementation (not a reimplementation)."""
    path = ROOT / "tools" / "_e41_s3_equalbias.py"
    spec = importlib.util.spec_from_file_location("s3mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    return mod, digest


def find_constants(obj, needle: str, found: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if needle in str(k).lower() and isinstance(v, (int, float)):
                found.append((str(k), float(v)))
            find_constants(v, needle, found)
    elif isinstance(obj, list):
        for v in obj:
            find_constants(v, needle, found)


def main() -> int:
    report: dict = {"round": 91, "axis": "s3_bootstrap_rng_robustness"}

    s3mod, module_md5 = load_guard_rail()
    print(f"guard rail implementation: tools/_e41_s3_equalbias.py md5={module_md5}")
    report["guardrail_module_md5_local"] = module_md5
    N_BOOT = s3mod.N_BOOT
    REG_SEED = s3mod.RNG_SEED

    # ---------------- inventory ----------------
    eval_files = sorted(EVDIR.glob("*.eval.json"))
    win_files = sorted(EVDIR.glob("*.windows.json"))
    check("eval tree has 21 eval reports", len(eval_files) == 21, f"{len(eval_files)}")
    check("eval tree has 21 window tables", len(win_files) == 21, f"{len(win_files)}")

    s3 = load_json(S3PATH)
    check("guardrail parameters untouched",
          s3["threshold_mm"] == 2.0 and s3["control"] == "B0"
          and s3["n_boot"] == N_BOOT and s3["rng_seed"] == REG_SEED
          and s3["epoch"] == "00",
          f"thr={s3['threshold_mm']} ctrl={s3['control']} n_boot={s3['n_boot']} "
          f"rng={s3['rng_seed']} epoch={s3['epoch']}")
    check("guardrail covers six arms", sorted(s3["arms"].keys()) == ARMS,
          ",".join(sorted(s3["arms"].keys())))

    # ---------------- headline table, recomputed from the 21 reports -----
    csi: dict[str, dict[int, float]] = {a: {} for a in ALL_ARMS}
    for f in eval_files:
        tag = f.name.split("_s")[0]
        seed = int(f.name.split("_s")[1].split("_")[0])
        if tag not in csi:
            check(f"unexpected arm tag in the eval tree: {tag}", False, f.name)
            continue
        csi[tag][seed] = load_json(f)["thresholds"][THR_KEY]["csi"]
    check("headline table has 21 cells", sum(len(v) for v in csi.values()) == 21)

    found: list = []
    find_constants(load_json(SUMPATH), "floor", found)
    if found:
        vals = {v for _, v in found}
        check("floor constant in the summary is 0.0924",
              any(abs(v - FLOOR) < 1e-9 for v in vals), str(found[:4]))
    found_o: list = []
    find_constants(load_json(SUMPATH), "oracle", found_o)
    if found_o:
        vals = {v for _, v in found_o}
        check("oracle constant in the summary is 0.1653",
              any(abs(v - ORACLE) < 1e-9 for v in vals), str(found_o[:4]))

    headline = {}
    for a in ALL_ARMS:
        vals = [csi[a][s] for s in SEEDS]
        headline[a] = {
            "median_csi_at_2.0": float(np.median(vals)),
            "pass_rate_over_floor": int(sum(v >= FLOOR for v in vals)),
            "n_seeds": len(vals),
            "seeds_over_oracle": int(sum(v > ORACLE for v in vals)),
        }
    report["headline"] = headline
    print("\n-- headline (median + pass rate, never a bare point score) --")
    for a in ALL_ARMS:
        h = headline[a]
        print(f"  {a:4s} median CSI@2.0 = {h['median_csi_at_2.0']:.6f}   "
              f"over floor {h['pass_rate_over_floor']}/3")

    # ---------------- the raw comparison: re-run the resampling --------
    print("\n-- raw comparison: reproduction + RNG-seed sweep --")
    ctrl = {s: s3mod.window_counts(EVDIR / f"B0_s{s}_ep00.windows.json")
            for s in SEEDS}
    ctrl_list = [ctrl[s] for s in SEEDS]

    raw_report: dict = {}
    for a in ARMS:
        raw_counts = [s3mod.window_counts(EVDIR / f"{a}_s{s}_ep00.windows.json")
                      for s in SEEDS]
        rep = s3mod.two_level_bootstrap(raw_counts, ctrl_list,
                                        np.random.default_rng(REG_SEED))
        stored = s3["arms"][a]["raw"]
        dev = max(abs(rep["point"] - stored["point"]),
                  abs(rep["ci_lo"] - stored["ci_lo"]),
                  abs(rep["ci_hi"] - stored["ci_hi"]))
        check(f"{a}: raw resampling reproduces the stored endpoints",
              dev < 1e-12, f"max|dev|={dev:.3e}")

        sweep = [s3mod.two_level_bootstrap(raw_counts, ctrl_list,
                                           np.random.default_rng(sd))
                 for sd in ALT_SEEDS]
        n_cross = sum(r["crosses_zero"] for r in sweep)
        raw_report[a] = {
            "stored": stored,
            "reproduction_max_abs_dev": dev,
            "n_seeds_tried": len(sweep),
            "n_seeds_with_zero_inside_interval": n_cross,
            "min_ci_lo": float(min(r["ci_lo"] for r in sweep)),
            "max_ci_hi": float(max(r["ci_hi"] for r in sweep)),
            "point_span": [float(min(r["point"] for r in sweep)),
                           float(max(r["point"] for r in sweep))],
        }
        print(f"  {a:4s} zero inside the interval in {n_cross}/{len(sweep)} seeds  "
              f"CI union [{raw_report[a]['min_ci_lo']:+.4f}, {raw_report[a]['max_ci_hi']:+.4f}]")
    report["raw_sweep"] = raw_report
    check("every arm's raw interval contains zero under every bootstrap seed",
          all(v["n_seeds_with_zero_inside_interval"] == v["n_seeds_tried"]
              for v in raw_report.values()))

    # ---------------- the equal-bias half (remote job, merged if present) --
    eq_report: dict = {}
    if REMOTE_RES.exists():
        eq_remote = load_json(REMOTE_RES)
        print("\n-- equal-bias sweep (remote job) --")
        check("equal-bias sweep used the registered parameters",
              eq_remote.get("threshold_mm") == 2.0 and eq_remote.get("n_boot") == N_BOOT
              and eq_remote.get("rng_seed") == REG_SEED,
              f"thr={eq_remote.get('threshold_mm')} n_boot={eq_remote.get('n_boot')} "
              f"rng={eq_remote.get('rng_seed')}")
        check("equal-bias sweep ran the same guard rail implementation",
              eq_remote.get("module_md5") == module_md5,
              f"remote={eq_remote.get('module_md5')} local={module_md5}")
        for a in ARMS:
            entry = eq_remote["arms"].get(a)
            if entry is None:
                continue
            stored = s3["arms"][a]["equal_bias"]
            rep = entry["reproduced"]
            dev = max(abs(rep["point"] - stored["point"]),
                      abs(rep["ci_lo"] - stored["ci_lo"]),
                      abs(rep["ci_hi"] - stored["ci_hi"]))
            sweep = entry["sweep"]
            n_cross = sum(r["crosses_zero"] for r in sweep)
            eq_report[a] = {
                "stored": stored,
                "reproduction_max_abs_dev": dev,
                "max_abs_bias_gap": entry.get("max_abs_bias_gap"),
                "n_seeds_tried": len(sweep),
                "n_seeds_with_zero_inside_interval": n_cross,
                "point_positive": stored["point"] > 0,
                "min_ci_lo": float(min(r["ci_lo"] for r in sweep)),
                "max_ci_hi": float(max(r["ci_hi"] for r in sweep)),
            }
            print(f"  {a:4s} zero inside the interval in {n_cross}/{len(sweep)} seeds  "
                  f"point={stored['point']:+.6f} (>0: {stored['point'] > 0})  "
                  f"CI union [{eq_report[a]['min_ci_lo']:+.4f}, {eq_report[a]['max_ci_hi']:+.4f}]")
        report["equal_bias_sweep"] = eq_report
        check("equal-bias resampling reproduces the stored endpoints",
              all(v["reproduction_max_abs_dev"] < 1e-12 for v in eq_report.values()),
              ", ".join(f"{a}:{v['reproduction_max_abs_dev']:.1e}"
                        for a, v in eq_report.items()))
        check("every arm's equalised interval contains zero under every bootstrap seed",
              all(v["n_seeds_with_zero_inside_interval"] == v["n_seeds_tried"]
                  for v in eq_report.values()))
        # The only arms that could flip are the ones with a positive
        # equalised point: the rule keeps an arm only when the interval
        # excludes zero AND the point is positive.
        flippable = [a for a, v in eq_report.items() if v["point_positive"]]
        report["flippable_arms"] = flippable
        print(f"\n  arms with a positive equalised point (the only ones that "
              f"could flip): {flippable}")
        check("no arm with a positive equalised point loses zero from the interval",
              all(eq_report[a]["n_seeds_with_zero_inside_interval"]
                  == eq_report[a]["n_seeds_tried"] for a in flippable))
        check("the remote re-computation kept the equalisation on target",
              all(v["max_abs_bias_gap"] < 1e-6 for v in eq_report.values()),
              ", ".join(f"{a}:{v['max_abs_bias_gap']:.1e}"
                        for a, v in eq_report.items()))
    else:
        print("\n-- equal-bias sweep: remote result not fetched yet --")
        report["equal_bias_sweep"] = None

    # ---------------- the equalisation itself (from the stored file) -----
    worst_bias_gap = 0.0
    for a in ARMS:
        for sd in SEEDS:
            e = s3["arms"][a]["seeds"][str(sd)]
            worst_bias_gap = max(worst_bias_gap,
                                 abs(e["achieved_bias"] - e["control_bias"]))
    report["max_abs_bias_gap_after_equalisation"] = worst_bias_gap
    check("the equalisation actually reached the control bias",
          worst_bias_gap < 1e-6, f"max |achieved - control| = {worst_bias_gap:.3e}")

    # ---------------- how close is each arm to flipping? ---------------
    # The pre-registered rule keeps an arm only when the equalised
    # interval EXCLUDES zero AND the point is positive.  So an arm with
    # a negative equalised point is immune: no interval can rescue it.
    # For the arms with a positive point, the distance from the nearest
    # interval endpoint to zero is what the bootstrap seed has to erase.
    # The measured seed-to-seed jitter of the endpoints (from the raw
    # half, same resampling scheme) says how big that jitter is.
    jitter = max(max(abs(v["min_ci_lo"] - v["stored"]["ci_lo"]),
                      abs(v["max_ci_hi"] - v["stored"]["ci_hi"]))
                 for v in raw_report.values())
    flip = {}
    for a in ARMS:
        e = s3["arms"][a]["equal_bias"]
        margin = min(abs(e["ci_lo"]), abs(e["ci_hi"]))
        flip[a] = {
            "equal_bias_point": e["point"],
            "point_positive": e["point"] > 0,
            "distance_of_nearest_endpoint_to_zero": margin,
            "margin_over_measured_jitter": (margin / jitter) if jitter > 0 else None,
            "immune_because_point_negative": e["point"] <= 0,
        }
    report["endpoint_jitter_measured_on_raw_half"] = jitter
    report["flip_risk"] = flip
    print(f"\n-- how close each arm is to flipping (measured endpoint jitter "
          f"{jitter:.6f}) --")
    for a, v in flip.items():
        if v["immune_because_point_negative"]:
            print(f"  {a:4s} IMMUNE  equalised point {v['equal_bias_point']:+.6f} <= 0 "
                  f"=> no interval can make it survive")
        else:
            print(f"  {a:4s} AT RISK  equalised point {v['equal_bias_point']:+.6f}, "
                  f"nearest endpoint {v['distance_of_nearest_endpoint_to_zero']:.6f} "
                  f"from zero = {v['margin_over_measured_jitter']:.1f}x the jitter")
    n_at_risk = [a for a, v in flip.items() if not v["immune_because_point_negative"]]
    report["arms_that_could_flip"] = n_at_risk
    # Two arms have a positive equalised point (A1 +0.000179, A2
    # +0.004086), so both are, in principle, able to flip.  The rest are
    # immune: the rule keeps an arm only when the equalised interval
    # EXCLUDES zero AND the point is positive, so a negative point
    # cannot be rescued by any interval.
    check("the arms that could flip are exactly those with a positive equalised point",
          sorted(n_at_risk) == sorted(a for a, v in flip.items() if v["point_positive"]),
          ",".join(n_at_risk))
    # A2's nearest endpoint is only ~1x the measured jitter away from
    # zero, so its equalised interval is the one that could move across
    # zero under a different bootstrap seed.  That is what the remote
    # seed-robustness job (343) settles; it is the only arm whose
    # equalised verdict is not already safe by a wide margin.
    tight = [a for a, v in flip.items()
             if not v["immune_because_point_negative"]
             and v["margin_over_measured_jitter"] is not None
             and v["margin_over_measured_jitter"] <= 2.0]
    report["arms_within_2x_jitter_of_flipping"] = tight
    print(f"  arms within 2x the jitter of flipping: {tight or 'none'}")
    check("no arm's equalised interval is closer to zero than the jitter is",
          all(v["immune_because_point_negative"]
              or v["distance_of_nearest_endpoint_to_zero"] > jitter
              for v in flip.values()),
          " ".join(f"{a}:{v['distance_of_nearest_endpoint_to_zero']:.6f}"
                   for a, v in flip.items()))

    # ---------------- summary ----------------
    n_pass = sum(1 for _, ok, _ in checks if ok)
    report["checks"] = [{"name": n, "ok": o, "detail": d} for n, o, d in checks]
    report["n_checks"] = len(checks)
    report["n_pass"] = n_pass
    report["verdict"] = "PASS" if n_pass == len(checks) else "FAIL"
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPOS {n_pass}/{len(checks)}  VERDICT: {report['verdict']}")
    print(f"written: {OUT}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
