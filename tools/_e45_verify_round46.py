#!/usr/bin/env python
"""E45 round 46: per-window level audit of the raw evaluation tables.

Read-only.  Never touches the remote, never writes into docs/.

What is new here (rounds 9-45 never looked at this layer):
  * the raw ``*.windows.json`` tables carry, for every one of the 21 cells
    (7 arms x 3 seeds), 538 window-level contingency tables
    (hits / misses / false_alarms / correct_negatives) at 4 thresholds.
    Every previous round judged only the *aggregate* metrics, the
    mechanism probe and the S1 structural metrics.
  * Therefore this round can (a) re-derive the published aggregate metrics
    from the raw window tables -- an end-to-end audit of the scoring chain --
    (b) check that all 21 cells are window-aligned (paired comparison is
    legitimate), and (c) run a *paired, window-level* bootstrap of the
    arm-minus-control delta at the pre-registered threshold 2.0 mm, which
    is an axis independent of the seed-level S3 equal-bias bootstrap.

Discipline (unchanged from rounds 40-45):
  * medians + pass rates only; no single score is ever a verdict;
  * paired and difference conventions are reported side by side;
  * post-hoc diagnostic -- it does NOT create a new criterion, so it can
    NOT change the E45 verdict and does NOT bump the MASTER version.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "_evals" / "e45" / "_raw" / "runs" / "e45" / "evals"
TARBALL = ROOT / "_evals" / "e45" / "e45_raw_audit.tar.gz"
OUT = ROOT / "_evals" / "e45" / "round46_window_level.json"

# The raw per-window tables live in a tarball downloaded from the remote
# (md5 9d8c1468d47c75df17600414fac28d2b).  Unpack it on demand so the audit
# is reproducible from the tarball alone.
if not RAW.exists():
    if not TARBALL.exists():
        raise SystemExit(
            f"raw tables missing: neither {RAW} nor {TARBALL} exists.\n"
            "Re-fetch with:\n"
            "  MSYS_NO_PATHCONV=1 python tools/remote.py -- "
            "'cd /home/wf/radarNWP && tar czf /tmp/e45_raw_audit.tar.gz "
            "runs/e45/evals/*.eval.json runs/e45/evals/*.windows.json'\n"
            "  MSYS_NO_PATHCONV=1 python tools/remote.py --download "
            "/tmp/e45_raw_audit.tar.gz --out tools/_evals/e45/e45_raw_audit.tar.gz")
    import tarfile
    # the tarball root is "runs/e45/evals/...", so it unpacks into "_raw/"
    # (RAW = _raw/runs/e45/evals  ->  _raw is RAW.parents[2])
    dest = RAW.parents[2]
    with tarfile.open(TARBALL) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # Python < 3.12 has no filter kwarg
            tf.extractall(dest)
    if not RAW.exists():
        raise SystemExit(f"unpacked {TARBALL} into {dest} but {RAW} still missing")
    print(f"unpacked {TARBALL} -> {RAW}")

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = ["s0", "s1", "s2"]
THR = "2.0"
FLOOR = 0.0924
ORACLE = 0.1653
N_BOOT = 10000
RNG_SEED = 20261113

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))
    print(f"[{'POS' if ok else 'NEG'}] {name} {detail}")


def load_cell(arm: str, seed: str) -> tuple[dict, dict]:
    tag = f"{arm}_{seed}_ep00"
    win = json.loads((RAW / f"{tag}.windows.json").read_text(encoding="utf-8"))
    ev = json.loads((RAW / f"{tag}.eval.json").read_text(encoding="utf-8"))
    return win, ev


def agg(windows: list[dict], thr: str) -> dict:
    """Pool the window-level contingency tables into the published metrics."""
    h = m = fa = cn = 0.0
    for w in windows:
        c = w[thr]
        h += c["hits"]
        m += c["misses"]
        fa += c["false_alarms"]
        cn += c["correct_negatives"]
    ref = h + m
    obs = h + fa
    out = {
        "hits": h, "misses": m, "false_alarms": fa, "correct_negatives": cn,
    }
    out["csi"] = h / (h + m + fa) if (h + m + fa) > 0 else float("nan")
    out["bias"] = obs / ref if ref > 0 else float("nan")
    out["pod"] = h / ref if ref > 0 else float("nan")
    out["far"] = fa / obs if obs > 0 else float("nan")
    base = obs / (ref + obs) if (ref + obs) > 0 else float("nan")
    out["ets"] = (h - cn * base) / (h + m + fa - cn * base) if (h + m + fa - cn * base) != 0 else float("nan")
    hss_den = (h + fa) * (h + m) + (m + cn) * (fa + cn)
    out["hss"] = 2 * (h * cn - fa * m) / hss_den if hss_den != 0 else float("nan")
    out["fss"] = 2 * h / (2 * h + m + fa) if (2 * h + m + fa) > 0 else float("nan")
    return out


def median(xs: list[float]) -> float:
    """Median with the standard even-length rule (lower/upper middle)."""
    s = sorted(xs)
    k = len(s)
    if k == 0:
        raise ValueError("median of empty sequence")
    if k % 2:
        return s[k // 2]
    return 0.5 * (s[k // 2 - 1] + s[k // 2])


def main() -> int:
    cells = {}
    for arm in ARMS:
        for seed in SEEDS:
            cells[f"{arm}_{seed}"] = load_cell(arm, seed)
    print(f"loaded {len(cells)} cells from {RAW}")

    # ---- (a) pairing integrity: all 21 cells must be window-aligned -------
    ref_keys = cells["B0_s0"][0]["window_keys"]
    same_order = all(cells[c][0]["window_keys"] == ref_keys for c in cells)
    same_set = all(set(cells[c][0]["window_keys"]) == set(ref_keys) for c in cells)
    check("window keys identical across all 21 cells (order)", same_order)
    check("window keys identical across all 21 cells (set)", same_set)
    check("n_windows == 538 everywhere",
          all(cells[c][0]["n_windows"] == len(ref_keys) == 538 for c in cells))

    # ---- (b) re-derive the published aggregates from the raw tables -------
    worst_csi = 0.0
    worst_bias = 0.0
    worst_pod = 0.0
    worst_far = 0.0
    n_metric_cmp = 0
    for c, (win, ev) in cells.items():
        for thr in ("0.1", "0.5", "2.0", "4.0"):
            got = agg(win["windows"], thr)
            pub = ev["thresholds"][thr]
            for key, acc in (("csi", "csi"), ("bias", "bias"), ("pod", "pod"), ("far", "far")):
                d = abs(got[key] - pub[key])
                n_metric_cmp += 1
                if key == "csi":
                    worst_csi = max(worst_csi, d)
                elif key == "bias":
                    worst_bias = max(worst_bias, d)
                elif key == "pod":
                    worst_pod = max(worst_pod, d)
                else:
                    worst_far = max(worst_far, d)
    check("re-derived aggregate CSI matches published (21 cells x 4 thr)",
          worst_csi < 1e-12, f"max|diff|={worst_csi:.3e}")
    check("re-derived aggregate BIAS matches published", worst_bias < 1e-12,
          f"max|diff|={worst_bias:.3e}")
    check("re-derived aggregate POD matches published", worst_pod < 1e-12,
          f"max|diff|={worst_pod:.3e}")
    check("re-derived aggregate FAR matches published", worst_far < 1e-12,
          f"max|diff|={worst_far:.3e}")
    print(f"  ({n_metric_cmp} metric comparisons, all thresholds)")

    # ---- (c) window-level paired analysis at the pre-registered 2.0 mm ----
    # per-window CSI for every cell
    win_csi: dict[str, list[float]] = {}
    for c, (win, _) in cells.items():
        vals = []
        for w in win["windows"]:
            cc = w[THR]
            den = cc["hits"] + cc["misses"] + cc["false_alarms"]
            vals.append(cc["hits"] / den if den > 0 else float("nan"))
        win_csi[c] = vals

    # case (storm event) decomposition: prefix of the window key
    case_of = [k.split(":")[0] for k in ref_keys]
    cases = sorted(set(case_of))
    idx_by_case: dict[str, list[int]] = defaultdict(list)
    for i, cs in enumerate(case_of):
        idx_by_case[cs].append(i)

    rng = random.Random(RNG_SEED)

    def boot_median_ci(deltas: list[float], n: int = N_BOOT) -> tuple[float, float]:
        n_obs = len(deltas)
        meds = []
        for _ in range(n):
            s = [deltas[rng.randrange(n_obs)] for _ in range(n_obs)]
            s.sort()
            mid = n_obs // 2
            meds.append(s[mid] if n_obs % 2 else 0.5 * (s[mid - 1] + s[mid]))
        meds.sort()
        lo = meds[int(0.025 * n)]
        hi = meds[int(0.975 * n) - 1]
        return lo, hi

    results: dict = {"n_windows": len(ref_keys), "n_cases": len(cases),
                     "cases": cases, "thr": THR, "floor": FLOOR,
                     "oracle": ORACLE, "n_boot": N_BOOT, "rng_seed": RNG_SEED,
                     "arms": {}}

    # control window-level aggregate per seed
    for arm in ARMS:
        if arm == "B0":
            continue
        per_seed = {}
        for seed in SEEDS:
            a = win_csi[f"{arm}_{seed}"]
            b = win_csi[f"B0_{seed}"]
            deltas = [x - y for x, y in zip(a, b)]
            finite = [d for d in deltas if not math.isnan(d)]
            sign_pos = sum(1 for d in finite if d > 0)
            sign_neg = sum(1 for d in finite if d < 0)
            sign_zero = sum(1 for d in finite if d == 0.0)
            med_delta = median(finite)
            lo, hi = boot_median_ci(finite)
            # case level: pooled contingency tables per case
            case_rows = []
            n_case_pos = 0
            for cs in cases:
                idxs = idx_by_case[cs]
                agg_a = agg([cells[f"{arm}_{seed}"][0]["windows"][i] for i in idxs], THR)
                agg_b = agg([cells[f"B0_{seed}"][0]["windows"][i] for i in idxs], THR)
                d = agg_a["csi"] - agg_b["csi"]
                if d > 0:
                    n_case_pos += 1
                case_rows.append({"case": cs, "n_windows": len(idxs),
                                  "csi_arm": agg_a["csi"], "csi_b0": agg_b["csi"],
                                  "delta": d, "bias_arm": agg_a["bias"],
                                  "bias_b0": agg_b["bias"]})
            # per-seed arm aggregate vs control aggregate (difference convention)
            agg_arm = agg(cells[f"{arm}_{seed}"][0]["windows"], THR)
            agg_b0 = agg(cells[f"B0_{seed}"][0]["windows"], THR)
            per_seed[seed] = {
                "median_window_delta_csi": med_delta,
                "window_sign_pos": sign_pos,
                "window_sign_neg": sign_neg,
                "window_sign_zero": sign_zero,
                "window_sign_pos_frac": sign_pos / len(finite),
                "window_bootstrap_ci_lo": lo,
                "window_bootstrap_ci_hi": hi,
                "window_ci_crosses_zero": (lo <= 0.0 <= hi),
                "aggregate_csi_arm": agg_arm["csi"],
                "aggregate_csi_b0": agg_b0["csi"],
                "aggregate_delta_csi": agg_arm["csi"] - agg_b0["csi"],
                "aggregate_bias_arm": agg_arm["bias"],
                "aggregate_bias_b0": agg_b0["bias"],
                "n_cases_positive": n_case_pos,
                "n_cases": len(cases),
                "case_rows": case_rows,
            }
        # cross-seed summary: medians + pass rates, no single-point verdict
        med_win = median([per_seed[s]["median_window_delta_csi"] for s in SEEDS])
        med_agg = median([per_seed[s]["aggregate_delta_csi"] for s in SEEDS])
        pass_floor = sum(1 for s in SEEDS
                         if agg(cells[f"{arm}_{s}"][0]["windows"], THR)["csi"] > FLOOR)
        ci_all_cross = all(per_seed[s]["window_ci_crosses_zero"] for s in SEEDS)
        results["arms"][arm] = {
            "per_seed": per_seed,
            "median_window_delta_csi": med_win,
            "median_aggregate_delta_csi": med_agg,
            "pass_rate_over_floor": pass_floor / len(SEEDS),
            "window_bootstrap_ci_crosses_zero_all_seeds": ci_all_cross,
            "n_cases_positive_per_seed": {s: per_seed[s]["n_cases_positive"] for s in SEEDS},
        }
        print(f"  {arm}: median window dCSI={med_win:+.6f}  median agg dCSI={med_agg:+.6f}  "
              f"floor pass {pass_floor}/3  window-CI-crosses-0 all seeds={ci_all_cross}")

    # ---- (d) concentration of the gain over the 8 storm cases --------------
    # Which case contributes most of the (arm - B0) aggregate delta?
    for arm in ARMS:
        if arm == "B0":
            continue
        contrib = []
        for cs in cases:
            idxs = idx_by_case[cs]
            num = 0.0
            den = 0.0
            for seed in SEEDS:
                aa = agg([cells[f"{arm}_{seed}"][0]["windows"][i] for i in idxs], THR)
                bb = agg([cells[f"B0_{seed}"][0]["windows"][i] for i in idxs], THR)
                num += aa["csi"] - bb["csi"]
                den += bb["csi"]
            contrib.append({"case": cs, "mean_delta": num / 3.0, "share": (num / 3.0) if den else float("nan")})
        tot = sum(c["mean_delta"] for c in contrib)
        best = max(contrib, key=lambda c: c["mean_delta"])
        results["arms"][arm]["case_contribution"] = contrib
        results["arms"][arm]["top_case"] = best
        results["arms"][arm]["top_case_share_of_total_delta"] = (
            best["mean_delta"] / tot if tot else float("nan"))
        print(f"  {arm}: top case {best['case']} delta={best['mean_delta']:+.6f} "
              f"(total {tot:+.6f}, share {results['arms'][arm]['top_case_share_of_total_delta']:.3f})")

    # ---- (e) does the window-level sign pattern agree with the aggregate? -
    # Informational: the window-level bootstrap CI is the new axis; whether
    # it excludes zero is evidence, not a script-integrity verdict.
    consistent = []
    for arm in ARMS:
        if arm == "B0":
            continue
        for seed in SEEDS:
            r = results["arms"][arm]["per_seed"][seed]
            agg_sign = r["aggregate_delta_csi"] > 0
            win_sign = r["window_sign_pos"] > r["window_sign_neg"]
            consistent.append(agg_sign == win_sign)
    n_sig = 0
    for arm in ARMS:
        if arm == "B0":
            continue
        for seed in SEEDS:
            r = results["arms"][arm]["per_seed"][seed]
            if not r["window_ci_crosses_zero"]:
                n_sig += 1
    results["window_level_ci_excludes_zero_count"] = n_sig
    results["sign_agreement"] = {
        "n_consistent": sum(consistent), "n_total": len(consistent)}
    print(f"\n  window-level CI excludes zero in {n_sig}/{6 * 3} arm-seed cells")
    print(f"  window sign majority agrees with aggregate sign in "
          f"{sum(consistent)}/{len(consistent)} arm-seed cells")

    # ---- (f) leave-one-case-out (LOCO) robustness --------------------------
    # Drop one storm case at a time and recompute the pooled (arm - B0)
    # aggregate CSI delta.  If an arm's gain survives every case removal it
    # is a broad effect; if it collapses, the gain rides on a single event.
    def loco_delta(arm: str, seed: str, drop: str | None) -> float:
        win_a = cells[f"{arm}_{seed}"][0]["windows"]
        win_b = cells[f"B0_{seed}"][0]["windows"]
        keep = [i for i, cs in enumerate(case_of) if cs != drop]
        return (agg([win_a[i] for i in keep], THR)["csi"]
                - agg([win_b[i] for i in keep], THR)["csi"])

    loco: dict[str, dict] = {}
    for arm in ARMS:
        if arm == "B0":
            continue
        per_drop = {}
        for cs in cases:
            vals = [loco_delta(arm, seed, cs) for seed in SEEDS]
            per_drop[cs] = {"per_seed": vals, "median": median(vals)}
        full = [loco_delta(arm, seed, None) for seed in SEEDS]
        meds = [per_drop[cs]["median"] for cs in cases]
        loco[arm] = {
            "per_drop_median": per_drop,
            "full_median": median(full),
            "loco_median_min": min(meds),
            "loco_median_max": max(meds),
            "n_drops_with_negative_median": sum(1 for m in meds if m < 0),
            "n_drops": len(cases),
        }
        results["arms"][arm]["loco"] = loco[arm]
        print(f"  LOCO {arm}: full median={loco[arm]['full_median']:+.6f}  "
              f"drop-one-case medians range [{loco[arm]['loco_median_min']:+.6f}, "
              f"{loco[arm]['loco_median_max']:+.6f}]  "
              f"negative in {loco[arm]['n_drops_with_negative_median']}/{len(cases)} drops")

    # ---- (g) B0-vs-B0 noise control for the case concentration -----------
    # The control arm evaluated under two different seeds is a null model:
    # whatever case concentration and LOCO instability it shows is the noise
    # floor against which the arms must be read.
    ctrl = {}
    for sa, sb in (("s0", "s1"), ("s0", "s2"), ("s1", "s2")):
        per_case = []
        tot = 0.0
        for cs in cases:
            idxs = idx_by_case[cs]
            aa = agg([cells[f"B0_{sa}"][0]["windows"][i] for i in idxs], THR)
            bb = agg([cells[f"B0_{sb}"][0]["windows"][i] for i in idxs], THR)
            d = aa["csi"] - bb["csi"]
            tot += d
            per_case.append({"case": cs, "delta": d})
        best = max(per_case, key=lambda r: r["delta"])
        # LOCO for this control pair
        drop_meds = []
        for cs in cases:
            keep = [i for i, c2 in enumerate(case_of) if c2 != cs]
            da = agg([cells[f"B0_{sa}"][0]["windows"][i] for i in keep], THR)["csi"]
            db = agg([cells[f"B0_{sb}"][0]["windows"][i] for i in keep], THR)["csi"]
            drop_meds.append(da - db)
        ctrl[f"B0_{sa}_vs_B0_{sb}"] = {
            "per_case": per_case,
            "sum_case_deltas": tot,
            "top_case": best["case"],
            "top_case_delta": best["delta"],
            "top_case_share": best["delta"] / tot if tot else float("nan"),
            "loco_median_min": min(drop_meds),
            "loco_median_max": max(drop_meds),
            "n_drops_with_negative_median": sum(1 for m in drop_meds if m < 0),
            "n_drops": len(cases),
        }
        print(f"  CTRL B0_{sa} vs B0_{sb}: top case {best['case']} "
              f"delta={best['delta']:+.6f} (share {best['delta']/tot if tot else float('nan'):.3f})  "
              f"LOCO medians [{min(drop_meds):+.6f}, {max(drop_meds):+.6f}]  "
              f"negative in {sum(1 for m in drop_meds if m < 0)}/{len(cases)} drops")
    results["control_seed_noise"] = ctrl

    # ---- (h) case inventory: how much truth does each case carry? --------
    inv = []
    for cs in cases:
        idxs = idx_by_case[cs]
        a = agg([cells["B0_s0"][0]["windows"][i] for i in idxs], THR)
        inv.append({"case": cs, "n_windows": len(idxs),
                    "truth_events_2mm": a["hits"] + a["misses"],
                    "csi_b0_s0": a["csi"], "bias_b0_s0": a["bias"]})
    results["case_inventory"] = inv
    print("\n  case inventory (truth event volume at 2.0 mm, B0_s0):")
    for r in sorted(inv, key=lambda x: -x["truth_events_2mm"]):
        print(f"    {r['case']}  n_win={r['n_windows']:3d}  truth_events={r['truth_events_2mm']:.0f}  "
              f"csi_b0={r['csi_b0_s0']:.4f}  bias_b0={r['bias_b0_s0']:.3f}")

    # ---- (i) the noise floor: pooled control-vs-control aggregate delta --
    # Same model (B0), two different seeds.  Whatever spread this shows is
    # pure seed noise and bounds what an arm-vs-control delta can mean.
    noise = []
    for sa, sb in (("s0", "s1"), ("s0", "s2"), ("s1", "s2")):
        for seed_arm in (sa, sb):
            a = agg(cells[f"B0_{seed_arm}"][0]["windows"], THR)["csi"]
            b = agg(cells[f"B0_{sa if seed_arm == sb else sb}"][0]["windows"], THR)["csi"]
            noise.append(abs(a - b))
    noise_floor = max(noise)
    results["control_noise_floor_max_abs_aggregate_delta"] = noise_floor
    print(f"\n  noise floor (B0 vs B0, different seeds, pooled aggregate |dCSI|): "
          f"max = {noise_floor:.6f}")

    # ---- (j) pseudo-replication check: window CI vs seed-level spread ----
    # The window bootstrap resamples 538 windows of ONE seed; it cannot see
    # seed-to-seed variability.  The seed is the pre-registered unit of
    # replication (n=3), so a narrow window CI must NOT be read as a
    # significant effect.
    pseudo = []
    for arm in ARMS:
        if arm == "B0":
            continue
        per = results["arms"][arm]["per_seed"]
        win_half = max(abs(per[s]["window_bootstrap_ci_hi"]
                           - per[s]["window_bootstrap_ci_lo"]) / 2.0 for s in SEEDS)
        seed_vals = [per[s]["aggregate_delta_csi"] for s in SEEDS]
        seed_spread = max(seed_vals) - min(seed_vals)
        pseudo.append({"arm": arm, "max_window_ci_halfwidth": win_half,
                       "seed_spread_aggregate_delta": seed_spread,
                       "ratio_seed_over_window": (seed_spread / win_half) if win_half else float("inf")})
        results["arms"][arm]["pseudo_replication"] = pseudo[-1]
    results["pseudo_replication"] = pseudo
    print("  pseudo-replication: seed-level spread of the aggregate delta vs "
          "window-CI half-width")
    for r in pseudo:
        print(f"    {r['arm']}: window CI half-width <= {r['max_window_ci_halfwidth']:.6f}  "
              f"seed spread {r['seed_spread_aggregate_delta']:.6f}  "
              f"ratio {r['ratio_seed_over_window']:.1f}x")

    # ---- write the record and the verdict --------------------------------
    results["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks]
    results["reproduce_max_abs_diff"] = {"csi": worst_csi, "bias": worst_bias,
                                          "pod": worst_pod, "far": worst_far}
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")

    n_pos = sum(1 for _, ok, _ in checks if ok)
    print(f"\nPOS {n_pos}/{len(checks)}")
    verdict = "PASS" if n_pos == len(checks) else "FAIL"
    print(f"VERDICT: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
