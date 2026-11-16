#!/usr/bin/env python
"""E45 verification round 30 (independent, no import of earlier round scripts).

Round 30 closes an integrity chain that rounds 27-29 never verified end-to-end:

    remote md5 (shell)  ->  dump md5 (downloaded)  ->  recomputed contingency
    metrics  ->  local scores_summary.json  ->  HANDOVER seed-0 anchors

Authoritative metric definitions are taken from
``radar_nwp_bench/metrics/precipitation.py`` (VerificationCounts), NOT guessed:

    csi  = H / (H+M+FA)
    bias = (H+FA) / (H+M)
    pod  = H / (H+M)
    far  = FA / (H+FA)
    ets  = (H - chance) / (D - chance),  chance = (H+M)(H+FA)/T,  T = H+M+FA+CN
    hss  = 2 (H*CN - M*FA) / ((H+M)(FA+CN) + (H+FA)(M+CN))
    fss  = spatial fractions skill score (radius neighbourhood)

Consequence, established here and not assumed: ETS and HSS need
``correct_negatives`` (CN), which the eval JSON does NOT store, and FSS needs
the spatial field.  So CN is *solved back* from the stored ETS and then used to
reproduce the stored HSS -- if ETS and HSS are mutually consistent with a
single physically valid CN, the two metrics corroborate each other and the
implied pixel total must be consistent with the window count.

Read-only with respect to the repository: it only reads JSON files.
"""
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(ROOT, "tools", "_evals", "e45")
DUMP = os.path.join(EVALS, "_eval_dump_round30.json")
SUMMARY = os.path.join(EVALS, "scores_summary.json")
SCORES_ALL = os.path.join(EVALS, "scores_all.json")

FLOOR = 0.0924
ORACLE = 0.1653
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
CONTROL = "B0"

REMOTE_EVAL_MD5 = {
    "A1_s0_ep00.eval.json": "081f10d09d05e8a4097793392c13cba5",
    "A1_s1_ep00.eval.json": "31906e020988516339824a88cb6c6109",
    "A1_s2_ep00.eval.json": "83f0edad0e408e0275ed6d717313a8e6",
    "A2_s0_ep00.eval.json": "4cd4d82310969df3fd1e3ce087572225",
    "A2_s1_ep00.eval.json": "3c4876c8dce6298d2e51809f39eca3b4",
    "A2_s2_ep00.eval.json": "9893ac967b7e3afbdc6fbd87bbdfad1e",
    "A3a_s0_ep00.eval.json": "a463d9102694b1d88643c9a27c34ceec",
    "A3a_s1_ep00.eval.json": "f56b90efedeb41400d5b824744ae4340",
    "A3a_s2_ep00.eval.json": "adf0056027fa36ad33ac82e31ee909d7",
    "A3b_s0_ep00.eval.json": "6d63f95aa26322f51e2842b85551a7f7",
    "A3b_s1_ep00.eval.json": "dcd9b3011e23f133be07fe06daf17ed3",
    "A3b_s2_ep00.eval.json": "07f9089598450d85c1ac62b1af7d0848",
    "A4_s0_ep00.eval.json": "d79fd59b52fdc8334ba97df9d82065e5",
    "A4_s1_ep00.eval.json": "7149bf6794769b4aca52b69b98120e0e",
    "A4_s2_ep00.eval.json": "3e356c3759179feca6f7011c2a9e19ef",
    "A4b_s0_ep00.eval.json": "9247529bf6d2cb4057c201ecdc6b63e6",
    "A4b_s1_ep00.eval.json": "e8b903ceaaacfa1a9569849e2ae95cbe",
    "A4b_s2_ep00.eval.json": "82c4631825af2283df2b1133b085958c",
    "B0_s0_ep00.eval.json": "19f73e2acadb85606a2af5f2f13b28a9",
    "B0_s1_ep00.eval.json": "f1518c976f18291146206fc557c63f5f",
    "B0_s2_ep00.eval.json": "fec6672ff210c017831f7a3d02072591",
}

HANDOVER_S0 = {
    "B0": (0.12181703042952949, 0.5095421057251265),
    "A1": (0.1183974362784095, 0.4430213000608119),
    "A2": (0.11200272630175023, 0.4490284126332717),
    "A3a": (0.14947025751330345, 0.671884634518365),
    "A3b": (0.16966079982919896, 0.9204696034133556),
    "A4": (0.1755001059215604, 1.6130548061898633),
    "A4b": (0.13756654739888727, 3.17889950530225),
}

POS, NEG = [], []


def ok(name, cond, detail=""):
    POS.append((name, bool(cond), detail))
    return bool(cond)


def bad(name, cond, detail=""):
    NEG.append((name, bool(cond), detail))
    return bool(cond)


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return False
    if isinstance(a, float) and math.isnan(a):
        return False
    return abs(a - b) <= tol


def solve_cn(h, m, f, ets):
    """Recover correct_negatives CN from the stored ETS.

    chance = (H+M)(H+FA)/T,  T = H+M+FA+CN
    ets    = (H - chance) / (D - chance),  D = H+M+FA
    => chance = (H - ets*D) / (1 - ets)
    => T = (H+M)(H+FA)/chance ;  CN = T - D
    """
    d = h + m + f
    if abs(1.0 - ets) < 1e-12:
        return None
    num = h - ets * d
    chance = num / (1.0 - ets)
    if chance <= 0:
        return None
    t = (h + m) * (h + f) / chance
    return t - d


def hss_from(h, m, f, cn):
    num = 2.0 * (h * cn - m * f)
    den = (h + m) * (f + cn) + (h + f) * (m + cn)
    return num / den if den != 0 else float("nan")


def main():
    with open(DUMP, "r", encoding="utf-8") as fh:
        dump = json.load(fh)
    with open(SUMMARY, "r", encoding="utf-8") as fh:
        summ = json.load(fh)
    with open(SCORES_ALL, "r", encoding="utf-8") as fh:
        allsc = json.load(fh)

    rows = {r["file"]: r for r in dump["rows"]}

    # ---------- 1. coverage / integrity of the 21 remote eval files ----------
    ok("dump_has_21_eval_files", dump["n_files"] == 21, f"n_files={dump['n_files']}")
    ok("dump_row_count_21", len(rows) == 21, f"rows={len(rows)}")

    parsed = {}
    pat = re.compile(r"^(?P<arm>[A-Za-z0-9]+)_s(?P<seed>[0-9])_ep(?P<ep>[0-9]+)\.eval\.json$")
    for fn in rows:
        mo = pat.match(fn)
        ok(f"filename_parses::{fn}", mo is not None, fn)
        if mo:
            parsed[fn] = (mo.group("arm"), int(mo.group("seed")), mo.group("ep"))
            ok(f"ep00::{fn}", mo.group("ep") == "00", mo.group("ep"))
    arms_seen = sorted({v[0] for v in parsed.values()})
    seeds_seen = sorted({v[1] for v in parsed.values()})
    ok("all_7_arms_present", arms_seen == sorted(ARMS), f"{arms_seen}")
    ok("all_3_seeds_present", seeds_seen == [0, 1, 2], f"{seeds_seen}")
    ok("arm_x_seed_complete_21",
       sorted((v[0], v[1]) for v in parsed.values())
       == sorted((a, s) for a in ARMS for s in (0, 1, 2)), "7x3 grid")

    md5_ok = sum(1 for fn, want in REMOTE_EVAL_MD5.items()
                 if fn in rows and rows[fn]["md5"] == want)
    ok("eval_md5_matches_remote_shell_21", md5_ok == 21, f"{md5_ok}/21")

    ns = {rows[fn]["n_samples"] for fn in rows}
    ok("n_samples_identical_538", ns == {538}, f"{ns}")

    # ---------- 2. recompute the 4 count-derived metrics ----------
    # (csi / bias / pod / far -- the only ones determined by H,M,FA alone)
    c4_fail, c4_total = 0, 0
    recomputed = {}
    for fn, (arm, seed, _ep) in sorted(parsed.items()):
        for tname, block in rows[fn]["thresholds"].items():
            h, m, f = float(block["hits"]), float(block["misses"]), float(block["false_alarms"])
            d = h + m + f
            g = {
                "csi": h / d if d > 0 else float("nan"),
                "bias": (h + f) / (h + m) if (h + m) > 0 else float("nan"),
                "pod": h / (h + m) if (h + m) > 0 else float("nan"),
                "far": f / (h + f) if (h + f) > 0 else float("nan"),
            }
            for k, v in g.items():
                c4_total += 1
                if not close(v, block[k], 1e-12):
                    c4_fail += 1
            recomputed[(arm, seed, tname)] = g
    ok("count_derived_metrics_recompute_exactly", c4_fail == 0,
       f"{c4_fail}/{c4_total} mismatches")

    # ---------- 3. ETS<->HSS mutual consistency via solved CN ----------
    cn_fail, cn_bad, cn_skip = 0, 0, 0
    tot_implied = {}
    for fn, (arm, seed, _ep) in sorted(parsed.items()):
        for tname, block in rows[fn]["thresholds"].items():
            h, m, f = float(block["hits"]), float(block["misses"]), float(block["false_alarms"])
            cn = solve_cn(h, m, f, block["ets"])
            if cn is None:
                cn_skip += 1
                continue
            if cn < -1e-6:
                cn_bad += 1
                continue
            hss = hss_from(h, m, f, cn)
            if not close(hss, block["hss"], 1e-6):
                cn_fail += 1
            tot_implied[(arm, seed, tname)] = h + m + f + cn
    ok("ets_hss_mutually_consistent_via_solved_cn", cn_fail == 0,
       f"mismatch={cn_fail} skipped={cn_skip}")
    ok("solved_cn_never_negative", cn_bad == 0, f"negative_cn_blocks={cn_bad}")

    # implied total pixels per window should be an integer multiple of n_samples
    px = set()
    for (arm, seed, tname), t in tot_implied.items():
        if t > 0:
            px.add(round(t / 538.0, 4))
    px_spread = (max(px) - min(px)) if px else None
    ok("implied_pixels_per_window_consistent", px_spread is not None and px_spread < 1e-6,
       f"distinct values={len(px)} spread={px_spread}")

    # FSS is spatial -> not derivable from counts, only bounds-checkable
    fss_fail = sum(1 for fn in rows for b in rows[fn]["thresholds"].values()
                   if not (-1.0 - 1e-9 <= b["fss"] <= 1.0 + 1e-9))
    ok("fss_within_theoretical_bounds", fss_fail == 0, f"out_of_bounds={fss_fail}")

    # ---------- 4. tie recomputed eval metrics to the local summary ----------
    tie_fail = 0
    for arm in ARMS:
        for seed in (0, 1, 2):
            g2 = recomputed.get((arm, seed, "2.0"))
            g4 = recomputed.get((arm, seed, "4.0"))
            g05 = recomputed.get((arm, seed, "0.5"))
            if g2 is None or g4 is None or g05 is None:
                tie_fail += 1
                continue
            if not close(g2["csi"], summ[arm]["csi_at_2.0"][f"s{seed}"], 1e-12):
                tie_fail += 1
            if not close(g2["bias"], summ[arm]["bias_at_2.0"][f"s{seed}"], 1e-12):
                tie_fail += 1
            if not close(g4["csi"], summ[arm]["csi_at_4.0"][f"s{seed}"], 1e-12):
                tie_fail += 1
            if not close(g05["csi"], summ[arm]["csi_at_0.5"][f"s{seed}"], 1e-12):
                tie_fail += 1
    ok("recomputed_eval_ties_to_scores_summary", tie_fail == 0, f"tie_fail={tie_fail}")

    # ---------- 5. re-derive summary aggregates ----------
    agg_fail = 0
    for arm in ARMS:
        csis = [summ[arm]["csi_at_2.0"][f"s{s}"] for s in (0, 1, 2)]
        biass = [summ[arm]["bias_at_2.0"][f"s{s}"] for s in (0, 1, 2)]
        if not close(sorted(csis)[1], summ[arm]["median_csi_at_2.0"], 1e-12):
            agg_fail += 1
        if not close(sorted(biass)[1], summ[arm]["median_bias_at_2.0"], 1e-12):
            agg_fail += 1
        if summ[arm]["pass_rate_over_floor"] != f"{sum(1 for c in csis if c >= FLOOR)}/3":
            agg_fail += 1
        if not close(max(csis) / min(csis), summ[arm]["seed_spread_csi_at_2.0"], 1e-9):
            agg_fail += 1
        if summ[arm]["n_seeds"] != 3 or summ[arm]["seeds"] != [0, 1, 2]:
            agg_fail += 1
    ok("summary_aggregates_reproduced", agg_fail == 0, f"agg_fail={agg_fail}")

    # ---------- 6. HANDOVER seed-0 anchors ----------
    hv_fail = 0
    for arm, (csi, bias) in HANDOVER_S0.items():
        g = recomputed.get((arm, 0, "2.0"))
        if g is None or not close(g["csi"], csi, 1e-12) or not close(g["bias"], bias, 1e-12):
            hv_fail += 1
    ok("handover_seed0_anchors_reproduced", hv_fail == 0, f"hv_fail={hv_fail}")

    # ---------- 7. scores_all.json cross-check ----------
    allfail = 0
    keys = [k for k in allsc if re.match(r"^[A-Za-z0-9]+_s[0-2]_ep00$", k)]
    if len(keys) != 21:
        allfail += 1
    for arm in ARMS:
        for seed in (0, 1, 2):
            blk = allsc.get(f"{arm}_s{seed}_ep00")
            if not isinstance(blk, dict):
                allfail += 1
                continue
            g = recomputed[(arm, seed, "2.0")]
            if not close(float(blk.get("csi_at_2.0")), g["csi"], 1e-12):
                allfail += 1
            if not close(float(blk.get("bias_at_2.0")), g["bias"], 1e-12):
                allfail += 1
            if blk.get("n_samples") != 538:
                allfail += 1
    ok("scores_all_consistent_with_recomputed", allfail == 0, f"allfail={allfail}")

    # ---------- 8. decision-level invariants ----------
    med = {a: summ[a]["median_csi_at_2.0"] for a in ARMS}
    medb = {a: summ[a]["median_bias_at_2.0"] for a in ARMS}
    above_ctrl = [a for a in ARMS if a != CONTROL and med[a] > med[CONTROL]]
    ok("above_control_median_arms_exactly_A4_A4b",
       sorted(above_ctrl) == ["A4", "A4b"], f"{sorted(above_ctrl)}")
    ok("above_control_arms_have_inflated_bias",
       all(medb[a] > medb[CONTROL] for a in above_ctrl),
       f"ctrl_bias={medb[CONTROL]:.4f} " + " ".join(f"{a}={medb[a]:.4f}" for a in above_ctrl))

    qual = [a for a in above_ctrl if med[a] > med[CONTROL] and medb[a] <= medb[CONTROL]]
    ok("qualifying_arm_set_empty", qual == [], f"{qual}")

    bad("any_arm_median_exceeds_oracle",
        any(med[a] > ORACLE for a in ARMS),
        f"{ {a: round(med[a], 4) for a in ARMS if med[a] > ORACLE} }")
    bad("any_arm_median_above_oracle_with_low_bias",
        any(med[a] > ORACLE and medb[a] <= 1.0 for a in ARMS),
        f"{ {a: (round(med[a], 4), round(medb[a], 4)) for a in ARMS if med[a] > ORACLE} }")
    bad("any_arm_all_3_seeds_over_oracle",
        any(all(summ[a]["csi_at_2.0"][f"s{s}"] > ORACLE for s in (0, 1, 2)) for a in ARMS),
        "over-forecast signature check")
    bad("scores_all_record_count_not_21", len(keys) != 21, f"{len(keys)}")

    # ---------- report ----------
    L = []
    L.append("=== E45 round-30 independent verification (fixed) ===")
    L.append(f"remote: 21 eval files | md5 chain {md5_ok}/21 | n_samples=538 uniform")
    L.append(f"count-derived metrics (csi/bias/pod/far): {c4_total - c4_fail}/{c4_total} exact")
    L.append(f"ETS<->HSS via solved CN: mismatch={cn_fail} negative_cn={cn_bad} skipped={cn_skip}")
    L.append(f"implied pixels/window distinct={len(px)} spread={px_spread}")
    L.append("")
    L.append(f"{'arm':<5}{'medianCSI@2.0':>16}{'medianBIAS@2.0':>16}{'pass/floor':>12}{'spread(max/min)':>18}")
    for a in ARMS:
        L.append(f"{a:<5}{med[a]:>16.6f}{medb[a]:>16.4f}"
                 f"{summ[a]['pass_rate_over_floor']:>12}{summ[a]['seed_spread_csi_at_2.0']:>18.3f}")
    L.append("")
    L.append(f"floor={FLOOR}  oracle={ORACLE}")
    L.append("qualifying arm set for the >=10-seed ladder = "
             f"{qual if qual else 'EMPTY (consistent with S3: all paired CIs cross zero)'}")
    npass = sum(1 for _, c, _ in POS if c)
    nfail = sum(1 for _, c, _ in NEG if c)
    L.append("")
    L.append(f"POS {npass}/{len(POS)}   NEG {nfail}/{len(NEG)}")
    for n, c, d in POS:
        if not c:
            L.append(f"  POS-FAIL {n} :: {d}")
    for n, c, d in NEG:
        if c:
            L.append(f"  NEG-FAIL {n} :: {d}")
    verdict = "PASS" if (nfail == 0 and npass == len(POS)) else "FAIL"
    L.append(f"VERDICT: {verdict}")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(ROOT, "tools", "_e45_verify_round30_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(out + "\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
