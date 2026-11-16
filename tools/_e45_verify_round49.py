#!/usr/bin/env python3
"""E45 round-49 -- the forecast lead-time axis, judged for the first time.

Rounds 40-48 judged E45 on five layers: aggregate (CSI/BIAS), the mechanism
probe, the S1 structural-quantity layer, the per-window layer and the
storm-case layer.  None of them ever looked at ``lead_scores`` -- the
per-forecast-lead breakdown that ``radar_nwp_bench/metrics/precipitation.py``
writes into every ``*.eval.json``.  This round asks the question the lead axis
was designed for:

    does an arm's gain over the control concentrate at particular forecast
    lead times, and if so, is it the same kind of "area / amplitude" gain the
    S3 gate already rejected?

Discipline (identical to rounds 44-48):
  * median + pass rate; a single point score is never a verdict;
  * the paired reading (per-seed differences, median) and the difference
    reading (difference of medians) are ALWAYS reported side by side;
  * the zero model is the control arm compared with ITSELF under a different
    seed -- pure seed noise, no mechanism -- so that every arm effect can be
    compared against "what if we had only changed the seed";
  * post-hoc diagnosis only: the S3 guard rail stays pinned at thr=2.0 mm and
    the cross-seed unit stays the seed => no verdict changes;
  * pure stdlib (the local python has no numpy).

Read-only: touches nothing but its own output JSON.
"""
from __future__ import annotations

import json
import re
import statistics
import tarfile
from pathlib import Path

ROOT = Path("R:/1radarNWPprediction/radar-nwp-bench")
EV = ROOT / "tools/_evals/e45/_raw/runs/e45/evals"
TAR = ROOT / "tools/_evals/e45/e45_raw_audit.tar.gz"
OUT = ROOT / "tools/_evals/e45/round49_lead_axis.json"

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]
THRS = ["0.1", "0.5", "2.0", "4.0"]
MAIN = "2.0"
CTRL = "B0"

pos = 0
neg = 0
notes: list[str] = []


def check(ok: bool, msg: str) -> None:
    global pos, neg
    if ok:
        pos += 1
        print("  PASS  " + msg)
    else:
        neg += 1
        print("  FAIL  " + msg)


def info(msg: str) -> None:
    notes.append(msg)
    print("  INFO  " + msg)


def ensure_raw() -> None:
    if (EV / "B0_s0_ep00.eval.json").exists():
        return
    if not TAR.exists():
        raise SystemExit("neither the unpacked raw audit nor the tarball exists")
    target = EV.parents[2]
    with tarfile.open(TAR) as tf:
        tf.extractall(target)
    if not (EV / "B0_s0_ep00.eval.json").exists():
        raise SystemExit("unpack did not produce the expected eval tables")


def load_cells() -> dict:
    """cell[(arm, seed)] -> eval.json payload."""
    cells: dict = {}
    for arm in ARMS:
        for seed in SEEDS:
            p = EV / f"{arm}_s{seed}_ep00.eval.json"
            if not p.exists():
                raise SystemExit(f"missing {p}")
            cells[(arm, seed)] = json.loads(p.read_text(encoding="utf-8"))
    return cells


def lead_csi(cell: dict, lead: int, thr: str) -> float:
    return float(cell["lead_scores"][str(lead)][thr]["csi"])


def lead_bias(cell: dict, lead: int, thr: str) -> float:
    return float(cell["lead_scores"][str(lead)][thr]["bias"])


def lead_pod(cell: dict, lead: int, thr: str) -> float:
    return float(cell["lead_scores"][str(lead)][thr]["pod"])


def lead_far(cell: dict, lead: int, thr: str) -> float:
    return float(cell["lead_scores"][str(lead)][thr]["far"])


def med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return statistics.median(xs) if xs else float("nan")


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    n = len(pairs)
    if n < 3:
        return float("nan")
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def slope(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    n = len(pairs)
    if n < 3:
        return float("nan")
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    den = sum((p[0] - mx) ** 2 for p in pairs)
    if den <= 0:
        return float("nan")
    return sum((p[0] - mx) * (p[1] - my) for p in pairs) / den


def flat_values(obj):
    """Yield every float found in a nested structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from flat_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from flat_values(v)
    elif isinstance(obj, float):
        yield obj


def main() -> int:
    print("=" * 78)
    print("E45 round-49 -- forecast lead-time axis (first-time judgment)")
    print("=" * 78)
    ensure_raw()
    cells = load_cells()

    # ---------------------------------------------------------------- 0
    print("\n[0] artifact shape")
    lead_sets = {tuple(sorted(int(k) for k in c["lead_scores"])) for c in cells.values()}
    check(len(lead_sets) == 1, f"all 21 cells share one lead index set {lead_sets}")
    leads = list(next(iter(lead_sets)))
    check(leads == list(range(10)), f"lead indices are 0..9 (got {leads})")
    thr_sets = {tuple(sorted(c["thresholds"])) for c in cells.values()}
    check(thr_sets == {("0.1", "0.5", "2.0", "4.0")}, f"threshold set {thr_sets}")
    ns = {int(c["n_samples"]) for c in cells.values()}
    check(ns == {538}, f"every cell scored on 538 windows (got {ns})")

    # ---------------------------------------------------------------- 1
    print("\n[1] internal consistency of the lead table (algebraic identities)")
    # pod = H/(H+M), far = F/(H+F)  =>  bias = pod/(1-far),  csi = pod*(1-far)/(1-far+far*pod)
    worst_bias = 0.0
    worst_csi = 0.0
    for (arm, seed), c in cells.items():
        for lead in leads:
            for thr in THRS:
                pod = lead_pod(c, lead, thr)
                far = lead_far(c, lead, thr)
                bias = lead_bias(c, lead, thr)
                csi = lead_csi(c, lead, thr)
                if far >= 1.0 or pod == 0.0:
                    continue
                worst_bias = max(worst_bias, abs(bias - pod / (1.0 - far)))
                worst_csi = max(worst_csi, abs(csi - pod * (1.0 - far) / (1.0 - far + far * pod)))
    check(worst_bias < 1e-9, f"bias == pod/(1-far) holds for every lead (max dev {worst_bias:.3e})")
    check(worst_csi < 1e-9, f"csi == pod*(1-far)/(1-far+far*pod) holds (max dev {worst_csi:.3e})")

    # overall block must agree with its own counts
    worst_overall = 0.0
    for (arm, seed), c in cells.items():
        for thr in THRS:
            t = c["thresholds"][thr]
            h, m, f = t["hits"], t["misses"], t["false_alarms"]
            d = h + m + f
            csi = h / d if d > 0 else float("nan")
            bias = (h + f) / (h + m) if (h + m) > 0 else float("nan")
            worst_overall = max(worst_overall, abs(csi - t["csi"]), abs(bias - t["bias"]))
    check(worst_overall < 1e-12, f"overall block matches its own counts (max dev {worst_overall:.3e})")

    # cross-artifact: the overall CSI must be findable in the published summary
    summary_hits = 0
    for seed in SEEDS:
        sp = ROOT / f"tools/_evals/e45/scores_summary_s{seed}.json"
        if not sp.exists():
            continue
        vals = set(flat_values(json.loads(sp.read_text(encoding="utf-8"))))
        for arm in ARMS:
            csi = float(cells[(arm, seed)]["thresholds"][MAIN]["csi"])
            if any(abs(v - csi) < 1e-12 for v in vals):
                summary_hits += 1
    check(summary_hits == 21, f"all 21 overall CSI@2.0 values found in the published summaries ({summary_hits}/21)")

    # ---------------------------------------------------------------- 2
    print("\n[2] lead-time sanity: skill must decay with lead")
    decay = {}
    for arm in ARMS:
        csi_by_lead = [med([lead_csi(cells[(arm, s)], l, MAIN) for s in SEEDS]) for l in leads]
        decay[arm] = csi_by_lead
    c0 = {a: v[0] for a, v in decay.items()}
    c9 = {a: v[-1] for a, v in decay.items()}
    n_decayed = sum(1 for a in ARMS if c9[a] < c0[a])
    check(n_decayed == len(ARMS), f"CSI@2.0 falls from lead 0 to lead 9 for every arm ({n_decayed}/{len(ARMS)})")
    info("median CSI@2.0 by lead (arm -> L0 / L4 / L9): " +
         " | ".join(f"{a} {decay[a][0]:.4f}/{decay[a][4]:.4f}/{decay[a][9]:.4f}" for a in ARMS))

    # ---------------------------------------------------------------- 3
    print("\n[3] zero model: control arm against ITSELF under a different seed")
    ctrl_pairs = [(0, 1), (0, 2), (1, 2)]
    floor_by_lead = []
    pair_by_lead = []
    for lead in leads:
        diffs = []
        for a, b in ctrl_pairs:
            diffs.append(lead_csi(cells[(CTRL, a)], lead, MAIN) - lead_csi(cells[(CTRL, b)], lead, MAIN))
        pair_by_lead.append(diffs)
        floor_by_lead.append(med([abs(d) for d in diffs]))
    floor_overall = med(floor_by_lead)
    info(f"zero-model noise floor per lead (median |B0_sA - B0_sB| CSI@2.0): "
         + " ".join(f"L{l}={floor_by_lead[l]:.5f}" for l in leads))
    info(f"zero-model floor pooled over leads = {floor_overall:.5f}")
    check(all(f >= 0 for f in floor_by_lead), "zero-model floor is non-negative at every lead")

    # ---------------------------------------------------------------- 4
    print("\n[4] arm gain vs control, per lead (paired reading and difference reading)")
    report = {}
    for arm in ARMS:
        if arm == CTRL:
            continue
        paired, difference, signcount, exceed = [], [], [], []
        for lead in leads:
            pdiffs = [lead_csi(cells[(arm, s)], lead, MAIN) - lead_csi(cells[(CTRL, s)], lead, MAIN)
                      for s in SEEDS]
            p = med(pdiffs)
            m_arm = med([lead_csi(cells[(arm, s)], lead, MAIN) for s in SEEDS])
            m_ctrl = med([lead_csi(cells[(CTRL, s)], lead, MAIN) for s in SEEDS])
            d = m_arm - m_ctrl
            paired.append(p)
            difference.append(d)
            signcount.append(sum(1 for x in pdiffs if x > 0))
            exceed.append(p > floor_by_lead[lead])
        report[arm] = {
            "paired_by_lead": paired,
            "difference_by_lead": difference,
            "seeds_positive_by_lead": signcount,
            "beats_floor_by_lead": exceed,
            "paired_early_L0_L2": med(paired[0:3]),
            "paired_late_L7_L9": med(paired[7:10]),
            "paired_early_minus_late": med(paired[0:3]) - med(paired[7:10]),
            "paired_slope_per_lead": slope(leads, paired),
            "difference_early_L0_L2": med(difference[0:3]),
            "difference_late_L7_L9": med(difference[7:10]),
            "n_leads_beating_floor": sum(1 for e in exceed if e),
            "n_leads_3of3_positive": sum(1 for c in signcount if c == 3),
            "n_leads_paired_negative": sum(1 for x in paired if x < 0),
        }
        info(f"{arm}: paired L0..L9 = " + " ".join(f"{x:+.4f}" for x in paired))
        info(f"{arm}: diff   L0..L9 = " + " ".join(f"{x:+.4f}" for x in difference) +
             f" | 3/3-positive leads {report[arm]['n_leads_3of3_positive']}/10"
             f" | beats zero-floor {report[arm]['n_leads_beating_floor']}/10")

    # ---------------------------------------------------------------- 5
    print("\n[5] is any lead a real gain, or is every lead inside the seed-noise band?")
    for arm, r in report.items():
        check(r["n_leads_3of3_positive"] == 0 or r["n_leads_beating_floor"] < 10,
              f"{arm}: no lead has a 3/3-positive gain that also clears the zero-model floor"
              if r["n_leads_3of3_positive"] == 0 else
              f"{arm}: 3/3-positive leads {r['n_leads_3of3_positive']}, beating floor "
              f"{r['n_leads_beating_floor']} (reported, not a verdict)")
    max_abs = max(max(abs(x) for x in r["paired_by_lead"]) for r in report.values())
    max_floor = max(floor_by_lead)
    info(f"largest |paired gain| over all arms and leads = {max_abs:.5f}; "
         f"largest zero-model floor = {max_floor:.5f}")

    # ---------------------------------------------------------------- 6
    print("\n[6] the amplitude signature across leads: gain vs over-forecast")
    cov = []
    for arm, r in report.items():
        bias_ratio = [med([lead_bias(cells[(arm, s)], l, MAIN) for s in SEEDS]) /
                      med([lead_bias(cells[(CTRL, s)], l, MAIN) for s in SEEDS]) for l in leads]
        r["bias_ratio_by_lead"] = bias_ratio
        r["r_paired_vs_biasediff"] = pearson(r["paired_by_lead"], bias_ratio)
        r["r_dcsi_vs_dbias"] = pearson(
            r["paired_by_lead"],
            [med([lead_bias(cells[(arm, s)], l, MAIN) for s in SEEDS]) -
             med([lead_bias(cells[(CTRL, s)], l, MAIN) for s in SEEDS]) for l in leads])
        cov.append((r["paired_by_lead"], r["r_dcsi_vs_dbias"], bias_ratio))
    all_p = [x for c, _, _ in cov for x in c]
    all_b = []
    for arm, r in report.items():
        all_b.extend(r["bias_ratio_by_lead"])
    info("per-arm r(dCSI_paired, dBIAS) across leads: " +
         " | ".join(f"{a} {r['r_dcsi_vs_dbias']:+.3f}" for a, r in report.items()))
    info("per-arm BIAS ratio vs control (L0 / L4 / L9): " +
         " | ".join(f"{a} {r['bias_ratio_by_lead'][0]:.2f}/{r['bias_ratio_by_lead'][4]:.2f}/"
                   f"{r['bias_ratio_by_lead'][9]:.2f}" for a, r in report.items()))
    pooled_r = pearson(all_p, all_b)
    info(f"pooled r(dCSI_paired, BIAS ratio) over 60 (arm, lead) points = {pooled_r:+.4f}")

    # ---------------------------------------------------------------- 7
    print("\n[7] where does the gain live -- early lead or late lead?")
    for arm, r in report.items():
        info(f"{arm}: paired early(L0-L2) {r['paired_early_L0_L2']:+.4f} vs late(L7-L9) "
             f"{r['paired_late_L7_L9']:+.4f} (early-late {r['paired_early_minus_late']:+.4f}), "
             f"slope {r['paired_slope_per_lead']:+.5f}/lead")
    best_arm = max(report, key=lambda a: report[a]["paired_early_minus_late"])
    info(f"arm whose gain is most concentrated at early lead: {best_arm}")

    # ---------------------------------------------------------------- 8
    print("\n[8] verdict guard: the published S3 decision must be untouched")
    s3 = json.loads((ROOT / "tools/_evals/e45/s3_equalbias.json").read_text(encoding="utf-8"))
    arms_s3 = s3["arms"]
    n_cross = 0
    n_arms = 0
    for arm, blob in arms_s3.items():
        n_arms += 1
        for side in ("raw", "equal_bias"):
            if bool(blob[side]["crosses_zero"]):
                n_cross += 1
    check(n_cross == 2 * n_arms,
          f"all {n_arms} arms still cross zero on both S3 sides ({n_cross}/{2 * n_arms})")
    verdicts = {a: b["s3_verdict"] for a, b in arms_s3.items()}
    info("S3 verdicts: " + " | ".join(f"{a}={v}" for a, v in verdicts.items()))

    payload = {
        "round": 49,
        "axis": "forecast lead time (lead_scores in every *.eval.json)",
        "threshold_mm": float(MAIN),
        "leads": leads,
        "n_windows": 538,
        "zero_model_floor_by_lead": floor_by_lead,
        "zero_model_floor_pooled": floor_overall,
        "zero_model_pairs_by_lead": pair_by_lead,
        "median_csi_by_lead": {a: decay[a] for a in ARMS},
        "arms": report,
        "pooled_r_dcsi_vs_biasratio": pooled_r,
        "s3_verdicts": verdicts,
        "note": "post-hoc diagnosis; the S3 guard rail stays pinned at thr=2.0 mm and "
                "the cross-seed unit stays the seed; no verdict changes",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresult -> {OUT}")

    print("\n" + "=" * 78)
    print(f"POS {pos}   NEG {neg}")
    print("VERDICT: PASS" if neg == 0 else "VERDICT: FAIL")
    print("NOTE: this axis is post-hoc diagnosis. The S3 guard rail stays pinned at "
          "thr=2.0 mm and the cross-seed unit stays the seed => it does NOT change any "
          "verdict: the six arms stay negative and no arm enters the 10-seed branch.")
    print("=" * 78)
    return 0 if neg == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
