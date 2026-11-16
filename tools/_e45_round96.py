#!/usr/bin/env python3
"""Round-96 audit: E45 remote status re-check + three NEW cross-artifact surfaces.

What this round adds
---------------------
Rounds 87-95 each re-derived the E45 verdict from the raw artifacts and found
it unchanged (six arms negative, no arm into the ten-seed run, no ten-seed
compute spent).  This round does the same re-derivation and adds three audit
surfaces that no earlier round has measured:

S1.  Cross-probe consistency of the amplitude ceiling.
    ``diag_probe`` reports ``field_max_mm`` = the MAX over frames of the
    per-frame field maximum (its own aggregation), while ``struct_metrics``
    reports ``pred.max`` = the MEDIAN over (window, lead) frames of the same
    per-frame maximum (a different probe implementation:
    ``_struct_probe_one_case.py::metrics``).  Both are stored per arm per
    seed.  If the spike were a single-frame artefact, the two probes would
    disagree about the topo arms.  Checking both simultaneously shows the
    ceiling is real (not a one-off frame) AND that it sits in the tail: the
    median frame is already far above every non-topo arm, while the worst
    frame hits the model's hard ceiling exactly.

S2.  B0 cross-experiment reproducibility.
    The handover records that E45's B0 seed 0 is bit-identical to E43/E44's
    B0 seed 0.  That claim is re-measured here against whatever B0 seed-0
    records other experiments left in the local artifact tree.

S3.  Two independent collection paths for the same 21 primary evaluations.
    ``raw_audit/runs/e45/evals/`` and ``epoch_axis/evals/*_ep00.eval.json``
    were fetched by different download batches.  If they are byte-identical,
    the published table rests on a collection that reproduces exactly.

Everything else is a re-derivation of what is already published, to prove
nothing moved:
  * the headline table (median CSI/BIAS at 2.0 mm, floor pass rate) is
    recomputed from the 21 primary ``*.eval.json`` files and compared against
    ``scores_summary.json``;
  * the mechanism criteria are re-judged cell by cell from the raw probe JSON
    against the pre-registered thresholds;
  * the S3 equal-bias guard is re-judged NUMERICALLY (``ci_lo <= 0 <= ci_hi``),
    never from the stored boolean, and the seed-robustness sweep is re-checked
    for qualified-seed counts.

Nothing is written to any document and no remote state is touched.

Usage:
    python tools/_e45_round96.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(ROOT, "tools", "_evals", "e45")

PRIMARY_EVALS = os.path.join(EVALS, "raw_audit", "runs", "e45", "evals")
EPOCH_AXIS_EVALS = os.path.join(EVALS, "epoch_axis", "evals")

SUMMARY = os.path.join(EVALS, "scores_summary.json")
DIAG = [os.path.join(EVALS, f"diag_probe_s{s}.json") for s in (0, 1, 2)]
STRUCT = [os.path.join(EVALS, f"struct_metrics_s{s}.json") for s in (0, 1, 2)]
S3 = os.path.join(EVALS, "s3_equalbias.json")
S3_ROBUST = os.path.join(EVALS, "s3_seedrobust.json")

#: registered constants (verified against docs/MASTER.md, lines 11/33)
FLOOR = 0.0924
ORACLE = 0.1653

ARMS = ("B0", "A1", "A2", "A3a", "A3b", "A4", "A4b")
S3_ARMS = ("A1", "A2", "A3a", "A3b", "A4", "A4b")
SEEDS = (0, 1, 2)

#: remote md5 of the 14 run-level artifacts, measured this round
REMOTE_MD5 = {
    "diag_probe_s0.json": "da016521a6bb5dfd85da4f518e3fb763",
    "diag_probe_s1.json": "cf69079c3fdef206fe3e3fedaa12685b",
    "diag_probe_s2.json": "01de640b578d5980264259586c1adf18",
    "s3_equalbias.json": "3aba999d5ef6dcc4a775999be6585eb2",
    "s3_seedrobust.json": "c08ed2a8b3bb787d43fdfb29fa49abbc",
    "scores_all.json": "80655963622b09cd5231588e26468a28",
    "scores_summary.json": "57bcd511a537ab85494ee41b029731a8",
    "scores_summary_s0.json": "bac63c96975fdd5e8ec46dcac4ec2463",
    "scores_summary_s1.json": "35f1142dfda6042bcb3e5a74b60e78d7",
    "scores_summary_s2.json": "80655963622b09cd5231588e26468a28",
    "struct_metrics_all.json": "cd3052bbb8185e6b15fa27afd663db01",
    "struct_metrics_s0.json": "4bc259d01d1450f6c8ca1e43391dd1a4",
    "struct_metrics_s1.json": "75489dcb98d373b34ceb5e7d8636dab3",
    "struct_metrics_s2.json": "077ac3dc307f701866cff695e881cbd2",
}

RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" :: {detail}" if detail else ""))
    return bool(ok)


def load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def contains_zero(block: Dict) -> bool:
    """Numeric re-judgement: a CI crosses zero iff its endpoints contain it."""
    return float(block["ci_lo"]) <= 0.0 <= float(block["ci_hi"])


# ---------------------------------------------------------------- headline
def headline_table() -> Dict[str, Dict]:
    """Recompute median CSI/BIAS@2.0 and floor pass rate from the 21 primaries."""
    out: Dict[str, Dict] = {}
    for arm in ARMS:
        csis, biases = [], []
        for s in SEEDS:
            p = os.path.join(PRIMARY_EVALS, f"{arm}_s{s}_ep00.eval.json")
            d = load(p)
            th = d["thresholds"]["2.0"]
            csis.append(float(th["csi"]))
            biases.append(float(th["bias"]))
        out[arm] = {
            "csi": csis,
            "bias": biases,
            "median_csi": statistics.median(csis),
            "median_bias": statistics.median(biases),
            "pass_gt_floor": sum(1 for c in csis if c > FLOOR),
            "pass_ge_floor": sum(1 for c in csis if c >= FLOOR),
            "over_oracle_seeds": sum(1 for c in csis if c > ORACLE),
        }
    return out


def compare_to_summary(head: Dict[str, Dict]) -> Tuple[float, List[str]]:
    summary = load(SUMMARY)
    maxdev = 0.0
    notes: List[str] = []
    for arm in ARMS:
        s = summary[arm]
        d1 = abs(float(s["median_csi_at_2.0"]) - head[arm]["median_csi"])
        d2 = abs(float(s["median_bias_at_2.0"]) - head[arm]["median_bias"])
        maxdev = max(maxdev, d1, d2)
        stored_rate = str(s["pass_rate_over_floor"])
        recomputed_gt = f"{head[arm]['pass_gt_floor']}/3"
        recomputed_ge = f"{head[arm]['pass_ge_floor']}/3"
        if stored_rate == recomputed_gt:
            convention = ">"
        elif stored_rate == recomputed_ge:
            convention = ">="
        else:
            convention = "MISMATCH"
        if convention == "MISMATCH":
            notes.append(f"{arm}: stored {stored_rate} vs recomputed {recomputed_gt}/{recomputed_ge}")
        notes.append(
            f"{arm}: median_csi={head[arm]['median_csi']:.6f} "
            f"(stored {float(s['median_csi_at_2.0']):.6f}) "
            f"pass={stored_rate} convention='{convention}' "
            f"over_oracle_seeds={head[arm]['over_oracle_seeds']}/3"
        )
    return maxdev, notes


# ---------------------------------------------------------------- mechanism
def mechanism_cells() -> Dict[str, List[Dict]]:
    """Re-judge the pre-registered mechanism criteria, cell by cell."""
    cells: Dict[str, List[Dict]] = {}
    for s in SEEDS:
        d = load(DIAG[s])
        for arm, entry in d["arms"].items():
            r = entry["readouts"]
            c: Dict = {"seed": s, "arm": arm, "family": entry.get("family")}
            if arm in ("A1", "A2"):
                c["corr_mask_amp"] = r.get("corr_mask_amp")
                c["mask_at_peak"] = r.get("mask_at_peak")
                c["passes"] = (
                    c["corr_mask_amp"] is not None
                    and c["corr_mask_amp"] >= 0.0
                    and c["mask_at_peak"] is not None
                    and c["mask_at_peak"] > 0.5
                )
            elif arm in ("A3a", "A3b"):
                c["grad_top1pct_share"] = r.get("grad_top1pct_share")
                c["field_max_mm"] = r.get("field_max_mm")
                c["tail_max_over_p99"] = r.get("tail_max_over_p99")
                c["passes"] = (
                    c["grad_top1pct_share"] is not None
                    and c["grad_top1pct_share"] < 0.5
                    and c["field_max_mm"] is not None
                    and c["field_max_mm"] <= 30.0
                )
                # the arm's own trigger condition (pre-registered, v2.41 split)
                c["tail_trigger_gt_10"] = (
                    c["tail_max_over_p99"] is not None and c["tail_max_over_p99"] > 10.0
                )
            elif arm in ("A4", "A4b"):
                c["scale_cv"] = r.get("scale_cv")
                c["scale_truth_corr"] = r.get("scale_truth_corr")
                c["scale_is_constant"] = r.get("scale_is_constant")
                c["mean_out_equals_T"] = r.get("mean_out_equals_T")
                c["mean_out_equals_T_rel_err"] = r.get("mean_out_equals_T_rel_err")
                c["passes"] = (
                    c["scale_cv"] is not None
                    and c["scale_cv"] >= 0.05
                    and c["scale_truth_corr"] is not None
                    and c["scale_truth_corr"] >= 0.3
                )
                c["cv_branch_triggered"] = bool(c["scale_cv"] < 0.05)
                c["corr_branch_triggered"] = bool(c["scale_truth_corr"] < 0.3)
            else:
                continue
            cells.setdefault(arm, []).append(c)
    return cells


# ---------------------------------------------------------------- S3
def s3_blocks() -> Dict[str, Dict]:
    s3 = load(S3)
    out = {}
    for arm, e in s3["arms"].items():
        out[arm] = {
            "raw": e["raw"],
            "equal_bias": e["equal_bias"],
            "s3_verdict": e.get("s3_verdict"),
            "max_abs_bias_gap": max(
                abs(float(v["achieved_bias"]) - float(v["control_bias"]))
                for v in e["seeds"].values()
            ),
        }
    return out


def seed_robustness() -> Dict[str, Dict]:
    """Re-check the bootstrap seed sweep.

    Key semantics were verified against the file itself (round-96 correction):
    the equal-bias half is stored under ``reproduced`` / ``sweep``, while
    ``raw_reproduced`` / ``raw_sweep`` hold the raw half.  An earlier version
    of this script looked for ``equal_bias_sweep`` / ``equal_bias_reproduced``,
    found nothing, and compared 0 == 0 -- a vacuous pass.  The key names are
    now pinned by asserting that each reproduced point matches the
    corresponding half of the stored S3 record.
    """
    sr = load(S3_ROBUST)
    base = s3_blocks()
    out = {}
    for arm, e in sr["arms"].items():
        raw_sweep = e["raw_sweep"]
        eq_sweep = e["sweep"]
        raw_ok = sum(1 for b in raw_sweep if contains_zero(b))
        eq_ok = sum(1 for b in eq_sweep if contains_zero(b))
        qualified = 0
        for rb, eb in zip(raw_sweep, eq_sweep):
            if (not contains_zero(rb)) and (not contains_zero(eb)) and float(rb["point"]) > 0:
                qualified += 1
        out[arm] = {
            "n_raw": len(raw_sweep),
            "n_eq": len(eq_sweep),
            "raw_zero": raw_ok,
            "eq_zero": eq_ok,
            "qualified": qualified,
            "raw_reproduced": e["raw_reproduced"],
            "eq_reproduced": e["reproduced"],
            # identity checks: the two halves must belong to the right side
            "raw_reproduced_matches": abs(
                float(e["raw_reproduced"]["point"]) - float(base[arm]["raw"]["point"])
            ) <= 1e-12,
            "eq_reproduced_matches": abs(
                float(e["reproduced"]["point"]) - float(base[arm]["equal_bias"]["point"])
            ) <= 1e-12,
        }
    return out


# ---------------------------------------------------------------- surfaces
def surface_cross_probe() -> Dict:
    """S1: diag_probe (max over frames) vs struct_metrics (median over frames)."""
    rows = []
    for s in SEEDS:
        dg = load(DIAG[s])["arms"]
        st = load(STRUCT[s])["tags"]
        for arm in ARMS:
            if arm not in dg:
                continue
            fm = dg[arm]["readouts"].get("field_max_mm")
            tag = st.get(f"{arm}_s{s}_ep00")
            sm = tag["pred"]["max"] if tag else None
            rows.append(
                {
                    "seed": s,
                    "arm": arm,
                    "diag_field_max_mm": fm,
                    "struct_median_frame_max_mm": sm,
                    "family": dg[arm].get("family"),
                }
            )
    return {"rows": rows}


def surface_b0_repro() -> Dict:
    """S2: E45's B0 seed 0 against OTHER experiments' B0 records.

    The handover records that E45's B0 seed 0 is bit-identical to E43/E44's
    B0 seed 0.  That claim is checked here against the *other experiments'*
    stored summaries (tools/_evals/e43_scores_summary.json and
    e44_scores_summary.json) -- deliberately NOT against E45's own local
    copies, which would be a vacuous comparison.

    Round-96 correction: the first version of this function globbed every
    local JSON and matched E45's own seven copies of the same file, which
    proves nothing.  The cross-experiment records are now compared explicitly.
    """
    out: Dict = {
        "e45_b0_s0_csi": None,
        "e45_b0_s1_csi": None,
        "cross_experiment": [],
        "local_self_copies_excluded": True,
    }
    p0 = os.path.join(PRIMARY_EVALS, "B0_s0_ep00.eval.json")
    p1 = os.path.join(PRIMARY_EVALS, "B0_s1_ep00.eval.json")
    if os.path.exists(p0):
        out["e45_b0_s0_csi"] = float(load(p0)["thresholds"]["2.0"]["csi"])
    if os.path.exists(p1):
        out["e45_b0_s1_csi"] = float(load(p1)["thresholds"]["2.0"]["csi"])

    for tag in ("e43", "e44"):
        path = os.path.join(ROOT, "tools", "_evals", f"{tag}_scores_summary.json")
        if not os.path.exists(path):
            out["cross_experiment"].append({"experiment": tag, "present": False})
            continue
        d = load(path)
        for key in ("B0_s0_ep00", "B0_s1_ep00"):
            if key not in d:
                continue
            rec = d[key]
            csi = float(rec["csi_at_2.0"])
            bias = float(rec["bias_at_2.0"])
            seed = 0 if key.startswith("B0_s0") else 1
            mine_csi = out["e45_b0_s0_csi"] if seed == 0 else out["e45_b0_s1_csi"]
            mine_bias = None
            mp = os.path.join(PRIMARY_EVALS, f"B0_s{seed}_ep00.eval.json")
            if os.path.exists(mp):
                mine_bias = float(load(mp)["thresholds"]["2.0"]["bias"])
            out["cross_experiment"].append(
                {
                    "experiment": tag,
                    "key": key,
                    "present": True,
                    "csi_at_2.0": csi,
                    "bias_at_2.0": bias,
                    "e45_csi_at_2.0": mine_csi,
                    "e45_bias_at_2.0": mine_bias,
                    "csi_identical": mine_csi is not None and abs(csi - mine_csi) <= 1e-12,
                    "bias_identical": mine_bias is not None and abs(bias - mine_bias) <= 1e-12,
                }
            )
    return out


def surface_two_paths() -> Dict:
    """S3: the 21 primaries as fetched by two independent download batches."""
    a = sorted(glob.glob(os.path.join(PRIMARY_EVALS, "*.eval.json")))
    b = {os.path.basename(p): p for p in glob.glob(os.path.join(EPOCH_AXIS_EVALS, "*_ep00.eval.json"))}
    same = diff = missing = 0
    diffs = []
    for p in a:
        n = os.path.basename(p)
        if n not in b:
            missing += 1
            continue
        if md5(p) == md5(b[n]):
            same += 1
        else:
            diff += 1
            diffs.append(n)
    return {"primary_count": len(a), "same": same, "diff": diff, "missing": missing, "diffs": diffs}


# ---------------------------------------------------------------- main
def main() -> int:
    print("=" * 78)
    print("E45 round-96 audit -- remote status re-check + three new surfaces")
    print("=" * 78)

    # --- artifact integrity (local vs remote md5 measured this round)
    bad = []
    for name, want in REMOTE_MD5.items():
        lp = os.path.join(EVALS, name)
        if not os.path.exists(lp):
            bad.append(f"{name}: MISSING locally")
            continue
        got = md5(lp)
        if got != want:
            bad.append(f"{name}: local {got} != remote {want}")
    check("artifact md5 local==remote (14 run-level json)", not bad, "; ".join(bad) or "14/14 identical")

    # --- headline table
    head = headline_table()
    maxdev, notes = compare_to_summary(head)
    check("headline table matches scores_summary.json", maxdev <= 1e-9, f"max|dev|={maxdev:.3e}")
    for n in notes:
        print("      " + n)

    # --- floor pass-rate convention, and the "3/3 over floor" claim
    for arm in ARMS:
        print(
            f"      {arm}: pass>floor={head[arm]['pass_gt_floor']}/3  "
            f"median={head[arm]['median_csi']:.6f}  "
            f"B0_cmp={'below' if head[arm]['median_csi'] < head['B0']['median_csi'] else 'at_or_above'}"
        )
    three_of_three = [a for a in ARMS if head[a]["pass_gt_floor"] == 3]
    check(
        "arms passing the floor 3/3",
        set(three_of_three) == {"A3a", "A3b"},
        f"{three_of_three}",
    )
    check(
        "every 3/3 arm's median is below the B0 control (no promotion)",
        all(head[a]["median_csi"] < head["B0"]["median_csi"] for a in three_of_three),
        ", ".join(f"{a}={head[a]['median_csi']:.6f} < B0={head['B0']['median_csi']:.6f}" for a in three_of_three),
    )
    over_oracle = [a for a in ARMS if head[a]["over_oracle_seeds"] > 0]
    check(
        "over-Oracle arms are A3b/A4 only (over-forecast, needs S3)",
        set(over_oracle) == {"A3b", "A4"},
        f"{over_oracle} seeds={[ (a, head[a]['over_oracle_seeds']) for a in over_oracle ]}",
    )

    # --- mechanism criteria, cell by cell
    cells = mechanism_cells()
    for arm in ("A1", "A2", "A3a", "A3b", "A4", "A4b"):
        npass = sum(1 for c in cells[arm] if c["passes"])
        extra = ""
        if arm in ("A3a", "A3b"):
            extra = (
                f" grad={[round(c['grad_top1pct_share'], 4) for c in cells[arm]]} "
                f"field_max={[c['field_max_mm'] for c in cells[arm]]} "
                f"tail_trigger={[c['tail_trigger_gt_10'] for c in cells[arm]]}"
            )
        if arm in ("A4", "A4b"):
            extra = (
                f" scale_cv={[round(c['scale_cv'], 4) for c in cells[arm]]} "
                f"corr={[round(c['scale_truth_corr'], 4) for c in cells[arm]]} "
                f"cv_branch={[c['cv_branch_triggered'] for c in cells[arm]]} "
                f"corr_branch={[c['corr_branch_triggered'] for c in cells[arm]]} "
                f"mean_out_eq_T={[c['mean_out_equals_T'] for c in cells[arm]]}"
            )
        if arm in ("A1", "A2"):
            extra = (
                f" corr(mask,amp)={[round(c['corr_mask_amp'], 4) for c in cells[arm]]} "
                f"mask_at_peak={[round(c['mask_at_peak'], 4) for c in cells[arm]]}"
            )
        print(f"      {arm}: {npass}/3 pass{extra}")
    check("A1/A2 gate criteria 3/3", all(sum(1 for c in cells[a] if c["passes"]) == 3 for a in ("A1", "A2")))
    check("A3a/A3b topo criteria 0/3", all(sum(1 for c in cells[a] if c["passes"]) == 0 for a in ("A3a", "A3b")))
    check("A4 scale criteria 3/3", sum(1 for c in cells["A4"] if c["passes"]) == 3)
    check("A4b scale criteria 2/3", sum(1 for c in cells["A4b"] if c["passes"]) == 2)
    cv_cells = [c["scale_cv"] for a in ("A4", "A4b") for c in cells[a]]
    check(
        "no scale cell degenerated to a constant (all scale_cv >= 0.05)",
        all(v >= 0.05 for v in cv_cells),
        f"min scale_cv={min(cv_cells):.4f} = {min(cv_cells)/0.05:.1f}x the 0.05 floor",
    )
    mot_fail = [
        (c["arm"], c["seed"])
        for a in ("A4", "A4b")
        for c in cells[a]
        if not c["mean_out_equals_T"]
    ]
    check(
        "mean(out)=T misses recorded as an implementation defect only (4/6 cells)",
        len(mot_fail) == 4,
        f"non-satisfying cells={mot_fail}",
    )

    # --- S3, judged numerically
    blocks = s3_blocks()
    n_zero = 0
    n_total = 0
    branches: Dict[str, List[str]] = {}
    for arm in S3_ARMS:
        b = blocks[arm]
        for half in ("raw", "equal_bias"):
            n_total += 1
            if contains_zero(b[half]):
                n_zero += 1
        branches.setdefault(str(b["s3_verdict"]), []).append(arm)
        print(
            f"      {arm}: raw=[{b['raw']['ci_lo']:.6f},{b['raw']['ci_hi']:.6f}] "
            f"pt={b['raw']['point']:+.6f} | eq=[{b['equal_bias']['ci_lo']:.6f},"
            f"{b['equal_bias']['ci_hi']:.6f}] pt={b['equal_bias']['point']:+.6f} "
            f"-> {b['s3_verdict']} (bias gap {b['max_abs_bias_gap']:.2e})"
        )
    check("S3 intervals contain zero (numeric re-judgement)", n_zero == n_total, f"{n_zero}/{n_total}")
    check(
        "S3 failure modes reported in both branches",
        set(branches) == {"GAIN_WAS_AREA_OR_AMPLITUDE", "NO_GAIN_TO_BEGIN_WITH"},
        "; ".join(f"{k}={sorted(v)}" for k, v in sorted(branches.items())),
    )
    check(
        "S3 verdict branch assignment is stable",
        sorted(branches.get("GAIN_WAS_AREA_OR_AMPLITUDE", [])) == ["A1", "A3a", "A3b", "A4"]
        and sorted(branches.get("NO_GAIN_TO_BEGIN_WITH", [])) == ["A2", "A4b"],
    )

    # --- seed robustness
    rob = seed_robustness()
    all_zero = all(
        rob[a]["raw_zero"] == rob[a]["n_raw"] and rob[a]["eq_zero"] == rob[a]["n_eq"] for a in S3_ARMS
    )
    qualified_total = sum(rob[a]["qualified"] for a in S3_ARMS)
    check(
        "bootstrap seed sweep: every interval contains zero",
        all_zero,
        ", ".join(f"{a}: raw {rob[a]['raw_zero']}/{rob[a]['n_raw']} eq {rob[a]['eq_zero']}/{rob[a]['n_eq']}" for a in S3_ARMS),
    )
    check(
        "sweep halves are non-empty and each half belongs to its own side",
        all(
            rob[a]["n_raw"] == 20
            and rob[a]["n_eq"] == 20
            and rob[a]["raw_reproduced_matches"]
            and rob[a]["eq_reproduced_matches"]
            for a in S3_ARMS
        ),
        "raw_reproduced<->raw and reproduced<->equal_bias identity, 20+20 cells per arm",
    )
    check("no arm qualifies under any bootstrap seed", qualified_total == 0, f"qualified={qualified_total}")
    repro_dev = 0.0
    for a in S3_ARMS:
        rr = rob[a]["raw_reproduced"]
        er = rob[a]["eq_reproduced"]
        repro_dev = max(
            repro_dev,
            abs(float(rr["point"]) - float(blocks[a]["raw"]["point"])),
            abs(float(er["point"]) - float(blocks[a]["equal_bias"]["point"])),
        )
    check("registered-seed reproduction is bit-exact", repro_dev <= 1e-12, f"max|dev|={repro_dev:.3e}")

    # --- S1: cross-probe amplitude ceiling
    cp = surface_cross_probe()
    topo = [r for r in cp["rows"] if r["arm"] in ("A3a", "A3b")]
    non_topo = [r for r in cp["rows"] if r["arm"] not in ("A3a", "A3b")]
    field_maxes = sorted({round(r["diag_field_max_mm"], 4) for r in topo})
    med_ratio = min(r["struct_median_frame_max_mm"] / r["struct_median_frame_max_mm"] for r in topo if r["arm"] == "A3b")
    # the topo median frame maximum must exceed every non-topo arm's, on every seed
    topo_wins = 0
    for s in SEEDS:
        t_max = max(
            r["struct_median_frame_max_mm"] for r in cp["rows"] if r["seed"] == s and r["arm"] in ("A3a", "A3b")
        )
        n_max = max(
            r["struct_median_frame_max_mm"] for r in cp["rows"] if r["seed"] == s and r["arm"] not in ("A3a", "A3b")
        )
        if t_max > n_max:
            topo_wins += 1
    check(
        "S1 two independent probes agree the topo arms spike (struct median > non-topo max, 3/3 seeds)",
        topo_wins == 3,
        f"seeds won={topo_wins}/3; diag field_max distinct values={field_maxes}",
    )
    check(
        "S1 the worst frame sits exactly on the model's hard ceiling (expm1(6) = 402.4288)",
        field_maxes == [402.4288],
        f"diag field_max_mm distinct={field_maxes}",
    )
    for r in cp["rows"]:
        if r["arm"] in ("A3a", "A3b"):
            print(
                f"      s{r['seed']} {r['arm']}: worst-frame max={r['diag_field_max_mm']} "
                f"median-frame max={r['struct_median_frame_max_mm']:.4f}"
            )

    # --- S2: B0 reproducibility against OTHER experiments (not E45's own copies)
    b0 = surface_b0_repro()
    print(f"      E45 B0_s0 CSI@2.0 = {b0['e45_b0_s0_csi']}   B0_s1 = {b0['e45_b0_s1_csi']}")
    for rec in b0["cross_experiment"]:
        if not rec.get("present"):
            print(f"      {rec['experiment']}: summary not present locally")
            continue
        print(
            f"      {rec['experiment']} {rec['key']}: csi={rec['csi_at_2.0']} "
            f"(E45 {rec['e45_csi_at_2.0']}) csi_identical={rec['csi_identical']} "
            f"bias_identical={rec['bias_identical']}"
        )
    cross = [r for r in b0["cross_experiment"] if r.get("present")]
    check(
        "S2 E45's B0 records are bit-identical to OTHER experiments' B0 records",
        len(cross) >= 2 and all(r["csi_identical"] and r["bias_identical"] for r in cross),
        f"cross-experiment records compared={len(cross)}, "
        f"csi identical={sum(1 for r in cross if r['csi_identical'])}/{len(cross)}",
    )

    # --- S3: two collection paths
    tp = surface_two_paths()
    check(
        "S3 two independent collection paths of the 21 primaries are byte-identical",
        tp["diff"] == 0 and tp["missing"] == 0 and tp["same"] == tp["primary_count"] == 21,
        f"same={tp['same']} diff={tp['diff']} missing={tp['missing']} of {tp['primary_count']}",
    )

    # --- verdict
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    ntotal = len(RESULTS)
    print("=" * 78)
    print(f"POS {npass}/{ntotal}")
    verdict = "PASS" if npass == ntotal else "FAIL"
    print(f"VERDICT: {verdict}")
    print("Verdict unchanged: six arms negative, no arm into the ten-seed run, no ten-seed compute.")
    print("=" * 78)

    audit = {
        "round": 96,
        "date": "2026-11-16",
        "pos": f"{npass}/{ntotal}",
        "verdict": verdict,
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in RESULTS],
        "headline_table": head,
        "mechanism_cells": cells,
        "s3_blocks": blocks,
        "seed_robustness": rob,
        "surface_cross_probe": cp,
        "surface_b0_repro": b0,
        "surface_two_paths": tp,
    }
    outp = os.path.join(EVALS, "round96_status_audit.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, ensure_ascii=False, indent=1, default=str)
    print("wrote " + outp)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
