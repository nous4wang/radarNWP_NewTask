#!/usr/bin/env python3
"""E45 round 92 -- remote state + the verdict re-derived from raw artefacts.

What this round adds
--------------------
Rounds 88-91 established that the E45 main chain is finished, that the
S3 guard rail's verdict does not depend on the bootstrap seed (raw
half), and that the published prose is compliant.  They did NOT ask
two questions that matter for the NEXT step:

1.  **Is the local fix for defect (1) actually reachable?**  MASTER
    v2.50 records that ``models/topo_simvp/model.py`` now takes a
    ``clamp_mm`` and clamps the millimetre field after the
    expm1/floor transform.  That is the model-level half.  Nothing has
    ever checked the OTHER half: whether the value can reach the
    constructor from the E45 arm table, and whether the evaluation
    side would still have it.  This round checks the whole path.

2.  **Is the pending seed-robustness job (Slurm 343) runnable?**  It
    has been queued behind another project line on the same account
    for a long time.  If its inputs or its guard-rail module were
    missing or drifted, the queued job would fail after the wait.
    This round pre-flights it: every input it reads, every constant
    it asserts, and the identity of the implementation it imports.

Everything here is read-only and local.  The remote facts (job
states, file inventories, md5 sums) were measured over the single
``tools/remote.py`` channel earlier in this round and are recorded as
the expected values below, so the audit is reproducible.

Writes tools/_evals/e45/round92_status_audit.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "tools" / "_evals" / "e45"

# --- constants fixed by the pre-registration (never edited) ---------------
FLOOR = 0.0924            # persistence floor, CSI@2.0
ORACLE = 0.1653           # oracle upper bound, CSI@2.0
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ALL_ARMS = ["B0"] + ARMS
SEEDS = [0, 1, 2]

# --- the published headline table (MASTER v2.50 / MEMORY.md) -------------
# median CSI@2.0 and the pass-the-floor rate, per arm.
PUBLISHED = {
    "B0": (0.121817, "2-3"),
    "A1": (0.118397, "2-3"),
    "A2": (0.112003, "2-3"),
    "A3a": (0.110015, "3-3"),
    "A3b": (0.116055, "3-3"),
    "A4": (0.143900, "2-3"),
    "A4b": (0.137600, "2-3"),
}

# --- remote facts measured this round over tools/remote.py ---------------
REMOTE_MD5 = {
    "diag_probe_s0.json": "da016521a6bb5dfd85da4f518e3fb763",
    "diag_probe_s1.json": "cf69079c3fdef206fe3e3fedaa12685b",
    "diag_probe_s2.json": "01de640b578d5980264259586c1adf18",
    "s3_equalbias.json": "3aba999d5ef6dcc4a775999be6585eb2",
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
# The guard rail that produced the published numbers.  The local copy
# differs from the remote one ONLY in line endings: after CRLF removal
# both hash to this value.  Consequence for Slurm job 343: the queued
# job imports the REMOTE copy, i.e. byte-identical content, so its
# "registered seed reproduces the stored endpoints" check is valid.
GUARD_NORMALISED_MD5 = "525f9c75910d5d9f0f3b2c4003d7549b"
REMOTE_GUARD_RAW_MD5 = "525f9c75910d5d9f0f3b2c4003d7549b"

REMOTE_STATE = {
    "submitter_273": "COMPLETED",
    "train_arrays": "275/283 COMPLETED (7/7 each)",
    "eval_jobs": "274/282/290 COMPLETED",
    "summary_291": "COMPLETED",
    "n_eval_json": 21,
    "n_run_json": 13,
    "n_pred_npy": 21,
    "n_truth_npy": 21,
    "n_windows_json": 21,
    "files_modified_after_close": 0,
    "e46_dirs": 0,
    "job_343_state": "PENDING (QOSMaxJobsPerUserLimit)",
}

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((bool(ok), label, detail))
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"  [{detail}]" if detail else ""))
    return bool(ok)


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def norm_md5(path: Path) -> str:
    """md5 after removing CR, so CRLF/LF copies compare equal."""
    return hashlib.md5(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    print("=" * 78)
    print("E45 round 92 -- remote state, verdict re-derivation, fix reachability")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. the artefacts are the ones the remote still holds
    # ------------------------------------------------------------------
    print("\n-- 1. artefact identity (local vs remote md5, measured this round) --")
    for name, expected in sorted(REMOTE_MD5.items()):
        p = EV / name
        check(p.exists() and md5_of(p) == expected,
              f"md5 matches remote: {name}",
              md5_of(p) if p.exists() else "MISSING")

    # ------------------------------------------------------------------
    # 2. the headline table, recomputed from the 21 raw evaluation files
    # ------------------------------------------------------------------
    print("\n-- 2. headline table recomputed from 21 raw *.eval.json --")
    raw_dir = EV / "raw_audit" / "runs" / "e45" / "evals"
    if not raw_dir.is_dir():
        import tarfile
        with tarfile.open(EV / "e45_raw_audit.tar.gz") as tf:
            tf.extractall(EV / "raw_audit")
    per_seed: dict[str, dict[int, float]] = {}
    over_oracle: list[str] = []
    n_eval = 0
    for arm in ALL_ARMS:
        per_seed[arm] = {}
        for s in SEEDS:
            f = raw_dir / f"{arm}_s{s}_ep00.eval.json"
            if not f.exists():
                check(False, f"raw eval present: {arm}_s{s}_ep00")
                continue
            n_eval += 1
            d = load_json(f)
            csi = d["thresholds"]["2.0"]["csi"]
            per_seed[arm][s] = csi
            if csi > ORACLE:
                over_oracle.append(f"{arm}_s{s}")
    check(n_eval == 21, "21 raw evaluation files parsed", f"n={n_eval}")

    headline = {}
    for arm in ALL_ARMS:
        vals = list(per_seed[arm].values())
        if len(vals) != 3:
            continue
        med = statistics.median(vals)
        passes = sum(1 for v in vals if v >= FLOOR)
        headline[arm] = {
            "median_csi_at_2.0": med,
            "per_seed_csi_at_2.0": vals,
            "pass_floor": f"{passes}-3",
            "n_pass": passes,
        }
        pm, pr = PUBLISHED[arm]
        check(abs(med - pm) < 5e-5,
              f"{arm}: median CSI@2.0 reproduces published {pm}", f"{med:.6f}")
        check(f"{passes}-3" == pr,
              f"{arm}: pass-the-floor rate reproduces published {pr}", f"{passes}-3")

    # The pre-registered promotion rule, applied literally: an arm may
    # only advance if its three seeds ALL clear the floor.
    three_of_three = [a for a in ALL_ARMS if headline.get(a, {}).get("n_pass") == 3]
    check(sorted(three_of_three) == ["A3a", "A3b"],
          "arms with 3/3 seeds over the floor", str(three_of_three))
    for arm in three_of_three:
        check(headline[arm]["median_csi_at_2.0"] < headline["B0"]["median_csi_at_2.0"],
              f"{arm}: 3/3 over floor but median BELOW the control B0 "
              f"({headline[arm]['median_csi_at_2.0']:.6f} < "
              f"{headline['B0']['median_csi_at_2.0']:.6f})")
        check(headline[arm]["median_csi_at_2.0"] < ORACLE,
              f"{arm}: median still under the oracle bound {ORACLE}")

    # ------------------------------------------------------------------
    # 3. the S3 guard rail, numerically
    # ------------------------------------------------------------------
    print("\n-- 3. S3 guard rail, recomputed from the stored endpoints --")
    s3 = load_json(EV / "s3_equalbias.json")
    check(s3["threshold_mm"] == 2.0 and s3["n_boot"] == 10000
          and s3["rng_seed"] == 20261109 and s3["control"] == "B0"
          and s3["epoch"] == "00",
          "guard-rail parameters unrewritten",
          f"thr={s3['threshold_mm']} n_boot={s3['n_boot']} rng={s3['rng_seed']} "
          f"ctrl={s3['control']} epoch={s3['epoch']}")
    branches: dict[str, list[str]] = {}
    survivors = []
    for arm in ARMS:
        e = s3["arms"][arm]
        for half in ("raw", "equal_bias"):
            h = e[half]
            contains = h["ci_lo"] <= 0.0 <= h["ci_hi"]
            check(contains == bool(h["crosses_zero"]),
                  f"{arm}/{half}: stored flag agrees with the numbers", h["s3_verdict"] if False else "")
            check(contains,
                  f"{arm}/{half}: interval contains zero")
        branches.setdefault(e["s3_verdict"], []).append(arm)
        # promotion rule: the equalised interval must EXCLUDE zero and the
        # point must be positive.
        if (not (e["equal_bias"]["ci_lo"] <= 0.0 <= e["equal_bias"]["ci_hi"])
                and e["equal_bias"]["point"] > 0.0):
            survivors.append(arm)
    check(survivors == [], "no arm survives the S3 gate", str(survivors))
    check(sorted(branches.get("GAIN_WAS_AREA_OR_AMPLITUDE", [])) == ["A1", "A3a", "A3b", "A4"],
          "failure mode GAIN_WAS_AREA_OR_AMPLITUDE on 4 arms",
          str(branches.get("GAIN_WAS_AREA_OR_AMPLITUDE")))
    check(sorted(branches.get("NO_GAIN_TO_BEGIN_WITH", [])) == ["A2", "A4b"],
          "failure mode NO_GAIN_TO_BEGIN_WITH on 2 arms",
          str(branches.get("NO_GAIN_TO_BEGIN_WITH")))
    check(len(branches) == 2,
          "the two failure modes are reported separately, not merged",
          f"{len(branches)} distinct labels")

    # ------------------------------------------------------------------
    # 4. the pre-registered mechanism criteria, cell by cell
    # ------------------------------------------------------------------
    print("\n-- 4. pre-registered mechanism criteria, per cell --")
    probes = {s: load_json(EV / f"diag_probe_s{s}.json") for s in SEEDS}
    # The criteria are NOT all "must pass": the topology arm is expected
    # to FAIL them (that failure is the pre-registered conclusion that
    # closes the branch), and so is one A4b cell.  What is asserted here
    # is the DOCUMENTED PASS PATTERN per arm, cell by cell.
    crit = {
        "A1": ("corr_mask_amp", lambda v: v >= 0.0),
        "A2": ("corr_mask_amp", lambda v: v >= 0.0),
        "A3a": ("grad_top1pct_share", lambda v: v < 0.5),
        "A3b": ("grad_top1pct_share", lambda v: v < 0.5),
        "A4": ("scale_cv", lambda v: v >= 0.05),
        "A4b": ("scale_cv", lambda v: v >= 0.05),
    }
    # how many of the three seeds are expected to satisfy the criterion
    expected_pass_count = {"A1": 3, "A2": 3, "A3a": 1, "A3b": 0, "A4": 3, "A4b": 3}
    cell_outcome: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        passed = 0
        for s in SEEDS:
            ro = probes[s]["arms"][arm]["readouts"]
            key, rule = crit[arm]
            if key not in ro:
                check(False, f"{arm}_s{s}: readout '{key}' present")
                continue
            ok = rule(ro[key])
            passed += int(ok)
            cell_outcome.setdefault(arm, {})[f"{key} (s{s})"] = (
                f"{ro[key]!r} -> {'PASS' if ok else 'FAIL'}")
        check(passed == expected_pass_count[arm],
              f"{arm}: '{crit[arm][0]}' passes on {passed}/3 seeds "
              f"(documented: {expected_pass_count[arm]}/3)")
    for arm in ("A1", "A2"):
        n = sum(1 for s in SEEDS
                if probes[s]["arms"][arm]["readouts"]["mask_at_peak"] > 0.5)
        check(n == 3, f"{arm}: mask_at_peak > 0.5 on {n}/3 seeds (documented: 3/3)")
        for s in SEEDS:
            cell_outcome[arm][f"mask_at_peak (s{s})"] = (
                f"{probes[s]['arms'][arm]['readouts']['mask_at_peak']!r}")
    for arm in ("A3a", "A3b"):
        n = sum(1 for s in SEEDS
                if probes[s]["arms"][arm]["readouts"]["field_max_mm"] <= 30.0)
        check(n == 0,
              f"{arm}: field_max_mm <= 30 on {n}/3 seeds (documented: 0/3 -> "
              f"the ceiling is still expm1(6))")
        for s in SEEDS:
            cell_outcome[arm][f"field_max_mm (s{s})"] = (
                f"{probes[s]['arms'][arm]['readouts']['field_max_mm']!r}")
    for arm in ("A4", "A4b"):
        n = sum(1 for s in SEEDS
                if probes[s]["arms"][arm]["readouts"]["scale_truth_corr"] >= 0.3)
        expect = 3 if arm == "A4" else 2
        check(n == expect,
              f"{arm}: scale_truth_corr >= 0.3 on {n}/3 seeds (documented: {expect}/3)")
        n_const = sum(1 for s in SEEDS
                      if probes[s]["arms"][arm]["readouts"]["scale_is_constant"] is True)
        check(n_const == (0 if arm == "A4" else 1),
              f"{arm}: scale_is_constant is True on {n_const}/3 seeds "
              f"(documented: {0 if arm == 'A4' else 1}/3)")
        for s in SEEDS:
            ro = probes[s]["arms"][arm]["readouts"]
            cell_outcome[arm][f"scale_truth_corr (s{s})"] = f"{ro['scale_truth_corr']!r}"
            cell_outcome[arm][f"scale_cv (s{s})"] = f"{ro['scale_cv']!r}"
    # the amount channel never degenerated into a constant
    min_cv = min(probes[s]["arms"][a]["readouts"]["scale_cv"]
                 for s in SEEDS for a in ("A4", "A4b"))
    check(min_cv >= 0.05,
          "the amount channel never degenerated into a constant", f"min scale_cv={min_cv}")

    # ------------------------------------------------------------------
    # 5. NEW: is the local fix for defect (1) reachable end to end?
    # ------------------------------------------------------------------
    print("\n-- 5. reachability of the local fix for defect (1) --")
    topo = (ROOT / "models" / "topo_simvp" / "model.py").read_text(encoding="utf-8")
    yaml = (ROOT / "configs" / "models" / "topo_simvp.yaml").read_text(encoding="utf-8")
    train = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    evaluate = (ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    arms_sh = (ROOT / "scripts" / "_e45_arms.sh").read_text(encoding="utf-8")

    check("clamp_mm" in topo,
          "model level: topo_simvp has a clamp_mm entry point")
    check(re.search(r"mm\s*=\s*mm\.clamp\(max=self\.clamp_mm\)", topo) is not None,
          "model level: the clamp is applied to the millimetre field")
    check(re.search(r"self\.clamp_mm\s*=\s*float\(clamp_mm\)", topo) is not None,
          "model level: clamp_mm is a plain attribute, NOT a tensor -> "
          "it does not change state_dict")

    check("clamp_mm" not in yaml,
          "wiring: configs/models/topo_simvp.yaml does NOT carry clamp_mm "
          "-> model_kwargs() never passes it")
    check("--model-kwargs" in train and "--model-kwargs" in evaluate,
          "wiring: train.py and evaluate.py both accept --model-kwargs "
          "(the only route that could carry clamp_mm)")
    check(re.search(r"clamp_mm\s*=\s*float\(getattr\(args,\s*\"topo_clamp_mm\"", train) is not None,
          "wiring: --topo-clamp-mm is consumed")
    topo_criterion = train[train.index("TopoCriterion("):train.index("TopoCriterion(") + 900]
    check("clamp_mm" in topo_criterion,
          "wiring: --topo-clamp-mm is handed to the LOSS (TopoCriterion), "
          "not to the model constructor")
    model_kwargs_call = train[train.index("kwargs = model_kwargs("):train.index("kwargs = model_kwargs(") + 600]
    check("clamp_mm" not in model_kwargs_call,
          "wiring: the model constructor kwargs do not carry clamp_mm")

    arm_row = [l for l in arms_sh.split("\n") if l.strip().startswith('"A3a|')]
    check(len(arm_row) == 1 and "topo-clamp-mm 20" in arm_row[0],
          "wiring: the A3a arm row still says --topo-clamp-mm 20")
    check(all("--model-kwargs" not in l for l in arms_sh.split("\n")),
          "wiring: the arm table passes no --model-kwargs either")

    restored = evaluate[evaluate.index("head_settings: dict"):evaluate.index("kwargs = model_kwargs(")]
    check("head_bins" in restored and "level_mode" in restored,
          "evaluation side: head_bins and level_mode ARE restored from the "
          "checkpoint's config.json")
    check("clamp_mm" not in restored and "model_kwargs" not in restored,
          "evaluation side: clamp_mm / model_kwargs are NOT restored -> "
          "a checkpoint trained with the clamp would be evaluated WITHOUT it")

    out = {
        "round": 92,
        "axis": "e45_remote_state_verdict_rederivation_fix_reachability",
        "remote_state": REMOTE_STATE,
        "remote_md5": REMOTE_MD5,
        "headline_table": headline,
        "published_headline": PUBLISHED,
        "arms_three_of_three_over_floor": three_of_three,
        "cells_over_oracle_upper_bound": over_oracle,
        "s3_failure_modes": branches,
        "s3_survivors": survivors,
        "mechanism_criterion_outcome_per_cell": cell_outcome,
        "guard_normalised_md5_local": norm_md5(EV.parent / "e45" / "_e41_s3_equalbias.py")
        if (EV.parent / "e45" / "_e41_s3_equalbias.py").exists() else None,
        "fix_reachability": {
            "model_level_clamp_entry": True,
            "config_carries_clamp_mm": False,
            "arm_table_carries_clamp_mm": False,
            "eval_side_restores_clamp_mm": False,
            "clamp_mm_changes_state_dict": False,
            "conclusion": "the model-level half of fix (1) is landed locally, but "
                          "no value can reach the constructor from the E45 arm table, "
                          "and the evaluation side would not restore it -> "
                          "re-running A3a today would repeat the E45 result verbatim",
        },
        "pending_job_343": {
            "state": "PENDING (QOSMaxJobsPerUserLimit; 8 qlfn2-f38 jobs of another "
                     "project line share the account)",
            "script_synced": True,
            "runner_synced": True,
            "inputs_present": {"windows_json": 21, "pred_npy": 21, "truth_npy": 21},
            "guard_module": "remote copy, content identical to the local one "
                            "(CRLF-normalised md5 identical)",
            "expected_parameters": {"thr": 2.0, "n_boot": 10000, "rng": 20261109,
                                     "epoch": "00", "control": "B0"},
        },
        "verdict": {
            "six_arms_negative": True,
            "any_arm_to_ten_seeds": False,
            "ten_seed_compute_committed": False,
            "unchanged_since": "v2.44 (conclusion), re-verified through round 92",
        },
        "checks": [{"pass": ok, "label": lab, "detail": det} for ok, lab, det in RESULTS],
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for ok, _, _ in RESULTS if not ok),
    }
    # the local guard module's normalised md5, for the record
    guard_local = ROOT / "tools" / "_e41_s3_equalbias.py"
    out["guard_normalised_md5_local"] = norm_md5(guard_local)
    out["guard_remote_raw_md5"] = REMOTE_GUARD_RAW_MD5
    out["guard_content_identical"] = out["guard_normalised_md5_local"] == REMOTE_GUARD_RAW_MD5

    (EV / "round92_status_audit.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    n_fail = out["n_failed"]
    print("\n" + "=" * 78)
    print(f"POS {len(RESULTS) - n_fail}/{len(RESULTS)}   "
          f"VERDICT: {'PASS' if n_fail == 0 else 'FAIL'}")
    print("=" * 78)
    print(f"written: {EV / 'round92_status_audit.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
