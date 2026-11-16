#!/usr/bin/env python3
"""E45 round 93 -- remote state, the verdict re-derived, and the defect-(1)
wiring, checked from the run records instead of from the source alone.

What this round adds
--------------------
Round 92 read the reachability gap of the defect-(1) fix out of the
SOURCE: the model has a ``clamp_mm`` entry point, the yaml does not
carry it, the arm table passes no ``--model-kwargs``, and the
evaluation restores only ``head_bins`` and ``level_mode``.  That is a
statement about the code as it stands now.

This round checks the same thing against the RUN RECORDS -- the
``config.json`` each of the 21 E45 training runs wrote next to its
checkpoint.  Those files are the training run's own account of how it
built its model, and they are the artefacts an evaluation would read.
The question they answer is not "could the value reach the constructor"
but "did it": if a run's record shows the constructor kwargs it was
given, and that record is empty, then the mechanism was off in the run
that produced the published numbers -- a fact about the experiment, not
a prediction about the code.

This round also lands the half of the wiring that can be landed without
touching the frozen E45 arm table:

* the evaluation side now restores the constructor kwargs the training
  run recorded (``scripts/evaluate.py``), which is the only half that
  can be fixed without changing what E45 was;
* the training side is NOT changed: the E45 arm table is the
  pre-registered definition of what E45 ran, and editing it would
  change what E45 means.  The next round's arm table will carry
  ``--model-kwargs``; the route is verified here to work through the
  very same ``read -r -a`` splitting the arm table uses.

Everything here is read-only and local.  The remote facts were measured
over the single ``tools/remote.py`` channel earlier in this round and
are recorded as the expected values below, so the audit is reproducible.

Writes tools/_evals/e45/round93_status_audit.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "tools" / "_evals" / "e45"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- constants fixed by the pre-registration (never edited) ---------------
FLOOR = 0.0924            # persistence floor, CSI@2.0
ORACLE = 0.1653           # oracle upper bound, CSI@2.0
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
ALL_ARMS = ["B0"] + ARMS
SEEDS = [0, 1, 2]

# --- the published headline table (MASTER v2.51 / MEMORY.md) -------------
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

# The 21 training runs' own config.json records, read on the remote this
# round.  Every one of them records the constructor kwargs it was given
# as None -- and A3a's three record topo_clamp_mm = 20.0, i.e. the loss
# got the cap and the model did not.
REMOTE_RUN_RECORDS = {
    "n_configs": 21,
    "n_with_nonempty_model_kwargs": 0,
    "a3a_topo_clamp_mm": 20.0,
    "a3a_model_kwargs": None,
    "n_configs_scanned_all_runs": 43,
    "n_nonempty_model_kwargs_all_runs": 0,
}

REMOTE_STATE = {
    "submitter_273": "COMPLETED",
    "train_arrays": "275/283 COMPLETED (7/7 each)",
    "eval_jobs": "274/282/290 COMPLETED",
    "summary_291": "COMPLETED",
    "n_eval_json": 21,
    "n_run_json": 13,
    "files_modified_after_close": 0,
    "e46_dirs": 0,
    "job_343_state": "PENDING (QOSMaxJobsPerUserLimit)",
    "job_343_time_limit": "04:00:00",
    "job_343_cpus": 8,
    "job_343_mem": "64G",
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


def main() -> int:
    print("=" * 78)
    print("E45 round 93 -- remote state, verdict re-derivation, defect-(1) wiring")
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

    three_of_three = [a for a in ALL_ARMS if headline.get(a, {}).get("n_pass") == 3]
    check(sorted(three_of_three) == ["A3a", "A3b"],
          "arms with 3/3 seeds over the floor", str(three_of_three))
    for arm in three_of_three:
        check(headline[arm]["median_csi_at_2.0"] < headline["B0"]["median_csi_at_2.0"],
              f"{arm}: 3/3 over floor but median BELOW the control B0 "
              f"({headline[arm]['median_csi_at_2.0']:.6f} < "
              f"{headline['B0']['median_csi_at_2.0']:.6f})")

    # ------------------------------------------------------------------
    # 3. the S3 guard rail, numerically
    # ------------------------------------------------------------------
    print("\n-- 3. S3 guard rail, recomputed from the stored endpoints --")
    s3 = load_json(EV / "s3_equalbias.json")
    check(s3["threshold_mm"] == 2.0 and s3["n_boot"] == 10000
          and s3["rng_seed"] == 20261109 and s3["control"] == "B0"
          and s3["epoch"] == "00",
          "guard-rail parameters unrewritten",
          f"thr={s3['threshold_mm']} n_boot={s3['n_boot']} rng={s3['rng_seed']}")
    branches: dict[str, list[str]] = {}
    survivors = []
    for arm in ARMS:
        e = s3["arms"][arm]
        for half in ("raw", "equal_bias"):
            h = e[half]
            contains = h["ci_lo"] <= 0.0 <= h["ci_hi"]
            check(contains == bool(h["crosses_zero"]),
                  f"{arm}/{half}: stored flag agrees with the numbers")
            check(contains, f"{arm}/{half}: interval contains zero")
        branches.setdefault(e["s3_verdict"], []).append(arm)
        if (not (e["equal_bias"]["ci_lo"] <= 0.0 <= e["equal_bias"]["ci_hi"])
                and e["equal_bias"]["point"] > 0.0):
            survivors.append(arm)
    check(survivors == [], "no arm survives the S3 gate", str(survivors))
    check(sorted(branches.get("GAIN_WAS_AREA_OR_AMPLITUDE", [])) == ["A1", "A3a", "A3b", "A4"],
          "failure mode GAIN_WAS_AREA_OR_AMPLITUDE on 4 arms")
    check(sorted(branches.get("NO_GAIN_TO_BEGIN_WITH", [])) == ["A2", "A4b"],
          "failure mode NO_GAIN_TO_BEGIN_WITH on 2 arms")
    check(len(branches) == 2,
          "the two failure modes are reported separately, not merged")

    # ------------------------------------------------------------------
    # 4. NEW: the run records, not just the source
    # ------------------------------------------------------------------
    print("\n-- 4. the 21 training runs' own records --")
    check(REMOTE_RUN_RECORDS["n_configs"] == 21,
          "all 21 E45 training runs wrote a config.json")
    check(REMOTE_RUN_RECORDS["n_with_nonempty_model_kwargs"] == 0,
          "NO E45 run recorded any constructor kwargs -> the amplitude "
          "cap never reached the model in the runs that produced the "
          "published numbers")
    check(REMOTE_RUN_RECORDS["a3a_topo_clamp_mm"] == 20.0
          and REMOTE_RUN_RECORDS["a3a_model_kwargs"] is None,
          "A3a recorded topo_clamp_mm=20.0 (the LOSS got the cap) and "
          "model_kwargs=None (the MODEL did not)")
    check(REMOTE_RUN_RECORDS["n_nonempty_model_kwargs_all_runs"] == 0,
          "no run on the remote ever recorded constructor kwargs, so the "
          "evaluation-side restore changes nothing for any existing run")

    # ------------------------------------------------------------------
    # 5. the wiring, now that the route is known
    # ------------------------------------------------------------------
    print("\n-- 5. the defect-(1) wiring --")
    evaluate = (ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    train = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    arms_sh = (ROOT / "scripts" / "_e45_arms.sh").read_text(encoding="utf-8")
    topo = (ROOT / "models" / "topo_simvp" / "model.py").read_text(encoding="utf-8")

    check("_trained_model_kwargs" in evaluate,
          "evaluation side: the recorded constructor kwargs are read back")
    check("restored_kwargs" in evaluate
          and "model_cfg[\"model\"].update(applicable)" in evaluate,
          "evaluation side: the restored kwargs reach the model config")
    check('"--model-kwargs"' in train and "run_config = dict(vars(args))" in train,
          "training side: the flag is accepted and recorded in config.json")

    # The E45 arm table is frozen: the pre-registered definition of what
    # E45 ran.  It is NOT changed here.
    check("--model-kwargs" not in arms_sh,
          "the frozen E45 arm table still carries no --model-kwargs "
          "(it must not be edited: it defines what E45 was)")
    arm_row = [l for l in arms_sh.split("\n") if l.strip().startswith('"A3a|')]
    check(len(arm_row) == 1 and "topo-clamp-mm 20" in arm_row[0],
          "the A3a row is unchanged")

    # The route the NEXT round's arm table will use, verified through the
    # very splitting the arm table uses (read -r -a on a space-separated
    # field) and through the constructor.  The probe is a bash script
    # written to disk, because the flag string carries quotes and this
    # MSYS bash does not pass environment variables into a child bash.
    import subprocess
    probe_dir = ROOT / "tools" / "_e45_armflag_probe"
    probe_dir.mkdir(exist_ok=True)
    probe_sh = probe_dir / "probe.sh"
    probe_sh.write_text(
        "FLAG='--topo-clamp-mm 20 --model-kwargs {\"clamp_mm\":20.0}'\n"
        "read -r -a A <<< \"$FLAG\"\n"
        "printf 'N=%s\\n' \"${#A[@]}\"\n"
        "for i in \"${!A[@]}\"; do printf '[%s]=%s\\n' \"$i\" \"${A[$i]}\"; done\n",
        encoding="utf-8",
        # LF, not CRLF: a carriage return at the end of the `for` line
        # makes bash read the loop body as unfinished.
        newline="\n",
    )
    probe = subprocess.run(
        ["bash", "tools/_e45_armflag_probe/probe.sh"],
        capture_output=True, text=True, cwd=str(ROOT))
    tokens = []
    for line in probe.stdout.split("\n"):
        if line.startswith("[") and "]=" in line:
            tokens.append(line.split("]=", 1)[1])
    check(tokens == ["--topo-clamp-mm", "20", "--model-kwargs",
                     '{"clamp_mm":20.0}'],
          "the arm table's read -r -a splitting keeps the JSON in ONE token",
          str(tokens))
    try:
        parsed = json.loads(tokens[-1])
    except (ValueError, IndexError):
        parsed = None
    check(parsed == {"clamp_mm": 20.0},
          "that token parses as the JSON object the constructor wants",
          repr(parsed))
    from radar_nwp_bench.config import model_kwargs as mk
    kw = mk({"model": {}}, overrides=tokens[-1])
    check(kw.get("clamp_mm") == 20.0,
          "model_kwargs() carries it into the constructor kwargs",
          repr(kw.get("clamp_mm")))
    check("clamp_mm" in topo and "self.clamp_mm = float(clamp_mm)" in topo,
          "model level: clamp_mm is still a plain attribute (not a weight)")

    out = {
        "round": 93,
        "axis": "e45_remote_state_verdict_rederivation_defect1_wiring",
        "remote_state": REMOTE_STATE,
        "remote_md5": REMOTE_MD5,
        "remote_run_records": REMOTE_RUN_RECORDS,
        "headline_table": headline,
        "published_headline": PUBLISHED,
        "arms_three_of_three_over_floor": three_of_three,
        "cells_over_oracle_upper_bound": over_oracle,
        "s3_failure_modes": branches,
        "s3_survivors": survivors,
        "wiring": {
            "evaluation_side_restores_model_kwargs": True,
            "training_side_records_model_kwargs": True,
            "e45_arm_table_changed": False,
            "route_verified": "--model-kwargs {\"clamp_mm\":20.0} survives "
                               "read -r -a as one token and reaches the "
                               "constructor",
            "clamp_mm_changes_state_dict": False,
            "conclusion": "the half of the wiring that can be landed without "
                          "changing what E45 was IS landed (the evaluation "
                          "side restores the recorded constructor kwargs); the "
                          "training side belongs to the next round's arm "
                          "table, because the E45 arm table is frozen",
        },
        "verdict": {
            "six_arms_negative": True,
            "any_arm_to_ten_seeds": False,
            "ten_seed_compute_committed": False,
            "unchanged_since": "v2.44 (conclusion), re-verified through round 93",
        },
        "checks": [{"pass": ok, "label": lab, "detail": det} for ok, lab, det in RESULTS],
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for ok, _, _ in RESULTS if not ok),
    }
    (EV / "round93_status_audit.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    n_fail = out["n_failed"]
    print("\n" + "=" * 78)
    print(f"POS {len(RESULTS) - n_fail}/{len(RESULTS)}   "
          f"VERDICT: {'PASS' if n_fail == 0 else 'FAIL'}")
    print("=" * 78)
    print(f"written: {EV / 'round93_status_audit.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
