#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round 58 -- training-dynamics axis (FIRST reading of the 21 train logs).

Why this axis exists
--------------------
Rounds 40..57 read fourteen layers of the EVALUATION products (aggregated
CSI/BIAS, mechanism probe, S1 structural quantities, window level, storm-
event level, lead axis, severity/difficulty axis, error composition, arm
omnibus, dose-response, metric families, guard calibration, gain-pattern
stability, aggregation operator + evaluation contract).  NOT ONE of them
ever opened the 21 TRAINING logs -- the only artifact of the experiment
that records what happened WHILE the model was being fitted.  Those logs
hold the per-epoch loss trajectory, the per-epoch wall clock, the model
size, the dataset size, the arm identity as the job actually saw it, the
S0-equivalence verdict as the job actually computed it, and -- for the two
topology arms -- the mechanism's OWN internal readouts (topo_loss,
topo_cldice, topo_pred_area, topo_truth_area, topo_pred_soft_area).

This axis matters because two of the three open implementation defects
are about what training did to the model:
  * TODO (i)   the 1e-12 guard in the amount channel (mean(F_mm)==0 frames)
  * TODO (ii)  the amplitude clamp that never reached the output field
and because E45's epoch choice ("ep00 is the peak, last.pt is the
collapse end") is a claim about a quantity that only the SKILL side can
confirm -- the training loss may or may not be able to witness it.

Everything here is READ-ONLY and LOCAL.  No remote call, no sbatch, no
write to docs/.

Implementation notes
--------------------
Pure standard library on purpose: the local interpreter has no numpy, and
a round script that needs a scientific stack is a round script that
cannot be re-run by the next session.  Correlation, median and a sign
test are all that is needed.

Idempotency: writes exactly one artifact, tools/_evals/e45/
round58_training_dynamics.json, and never touches any other file.
"""
import io
import json
import math
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(REPO, "tools", "_evals", "e45", "_train_logs",
                      "run_logs", "slurm")
OUTPATH = os.path.join(REPO, "tools", "_evals", "e45",
                        "round58_training_dynamics.json")
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]

# --------------------------------------------------------------- helpers
RE_HDR = re.compile(r"arm\s*:\s*(\S+)\s+model:\s*(\S+)\s+loss:\s*(\S+)\s+seed:\s*(\d+)")
RE_TOPO_HDR = re.compile(r"topo flags\s*:\s*(.*?)\s*$")
RE_EPOCHS = re.compile(r"epochs\s*:\s*(\d+)")
RE_OUTDIR = re.compile(r"out\s*:\s*(\S+)")
RE_DATASET = re.compile(r"dataset:\s*(\d+)\s+samples")
RE_PARAMS = re.compile(r"model:\s*(\S+)\s+params:\s*([\d,]+)")
RE_S0 = re.compile(r"^(\w+):\s+max_abs_diff\s*=\s*([0-9.eE+-]+)\s+bitwise=(\w+)\s+(\w+)\s*$")
RE_STEP = re.compile(r"^\s*epoch\s+(\d+)\s+step\s+(\d+)\s+loss=([0-9.eE+-]+)(.*)$")
RE_EPOCH_SUM = re.compile(r"^epoch\s+(\d+):\s+loss=([0-9.eE+-]+)\s+\(([0-9.]+)s\)")
RE_KEPT = re.compile(r"kept\s+(\S+)/epoch_(\d+)\.pt")
RE_CKPTS = re.compile(r"checkpoints written:\s*(\d+)")
RE_ART = re.compile(r"artifacts written to\s+(\S+)")
RE_TOPO_FIELD = re.compile(r"(topo_\w+)=([0-9.eE+-]+)")
RE_NAN = re.compile(r"\bnan\b|\binf\b", re.IGNORECASE)


def read(path):
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def walk(node, key=None):
    """Yield (key, value) for every scalar in a nested JSON structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            for item in walk(v, k):
                yield item
    elif isinstance(node, list):
        for v in node:
            for item in walk(v, key):
                yield item
    else:
        yield (key, node)


def pick_csi(pairs):
    """Pick the CSI-at-2.0mm number out of a flat (key, value) pair list.

    Structure-free on purpose: the published per-seed summaries have been
    renamed more than once, so the script must not hard-code a key name.
    Preference order: a key that mentions 'csi' AND a threshold of 2
    (optionally 'median'-prefixed), then any 'csi' key that is numeric.
    """
    cands = [(k, v) for k, v in pairs
             if k and "csi" in k.lower() and isinstance(v, (int, float))
             and not isinstance(v, bool)]
    two = [(k, v) for k, v in cands
           if re.search(r"(^|[^0-9])2(\.0)?([^0-9]|$)", k)]
    if two:
        pref = [kv for kv in two if "median" in kv[0].lower()]
        return float((pref or two)[0][1])
    if cands:
        return float(cands[0][1])
    return None


def find_arm_block(obj):
    """Locate the dict whose keys contain every arm name."""
    if isinstance(obj, dict):
        keys = set(obj.keys())
        if set(ARMS).issubset(keys):
            return obj
        for v in obj.values():
            got = find_arm_block(v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = find_arm_block(v)
            if got is not None:
                return got
    return None


def find_number(node, keyname):
    """Depth-first search for a numeric field named keyname."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == keyname and isinstance(v, (int, float)):
                return float(v)
        for v in node.values():
            got = find_number(v, keyname)
            if got is not None:
                return got
    elif isinstance(node, list):
        for v in node:
            got = find_number(v, keyname)
            if got is not None:
                return got
    return None


# ---------------------------------------------------------- arm table
def load_arm_table():
    path = os.path.join(REPO, "scripts", "_e45_arms.sh")
    txt = read(path)
    rows = []
    for m in re.finditer(r'"([^"|]*)\|([^"|]*)\|([^"|]*)\|([^"|]*)\|([^"]*)"', txt):
        arm, model, loss, train_flags, probe_flags = m.groups()
        if arm in ARMS:
            rows.append((arm, model, loss, train_flags.strip(), probe_flags.strip()))
    return rows


# ------------------------------------------------------- log parsing
def parse_train_log(path, fname):
    txt = read(path)
    lines = txt.splitlines()
    info = {"file": fname, "header_arm": None, "model": None, "loss": None,
            "seed": None, "topo_flags_hdr": None, "epochs": None,
            "out_dir": None, "dataset_samples": None, "params": None,
            "params_model": None, "step_loss": [], "step_epoch": [],
            "step_extra": [], "epoch_loss": {}, "epoch_sec": {},
            "kept": [], "ckpts": None, "artifacts_dir": None,
            "s0_lines": [], "s0_new_head_keys": {}, "s0_shared_keys": {},
            "s0_switches_off": {}, "s0_pass_count": 0, "s0_audit_ok_count": 0,
            "nan_lines": [], "in_training_section": False}

    for idx, line in enumerate(lines):
        m = RE_HDR.search(line)
        if m:
            info["header_arm"], info["model"], info["loss"], sd = m.groups()
            info["seed"] = int(sd)
            continue
        m = RE_TOPO_HDR.search(line)
        if m and info["header_arm"] is not None:
            info["topo_flags_hdr"] = m.group(1)
            continue
        m = RE_EPOCHS.search(line)
        if m:
            info["epochs"] = int(m.group(1))
            continue
        m = RE_OUTDIR.search(line)
        if m:
            info["out_dir"] = m.group(1)
            continue
        m = RE_DATASET.search(line)
        if m:
            info["dataset_samples"] = int(m.group(1))
            continue
        m = RE_PARAMS.search(line)
        if m:
            info["params_model"] = m.group(1)
            info["params"] = [int(x.replace(",", "")) for x in m.group(2).split(",")]
            continue
        m = RE_S0.match(line.strip())
        if m:
            model, diff, bitwise, verdict = m.groups()
            info["s0_lines"].append({"model": model, "max_abs_diff": float(diff),
                                     "bitwise": bitwise, "verdict": verdict})
            continue
        m = RE_EPOCH_SUM.match(line.strip())
        if m:
            e, loss, sec = m.groups()
            info["epoch_loss"][int(e)] = float(loss)
            info["epoch_sec"][int(e)] = float(sec)
            continue
        m = RE_STEP.match(line)
        if m:
            e, st, loss, extra = m.groups()
            info["step_epoch"].append(int(e))
            info["step_loss"].append(float(loss))
            info["step_extra"].append({k: float(v)
                                       for k, v in RE_TOPO_FIELD.findall(extra)})
            continue
        m = RE_KEPT.search(line)
        if m:
            info["kept"].append({"path": m.group(1), "epoch": int(m.group(2))})
            continue
        m = RE_CKPTS.search(line)
        if m:
            info["ckpts"] = int(m.group(1))
            continue
        m = RE_ART.search(line)
        if m:
            info["artifacts_dir"] = m.group(1)
            continue
        if line.strip().startswith("--- training"):
            info["in_training_section"] = True
            continue
        if '"pass": true' in line:
            info["s0_pass_count"] += 1
        if '"audit_ok": true' in line:
            info["s0_audit_ok_count"] += 1

    # The S0 report is printed as ONE JSON object whose "arms" map holds
    # one block per mechanism model.  Reading it as "the text that follows
    # each model line" is wrong: every model block lives inside that one
    # object, so the text-segment trick attributes all of them to the LAST
    # model name and the other four look absent.  Parse the JSON instead.
    def extract_json_blocks(text):
        blocks = []
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    blocks.append(text[start:i + 1])
                    start = None
        return blocks

    for blk in extract_json_blocks(txt):
        try:
            obj = json.loads(blk)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        arms_obj = obj.get("arms")
        if not isinstance(arms_obj, dict):
            continue
        info["s0_json_blocks"] = info.get("s0_json_blocks", 0) + 1
        for model, v in arms_obj.items():
            if not isinstance(v, dict):
                continue
            info.setdefault("s0_models", {})[model] = v
            if "n_new_head_keys" in v:
                info["s0_new_head_keys"][model] = int(v["n_new_head_keys"])
            if "n_shared_keys" in v:
                info["s0_shared_keys"][model] = int(v["n_shared_keys"])
            if "switches_off" in v:
                info["s0_switches_off"][model] = json.dumps(v["switches_off"])

    # NaN / inf anywhere in the training section
    in_tr = False
    for line in lines:
        if line.strip().startswith("--- training"):
            in_tr = True
        if in_tr and RE_NAN.search(line):
            info["nan_lines"].append(line.strip()[:120])
    return info


def parse_stage2_log(path, fname):
    txt = read(path)
    out = {"file": fname, "run_paths": [], "epoch_mentions": []}
    for m in re.finditer(r"/home/wf/radarNWP/runs/e45/[^\s'\"]+", txt):
        p = m.group(0)
        if p not in out["run_paths"]:
            out["run_paths"].append(p)
    for m in re.finditer(r"epoch_\d+\.pt", txt):
        if m.group(0) not in out["epoch_mentions"]:
            out["epoch_mentions"].append(m.group(0))
    return out


# ---------------------------------------------------------- main
def main():
    checks = []      # (name, ok, detail)
    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    table = load_arm_table()
    table_by_arm = {r[0]: r for r in table}
    check("arm_table_rows_7", len(table) == 7, "rows=%d" % len(table))

    # ---- parse the 21 training logs
    logs = {}
    for fname in sorted(os.listdir(LOGDIR)):
        if not (fname.startswith("e45-train-") and fname.endswith(".out")):
            continue
        idx = int(fname.split("_")[1].split(".")[0])
        info = parse_train_log(os.path.join(LOGDIR, fname), fname)
        info["array_idx"] = idx
        info["arm_from_table"] = table_by_arm[table[idx][0]][0] if idx < len(table) else None
        logs[(info["header_arm"], info["seed"])] = info

    check("train_logs_21", len(logs) == 21, "parsed=%d" % len(logs))
    check("cells_7x3", sorted(set(a for a, _ in logs)) == sorted(ARMS) and
          sorted(set(s for _, s in logs)) == SEEDS, "arms/seeds grid")

    # ---- 1. arm-table conformance from the job's own header
    bad = []
    for (arm, seed), info in sorted(logs.items()):
        row = table_by_arm[arm]
        exp_flags = row[3] if row[3] else "none"
        got_flags = info["topo_flags_hdr"]
        if got_flags is None:
            got_flags = "none"
        got_flags_norm = " ".join(got_flags.split())
        exp_flags_norm = " ".join(exp_flags.split())
        if (info["header_arm"] != row[0] or info["model"] != row[1]
                or info["loss"] != row[2] or got_flags_norm != exp_flags_norm
                or info["array_idx"] != ARMS.index(arm)):
            bad.append({"arm": arm, "seed": seed,
                        "got": [info["header_arm"], info["model"], info["loss"], got_flags_norm],
                        "exp": [row[0], row[1], row[2], exp_flags_norm]})
    check("arm_table_conformance_21", not bad, "mismatches=%d %s" % (len(bad), bad[:2]))

    # ---- 2. S0 equivalence gate as computed inside every job
    s0_total = 0
    s0_bad = []
    for (arm, seed), info in sorted(logs.items()):
        for rec in info["s0_lines"]:
            s0_total += 1
            if not (rec["max_abs_diff"] == 0.0 and rec["bitwise"] == "True"
                    and rec["verdict"] == "PASS"):
                s0_bad.append({"arm": arm, "seed": seed, **rec})
    check("s0_gate_105_checks", s0_total == 105 and not s0_bad,
          "checks=%d bad=%d" % (s0_total, len(s0_bad)))
    s0_json_pass = sum(info["s0_pass_count"] for info in logs.values())
    s0_audit = sum(info["s0_audit_ok_count"] for info in logs.values())
    s0_model_blocks = 0
    s0_model_bad = []
    miss_by_model = {}
    for (arm, seed), info in sorted(logs.items()):
        for model, v in sorted(info.get("s0_models", {}).items()):
            s0_model_blocks += 1
            # NOTE: missing_after_copy is NOT required to be empty.  A model
            # that adds a NEW head has keys the control checkpoint cannot
            # contain, so that list is non-empty BY CONSTRUCTION for the
            # four models that add one -- the gate's claim is bitwise
            # equality of the SHARED part, which is exactly what
            # max_abs_diff / bitwise_equal assert.
            ok = (v.get("max_abs_diff") == 0.0 and v.get("bitwise_equal") is True
                  and v.get("pass") is True and v.get("audit_ok") is True)
            if not ok:
                s0_model_bad.append({"arm": arm, "seed": seed, "model": model, **v})
            miss_by_model.setdefault(model, set()).add(
                (len(v.get("missing_after_copy") or []),
                 len(v.get("unexpected_after_copy") or []),
                 v.get("n_new_head_keys")))
    check("s0_json_105_model_blocks",
          s0_model_blocks == 105 and not s0_model_bad,
          "blocks=%d bad=%d (text pass=%d audit_ok=%d)"
          % (s0_model_blocks, len(s0_model_bad), s0_json_pass, s0_audit))
    # a model that adds NO key must have nothing missing and nothing extra
    zero_key_models = [m for m, s in miss_by_model.items()
                       if all(t[2] == 0 for t in s)]
    clean = all(t[0] == 0 and t[1] == 0 for m in zero_key_models
                for t in miss_by_model[m])
    check("zero_new_key_models_have_clean_state_transfer",
          bool(zero_key_models) and clean,
          "models=%s" % {m: sorted(miss_by_model[m]) for m in zero_key_models})

    # ---- 3. checkpoints / epochs
    ck_bad = []
    for (arm, seed), info in sorted(logs.items()):
        eps = sorted(e["epoch"] for e in info["kept"])
        if info["ckpts"] != 3 or eps != [0, 1, 2] or info["epochs"] != 3:
            ck_bad.append({"arm": arm, "seed": seed, "ckpts": info["ckpts"], "eps": eps})
    check("checkpoints_3_epochs_0_2", not ck_bad, "bad=%d %s" % (len(ck_bad), ck_bad[:2]))

    # ---- 4. loss trajectory
    traj = {}
    monotone = 0
    rising = 0
    for (arm, seed), info in sorted(logs.items()):
        losses = [info["epoch_loss"][e] for e in (0, 1, 2)]
        secs = [info["epoch_sec"][e] for e in (0, 1, 2)]
        steps = [info["step_loss"][i] for i in range(len(info["step_loss"]))
                 if info["step_epoch"][i] == 0]
        traj["%s_s%d" % (arm, seed)] = {
            "arm": arm, "seed": seed, "epoch_loss": losses, "epoch_sec": secs,
            "epoch_loss_step_median": [statistics.median(
                [info["step_loss"][i] for i in range(len(info["step_loss"]))
                 if info["step_epoch"][i] == e]) for e in (0, 1, 2)],
            "step_counts": {str(e): sum(1 for x in info["step_epoch"] if x == e)
                            for e in (0, 1, 2)},
            "loss_ratio_ep2_over_ep0": losses[2] / losses[0],
            "loss_delta_ep2_minus_ep0": losses[2] - losses[0],
            "argmin_epoch": min(range(3), key=lambda e: losses[e]),
            "total_sec": sum(secs),
            "n_steps": len(info["step_loss"]),
            "nan_lines": len(info["nan_lines"]),
        }
        if losses[2] < losses[1] < losses[0]:
            monotone += 1
        if losses[2] >= losses[0]:
            rising += 1
    check("loss_monotone_decreasing_21", monotone == 21, "monotone=%d/21" % monotone)
    check("loss_never_rises_21", rising == 0, "rising=%d/21" % rising)
    check("no_nan_in_training_21", all(v["nan_lines"] == 0 for v in traj.values()),
          "nan_total=%d" % sum(v["nan_lines"] for v in traj.values()))

    # ---- 5. dataset / model size
    ds = set(info["dataset_samples"] for info in logs.values())
    check("dataset_size_identical_21", len(ds) == 1, "sizes=%s" % sorted(ds))
    params_by_model = {}
    for info in logs.values():
        params_by_model.setdefault(info["params_model"], set()).add(tuple(info["params"]))
    flat = {k: sorted(v) for k, v in params_by_model.items()}
    check("params_unique_per_model", all(len(v) == 1 for v in flat.values()), json.dumps(flat))

    # ---- 6. new-head accounting (A4 vs A4b) from the S0 JSON
    nh = {}
    for (arm, seed), info in sorted(logs.items()):
        for model, n in info["s0_new_head_keys"].items():
            nh.setdefault(model, set()).add(n)
    nh_flat = {k: sorted(v) for k, v in nh.items()}
    a4 = nh_flat.get("scaled_structure")
    a4b = nh_flat.get("scaled_structure_own")
    topo_nh = nh_flat.get("topo_simvp")
    g1 = nh_flat.get("amp_monotone_gate")
    g2 = nh_flat.get("amp_hard_support")
    # MEASURED (21/21 logs agree): the topology arm adds NO parameter key
    # at all; the "own" amount channel adds the SAME two keys as the two
    # gate arms (a small learnable block, not a separate head); only the
    # dedicated-head arm adds a new module group.  So "A4b does not add a
    # head" is true at the MODULE level, but not at the PARAMETER level --
    # and the handover's "own = no new head" must be read that way.
    check("topo_arm_adds_no_parameters",
          topo_nh is not None and max(topo_nh) == 0, "topo_simvp=%s" % topo_nh)
    check("A4_has_more_new_keys_than_A4b",
          a4 is not None and a4b is not None and min(a4) > max(a4b) > 0,
          "A4=%s A4b=%s" % (a4, a4b))
    check("A4b_param_cost_equals_gate_arms",
          g1 is not None and g2 is not None and a4b is not None
          and max(a4b) == max(g1) == max(g2),
          "A4b=%s gates=%s/%s" % (a4b, g1, g2))
    own_params = flat.get("scaled_structure_own", [[]])[0]
    ctrl_params = flat.get("simvp_v2", [[]])[0]
    head_params = flat.get("scaled_structure", [[]])[0]
    topo_params = flat.get("topo_simvp", [[]])[0]
    check("A4b_no_new_module_group_A4_has_one",
          len(own_params) == 3 and len(ctrl_params) == 3
          and own_params[1] == ctrl_params[1]
          and head_params[1] > ctrl_params[1]
          and topo_params == ctrl_params,
          "ctrl=%s topo=%s own=%s A4=%s" % (ctrl_params, topo_params,
                                            own_params, head_params))

    # ---- 7. skill linkage: pull the published per-seed CSI@2.0
    #         The per-seed summaries are keyed "<ARM>_s<SEED>_ep00" and are
    #         CUMULATIVE (the s1 file repeats s0, the s2 file repeats s0
    #         and s1), so the seed must be taken from the KEY, never from
    #         the file name -- reading the s2 file as "seed 2 only" is a
    #         silent way to score the wrong cell.
    KEY_RX = re.compile(r"^(B0|A1|A2|A3a|A3b|A4|A4b)_s(\d+)(?:_ep\d+)?$")
    csi = {}
    for s in SEEDS:
        p = os.path.join(REPO, "tools", "_evals", "e45", "scores_summary_s%d.json" % s)
        obj = json.loads(read(p))
        found = 0
        for k, v in obj.items():
            m = KEY_RX.match(k)
            if not m or int(m.group(2)) != s:
                continue
            val = pick_csi(list(walk(v)))
            if val is not None:
                csi["%s_s%d" % (m.group(1), s)] = val
                found += 1
        check("scores_summary_s%d_parse" % s, found == 7, "cells=%d" % found)
    check("csi_21_cells", len(csi) == 21, "cells=%d" % len(csi))

    # published headline median must be reproducible from the per-seed files
    head = json.loads(read(os.path.join(REPO, "tools", "_evals", "e45",
                                         "scores_summary.json")))
    hblk = find_arm_block(head)
    if hblk is None:
        hblk = {}
        for k, v in head.items():
            m = re.match(r"^(B0|A1|A2|A3a|A3b|A4|A4b)", k)
            if m:
                hblk.setdefault(m.group(1), v)
    head_ok = 0
    head_bad = []
    for arm in ARMS:
        vals = [csi["%s_s%d" % (arm, s)] for s in SEEDS if "%s_s%d" % (arm, s) in csi]
        if len(vals) != 3 or arm not in hblk:
            continue
        pub = pick_csi(list(walk(hblk[arm])))
        if pub is None:
            continue
        if abs(statistics.median(vals) - pub) <= 1e-9:
            head_ok += 1
        else:
            head_bad.append({"arm": arm, "recomputed": statistics.median(vals), "published": pub})
    check("headline_median_recompute_7", head_ok == 7 and not head_bad,
          "ok=%d bad=%s" % (head_ok, head_bad[:2]))

    # ---- 8. loss <-> skill coupling
    keys = sorted(csi.keys())
    l0 = [traj[k]["epoch_loss"][0] for k in keys]
    l2 = [traj[k]["epoch_loss"][2] for k in keys]
    skill = [csi[k] for k in keys]
    ratio = [traj[k]["loss_ratio_ep2_over_ep0"] for k in keys]
    non_ctrl = [k for k in keys if not k.startswith("B0_")]
    r_loss_skill_all = pearson(l0, skill)
    r_loss_skill_arm = pearson([traj[k]["epoch_loss"][0] for k in non_ctrl],
                               [csi[k] for k in non_ctrl])
    r_ratio_skill_all = pearson(ratio, skill)
    r_ratio_skill_arm = pearson([traj[k]["loss_ratio_ep2_over_ep0"] for k in non_ctrl],
                                [csi[k] for k in non_ctrl])

    # ---- 9. arm-level medians
    arm_agg = {}
    for arm in ARMS:
        cells = [traj["%s_s%d" % (arm, s)] for s in SEEDS]
        arm_agg[arm] = {
            "median_csi_at_2.0": statistics.median([csi["%s_s%d" % (arm, s)] for s in SEEDS]),
            "csi_min": min(csi["%s_s%d" % (arm, s)] for s in SEEDS),
            "csi_max": max(csi["%s_s%d" % (arm, s)] for s in SEEDS),
            "floor_pass_count": sum(1 for s in SEEDS if csi["%s_s%d" % (arm, s)] >= 0.0924),
            "median_epoch_loss": [statistics.median([c["epoch_loss"][e] for c in cells])
                                  for e in (0, 1, 2)],
            "median_epoch_sec": [statistics.median([c["epoch_sec"][e] for c in cells])
                                 for e in (0, 1, 2)],
            "median_total_sec": statistics.median([c["total_sec"] for c in cells]),
            "median_loss_ratio_ep2_ep0": statistics.median(
                [c["loss_ratio_ep2_over_ep0"] for c in cells]),
            "argmin_epoch_per_seed": [c["argmin_epoch"] for c in cells],
            "paired_gain_vs_control": statistics.median(
                [csi["%s_s%d" % (arm, s)] - csi["B0_s%d" % s] for s in SEEDS]),
        }
    ctrl_sec = arm_agg["B0"]["median_epoch_sec"][0]
    for arm in ARMS:
        arm_agg[arm]["train_time_ratio_vs_control"] = (
            arm_agg[arm]["median_epoch_sec"][0] / ctrl_sec)

    # ---- 10. topology readouts (mechanism's own internal area bookkeeping)
    topo = {}
    for (arm, seed), info in sorted(logs.items()):
        if info["step_extra"] and any(info["step_extra"][0].keys()):
            pass
    for arm in ("A3a", "A3b"):
        for seed in SEEDS:
            info = logs[(arm, seed)]
            fields = {}
            for d in info["step_extra"]:
                for k, v in d.items():
                    fields.setdefault(k, []).append(v)
            if not fields:
                continue
            med = {k: statistics.median(v) for k, v in fields.items()}
            n = len(fields.get("topo_loss", []))
            if n == 0:
                continue
            ratio = (med["topo_pred_area"] / med["topo_truth_area"]
                     if med.get("topo_truth_area") else float("nan"))
            topo["%s_s%d" % (arm, seed)] = {
                "arm": arm, "seed": seed, "n_steps": n,
                "median": {k: med[k] for k in sorted(med)},
                "pred_over_truth_area": ratio,
                "note": ("A3b runs with --topo-area-normalize, so its area "
                         "quantities are normalised: the pred/truth ratio is "
                         "comparable WITHIN the arm, the absolute values are "
                         "NOT comparable with A3a"),
            }
    check("topo_readouts_6_cells", len(topo) == 6, "cells=%d" % len(topo))

    # ---- 11. provenance: stage2 log must point at the very run dir the
    #          training log wrote, and at epoch_00.pt
    stg = {}
    for fname in sorted(os.listdir(LOGDIR)):
        if fname.startswith("e45-stage2-") and fname.endswith(".out"):
            stg[fname] = parse_stage2_log(os.path.join(LOGDIR, fname), fname)
    prov = []
    for fname, rec in stg.items():
        for p in rec["run_paths"]:
            m = re.search(r"runs/e45/([A-Za-z0-9]+)_s(\d+)/train/(\w+)_(\d+)", p)
            if m:
                prov.append({"stage2": fname, "arm": m.group(1), "seed": int(m.group(2)),
                             "model": m.group(3), "stamp": m.group(4), "path": p})
    prov_ok = 0
    prov_bad = []
    for rec in prov:
        key = (rec["arm"], rec["seed"])
        if key not in logs:
            continue
        art = logs[key]["artifacts_dir"] or ""
        if rec["stamp"] in art:
            prov_ok += 1
        else:
            prov_bad.append(rec)
    check("provenance_stage2_matches_train_log", prov_ok >= 21 and not prov_bad,
          "matched=%d bad=%d" % (prov_ok, len(prov_bad)))

    # ---- 12. S3 guard (soft): the published verdict must not have moved
    guard = {"status": "SKIPPED", "intervals": 0, "crossing_zero": 0, "verdicts": {}}
    try:
        s3 = json.loads(read(os.path.join(REPO, "tools", "_evals", "e45",
                                          "s3_equalbias.json")))
        verts = {}
        for k, v in walk(s3):
            if isinstance(v, str) and "verdict" in (k or "").lower():
                verts[v] = verts.get(v, 0) + 1
        ivs = []
        flags = []

        def collect(node):
            if isinstance(node, dict):
                lo = hi = None
                for k, v in node.items():
                    if not isinstance(v, (int, float)) or isinstance(v, bool):
                        continue
                    kl = k.lower()
                    if re.search(r"(^|_)(ci_)?lo(w)?$", kl):
                        lo = v
                    elif re.search(r"(^|_)(ci_)?hi(g(h)?)?$", kl):
                        hi = v
                if lo is not None and hi is not None:
                    ivs.append((float(lo), float(hi)))
                cz = node.get("crosses_zero")
                if isinstance(cz, bool):
                    flags.append(cz)
                for v in node.values():
                    collect(v)
            elif isinstance(node, list):
                for v in node:
                    collect(v)
        collect(s3)
        sys.stderr.write("STRUCT s3_equalbias.json -> %s\n"
                         % json.dumps(s3, ensure_ascii=False)[:900])
        guard = {"status": "OK", "intervals": len(ivs),
                 "crossing_zero": sum(1 for lo, hi in ivs if lo <= 0.0 <= hi),
                 "n_published_flags": len(flags),
                 "published_flags_all_true": bool(flags) and all(flags),
                 "verdicts": verts}
    except Exception as exc:  # pragma: no cover
        guard["status"] = "ERROR: %s" % exc

    # hard guard: the twelve published S3 intervals must still all cross zero
    if guard.get("intervals", 0) >= 12:
        check("s3_12_intervals_all_cross_zero",
              guard["crossing_zero"] == 12 and guard["published_flags_all_true"],
              "intervals=%d crossing=%d published_flags_all_true=%s"
              % (guard["intervals"], guard["crossing_zero"],
                 guard["published_flags_all_true"]))
    else:
        check("s3_12_intervals_all_cross_zero", False,
              "intervals not recognised (%d)" % guard.get("intervals", 0))

    # ---- assemble
    n_ok = sum(1 for c in checks if c["ok"])
    result = {
        "round": 58,
        "axis": "training dynamics (21 Slurm training logs, first reading)",
        "verdict": "PASS" if n_ok == len(checks) else "FAIL",
        "checks_passed": n_ok,
        "checks_total": len(checks),
        "checks": checks,
        "guard_s3": guard,
        "provenance": {"stage2_files": sorted(stg.keys()), "records": prov},
        "arm_table": {a: {"model": t[1], "loss": t[2], "train_flags": t[3]}
                      for a, t in table_by_arm.items()},
        "arm_aggregate": arm_agg,
        "per_cell": traj,
        "skill_per_cell_csi_at_2.0": csi,
        "loss_skill_coupling": {
            "r_ep0loss_vs_csi_all21": r_loss_skill_all,
            "r_ep0loss_vs_csi_18arms": r_loss_skill_arm,
            "r_lossratio_vs_csi_all21": r_ratio_skill_all,
            "r_lossratio_vs_csi_18arms": r_ratio_skill_arm,
        },
        "topo_readouts": topo,
        "model_params": flat,
        "new_head_keys": nh_flat,
        "dataset_samples": sorted(ds),
        "s0_gate": {"total_checks": s0_total, "bad": s0_bad,
                    "json_pass_count": s0_json_pass, "audit_ok_count": s0_audit},
    }
    with io.open(OUTPATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))

    print("=" * 78)
    print("E45 round 58 -- training dynamics axis")
    print("=" * 78)
    for c in checks:
        print("[%s] %-38s %s" % ("PASS" if c["ok"] else "FAIL", c["name"], c["detail"]))
    print("-" * 78)
    print("arm   model                 loss      ep0loss ep1loss ep2loss  ratio  sec/ep  t/c   "
          "csi_med   csi_rng      gain_paired  floor")
    for arm in ARMS:
        a = arm_agg[arm]
        print("%-5s %-20s %-9s %7.4f %7.4f %7.4f %6.4f %7.1f %5.2f  %8.6f  %s  %d/3  %+.6f" % (
            arm, table_by_arm[arm][1], table_by_arm[arm][2],
            a["median_epoch_loss"][0], a["median_epoch_loss"][1], a["median_epoch_loss"][2],
            a["median_loss_ratio_ep2_ep0"], a["median_epoch_sec"][0],
            a["train_time_ratio_vs_control"], a["median_csi_at_2.0"],
            "%.6f-%.6f" % (a["csi_min"], a["csi_max"]), a["floor_pass_count"],
            a["paired_gain_vs_control"]))
    print("-" * 78)
    print("loss<->skill coupling : r(ep0loss,CSI) all21=%.4f  arms18=%.4f | "
          "r(ratio,CSI) all21=%.4f  arms18=%.4f"
          % (r_loss_skill_all, r_loss_skill_arm, r_ratio_skill_all, r_ratio_skill_arm))
    print("S3 guard               : %s" % json.dumps(guard, ensure_ascii=False))
    print("topo readouts          :")
    for k in sorted(topo):
        t = topo[k]
        print("   %-8s steps=%4d  pred/truth_area=%.4f  topo_loss=%.5f  cldice=%.5f" % (
            k, t["n_steps"], t["pred_over_truth_area"],
            t["median"].get("topo_loss", float("nan")),
            t["median"].get("topo_cldice", float("nan"))))
    print("-" * 78)
    print("per-cell ep0 loss vs CSI (21 cells):")
    for k in sorted(csi):
        print("   %-8s loss=%.4f  csi=%.6f  ratio=%.4f" % (
            k, traj[k]["epoch_loss"][0], csi[k], traj[k]["loss_ratio_ep2_over_ep0"]))
    print("-" * 78)
    print("VERDICT: %s  (%d/%d)" % (result["verdict"], n_ok, len(checks)))
    print("artifact: %s" % OUTPATH)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
