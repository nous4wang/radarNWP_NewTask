#!/usr/bin/env python3
"""F38 verdict reader: architecture-matched control + 4 seeds (10..13).

Pre-registered constants (frozen in MASTER_CONTROL_ROOM.md F38 (2) BEFORE
any GPU time is spent; no CLI flags to relax them):

    FLOOR        = 0.002   (F30 nondeterminism floor)
    MEAN_TARGET  = 0.005   (4-seed mean & median threshold = 2.5x FLOOR)
    SLOPE_MIN    = 0.005   (late minus early accumulation slope threshold)
    SEEDS        = (10, 11, 12, 13)
    CONTROL      = "no_nwp_matched" (85,019 params, identical state_dict keys
                   to src_only, with all-zero NWP via --no-nwp)
    ARMS         = ("src_only", "evol_both")

Per-arm M_4S pass criteria (all 4 must hold):
    G0_4S     4-seed mean of Δ_early (Lead 0..2) >= -FLOOR
    G1_4S     >= 3/4 seeds have Δ_late (Lead 5..9) > +FLOOR, AND
              4-seed mean(Δ_late) > MEAN_TARGET, AND median(Δ_late) > MEAN_TARGET
    G2_4S     >= 3/4 seeds have Δ_all (pooled) > +FLOOR, AND
              4-seed mean(Δ_all) > MEAN_TARGET, AND median(Δ_all) > MEAN_TARGET
    G3_SLOPE  4-seed mean(Δ_late) - mean(Δ_early) >= SLOPE_MIN

Terminal verdict (wording capped by F37 (4)-6 / F38 (4)):
    POSITIVE_4S_REPRODUCED  at least one arm passes M_4S
    NOT_REPRODUCED_4S       no arm passes M_4S

Exit codes: 0 = verdict produced, 2 = data not ready, 3 = INVALID gate.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from summarize_curve_statistics import load as load_runs  # noqa: E402

# --- Pre-registered constants; do not edit after seeing the results. -------
FLOOR: float = 0.002
MEAN_TARGET: float = 0.005
SLOPE_MIN: float = 0.005
SEEDS: tuple[int, ...] = (10, 11, 12, 13)
REUSABLE_F37_ARMS: tuple[str, ...] = ("src_only", "evol_both")
REUSABLE_F37_SEEDS: tuple[int, ...] = (10, 11)
ARMS: tuple[str, ...] = ("src_only", "evol_both")
CONTROL: str = "no_nwp_matched"
ALL_ARMS: tuple[str, ...] = (*ARMS, CONTROL)
N_EPOCHS: int = 10
N_LEADS: int = 10
EXPECTED_GEOMETRY: dict[str, int] = {"patch_size": 64, "slide_stride": 64}
EXPECTED_N_SAMPLES: int = 538
RUN_TOKEN_F38: str = "f38_nwp"
RUN_TOKEN_F37: str = "f37_nwp"
DEFAULT_ROOT: Path = ROOT / "artifacts" / "f38_nwp" / "runs" / "f38_nwp"


def read_lead_csi2(report: dict[str, Any]) -> list[float] | None:
    """Read the 10-lead CSI@2.0 vector from the REAL evaluator report."""
    lc = report.get("lead_csi")
    if not isinstance(lc, dict):
        return None
    out: list[float] = []
    for k in range(N_LEADS):
        row = lc.get(str(k))
        if not isinstance(row, dict):
            return None
        value = row.get("2.0")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        out.append(float(value))
    return out


def _preconditions(summary: dict, label: str, allow_f37_token: bool) -> list[str]:
    problems: list[str] = []
    if summary.get("geometry") != EXPECTED_GEOMETRY:
        problems.append(
            f"{label}: geometry {summary.get('geometry')} != {EXPECTED_GEOMETRY}")
    ckpt = str(summary.get("checkpoint_dir", ""))
    allowed_tokens = (RUN_TOKEN_F38, RUN_TOKEN_F37) if allow_f37_token else (RUN_TOKEN_F38,)
    if not any(tok in ckpt for tok in allowed_tokens):
        problems.append(
            f"{label}: checkpoint_dir {ckpt!r} does not match allowed tokens {allowed_tokens}")
    rows = [r for r in summary.get("rows", []) if not r.get("is_last")]
    if len(rows) != N_EPOCHS:
        problems.append(f"{label}: {len(rows)} epoch rows != {N_EPOCHS}")
    counts = {r.get("n_samples") for r in rows}
    if counts != {EXPECTED_N_SAMPLES}:
        problems.append(
            f"{label}: n_samples {sorted(map(str, counts))} != "
            f"[{EXPECTED_N_SAMPLES}] -- a different stitching geometry")
    if any(not isinstance(r.get("csi_2.0"), (int, float))
           or not math.isfinite(float(r["csi_2.0"])) for r in rows):
        problems.append(f"{label}: a scored row has no finite CSI@2.0")
    return problems


def load_run_data(
    root: Path,
    reuse_f37_root: Path | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Load 12 summaries + 120 per-lead reports, with hard INVALID gates."""
    if not root.is_dir():
        print(f"WAITING: no pulled results at {root}")
        raise SystemExit(2)

    runs_f38 = load_runs(root)
    runs_f37 = load_runs(reuse_f37_root) if (reuse_f37_root and reuse_f37_root.is_dir()) else {}

    missing: list[str] = []
    resolved_sources: dict[tuple[str, int], tuple[Path, dict, bool]] = {}

    for seed in SEEDS:
        for arm in ALL_ARMS:
            can_reuse = (
                reuse_f37_root is not None
                and arm in REUSABLE_F37_ARMS
                and seed in REUSABLE_F37_SEEDS
            )
            if seed in runs_f38.get(arm, {}):
                # Present in root (may be a copied F37 curve if allow_f37_token is True)
                resolved_sources[(arm, seed)] = (root, runs_f38[arm][seed], can_reuse)
            elif can_reuse and seed in runs_f37.get(arm, {}):
                resolved_sources[(arm, seed)] = (reuse_f37_root, runs_f37[arm][seed], True)
            else:
                missing.append(f"{arm}/seed{seed}")

    if missing:
        print("WAITING: missing curve summaries: " + ", ".join(missing))
        raise SystemExit(2)

    problems: list[str] = []
    seen_ckpt: set[str] = set()
    data: dict[str, dict[int, dict[str, Any]]] = {a: {} for a in ALL_ARMS}

    for arm in ALL_ARMS:
        for seed in SEEDS:
            label = f"{arm}/seed{seed}"
            src_root, summary, allow_f37_token = resolved_sources[(arm, seed)]
            problems += _preconditions(summary, label, allow_f37_token=allow_f37_token)
            ckpt = str(summary.get("checkpoint_dir", ""))
            if ckpt in seen_ckpt:
                problems.append(f"{label}: duplicate checkpoint_dir {ckpt!r}")
            seen_ckpt.add(ckpt)

            rows = [r for r in summary["rows"] if not r.get("is_last")]
            pooled = [float(r["csi_2.0"]) for r in rows
                      if isinstance(r.get("csi_2.0"), (int, float))]

            cdir = src_root / arm / f"seed{seed}" / "curve"
            epoch_files = sorted(
                p for p in cdir.glob("qlfn2_epoch_*.json")
                if "epoch_curve" not in p.name and "last" not in p.name
            )
            if len(epoch_files) != N_EPOCHS:
                problems.append(
                    f"{label}: {len(epoch_files)} per-lead reports != {N_EPOCHS}; "
                    f"G0_4S/G1_4S cannot be evaluated without them")
            lead_matrix: list[list[float]] = []
            for p in epoch_files:
                try:
                    rep = json.loads(p.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    problems.append(f"{label}: cannot read {p.name}: {exc}")
                    continue
                vec = read_lead_csi2(rep)
                if vec is None:
                    problems.append(
                        f"{label}: {p.name} has no complete 10-lead "
                        f"lead_csi[k]['2.0'] vector")
                else:
                    lead_matrix.append(vec)

            lead_medians = None
            if len(lead_matrix) == N_EPOCHS:
                lead_medians = [
                    statistics.median(lead_matrix[e][k] for e in range(N_EPOCHS))
                    for k in range(N_LEADS)
                ]

            data[arm][seed] = {
                "median": statistics.median(pooled) if len(pooled) == N_EPOCHS else None,
                "peak": float(summary.get("peak_csi2", float("nan"))),
                "lead_medians": lead_medians,
                "reused_from_f37": (RUN_TOKEN_F37 in ckpt),
            }

    if problems:
        for p in problems:
            print(f"PRECONDITION VIOLATED  {p}")
        print("\nVERDICT: INVALID -- fix the harness; do NOT interpret the "
              "numbers.")
        raise SystemExit(3)
    return data


def evaluate_criteria(data: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Apply the frozen M_4S criteria across 4 seeds.  No silent fallback."""
    arm_evals: dict[str, Any] = {}
    any_pass = False

    for arm in ARMS:
        d_all: list[float] = []
        d_peak: list[float] = []
        d_early: list[float] = []
        d_late: list[float] = []
        d_lead0: list[float] = []

        for seed in SEEDS:
            base, cur = data[CONTROL][seed], data[arm][seed]
            d_all.append(cur["median"] - base["median"])
            d_peak.append(cur["peak"] - base["peak"])
            diffs = [cur["lead_medians"][k] - base["lead_medians"][k]
                     for k in range(N_LEADS)]
            d_lead0.append(diffs[0])
            d_early.append(statistics.mean(diffs[:3]))   # Lead 0..2
            d_late.append(statistics.mean(diffs[5:]))    # Lead 5..9

        mean_early = statistics.mean(d_early)
        mean_late = statistics.mean(d_late)
        med_late = statistics.median(d_late)
        mean_all = statistics.mean(d_all)
        med_all = statistics.median(d_all)
        slope = mean_late - mean_early

        n_pos_late = sum(1 for x in d_late if x > FLOOR)
        n_pos_all = sum(1 for x in d_all if x > FLOOR)

        g0 = mean_early >= -FLOOR
        g1 = (n_pos_late >= 3) and (mean_late > MEAN_TARGET) and (med_late > MEAN_TARGET)
        g2 = (n_pos_all >= 3) and (mean_all > MEAN_TARGET) and (med_all > MEAN_TARGET)
        g3 = slope >= SLOPE_MIN
        pass_m4s = g0 and g1 and g2 and g3
        if pass_m4s:
            any_pass = True

        arm_evals[arm] = {
            "delta_all": d_all,
            "delta_peak": d_peak,
            "delta_early_L0_2": d_early,
            "delta_late_L5_9": d_late,
            "delta_lead0": d_lead0,
            "mean_all": mean_all,
            "median_all": med_all,
            "mean_early_L0_2": mean_early,
            "mean_late_L5_9": mean_late,
            "median_late_L5_9": med_late,
            "slope_late_minus_early": slope,
            "n_seeds_pos_late": n_pos_late,
            "n_seeds_pos_all": n_pos_all,
            "G0_4S_zero_harm_early": g0,
            "G1_4S_positive_late": g1,
            "G2_4S_positive_all": g2,
            "G3_SLOPE_accumulating": g3,
            "pass_M_4S": pass_m4s,
        }

    if any_pass:
        direction = "POSITIVE_4S_REPRODUCED"
        dir_text = ("PASS: 在架构严格同构对照 (no_nwp_matched) 与 4 seeds 下，"
                    "本配方在远时效 (Lead 5..9) 及全域复现出可分辨且随预报时效累积的正增量")
    else:
        direction = "NOT_REPRODUCED_4S"
        dir_text = ("NOT_REPRODUCED: 在架构严格同构对照与 4 seeds 下未达到 M_4S 一致性门槛；"
                    "F37 在 2 seeds 下的观测值相对其自身对照仍成立，但归因强度与跨种子稳健性受限")

    return {
        "direction": direction,
        "direction_text": dir_text,
        "floor": FLOOR,
        "mean_target": MEAN_TARGET,
        "slope_min": SLOPE_MIN,
        "seeds": list(SEEDS),
        "control": CONTROL,
        "arms": arm_evals,
    }


def _run_self_test() -> int:
    def _mk(d_early_list, d_late_list, d_all_list):
        out: dict[str, dict[int, dict[str, Any]]] = {a: {} for a in ALL_ARMS}
        for i, s in enumerate(SEEDS):
            base_leads = [0.22 - 0.015 * k for k in range(N_LEADS)]
            out[CONTROL][s] = {"median": 0.135, "peak": 0.160,
                               "lead_medians": base_leads}
            for arm in ARMS:
                leads = [
                    base_leads[k] + (d_early_list[i] if k < 3
                                     else (d_late_list[i] if k >= 5 else 0.005))
                    for k in range(N_LEADS)
                ]
                out[arm][s] = {"median": 0.135 + d_all_list[i],
                               "peak": 0.160 + d_all_list[i],
                               "lead_medians": leads}
        return out

    cases = [
        ("4/4 positive", _mk([.001, .002, .000, .001], [.020, .018, .015, .022], [.012, .010, .009, .014]), "POSITIVE_4S_REPRODUCED"),
        ("3/4 positive", _mk([.000, -.001, .001, .000], [.018, .015, .001, .016], [.010, .009, .001, .011]), "POSITIVE_4S_REPRODUCED"),
        ("only 2/4 positive", _mk([.0, .0, .0, .0], [.020, .018, -.005, -.004], [.015, .012, -.003, -.002]), "NOT_REPRODUCED_4S"),
        ("flat slope rejected", _mk([.015, .015, .015, .015], [.016, .016, .016, .016], [.015, .015, .015, .015]), "NOT_REPRODUCED_4S"),
        ("G0 early harm rejected", _mk([-.010, -.008, -.005, -.006], [.020, .020, .020, .020], [.010, .010, .010, .010]), "NOT_REPRODUCED_4S"),
        ("below mean_target rejected", _mk([.0, .0, .0, .0], [.003, .004, .003, .004], [.003, .003, .003, .003]), "NOT_REPRODUCED_4S"),
    ]
    bad = 0
    for name, data, want in cases:
        got = evaluate_criteria(data)["direction"]
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {name}: {got} (want {want})")

    import tempfile
    good = {"lead_csi": {str(k): {"2.0": 0.25 - 0.02 * k} for k in range(10)}}
    ok7 = read_lead_csi2(good) is not None and read_lead_csi2({"lead_csi": {"2.0": [0.1] * 10}}) is None
    bad += not ok7
    print(f"{'ok  ' if ok7 else 'FAIL'}  lead_csi parser accepts real layout / rejects guessed layout")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for arm in ALL_ARMS:
            for seed in SEEDS:
                cdir = root / arm / f"seed{seed}" / "curve"
                cdir.mkdir(parents=True)
                rows = [{"epoch": e, "is_last": False, "n_samples": 538,
                         "csi_2.0": 0.13 + 0.001 * e} for e in range(10)]
                rows.append({"epoch": None, "is_last": True, "n_samples": 538,
                             "csi_2.0": 0.12})
                tok = "f37_nwp" if (arm in REUSABLE_F37_ARMS and seed in REUSABLE_F37_SEEDS) else "f38_nwp"
                (cdir / "qlfn2_epoch_curve.json").write_text(json.dumps({
                    "geometry": {"patch_size": 64, "slide_stride": 64},
                    "checkpoint_dir": f"/home/wf/radarNWPQML/runs/{tok}/{arm}/seed{seed}/train/x",
                    "rows": rows, "peak_csi2": 0.139,
                }), encoding="utf-8")
                for e in range(10):
                    (cdir / f"qlfn2_epoch_{e:02d}.json").write_text(
                        json.dumps(good), encoding="utf-8")

        # 8. without --reuse-f37-root, f37 token in root must be rejected (rc=3)
        try:
            load_run_data(root, reuse_f37_root=None)
            ok8 = False
        except SystemExit as exc:
            ok8 = exc.code == 3
        bad += not ok8
        print(f"{'ok  ' if ok8 else 'FAIL'}  f37 token rejected when reuse is not enabled (rc=3)")

        # 9. with reuse_f37_root=root, all 12 runs load cleanly
        try:
            loaded = load_run_data(root, reuse_f37_root=root)
            ok9 = all(loaded[a][s]["lead_medians"] is not None
                      for a in ALL_ARMS for s in SEEDS)
        except SystemExit:
            ok9 = False
        bad += not ok9
        print(f"{'ok  ' if ok9 else 'FAIL'}  load_run_data reads all 12 runs with explicit F37 reuse")

        # 10. remove one per-lead report -> must be INVALID (rc=3)
        (root / CONTROL / "seed12" / "curve" / "qlfn2_epoch_09.json").unlink()
        try:
            load_run_data(root, reuse_f37_root=root)
            ok10 = False
        except SystemExit as exc:
            ok10 = exc.code == 3
        bad += not ok10
        print(f"{'ok  ' if ok10 else 'FAIL'}  missing per-lead report -> INVALID (rc=3), no fallback")

    total = 10
    print(f"self-test: {total - bad}/{total} branches passed")
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="F38 预注册 4-seed 判据裁决器")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--reuse-f37-root",
        type=Path,
        default=None,
        help="Explicitly allow reusing F37 s10/s11 curves for src_only and evol_both from this root",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(_run_self_test())

    data = load_run_data(args.root, reuse_f37_root=args.reuse_f37_root)
    verdict = evaluate_criteria(data)
    verdict["root"] = str(args.root)
    verdict["reuse_f37_root"] = str(args.reuse_f37_root) if args.reuse_f37_root else None

    print(f"root          : {args.root}")
    print(f"reuse_f37_root: {args.reuse_f37_root}")
    print(f"criterion     : floor={FLOOR}, mean_target={MEAN_TARGET}, slope_min={SLOPE_MIN}, seeds={SEEDS}")
    for arm in ARMS:
        ev = verdict["arms"][arm]
        print(
            f"{arm:<10} | mean_Δ_all={ev['mean_all']:+.4f} (med={ev['median_all']:+.4f}, pos={ev['n_seeds_pos_all']}/4, G2={ev['G2_4S_positive_all']}) "
            f"| mean_Δ_early={ev['mean_early_L0_2']:+.4f} (G0={ev['G0_4S_zero_harm_early']}) "
            f"| mean_Δ_late={ev['mean_late_L5_9']:+.4f} (med={ev['median_late_L5_9']:+.4f}, pos={ev['n_seeds_pos_late']}/4, G1={ev['G1_4S_positive_late']}) "
            f"| slope={ev['slope_late_minus_early']:+.4f} (G3={ev['G3_SLOPE_accumulating']}) -> pass={ev['pass_M_4S']}"
        )
    print(f"VERDICT [{verdict['direction']}]: {verdict['direction_text']}")
    print("Reminder: even if M_4S passes, only state 'a distinguishable late-lead "
          "positive increment is reproduced under this 4-seed recipe'.")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(verdict, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
