"""Measure what the advection model actually does with the physics.

The score says whether the model is better. It does not say whether the
model is using the advection at all. The model records two numbers on
every forward pass -- the mean gate weight and the mass ratio -- and
nothing in the evaluation writes them down, so a reader of the score
cannot tell a model that advects from one that has learned to set its
gate to zero and ignore the flow.

This measures the two numbers directly. For each checkpoint it loads the
model, runs it over a number of batches of the test split, and averages
what the model recorded.

The gate weight is the fraction of the forecast that comes from the
advection rather than from the residual head. A gate that sits at zero
means the model has decided that the flow is not worth using, and the
score of that model has to be read as the score of the residual arm. A
mass ratio near one means the advection kept the rain; a ratio below one
means the bilinear resampling lost it, which is what the Oracle
measured on the analytic flow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from radar_nwp_bench.config import load_config, model_kwargs  # noqa: E402
from radar_nwp_bench.data import (  # noqa: E402
    NWPAlignment,
    RadarNWPDataset,
    build_dataloader,
)
from radar_nwp_bench.models import build_model  # noqa: E402


def build_dataset(args, nwp_vars, radar_vars):
    kwargs = dict(
        packed=args.packed,
        in_len=args.in_len,
        out_len=args.out_len,
        patch_size=args.patch_size,
        alignment=NWPAlignment(args.alignment),
        random_crop=False,
        split=None if args.split == "all" else args.split,
        max_samples=args.max_samples,
        slide_stride=args.slide_stride,
        blank_nwp=(args.no_nwp),
        tolerate_one_missing_frame=not args.no_interpolate,
    )
    if nwp_vars:
        kwargs["nwp_vars"] = tuple(nwp_vars)
    if radar_vars:
        kwargs["radar_vars"] = tuple(radar_vars)
    return RadarNWPDataset(args.data_root, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="the directory holding the arm directories")
    ap.add_argument("--model", default="simvp_v2_lagrangian")
    ap.add_argument("--data-root", default="/home/wf/data/radarNWP/BJ_mmap")
    ap.add_argument("--split", default="test")
    ap.add_argument("--packed", action="store_true", default=True)
    ap.add_argument("--no-packed", dest="packed", action="store_false")
    ap.add_argument("--in-len", type=int, default=None,
                    help="the history length; by default it is read from "
                         "the runs' own configurations, because a model "
                         "built for a different length will not load its "
                         "weights")
    ap.add_argument("--out-len", type=int, default=None,
                    help="the forecast length; by default it is read from "
                         "the runs' own configurations")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--slide-stride", type=int, default=64)
    ap.add_argument("--alignment", default="forecast")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--no-nwp", action="store_true")
    ap.add_argument("--no-interpolate", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batches", type=int, default=6,
                    help="how many batches to average over")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--nwp-vars", nargs="*", default=None)
    ap.add_argument("--radar-vars", nargs="*", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    device = torch.device(args.device)

    # The variable sets come from the training run's own record, not
    # from a guess: they decide the channel count of the model's input,
    # and a wrong guess builds a model whose first layer does not match
    # the checkpoint. The first arm's record is read before the
    # dataset is built, because the dataset is built from them.
    arms = sorted(p for p in root.glob("*_s*") if p.is_dir())
    if not arms:
        print(f"no arm directories under {root}")
        return 1
    first_config = None
    for run_dir in sorted(arms[0].glob("*/")):
        cfg_path = run_dir / "config.json"
        if cfg_path.is_file():
            first_config = json.loads(cfg_path.read_text(encoding="utf-8"))
            break
    if first_config is None:
        print(f"no readable configuration under {arms[0]}")
        return 1
    nwp_vars = args.nwp_vars or first_config.get("nwp_vars")
    radar_vars = args.radar_vars or first_config.get("radar_vars")
    # The lengths come from the runs' own record as well. The model is
    # built for a history of a given length and a forecast of a given
    # length, and the output head is shaped by the second of them, so a
    # model built for the wrong one will not load its weights at all.
    if args.in_len is None:
        args.in_len = int(first_config.get("in_len", 11))
    if args.out_len is None:
        args.out_len = int(first_config.get("out_len", 10))
    if isinstance(nwp_vars, str):
        nwp_vars = [nwp_vars]
    if isinstance(radar_vars, str):
        radar_vars = [radar_vars]
    print(f"the training used {len(radar_vars or [])} radar variables "
          f"and {len(nwp_vars or [])} NWP variables")

    dataset = build_dataset(args, nwp_vars, radar_vars)
    print(f"the split holds {len(dataset)} samples")

    # The loader yields the same dictionaries the evaluation uses.
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        seed=0,
    )

    results = []
    print(f"{len(arms)} arms under {root}")
    for arm_dir in arms:
        run_dirs = sorted(p for p in arm_dir.glob("*/") if (p / "config.json").is_file())
        for run_dir in run_dirs:
            ckpts = sorted(run_dir.glob("epoch_*.pt"))
            if not ckpts:
                continue
            ckpt = ckpts[0]
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            kwargs_raw = cfg.get("model_kwargs") or {}
            if isinstance(kwargs_raw, str):
                try:
                    kwargs_raw = json.loads(kwargs_raw)
                except json.JSONDecodeError:
                    kwargs_raw = {}

            model_cfg = cfg
            if not model_cfg.get("model"):
                try:
                    model_cfg = load_config(cfg.get("model", args.model))
                except FileNotFoundError:
                    model_cfg = {}
            kwargs = model_kwargs(
                model_cfg,
                shapes={
                    "in_radar_c": len(dataset.radar_vars),
                    "in_nwp_c": len(dataset.nwp_vars),
                    # This run's own lengths, not the ones the dataset was
                    # built with: they decide the shape of the output
                    # head, and a model built for a different forecast
                    # length will not load its weights.
                    "in_len": int(cfg.get("in_len", args.in_len)),
                    "out_len": int(cfg.get("out_len", args.out_len)),
                },
                overrides=kwargs_raw,
            )
            model = build_model(args.model, **kwargs).to(device)
            model.nwp_time_len = dataset.nwp_frames
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
            model.eval()

            gates: list[float] = []
            masses: list[float] = []
            with torch.no_grad():
                for bi, batch in enumerate(loader):
                    if bi >= args.batches:
                        break
                    x_radar = batch["x_radar"].to(device)
                    x_nwp = batch["x_nwp"].to(device)
                    model(x_radar, x_nwp)
                    gate = getattr(model, "last_gate_mean", None)
                    mass = getattr(model, "last_mass_ratio", None)
                    if gate is not None:
                        gates.append(float(gate))
                    if mass is not None:
                        masses.append(float(mass))

            record = {
                "arm": arm_dir.name,
                "run_dir": run_dir.name,
                "checkpoint": str(ckpt),
                "model_kwargs": kwargs_raw,
                "n_batches": len(gates),
                "gate_mean": float(np.mean(gates)) if gates else None,
                "gate_min": float(np.min(gates)) if gates else None,
                "gate_max": float(np.max(gates)) if gates else None,
                "mass_ratio_mean": float(np.mean(masses)) if masses else None,
                "mass_ratio_min": float(np.min(masses)) if masses else None,
                "mass_ratio_max": float(np.max(masses)) if masses else None,
            }
            results.append(record)
            print(
                f"  {arm_dir.name:<12} gate {record['gate_mean']:.4f} "
                f"[{record['gate_min']:.4f}, {record['gate_max']:.4f}]   "
                f"mass ratio {record['mass_ratio_mean']:.4f}"
            )
            del model

    print()
    print("=" * 92)
    print("the gate weight and the mass ratio, per arm")
    print("=" * 92)
    print()
    by_arm: dict[str, list[dict]] = {}
    for r in results:
        by_arm.setdefault(r["arm"], []).append(r)
    print(f"  {'arm':<12} {'n':<4} {'gate mean':>12} {'gate range':>20} "
          f"{'mass ratio':>12}")
    for arm, records in sorted(by_arm.items()):
        gates = [r["gate_mean"] for r in records if r["gate_mean"] is not None]
        masses = [r["mass_ratio_mean"] for r in records
                  if r["mass_ratio_mean"] is not None]
        if not gates:
            print(f"  {arm:<12} {len(records):<4} (the model records no gate)")
            continue
        print(
            f"  {arm:<12} {len(records):<4} {np.mean(gates):>12.4f} "
            f"{f'[{min(gates):.4f}, {max(gates):.4f}]':>20} "
            f"{np.mean(masses) if masses else float('nan'):>12.4f}"
        )
    print()

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps({"records": results}, indent=2), encoding="utf-8"
        )
        print(f"written to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
