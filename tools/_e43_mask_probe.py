#!/usr/bin/env python3
"""E43 -- the discriminating read-out: the shape of the learned support mask.

E43 section 3 added one read-out that is not a fifth criterion but a
guard on how S1's first four numbers are to be read:

    **the spatial form of the mask.**  If the mask comes out
    non-zero almost everywhere, the sparse branch has degenerated into
    "report a little everywhere", and then a rising ``kernel_px`` is only
    area inflation -- the same failure E41's A2 arm showed.  This read-out
    has to be reported together with S1's four numbers so that an area
    effect is not mistaken for a structural repair.

This script runs the trained A3 checkpoints on the test split, hooks the
mask out of the model, and measures it.  It trains nothing and it
produces no score: it reads the checkpoints and the data, so it is a
read-only probe.

The model is NOT modified.  Changing ``models/lr_sparse/model.py`` would
break the S0 bit-equality against ``simvp_v2`` that the whole
single-variable claim rests on.  Instead the forward path is replayed up
to ``body`` -- the same tensors the model's own forward computes -- and
the mask is recomputed with the model's own expression and weights:

    mask = sigmoid((gate - tau) / T)

so the measured mask is the one the model actually used.

For each (window, lead) frame it reports the median over frames of

* ``mask_dry_frac_lt_0p1``   -- the fraction of pixels whose mask value
  is below 0.1, i.e. pixels the mask effectively switched OFF.  A
  degenerate mask has this near 0; a sparse mask has it high.
* ``mask_mean``              -- the average mask value.  A mask stuck at
  1 everywhere has this near 1.
* ``mask_active_frac_gt_0p5`` -- the fraction of pixels the mask
  switched ON (value above 0.5).
* ``mask_frac_gt_0p9``       -- the fraction of pixels with a nearly
  saturated ON mask.
* ``mask_frac_lt_0p01``      -- the fraction of pixels with a nearly
  saturated OFF mask.
* ``tau_mean``               -- the learned threshold itself, so one can
  see whether the network moved the threshold or the shape.
* ``gate_std``               -- how much the gate varies spatially.  A
  gate with no spatial structure cannot express "where it rains", and
  then the threshold is doing all the work.

The probe runs on the same window set as the evaluation (538 test
windows, ep00) and, like the evaluation, on the remote CUDA device.

Usage:
    python tools/_e43_mask_probe.py --runs /home/wf/radarNWP/runs/e43 \
        --arms A3 --seeds 0 1 --epoch 00 --out <path.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from radar_nwp_bench.config import load_config
from radar_nwp_bench.data.dataset import RadarNWPDataset, build_dataloader
from radar_nwp_bench.models.registry import build_model
from models.lr_sparse.model import _lowest_dct_modes


def _replay(model, x_radar: torch.Tensor, x_nwp: torch.Tensor):
    """Run the model's own forward path and return the mask it used.

    Returns ``(pred, mask, tau, gate)`` with ``mask`` shaped
    ``(B, T, 1, H, W)``.
    """
    B, T_in, _, H, W = x_radar.shape
    nwp_flat = x_nwp.view(B, 1, -1, H, W).expand(-1, T_in, -1, -1, -1)
    x_concat = torch.cat([x_radar, nwp_flat], dim=2)

    feats = model.enc(x_concat.view(B * T_in, -1, H, W))
    _, C_f, H_f, W_f = feats.shape
    feats_time = feats.view(B, T_in * C_f, H_f, W_f)
    h = model.in_time_proj(feats_time)
    h = model.translator(h)
    future_feats = model.out_time_proj(h).view(B * model.out_len, C_f, H_f, W_f)

    body = model.dec[:-1](future_feats)
    amp = model.dec[-1](body)

    gate = model.sparse_gate(body)
    thr = model.gate_threshold(body).mean(dim=(2, 3)).view(-1, 1, 1, 1)
    mask = torch.sigmoid((gate - thr) / model.gate_temp)

    basis = _lowest_dct_modes(H, W, model.r_low).to(
        device=amp.device, dtype=amp.dtype)
    flat = amp.reshape(amp.shape[0], -1)
    coeff = flat @ basis.reshape(model.r_low, -1).T
    low = (coeff @ basis.reshape(model.r_low, -1)).reshape(amp.shape[0], 1, H, W)
    rain = mask * amp
    out = low + rain
    pred = torch.floor(torch.expm1(
        out.view(B, model.out_len, 1, H, W).clamp(-10.0, 6.0)))
    return pred, mask.view(B, model.out_len, 1, H, W), thr.view(-1), gate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="/home/wf/radarNWP/runs/e43")
    ap.add_argument("--arms", nargs="+", default=["A3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epoch", default="00")
    ap.add_argument("--data-root", default="/home/wf/data/radarNWP/BJ_mmap")
    ap.add_argument("--model", default="lr_sparse")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--slide-stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = Path(args.runs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # The model's own configuration file supplies its hyperparameters, so
    # the checkpoint is loaded into exactly the architecture it was trained
    # with -- the same rule scripts/evaluate.py follows.
    cfg: Dict[str, Any] = load_config(args.model)
    model_kwargs = dict(cfg.get("model") or {})
    print(f"model kwargs: {model_kwargs}")

    out: dict = {
        "split": "test",
        "epoch": args.epoch,
        "model": args.model,
        "model_kwargs": model_kwargs,
        "mask_definition": "sigmoid((gate - tau) / gate_temp)",
        "read_out_note": (
            "mask_dry_frac_lt_0p1 near 0 together with mask_mean near 1 "
            "means the mask degenerated into 'on everywhere': a rising "
            "kernel_px would then be area inflation, not a structural "
            "repair."
        ),
        "tags": {},
    }

    for arm in args.arms:
        for seed in args.seeds:
            tag = f"{arm}_s{seed}_ep{args.epoch}"
            ckpts = sorted((runs / f"{arm}_s{seed}" / "train").glob("*/epoch_*.pt"))
            if not ckpts:
                print(f"MISSING checkpoint for {arm} seed {seed}")
                continue
            exact = [p for p in ckpts if p.stem == f"epoch_{args.epoch}"]
            ckpt = (exact or ckpts)[0]
            print(f"{tag}: checkpoint {ckpt}")

            model = build_model(args.model, **model_kwargs)
            obj = torch.load(ckpt, map_location="cpu")
            sd = obj.get("model", obj.get("state_dict", obj))
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"  missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
            model.eval().to(device)

            dataset = RadarNWPDataset(
                args.data_root,
                packed=True,
                patch_size=args.patch_size,
                random_crop=False,
                split="test",
                slide_stride=args.slide_stride,
            )
            loader = build_dataloader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, drop_last=False, seed=0,
            )

            rows: list[Dict[str, float]] = []
            seen = 0
            with torch.no_grad():
                for batch in loader:
                    x_radar = batch["x_radar"].to(device)
                    x_nwp = batch["x_nwp"].to(device)
                    pred, mask, tau, gate = _replay(model, x_radar, x_nwp)
                    m = mask[:, :, 0].reshape(-1).float().cpu().numpy()
                    g = gate[:, 0].reshape(-1).float().cpu().numpy()
                    t = tau.float().cpu().numpy()
                    rows.append({
                        "mask_dry_frac_lt_0p1": float(np.mean(m < 0.1)),
                        "mask_mean": float(np.mean(m)),
                        "mask_active_frac_gt_0p5": float(np.mean(m > 0.5)),
                        "mask_frac_gt_0p9": float(np.mean(m > 0.9)),
                        "mask_frac_lt_0p01": float(np.mean(m < 0.01)),
                        "tau_mean": float(np.mean(t)),
                        "gate_std": float(np.std(g)),
                        "pred_max": float(pred.max().cpu()),
                        "pred_mean": float(pred.mean().cpu()),
                    })
                    seen += int(mask.shape[0])

            keys = list(rows[0].keys())
            agg = {k: float(np.median([r[k] for r in rows])) for k in keys}
            agg["n_batches"] = len(rows)
            agg["n_windows"] = seen
            out["tags"][tag] = agg
            print("  " + tag + ": "
                  + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()
                              if isinstance(v, float)))

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nmask probe written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
