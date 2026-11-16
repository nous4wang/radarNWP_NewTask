#!/usr/bin/env python3
"""E44 -- the discriminating read-outs for A1 and A3.

E44's S1 gave two results that need a mechanism read-out before the
verdict can be written, and neither read-out needs training.

**A1 (low-rank constrains the background only).**  S1 came out 3 of 4:
the tail fell (3.2245 -> 2.9414) while the background did NOT get
flatter (``tv_z`` 0.0401 -> 0.0419, ``lap_z`` 0.1007 -> 0.1110).  The
arm's premise was that the rain region bypasses the projection, so the
tail should have survived.  It did not.  The composition is::

    out = low + mask * (amp - low)

which only bypasses the projection where ``mask == 1`` exactly.  A soft
sigmoid mask that is mostly a little below one blends the tail-poor
low-rank component back into the rain region.  This probe replays the
trained model's own forward path and compares three compositions of the
SAME body:

    (a) the arm as trained          : low + mask * (amp - low)
    (b) full bypass                  : low + (amp - low)  == amp
    (c) E43's composition            : low + mask * amp

If (b) has a much higher tail than (a) at the same weights, the tail was
shaved by the mask's softness and not by the projection itself -- which
is a statement about the mechanism's implementation, and it is exactly
the difference between "the projection kills the tail" (E43's reading)
and "a soft gate cannot express a hard bypass" (this one).

**A3 (topology).**  S1 came out 1 of 4 and the field is pathological:
``tail_max_over_p99`` 200.33 against the truth's 4.55, a field maximum
of 306 mm, ``kernel_px`` 18.33 against the truth's 80.79, and
``hf_share`` 0.58 against the truth's 0.0126.  The model produced
isolated needles.  Two things have to be separated:

  * is the needle a way to LOWER the connectivity term (then the term
    is gameable and the arm says nothing about topology), or
  * is the needle something else entirely (a divergence of the
    objective) that the term merely rides along with.

The test is local: take a 64x64 crop around a real spike from the
dumped forecast, evaluate the connectivity term on it, then evaluate it
again with the spike clipped to the frame's 99th percentile.  If
clipping LOWERS the term, the term prefers the needle and it is
gameable; if clipping RAISES it, the term is doing its job and the
needle comes from elsewhere.

Both parts read the checkpoints and the dumped fields.  Nothing is
trained, nothing is scored, and the model is not modified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from radar_nwp_bench.config import load_config
from radar_nwp_bench.data.dataset import RadarNWPDataset, build_dataloader
from radar_nwp_bench.metrics.topology_losses import ClDiceLoss
from radar_nwp_bench.models.common import floor_at_zero
from radar_nwp_bench.models.registry import build_model
from models.lr_sparse.model import _lowest_dct_modes


# --------------------------------------------------------------------------
# A1: replay the trained forward path and compare three compositions
# --------------------------------------------------------------------------


def _replay_a1(model, x_radar: torch.Tensor, x_nwp: torch.Tensor):
    """Return the tensors the model used, without changing the model.

    Returns ``(amp, low, mask, thr, gate)``.  ``amp`` and ``low`` are
    ``(B*T, 1, H, W)`` in log-mm space, exactly as the model's own
    forward computes them.
    """
    B, T_in, _, H, W = x_radar.shape
    nwp_flat = x_nwp.view(B, 1, -1, H, W).expand(-1, T_in, -1, -1, -1)
    x_concat = torch.cat([x_radar, nwp_flat], dim=2)

    feats = model.enc(x_concat.view(B * T_in, -1, H, W))
    _, C_f, H_f, W_f = feats.shape
    feats_time = feats.view(B, T_in * C_f, H_f, W_f)
    h = model.in_time_proj(feats_time)
    h = model.translator(h)
    future_feats = model.out_time_proj(h).view(
        B * model.out_len, C_f, H_f, W_f)

    body = model.dec[:-1](future_feats)
    amp = model.dec[-1](body)

    basis = _lowest_dct_modes(H, W, model.r_low).to(
        device=amp.device, dtype=amp.dtype)
    flat = amp.reshape(amp.shape[0], -1)
    coeff = flat @ basis.reshape(model.r_low, -1).T
    low = (coeff @ basis.reshape(model.r_low, -1)).reshape(amp.shape[0], 1, H, W)

    gate = model.sparse_gate(body)
    thr = model.gate_threshold(body).mean(dim=(2, 3)).view(-1, 1, 1, 1)
    mask = torch.sigmoid((gate - thr) / model.gate_temp)
    return amp, low, mask, thr, gate


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """The Pearson correlation of two equal-length float sequences.

    The estimator is the one E44 used for the gate's read-outs, written
    out once so that E45's probe measures its own correlations with the
    same code: the sequences are centred, and the correlation is the
    centred covariance divided by the geometric mean of the two centred
    norms.

    A sequence with no variation has no correlation to report, so NaN is
    returned rather than zero: a zero would say "uncorrelated", which is
    a different statement from "not defined".

    The arithmetic is done in the INPUT dtype, which is what E44's
    inline computation did: the read-out must not move when it is
    refactored into a function.

    Args:
        a: the first sequence.
        b: the second sequence, the same length as ``a``.

    Returns:
        The correlation, or NaN when it is not defined.
    """
    x = a.reshape(-1)
    y = b.reshape(-1)
    if x.numel() < 2 or y.numel() != x.numel():
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(torch.sqrt((xc * xc).sum() * (yc * yc).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def gate_frame_stats(
    amp: torch.Tensor, mask: torch.Tensor
) -> Tuple[List[float], List[float]]:
    """The gate read-outs of one batch, per frame.

    This is E44's own definition of the two gate read-outs, and it lives
    here so that E45's probe can use the SAME one instead of writing a
    second implementation of the same number (the project's rule: one
    metric, one implementation).

    Two quantities come out, both per (window, lead time) frame:

    * ``peak_mask`` -- the mask's value at the frame's amplitude peak.
      The peak is the ARGMAX of the frame's flattened amplitude, and the
      mask is read at that one pixel.  A frame whose amplitude peak sits
      under a closed gate is the signature of E43/E44's failure, so the
      read-out is a single number per frame rather than an average.
    * ``peak_corr`` -- the Pearson correlation between the frame's
      amplitude and the frame's mask, both flattened over the frame's
      pixels.  The correlation is computed WITHIN a frame (the gate's
      threshold is one number per frame, so a within-frame correlation is
      the quantity the mechanism is about), and a frame that carries no
      variation -- a constant amplitude or a constant mask -- has no
      correlation to report and is skipped rather than reported as zero.

    Args:
        amp: the amplitude field, ``(N, 1, H, W)``, one frame per row.
        mask: the gate's mask, same shape as ``amp``.

    Returns:
        ``(peak_mask, peak_corr)``: two lists of per-frame floats, the
        second one possibly shorter because degenerate frames are skipped.
    """
    a = amp.detach().reshape(amp.shape[0], -1).float()
    mk = mask.detach().reshape(mask.shape[0], -1).float()
    peak_mask: List[float] = []
    peak_corr: List[float] = []
    for av, mv in zip(a, mk):
        if av.numel() == 0:
            continue
        idx = int(torch.argmax(av))
        peak_mask.append(float(mv[idx]))
        if av.numel() > 2 and float(av.std()) > 0 and float(mv.std()) > 0:
            value = pearson(av, mv)
            if not math.isnan(value):
                peak_corr.append(value)
    return peak_mask, peak_corr


def grad_top1pct_share_of(
    loss, pred_crop: np.ndarray, truth_crop: np.ndarray
) -> float:
    """The share of a term's gradient L1 that the top 1 % of the pixels carry.

    This is E44's own definition of ``grad_top1pct_share``, extracted so
    that E45's probe measures it with the same code.  The quantity is a
    property of the TERM, not of the model: a term whose gradient sits on
    a handful of pixels can be driven by the optimiser through those
    pixels alone, which is a statement about the implementation of the
    term rather than about the idea behind it.

    The gradient is taken on a CROP around a real spike, because that is
    where the term's attention is: E44 measured it on the 64x64 crop
    around the forecast's needle.

    Args:
        loss: the criterion to differentiate, called as
            ``loss(pred[None], truth[None])``.
        pred_crop: the predicted crop, in millimetres.
        truth_crop: the matching truth crop, in millimetres.

    Returns:
        The fraction of the gradient's L1 norm carried by the strongest
        1 % of the crop's pixels, or NaN when the gradient is zero.
    """
    gp = torch.from_numpy(np.ascontiguousarray(pred_crop)).requires_grad_(True)
    tt = torch.from_numpy(np.ascontiguousarray(truth_crop))[None]
    # The gradient is what is being measured, so it has to be enabled
    # even when the caller is inside a no_grad block.  E44's own probe
    # measured this outside one, so the numbers are unchanged.
    with torch.enable_grad():
        vg, _ = loss(gp[None], tt)
        vg.backward()
    abs_g = gp.grad.detach().abs().reshape(-1)
    total = float(abs_g.sum())
    if total <= 0:
        return float("nan")
    order = torch.argsort(abs_g, descending=True)
    top = float(abs_g[order[: max(1, abs_g.numel() // 100)]].sum())
    return top / total


def _tail_ratio(out: torch.Tensor) -> float:
    """The median over frames of ``max / p99``, the tail measure.

    The frame is the unit, exactly as in the structure metrics: the
    quantity is computed per (window, lead) frame and the median over
    frames is reported.  A batch-level ratio would mix frames and would
    not be comparable with the numbers in the structure file.

    Frames whose 99th percentile is zero carry no tail to measure -- a
    64x64 patch of a mostly dry field is often entirely below the
    threshold -- so they are skipped.  That is why the robust
    ``max_ratio`` measure below is the primary one here.
    """
    v = out.reshape(out.shape[0], -1).float()
    ratios: List[float] = []
    for row in v:
        if row.numel() == 0:
            continue
        finite = row[torch.isfinite(row)]
        if finite.numel() == 0:
            continue
        p99 = torch.quantile(finite, 0.99)
        if float(p99) <= 0.0:
            continue
        ratios.append(float(finite.max() / p99))
    if not ratios:
        return float("nan")
    return float(np.median(ratios))


def _max_ratio(out: torch.Tensor, ref: torch.Tensor) -> float:
    """The median over frames of ``max(out) / max(ref)``.

    This is the robust version of the tail comparison: unlike
    ``max / p99`` it does not need a non-zero percentile, so it is
    defined on every frame, and it says directly how much of the peak
    amplitude survives a composition.
    """
    v = out.reshape(out.shape[0], -1).float()
    r = ref.reshape(ref.shape[0], -1).float()
    ratios: List[float] = []
    for a, b in zip(v, r):
        if a.numel() == 0 or b.numel() == 0:
            continue
        fa, fb = a[torch.isfinite(a)], b[torch.isfinite(b)]
        if fa.numel() == 0 or fb.numel() == 0:
            continue
        mb = float(fb.max())
        if mb <= 0.0:
            continue
        ratios.append(float(fa.max()) / mb)
    if not ratios:
        return float("nan")
    return float(np.median(ratios))


def _mm(out: torch.Tensor) -> torch.Tensor:
    """The model's own decode: log-mm -> millimetres, floored at zero.

    The floor matters.  Without it the negative values of the dry
    majority pull the frame maximum towards zero and every ratio that
    divides by a maximum becomes undefined.
    """
    return floor_at_zero(torch.expm1(out.clamp(-10.0, 6.0)))


def probe_a1(args) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A1 probe, device {device}")

    cfg: Dict[str, Any] = load_config(args.a1_model)
    model_kwargs = dict(cfg.get("model") or {})
    print(f"  model kwargs: {model_kwargs}")

    runs = Path(args.runs)
    ckpts = sorted((runs / f"{args.a1_arm}_s0" / "train").glob("*/epoch_*.pt"))
    exact = [p for p in ckpts if p.stem == f"epoch_{args.epoch}"]
    ckpt = (exact or ckpts)[0]
    print(f"  checkpoint: {ckpt}")

    model = build_model(args.a1_model, **model_kwargs)
    obj = torch.load(ckpt, map_location="cpu")
    sd = obj.get("model", obj.get("state_dict", obj))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    model.eval().to(device)

    dataset = RadarNWPDataset(
        args.data_root, packed=True, patch_size=args.patch_size,
        random_crop=False, split="test", slide_stride=args.slide_stride,
    )
    loader = build_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, seed=0,
    )

    rows: List[Dict[str, float]] = []
    seen = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            x_radar = batch["x_radar"].to(device)
            x_nwp = batch["x_nwp"].to(device)
            amp, low, mask, thr, gate = _replay_a1(model, x_radar, x_nwp)

            # the three compositions of the SAME body
            as_trained = low + mask * (amp - low)      # (a) the arm
            full_bypass = low + (amp - low)            # (b) mask == 1
            e43_style = low + mask * amp               # (c) E43's form

            m = mask.reshape(-1).float().cpu().numpy()
            g = gate.reshape(-1).float().cpu().numpy()
            t = thr.reshape(-1).float().cpu().numpy()

            # Does the gate cover the amplitude peak?  The composition
            # is  out = low + mask * (amp - low), and the projection
            # smooths the peak away, so out_peak is about
            # mask_peak * amp_peak.  If the gate is soft the peak is
            # scaled down; if the gate is ANTI-ALIGNED with the
            # amplitude the peak is scaled down even harder, and that is
            # a different mechanism with a different fix.
            #
            # The two read-outs come from the shared helper, so that E45's
            # probe measures them with exactly this code.
            peak_mask, peak_corr = gate_frame_stats(amp, mask)

            rows.append({
                "mask_dry_frac_lt_0p1": float(np.mean(m < 0.1)),
                "mask_mean": float(np.mean(m)),
                "mask_active_frac_gt_0p5": float(np.mean(m > 0.5)),
                "mask_frac_gt_0p9": float(np.mean(m > 0.9)),
                "mask_frac_lt_0p01": float(np.mean(m < 0.01)),
                "tau_mean": float(np.mean(t)),
                "gate_std": float(np.std(g)),
                # how much of the rain region still receives the low-rank
                # component: the effective bypass factor
                "bypass_factor_mean": float(np.mean(1.0 - m)),
                "bypass_factor_p10": float(np.quantile(1.0 - m, 0.10)),
                # the gate value AT the amplitude peak, and how the gate
                # correlates with the amplitude
                "mask_at_peak": (float(np.median(peak_mask))
                                 if peak_mask else float("nan")),
                "corr_mask_amp": (float(np.median(peak_corr))
                                  if peak_corr else float("nan")),
                # the tail of each composition, in mm
                "tail_as_trained": _tail_ratio(_mm(as_trained)),
                "tail_full_bypass": _tail_ratio(_mm(full_bypass)),
                "tail_e43_style": _tail_ratio(_mm(e43_style)),
                "tail_raw_amp": _tail_ratio(_mm(amp)),
                # the robust version: how much of the peak survives each
                # composition, measured against the full bypass
                "maxratio_as_trained": _max_ratio(_mm(as_trained),
                                                  _mm(full_bypass)),
                "maxratio_e43_style": _max_ratio(_mm(e43_style),
                                                 _mm(full_bypass)),
                "maxratio_raw_amp": _max_ratio(_mm(amp), _mm(full_bypass)),
            })
            seen += int(mask.shape[0])
            if (bi + 1) % 10 == 0:
                print(f"  batch {bi + 1}: {seen} frames")

    keys = list(rows[0].keys())
    agg = {k: float(np.median([r[k] for r in rows])) for k in keys}
    agg["n_batches"] = len(rows)
    agg["n_windows"] = seen
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()
                           if isinstance(v, float)))
    return agg


# --------------------------------------------------------------------------
# A3: is the connectivity term gameable by a needle?
# --------------------------------------------------------------------------


def probe_a3(args) -> Dict[str, Any]:
    runs = Path(args.runs)
    field = runs / "fields" / f"{args.a3_arm}_s0_ep{args.epoch}.pred.npy"
    truth = runs / "fields" / f"{args.a3_arm}_s0_ep{args.epoch}.truth.npy"
    print(f"A3 probe, field {field}")
    pred = np.load(field, mmap_mode="r")
    tru = np.load(truth, mmap_mode="r")

    loss = ClDiceLoss(threshold=args.threshold)

    # locate the spikes: the largest values in the whole field
    print("  scanning for spikes ...")
    spike_list: List[Dict[str, Any]] = []
    n_win, n_lead = pred.shape[0], pred.shape[1]
    step = max(1, n_win // args.scan_windows)
    for w in range(0, n_win, step):
        block = np.asarray(pred[w], dtype=np.float64)
        mx = float(block.max())
        if mx < args.spike_mm:
            continue
        # the location of the maximum
        idx = int(np.argmax(block))
        lead = idx // (block.shape[1] * block.shape[2])
        rest = idx % (block.shape[1] * block.shape[2])
        y = rest // block.shape[2]
        x = rest % block.shape[2]
        spike_list.append({"window": w, "lead": lead, "y": y, "x": x,
                           "value": mx})
        if len(spike_list) >= args.max_spikes:
            break

    print(f"  found {len(spike_list)} windows with a value above "
          f"{args.spike_mm} mm")

    # for each spike, evaluate the connectivity term with and without it
    rows: List[Dict[str, Any]] = []
    half = args.crop // 2
    for sp in spike_list:
        w, lead = sp["window"], sp["lead"]
        y, x = sp["y"], sp["x"]
        p = np.asarray(pred[w, lead], dtype=np.float64)
        t = np.asarray(tru[w, lead], dtype=np.float64)
        # a crop around the spike, clipped to the frame
        y0, y1 = max(0, y - half), min(p.shape[0], y + half)
        x0, x1 = max(0, x - half), min(p.shape[1], x + half)
        pc = p[y0:y1, x0:x1]
        tc = t[y0:y1, x0:x1]
        if pc.size == 0:
            continue

        pt = torch.from_numpy(pc)[None]
        tt = torch.from_numpy(tc)[None]
        v_raw, comp_raw = loss(pt, tt)

        # clip the spike to the frame's 99th percentile
        flat = np.sort(pc.reshape(-1))
        p99 = float(flat[int(0.99 * (len(flat) - 1))]) if len(flat) else 0.0
        clipped = np.minimum(pc, p99)
        v_clip, comp_clip = loss(
            torch.from_numpy(clipped)[None], tt)

        # where does the gradient of the term live?  A term whose
        # gradient sits on a handful of pixels can be driven by the
        # optimiser through those pixels alone, which is a property of
        # the implementation rather than of the connectivity idea.
        grad_top1pct_share = grad_top1pct_share_of(loss, pc, tc)

        rows.append({
            "window": w, "lead": lead,
            "spike_value": float(pc.max()),
            "crop_p99": p99,
            "term_with_spike": float(v_raw),
            "term_without_spike": float(v_clip),
            "delta_clipped_minus_raw": float(v_clip) - float(v_raw),
            "pred_area_with": comp_raw["pred_area"],
            "pred_area_without": comp_clip["pred_area"],
            "grad_top1pct_share": grad_top1pct_share,
        })

    if not rows:
        return {"n_spikes": 0, "note": "no spike above the threshold found"}

    deltas = [r["delta_clipped_minus_raw"] for r in rows]
    n_lowers = int(sum(1 for d in deltas if d < 0.0))
    out = {
        "threshold_mm": args.threshold,
        "n_spikes": len(rows),
        "delta_mean": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "delta_min": float(np.min(deltas)),
        "delta_max": float(np.max(deltas)),
        "n_clipping_lowers_term": n_lowers,
        "rows": rows[:20],
    }
    print(f"  clipping the spike changes the term by "
          f"{out['delta_median']:+.6f} (median), "
          f"{n_lowers}/{out['n_spikes']} cases lower it")
    # delta = term_without_spike - term_with_spike, and the term is a
    # loss to be MINIMISED.
    #   delta < 0  -> removing the spike LOWERS the loss -> the term
    #                 wants the needle GONE -> the term punishes it.
    #   delta > 0  -> keeping the spike lowers the loss  -> the term
    #                 wants the needle   -> the term is gameable.
    n_prefers_needle = out["n_spikes"] - n_lowers
    gameable = n_prefers_needle > out["n_spikes"] / 2
    out["n_term_prefers_needle"] = n_prefers_needle
    out["term_is_gameable_by_needle"] = bool(gameable)
    shares = [r["grad_top1pct_share"] for r in rows
              if not math.isnan(r["grad_top1pct_share"])]
    if shares:
        out["grad_top1pct_share_median"] = float(np.median(shares))
    print("  -> " + ("the term PREFERS the needle (gameable): "
                     "the arm says nothing about topology"
                     if gameable else
                     "the term PUNISHES the needle: the needle is not "
                     "what the term wanted"))
    if shares:
        print(f"  gradient concentration: the top 1 % of pixels carry "
              f"{out['grad_top1pct_share_median']:.4f} of the gradient L1")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="/home/wf/radarNWP/runs/e44")
    ap.add_argument("--data-root", default="/home/wf/data/radarNWP/BJ_mmap")
    ap.add_argument("--a1-arm", default="A1")
    ap.add_argument("--a1-model", default="lr_sparse_bg")
    ap.add_argument("--a3-arm", default="A3")
    ap.add_argument("--epoch", default="00")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--slide-stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="0 = the whole test split")
    ap.add_argument("--scan-windows", type=int, default=60)
    ap.add_argument("--max-spikes", type=int, default=25)
    ap.add_argument("--spike-mm", type=float, default=20.0)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out: Dict[str, Any] = {
        "split": "test",
        "epoch": args.epoch,
        "note": (
            "read-outs for E44's two anomalies: A1's tail falling while its "
            "background does not flatten, and A3's needle-shaped field.  "
            "Nothing is trained and nothing is scored."
        ),
    }

    if args.a1_arm:
        out["a1"] = probe_a1(args)
    if args.a3_arm:
        out["a3"] = probe_a3(args)

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nprobe written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
