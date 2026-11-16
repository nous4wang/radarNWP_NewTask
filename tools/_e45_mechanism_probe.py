#!/usr/bin/env python3
"""E45 / MASTER 4.47 -- the mechanism read-outs of E45's arms.

E45's pre-registration asks for three DIFFERENTIATING read-outs per arm
(E45 doc section 3.1), and each of them is a statement about the
MECHANISM rather than about the score:

    A1 / A2   corr(mask, amp) >= 0, mask_at_peak > 0.5,
              maxratio_as_trained > 0.9
    A3        grad_top1pct_share < 0.5, field max <= 30 mm,
              tail_max_over_p99 <= 10
    A4        the amount channel must not degenerate into a constant

This tool reads them off a TRAINED checkpoint.  It does not train, it
does not score, it does not modify the checkpoint, and it writes nothing
into any data directory: it loads a model, runs it forwards over the
test split, and reports the read-outs.

THE DEFINITIONS ARE NOT NEW
---------------------------
Every read-out here is the one E44 (or the section 2 probe) already
defined, and the code that computes them is IMPORTED from there rather
than copied, so that the two cannot drift apart:

* ``corr_mask_amp`` and ``mask_at_peak`` come from
  ``tools/_e44_diag_probe.py::gate_frame_stats`` -- E44's own definition:
  a WITHIN-FRAME Pearson correlation between the flattened amplitude and
  the flattened mask (a frame with no variation is skipped, not reported
  as zero), the mask's value at the frame's amplitude peak, the median
  over the frames of a batch, and the median over the batches.  E44
  measured ``corr(mask, amp) = -0.858320`` and ``mask_at_peak =
  0.195087`` with it, so the E45 numbers are comparable with the E44
  numbers by construction.
* ``maxratio_as_trained`` (and its two controls) come from E44's
  ``_max_ratio`` and ``_mm``: the median over the frames of the ratio of
  the field's maximum to the maximum of the ``mask == 1`` bypass.  The
  controls are the numbers E44 quotes: the full bypass and the raw
  amplitude are BOTH 1.000, which is what says the loss of the peak
  belongs to the gate and not to the projection.
* ``grad_top1pct_share`` comes from
  ``tools/_e44_diag_probe.py::grad_top1pct_share_of``: the share of the
  topology term's gradient L1 that the strongest 1 % of the pixels carry,
  measured on a 64x64 crop around a real spike, exactly as E44 measured
  the 0.9399 that condemned the E44 term.
* ``tail_max_over_p99`` comes from
  ``tools/_struct_probe_one_case.py::metrics`` -- the section 2 probe's
  own implementation, imported and not copied.

THE A3 TERM IS REBUILT PER ARM
-------------------------------
A3's two arms are the SAME MODEL with different loss flags, so a probe
that measured one topology term would say nothing about the other.  The
term is therefore rebuilt from the arm's own flags (``--topo-clamp-mm``,
``--topo-truth-smooth-sigma``, ``--topo-area-normalize``) and the
read-outs are taken against THAT term.  The flags follow the training
script's convention: a clamp of zero means "no clamp".

THE A4 AMOUNT IS AUDITED, NOT ASSUMED
--------------------------------------
``mean(out) == T`` is the arm's arithmetic guarantee, and a guarantee
that is only written down is not a guarantee: the probe measures the
output's spatial mean against ``T`` on the real data and reports the
relative error.  ``scale_is_constant`` is a DEGENERATION FLOOR, not a
target: ``scale_cv < 0.05`` or ``scale_truth_corr < 0.3`` means the
amount channel has stopped carrying information about the rain and the
arm has degenerated into a global multiplier.  A healthy arm is above
both floors; the floors are where the mechanism is judged to have
failed, and they are not values to aim at.

Usage:
    python tools/_e45_mechanism_probe.py \
        --arm A1 --checkpoint runs/e45/A1_s0/last.pt \
        --data-root /home/wf/data/radarNWP/BJ_mmap \
        --out tools/_evals/e45_A1_mechanism.json

    # two arms in one run, the second one with its own topology flags
    python tools/_e45_mechanism_probe.py \
        --arm "A3|topo_clamp_mm=20,topo_area_normalize=1" \
        --arm "A3b|topo_clamp_mm=20,topo_truth_smooth_sigma=1,topo_area_normalize=1" \
        --model topo_simvp --model topo_simvp \
        --checkpoint runs/e45/A3_s0/last.pt --checkpoint runs/e45/A3b_s0/last.pt \
        --data-root /home/wf/data/radarNWP/BJ_mmap --out tools/_evals/e45_A3.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from radar_nwp_bench.config import load_config, model_kwargs  # noqa: E402
from radar_nwp_bench.data.dataset import (  # noqa: E402
    RadarNWPDataset,
    build_dataloader,
)
from radar_nwp_bench.metrics.stitching import PatchStitcher  # noqa: E402
from radar_nwp_bench.metrics.topology_losses import ClDiceLoss  # noqa: E402
from radar_nwp_bench.models.registry import build_model  # noqa: E402

# E44's own read-out implementations.  These are imported, not copied:
# one metric, one implementation.
from tools._e44_diag_probe import (  # noqa: E402
    _max_ratio,
    _mm,
    _tail_ratio,
    gate_frame_stats,
    grad_top1pct_share_of,
    pearson,
)
# The section 2 probe's structure metrics.  Imported, not copied.
from tools._struct_probe_one_case import (  # noqa: E402
    metrics as struct_metrics,
    radial_grid,
)

#: The arms and what they are.  ``family`` says which read-outs apply;
#: ``model`` is the default model name, and it may be overridden on the
#: command line.
ARMS: Dict[str, Dict[str, str]] = {
    "A1": {"model": "amp_monotone_gate", "family": "gate"},
    "A2": {"model": "amp_hard_support", "family": "gate"},
    # A3a and A3b are the SAME model with different loss flags: A3a
    # caps the amplitude, A3b redistributes the gradient.  The flags are
    # given inline on the command line, because the read-out has to
    # measure the term that was actually trained.
    "A3a": {"model": "topo_simvp", "family": "topo"},
    "A3": {"model": "topo_simvp", "family": "topo"},
    "A3b": {"model": "topo_simvp", "family": "topo"},
    "A4": {"model": "scaled_structure", "family": "scale"},
    "A4b": {"model": "scaled_structure_own", "family": "scale"},
}

#: The per-arm options that may be given inline, after a ``|``:
#: ``--arm "A3b|topo_clamp_mm=20,topo_area_normalize=1"``.
ARM_OPTIONS: Tuple[str, ...] = (
    "topo_threshold",
    "topo_clamp_mm",
    "topo_truth_smooth_sigma",
    "topo_area_normalize",
)

#: The DEGENERATION FLOORS of the amount channel.  They are thresholds
#: below which the arm is judged to have degenerated, not values the arm
#: is meant to reach.
SCALE_CV_FLOOR = 0.05
SCALE_TRUTH_CORR_FLOOR = 0.3


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _median(values: List[float]) -> float:
    """The nan-aware median of a list of floats, NaN when all are NaN/empty."""
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def _parse_bool(text: str) -> bool:
    """Read a boolean written as a flag value."""
    lowered = str(text).strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise SystemExit(
        f"ERROR: not a boolean: {text!r} (use 1/0, true/false, yes/no)"
    )


def _parse_arm_spec(spec: str) -> Tuple[str, Dict[str, Any]]:
    """Split an ``--arm`` value into its name and its inline options.

    The form is ``NAME`` or ``NAME|key=value,key=value``.  The inline
    options are the arm's OWN parameters -- the topology flags for A3 --
    and they exist because A3's two arms are the same model with
    different loss flags, so one run has to be able to carry both.
    """
    name, _, rest = spec.partition("|")
    name = name.strip()
    options: Dict[str, Any] = {}
    if not rest.strip():
        return name, options
    for item in rest.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in ARM_OPTIONS:
            raise SystemExit(
                f"ERROR: unknown arm option {key!r}; known options: "
                + ", ".join(ARM_OPTIONS)
            )
        options[key] = (
            _parse_bool(value) if key == "topo_area_normalize"
            else float(value)
        )
    return name, options


def _pair_with_arms(
    values: List[str], arms: List[str], what: str
) -> List[str | None]:
    """Line a list of per-arm values up with the arms.

    One value applies to every arm; as many values as there are arms are
    matched positionally.  Anything else is an error rather than a guess:
    a silently mismatched checkpoint would produce a read-out of the
    wrong arm, which is worse than no read-out at all.
    """
    if not values:
        return [None] * len(arms)
    if len(values) == 1:
        return list(values) * len(arms)
    if len(values) != len(arms):
        raise SystemExit(
            f"ERROR: {len(values)} values for --{what} but {len(arms)} arms; "
            f"give one value per arm or exactly one value for all of them"
        )
    return list(values)


# --------------------------------------------------------------------------
# the model and the data
# --------------------------------------------------------------------------


def _build_model(
    model_name: str,
    dataset: RadarNWPDataset,
    overrides: str | None,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Build the arm's model with the kwargs its own configuration file gives.

    The shape arguments come from the DATASET rather than from the
    configuration file: the file records what the run was trained with,
    and the data records what is actually being fed in.  When they
    disagree the run would be scored on the wrong thing.
    """
    cfg = load_config(model_name)
    kwargs = model_kwargs(
        cfg,
        shapes={
            "in_radar_c": len(dataset.radar_vars),
            "in_nwp_c": len(dataset.nwp_vars),
            "in_len": dataset.in_len,
            "out_len": dataset.out_len,
        },
        overrides=overrides,
    )
    model = build_model(model_name, **kwargs).to(device)
    return model, kwargs


def _load_checkpoint(
    model: torch.nn.Module, path: str, device: torch.device
) -> Dict[str, Any]:
    """Load a checkpoint into the model and report what did not line up.

    The checkpoint is a plain ``state_dict`` in this project's format,
    but a wrapper is accepted too.  A checkpoint that does not fit the
    model is reported rather than ignored: a read-out of a model that is
    not the one that was trained would look like a result.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        state = obj.get("model", obj.get("state_dict", obj))
    else:
        state = obj
    try:
        model.load_state_dict(state)
        return {"loaded_strictly": True, "missing": [], "unexpected": []}
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state, strict=False)
        return {
            "loaded_strictly": False,
            "missing": sorted(missing),
            "unexpected": sorted(unexpected),
        }


def _build_dataset(args) -> RadarNWPDataset:
    """The test split, with the patches the read-outs are measured on."""
    return RadarNWPDataset(
        args.data_root,
        packed=args.packed,
        patch_size=args.patch_size,
        random_crop=False,
        split=args.split,
        slide_stride=args.slide_stride,
    )


# --------------------------------------------------------------------------
# A1 / A2: the gate
# --------------------------------------------------------------------------


def probe_gate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args,
) -> Dict[str, Any]:
    """The gate read-outs of one arm.

    What is read: the model's own ``last_debug``, which holds the
    amplitude (``amp``, the decoder's log-millimetre output), the gate's
    mask (``mask``) and the gate's per-frame threshold (``thr``).

    What is computed, and why:

    * ``corr_mask_amp`` -- the Pearson correlation between the mask and
      the amplitude WITHIN a frame, median over the frames and then over
      the batches.  This is E44's definition, imported from E44: the
      gate's threshold is one number per (window, lead time), so the
      within-frame correlation is the quantity the mechanism is about.
      E44 measured ``-0.858320`` with it, which is the number A1 and A2
      are measured against.
    * ``mask_at_peak`` -- the mask's value at the frame's amplitude
      peak.  A gate that is closed at the peak shaves the tail, so this
      is the read-out that says whether the peak survives.
    * ``maxratio_as_trained`` -- how much of the peak the composition
      ``low + mask * resid`` keeps, measured against the ``mask == 1``
      bypass, which is the same field the amplitude is.  The two
      controls (``maxratio_full_bypass`` and ``maxratio_raw_amp``) are
      both 1.000 by construction and are reported anyway: they are the
      statement that the loss of the peak is the gate's doing and not
      the projection's.
    * ``mask_dry_frac_lt_0p1`` -- the fraction of pixels whose mask is
      below 0.1.  E44 used it to call a mask degenerate; a gate that is
      closed almost everywhere is a gate that is not pointing at the
      rain.
    * ``mask_is_binary`` (A2 only) -- whether the mask takes only the
      values 0 and 1, which is the straight-through construction's
      signature: the VALUE is the hard indicator, the GRADIENT is the
      soft one.
    """
    rows: List[Dict[str, float]] = []
    binary_frames = 0
    non_binary_frames = 0
    n_frames = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            x_radar = batch["x_radar"].to(device)
            x_nwp = batch["x_nwp"].to(device)
            model(x_radar, x_nwp)

            debug = model.last_debug
            amp = debug["amp"]
            mask = debug["mask"]
            thr = debug["thr"]
            low = debug["low"]
            resid = debug["resid"]

            # The arm's composition, in log-millimetre space, and the
            # two controls it is measured against.  ``low + resid`` is
            # the ``mask == 1`` bypass; ``amp`` is the raw amplitude.
            as_trained = low + mask * resid
            full_bypass = low + resid

            m = mask.reshape(-1).float().cpu().numpy()
            t = thr.reshape(-1).float().cpu().numpy()

            peak_mask, peak_corr = gate_frame_stats(amp, mask)
            _, corr_resid = gate_frame_stats(resid, mask)

            values = torch.unique(mask)
            binary = bool(
                torch.all((values == 0.0) | (values == 1.0)).item()
            )
            n_frames += int(mask.shape[0])
            if binary:
                binary_frames += int(mask.shape[0])
            else:
                non_binary_frames += int(mask.shape[0])

            rows.append({
                "mask_dry_frac_lt_0p1": float(np.mean(m < 0.1)),
                "mask_mean": float(np.mean(m)),
                "mask_active_frac_gt_0p5": float(np.mean(m > 0.5)),
                "mask_frac_gt_0p9": float(np.mean(m > 0.9)),
                "mask_frac_lt_0p01": float(np.mean(m < 0.01)),
                "thr_mean": float(np.mean(t)),
                "thr_std": float(np.std(t)),
                "mask_at_peak": _median(peak_mask),
                "corr_mask_amp": _median(peak_corr),
                "corr_mask_resid": _median(corr_resid),
                "maxratio_as_trained": _max_ratio(
                    _mm(as_trained), _mm(full_bypass)),
                "maxratio_full_bypass": _max_ratio(
                    _mm(full_bypass), _mm(amp)),
                "maxratio_raw_amp": _max_ratio(_mm(amp), _mm(full_bypass)),
                "tail_as_trained": _tail_ratio(_mm(as_trained)),
                "tail_full_bypass": _tail_ratio(_mm(full_bypass)),
                "tail_raw_amp": _tail_ratio(_mm(amp)),
            })

    if not rows:
        return {"note": "no batch was read"}

    keys = list(rows[0].keys())
    agg = {k: _median([r[k] for r in rows]) for k in keys}
    # Defensive .get(): not every probe row carries maxratio_as_trained.
    # A missing key counts as an invalid batch, so the statistic also
    # self-reports the NaN-contamination this fix is meant to expose.
    agg["n_valid_maxratio_batches"] = int(
        sum(
            not math.isnan(float(r.get("maxratio_as_trained", float("nan"))))
            for r in rows
        )
    )
    agg["n_batches"] = len(rows)
    agg["n_frames"] = n_frames
    agg["mask_is_binary"] = non_binary_frames == 0
    agg["n_frames_binary"] = binary_frames
    agg["n_frames_non_binary"] = non_binary_frames
    return agg


# --------------------------------------------------------------------------
# A3: the topology term
# --------------------------------------------------------------------------


def _topo_loss(args, options: Dict[str, Any]) -> ClDiceLoss:
    """The topology term with THIS arm's flags.

    The flags follow the training script: a clamp of zero means "no
    clamp", and the threshold is the strong-rain one.
    """
    clamp = options.get("topo_clamp_mm", args.topo_clamp_mm)
    sigma = options.get("topo_truth_smooth_sigma", args.topo_truth_smooth_sigma)
    area = options.get("topo_area_normalize", args.topo_area_normalize)
    threshold = options.get("topo_threshold", args.topo_threshold)
    return ClDiceLoss(
        threshold=float(threshold),
        clamp_mm=float(clamp) if float(clamp) > 0.0 else None,
        truth_smooth_sigma=float(sigma),
        area_normalize=bool(area),
    )


def probe_topo(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """The topology read-outs of one arm.

    What is read: the arm's forecast, stitched back into the whole
    field, and the topology term rebuilt with the arm's own flags.

    What is computed, and why:

    * ``grad_top1pct_share`` -- the share of the term's gradient L1 that
      the strongest 1 % of the pixels carry, measured on a 64x64 crop
      around the field's real spike, exactly as E44 measured the 0.9399
      that condemned the E44 term.  The measurement is made against the
      ARM'S term: a term whose gradient sits on a handful of pixels can
      be driven through those pixels alone, which is a statement about
      the term and not about the connectivity idea.
    * ``field_max_mm`` -- the largest value the forecast takes, in
      millimetres.  The label's own maximum is 8.6 mm, so a forecast
      that reaches tens of millimetres is a forecast whose amplitude is
      being driven by something other than the rain.
    * ``tail_max_over_p99`` -- the section 2 probe's own tail measure,
      imported from ``tools/_struct_probe_one_case.py`` and applied to
      the forecast and to the truth, so the two are comparable.
    * ``term_value`` -- the value of the ARM'S topology term on the
      measured crops.
    """
    loss = _topo_loss(args, options)
    acc = _TopoAccumulator(args)

    # The forecast is stitched back into the whole field, exactly as the
    # evaluation does it, because the crop the gradient is measured on
    # has to be a crop of the real field and not of a tile.
    stitch = PatchStitcher(
        (args.grid_h, args.grid_w),
        n_frames=args.out_len,
        n_positions=args.n_positions,
    ) if args.n_positions > 1 else None

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            x_radar = batch["x_radar"].to(device)
            x_nwp = batch["x_nwp"].to(device)
            pred = model(x_radar, x_nwp)
            truth = batch["y_rain"]

            pred_np = pred.clamp_min(0.0).cpu().numpy()
            truth_np = truth.clamp_min(0.0).cpu().numpy()

            if stitch is None:
                for b in range(pred_np.shape[0]):
                    acc.measure_window(
                        pred_np[b, :, 0], truth_np[b, :, 0], loss
                    )
                    if acc.n_windows >= args.scan_windows:
                        break
                if acc.n_windows >= args.scan_windows:
                    break
                continue

            sample_idx = batch["case_index"].cpu().numpy()
            position_idx = batch["position_index"].cpu().numpy()
            origin = batch["patch_origin"].cpu().numpy()
            for b in range(pred_np.shape[0]):
                stitch.add(
                    int(sample_idx[b]),
                    int(position_idx[b]),
                    (int(origin[b][0]), int(origin[b][1])),
                    pred_np[b, :, 0],
                    truth_np[b, :, 0],
                )
            for sample in list(stitch.pending):
                if not stitch.is_complete(sample):
                    continue
                field, truth_field = stitch.finalize(sample)
                acc.measure_window(field, truth_field, loss)
                stitch.release(sample)
                if acc.n_windows >= args.scan_windows:
                    break
            if acc.n_windows >= args.scan_windows:
                break

    if not acc.tails and not acc.shares:
        return {"note": "no window was measured"}

    out: Dict[str, Any] = {
        "grad_top1pct_share": _median(acc.shares),
        "term_value": _median(acc.terms),
        "field_max_mm": acc.field_max,
        "tail_max_over_p99": _median(acc.tails),
        "tail_max_over_p99_truth": _median(acc.tails_truth),
        "n_windows": acc.n_windows,
        "n_spikes": acc.n_spikes,
        "crop": args.crop,
        "spike_mm": args.spike_mm,
        "topo_threshold": loss.threshold,
        "topo_clamp_mm": loss.clamp_mm,
        "topo_truth_smooth_sigma": loss.truth_smooth_sigma,
        "topo_area_normalize": loss.area_normalize,
    }
    if acc.n_windows < args.scan_windows:
        # The window is stitched out of one tile per patch position, so
        # the read-out needs many more batches than it has windows: a
        # truncated run is reported rather than passed off as the full
        # scan.
        out["note"] = (
            f"only {acc.n_windows} of the {args.scan_windows} requested "
            "windows were measured; the run was cut short by --max-batches "
            "or by the end of the split"
        )
    return out


class _TopoAccumulator:
    """The measurements taken over the topology arm's windows.

    It is a holder rather than a set of loose lists because the
    measurement is a LOOP: the window's spike is located, the crop
    around it is taken, and the term is evaluated on that crop with the
    arm's own flags.
    """

    def __init__(self, args) -> None:
        self.args = args
        self.shares: List[float] = []
        self.terms: List[float] = []
        self.tails: List[float] = []
        self.tails_truth: List[float] = []
        self.field_max = 0.0
        self.n_windows = 0
        self.n_spikes = 0
        self._grid: np.ndarray | None = None

    def measure_window(
        self,
        field: np.ndarray,
        truth_field: np.ndarray,
        loss: ClDiceLoss,
    ) -> None:
        """Measure one stitched window: the spike, the crop, and the term.

        The spike is located the way E44 located it: the largest value in
        the whole window, and the lead time it belongs to.  The crop is a
        ``crop``-sided window around it, clipped to the field, and it is
        where the gradient share and the term value are measured.
        """
        args = self.args
        height, width = field.shape[-2], field.shape[-1]
        block_max = float(field.max())
        self.field_max = max(self.field_max, block_max)
        self.n_windows += 1

        # The section 2 probe's structure metrics, on every lead frame of
        # the window and on the truth's frame.
        if self._grid is None:
            self._grid = radial_grid((height, width))
        grid = self._grid
        for lead in range(field.shape[0]):
            self.tails.append(
                struct_metrics(
                    np.asarray(field[lead], dtype=np.float64), grid
                )["tail_max_over_p99"]
            )
            self.tails_truth.append(
                struct_metrics(
                    np.asarray(truth_field[lead], dtype=np.float64), grid
                )["tail_max_over_p99"]
            )

        if block_max < args.spike_mm:
            return

        idx = int(np.argmax(field))
        lead = idx // (height * width)
        rest = idx % (height * width)
        y, x = rest // width, rest % width
        half = args.crop // 2
        y0, y1 = max(0, y - half), min(height, y + half)
        x0, x1 = max(0, x - half), min(width, x + half)
        pc = np.asarray(field[lead][y0:y1, x0:x1], dtype=np.float64)
        tc = np.asarray(truth_field[lead][y0:y1, x0:x1], dtype=np.float64)
        if pc.size == 0:
            return

        share = grad_top1pct_share_of(loss, pc, tc)
        if not math.isnan(share):
            self.shares.append(share)
        value, _ = loss(
            torch.from_numpy(pc).float()[None],
            torch.from_numpy(tc).float()[None],
        )
        self.terms.append(float(value.detach()))
        self.n_spikes += 1


# --------------------------------------------------------------------------
# A4: the amount channel
# --------------------------------------------------------------------------


def probe_scale(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args,
) -> Dict[str, Any]:
    """The amount-channel read-outs of one arm.

    What is read: the model's own ``last_debug``: ``T``, the amount (one
    scalar per (window, lead time)) and ``S``, the unit-mean shape field
    the amount multiplies.

    What is computed, and why:

    * ``scale_cv`` -- the coefficient of variation of ``T`` over every
      frame of the test split.  A channel that does not vary is a
      constant, and a constant amount is a global multiplier: the arm
      would then be a calibration and nothing else.
    * ``scale_truth_corr`` -- the Pearson correlation between ``T`` and
      the TRUTH's mean rain intensity of the same window and the same
      lead time, taken from the data loader.  An amount that does not
      follow the rain is not an amount.
    * ``scale_is_constant`` -- the DEGENERATION FLOOR: true when
      ``scale_cv < 0.05`` or ``scale_truth_corr < 0.3``.  Both are
      thresholds below which the channel is judged to have stopped
      carrying information; they are not targets.
    * ``mean_out_equals_T`` -- the arm's arithmetic guarantee, MEASURED:
      the output's spatial mean against ``T``, with the relative error.
      The guarantee is a property of the construction, and a
      construction that does not hold on the real data is a broken one.
    """
    ts: List[float] = []
    truth_means: List[float] = []
    out_means: List[float] = []
    s_mean_devs: List[float] = []
    n_frames = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            x_radar = batch["x_radar"].to(device)
            x_nwp = batch["x_nwp"].to(device)
            out = model(x_radar, x_nwp)
            truth = batch["y_rain"]

            debug = model.last_debug
            if "T" not in debug or "S" not in debug:
                raise RuntimeError(
                    "the model does not publish the amount channel: "
                    f"last_debug has {sorted(debug)}"
                )
            amount = debug["T"]
            shape = debug["S"]

            batch_size, n_lead = out.shape[0], out.shape[1]
            if amount.numel() != batch_size * n_lead:
                raise RuntimeError(
                    f"T has {amount.numel()} entries but the output has "
                    f"{batch_size * n_lead} frames"
                )

            # The output's spatial mean per frame, the truth's mean rain
            # intensity per frame, and the shape field's own spatial
            # mean.  All three are indexed the same way as ``T``: the
            # model lays the frames out window-major.
            out_mean = out.mean(dim=(2, 3, 4)).reshape(-1)
            truth_mean = truth.mean(dim=(2, 3, 4)).reshape(-1)
            s_mean = shape.mean(dim=(2, 3)).reshape(-1)

            ts.extend(float(v) for v in amount.detach().cpu().numpy())
            out_means.extend(float(v) for v in out_mean.cpu().numpy())
            truth_means.extend(float(v) for v in truth_mean.cpu().numpy())
            s_mean_devs.extend(
                float(abs(float(v) - 1.0)) for v in s_mean.cpu().numpy()
            )
            n_frames += int(amount.numel())

    if not ts:
        return {"note": "no batch was read"}

    t_arr = np.asarray(ts, dtype=np.float64)
    mean_t = float(t_arr.mean())
    std_t = float(t_arr.std())
    cv = float(std_t / mean_t) if mean_t != 0.0 else float("nan")
    corr = pearson(
        torch.as_tensor(t_arr, dtype=torch.float64),
        torch.as_tensor(np.asarray(truth_means, dtype=np.float64),
                        dtype=torch.float64),
    )

    out_arr = np.asarray(out_means, dtype=np.float64)
    max_abs_err = float(np.max(np.abs(out_arr - t_arr)))
    scale = float(np.mean(np.abs(t_arr)))
    rel_err = float(max_abs_err / scale) if scale > 0.0 else float("nan")

    return {
        "scale_cv": cv,
        "scale_mean": mean_t,
        "scale_std": std_t,
        "scale_min": float(t_arr.min()),
        "scale_max": float(t_arr.max()),
        "scale_truth_corr": corr,
        "scale_is_constant": bool(
            (not math.isnan(cv) and cv < SCALE_CV_FLOOR)
            or (not math.isnan(corr) and corr < SCALE_TRUTH_CORR_FLOOR)
        ),
        "mean_out_equals_T": bool(rel_err <= 1e-5),
        "mean_out_equals_T_rel_err": rel_err,
        "mean_out_equals_T_max_abs_err": max_abs_err,
        "shape_S_unit_mean_max_dev": (
            float(np.max(s_mean_devs)) if s_mean_devs else float("nan")
        ),
        "n_frames": n_frames,
        "scale_cv_floor": SCALE_CV_FLOOR,
        "scale_truth_corr_floor": SCALE_TRUTH_CORR_FLOOR,
    }


# --------------------------------------------------------------------------
# the definitions that travel with the numbers
# --------------------------------------------------------------------------

DEFINITIONS: Dict[str, Dict[str, str]] = {
    "gate": {
        "corr_mask_amp": (
            "Pearson correlation between the gate's mask and the decoder's "
            "amplitude, computed WITHIN each (window, lead time) frame over "
            "the frame's pixels; frames without variation are skipped; the "
            "median over the frames of a batch and then over the batches. "
            "E44's definition, imported from tools/_e44_diag_probe.py; E44 "
            "measured -0.858320 with it."
        ),
        "mask_at_peak": (
            "the mask's value at the pixel of the frame's amplitude peak; "
            "the median over the frames of a batch and then over the "
            "batches. E44's definition."
        ),
        "maxratio_as_trained": (
            "the median over the frames of max(low + mask*resid) divided by "
            "max(low + resid), both in millimetres: how much of the peak "
            "the arm's own composition keeps, measured against the "
            "mask == 1 bypass. E44's definition."
        ),
        "maxratio_full_bypass": (
            "the same ratio with the full bypass in the numerator: 1.000 by "
            "construction, and reported as the control that says the peak's "
            "loss belongs to the gate and not to the projection. E44 quoted "
            "1.000000."
        ),
        "maxratio_raw_amp": (
            "the same ratio with the raw amplitude in the numerator: 1.000 "
            "by construction. E44 quoted 1.000000."
        ),
        "mask_dry_frac_lt_0p1": (
            "the fraction of pixels whose mask is below 0.1; the median "
            "over the batches. E44's mask-degeneration measure."
        ),
        "mask_is_binary": (
            "true when every frame's mask takes only the values 0 and 1: "
            "the straight-through construction's signature (the value is "
            "the hard indicator, the gradient is the soft one). A2's "
            "mask is binary by construction; A1's sigmoid gate is not."
        ),
        "n_frames_binary": (
            "the number of frames whose mask is binary, counted frame by "
            "frame rather than batch by batch."
        ),
        "corr_mask_resid": (
            "the same within-frame correlation, between the mask and the "
            "residual (amp - low) the gate is actually a function of."
        ),
        "tail_as_trained": (
            "the median over the frames of max/p99 of the composition, in "
            "millimetres: the heavy-tail measure. E44's definition."
        ),
    },
    "topo": {
        "grad_top1pct_share": (
            "the share of the topology term's gradient L1 carried by the "
            "strongest 1 % of the pixels of a 64x64 crop around the "
            "field's real spike, measured against THIS arm's term. E44's "
            "definition and algorithm, imported from "
            "tools/_e44_diag_probe.py; E44 measured 0.9399 with it."
        ),
        "term_value": (
            "the value of this arm's topology term on the measured crops; "
            "the median over the crops."
        ),
        "field_max_mm": (
            "the largest value the forecast takes over the measured "
            "windows, in millimetres. The label's own maximum is 8.6 mm."
        ),
        "tail_max_over_p99": (
            "the section 2 probe's tail measure (max divided by the 99th "
            "percentile) on the forecast, imported from "
            "tools/_struct_probe_one_case.py::metrics."
        ),
        "tail_max_over_p99_truth": (
            "the same measure on the truth's frame, so that the forecast's "
            "number has a reference next to it."
        ),
        "n_spikes": (
            "the number of windows whose maximum reached the spike "
            "threshold and that were therefore measured."
        ),
    },
    "scale": {
        "scale_cv": (
            "the coefficient of variation of the amount T over every frame "
            "of the test split: std(T) / mean(T)."
        ),
        "scale_truth_corr": (
            "the Pearson correlation between the amount T and the truth's "
            "mean rain intensity (mm) of the same window and the same lead "
            "time, taken from the data loader."
        ),
        "scale_is_constant": (
            "the DEGENERATION FLOOR: true when scale_cv < 0.05 or "
            "scale_truth_corr < 0.3, i.e. when the amount channel has "
            "stopped carrying information about the rain. These are "
            "thresholds below which the mechanism is judged to have "
            "failed; they are not values to aim at."
        ),
        "mean_out_equals_T": (
            "the arm's arithmetic guarantee, MEASURED on the real data: "
            "true when the output's spatial mean equals the amount T to "
            "within a relative error of 1e-5."
        ),
        "mean_out_equals_T_rel_err": (
            "the relative error of that guarantee: the largest absolute "
            "difference between the output's spatial mean and T, divided "
            "by the mean of |T|."
        ),
        "shape_S_unit_mean_max_dev": (
            "the largest deviation of the shape field's spatial mean from "
            "one: the normalisation the guarantee rests on, measured."
        ),
    },
}


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--arm", action="append", required=True,
        help="the arm to probe; repeatable.  The form is NAME or "
             "NAME|key=value,key=value, where the inline options are the "
             "arm's own parameters (the topology flags).",
    )
    ap.add_argument(
        "--model", action="append", default=[],
        help="the model name; one per arm, or one for all of them.  "
             "Defaults to the arm's own model.",
    )
    ap.add_argument(
        "--checkpoint", action="append", default=[],
        help="the checkpoint to read; one per arm, or one for all of them.",
    )
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    packed = ap.add_mutually_exclusive_group()
    packed.add_argument(
        "--packed", dest="packed", action="store_true", default=True,
        help="the data root holds packed cases (the default).",
    )
    packed.add_argument(
        "--no-packed", dest="packed", action="store_false",
        help="the data root holds raw CP_* directories.",
    )
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--slide-stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--max-batches", type=int, default=0,
        help="stop after this many batches; 0 reads the whole split.",
    )
    ap.add_argument("--model-kwargs", default=None,
                    help="a JSON object of extra constructor kwargs.")
    # The topology flags: the arm's own parameters.
    ap.add_argument("--topo-threshold", type=float, default=2.0)
    ap.add_argument("--topo-clamp-mm", type=float, default=0.0,
                    help="the physical ceiling in mm; 0 disables it.")
    ap.add_argument("--topo-truth-smooth-sigma", type=float, default=0.0,
                    help="the width of the Gaussian blur on the truth; 0 "
                         "disables it.")
    ap.add_argument("--topo-area-normalize", action="store_true")
    # The measurement knobs, at E44's values.
    ap.add_argument("--spike-mm", type=float, default=20.0)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--scan-windows", type=int, default=60,
                    help="how many windows the topology arm measures.")
    args = ap.parse_args()

    # The arms, their inline options, and the models and checkpoints.
    names: List[str] = []
    options: Dict[str, Dict[str, Any]] = {}
    for spec in args.arm:
        name, opts = _parse_arm_spec(spec)
        if name not in ARMS:
            raise SystemExit(
                f"ERROR: unknown arm {name!r}; known arms: "
                + ", ".join(sorted(ARMS))
            )
        if name in names:
            raise SystemExit(f"ERROR: arm {name!r} was given twice")
        names.append(name)
        options[name] = opts

    models = _pair_with_arms(args.model, names, "model")
    checkpoints = _pair_with_arms(args.checkpoint, names, "checkpoint")

    # One explicit model for several arms whose default models differ is
    # almost always a mistake: the read-out would be taken from the wrong
    # architecture and would look like a result.  The case that IS meant
    # -- A3 and A3b, which are the same model with different loss flags
    # -- has arms that agree, so it passes.
    if len(args.model) == 1 and len(names) > 1:
        defaults = {ARMS[n]["model"] for n in names}
        if len(defaults) > 1:
            raise SystemExit(
                "ERROR: one --model was given for arms with different "
                f"default models ({', '.join(sorted(defaults))}); give one "
                "--model per arm"
            )

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )

    # The checkpoints are checked before the data is opened: a missing
    # path is a mistake in the command line, and it should be reported
    # as one rather than as a failure to read the archive.
    for name, checkpoint in zip(names, checkpoints):
        if checkpoint is None:
            raise SystemExit(f"ERROR: arm {name!r} has no --checkpoint")
        if not Path(checkpoint).is_file():
            raise SystemExit(
                f"ERROR: the checkpoint of arm {name!r} does not exist: "
                f"{checkpoint}"
            )

    dataset = _build_dataset(args)
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        seed=0,
    )
    grid_h, grid_w = dataset.grid_shape

    # The knobs the topology measurement passes around.
    args.grid_h, args.grid_w = grid_h, grid_w
    args.out_len = dataset.out_len
    args.n_positions = dataset.n_patch_positions

    print(
        f"probing {len(names)} arm(s) on the {args.split} split, "
        f"device {device}, {len(dataset)} samples"
    )

    results: Dict[str, Any] = {}
    for name, model_name, checkpoint in zip(names, models, checkpoints):
        model_name = model_name or ARMS[name]["model"]
        family = ARMS[name]["family"]
        print(f"\n=== {name} ({model_name}) ===")
        model, kwargs = _build_model(
            model_name, dataset, args.model_kwargs, device
        )
        loaded = _load_checkpoint(model, checkpoint, device)
        if not loaded["loaded_strictly"]:
            print(
                f"  WARNING: the checkpoint does not fit the model exactly: "
                f"missing={loaded['missing'][:6]} "
                f"unexpected={loaded['unexpected'][:6]}"
            )
        model.eval()

        if family == "gate":
            readouts = probe_gate(model, loader, device, args)
        elif family == "topo":
            readouts = probe_topo(model, loader, device, args, options[name])
        elif family == "scale":
            readouts = probe_scale(model, loader, device, args)
        else:
            raise SystemExit(f"ERROR: unknown family {family!r}")

        results[name] = {
            "arm": name,
            "model": model_name,
            "checkpoint": str(checkpoint),
            "family": family,
            "model_kwargs": kwargs,
            "checkpoint_load": loaded,
            "readouts": readouts,
            "definitions": DEFINITIONS[family],
        }
        for key, value in readouts.items():
            if isinstance(value, float):
                print(f"  {key} = {value:.6f}")
            else:
                print(f"  {key} = {value}")

    out = {
        "tool": "tools/_e45_mechanism_probe.py",
        "purpose": (
            "the mechanism read-outs of E45's arms: a read-only probe of "
            "trained checkpoints. Nothing is trained, nothing is scored, "
            "no checkpoint is modified, and no data directory is written."
        ),
        "data_root": str(args.data_root),
        "split": args.split,
        "patch_size": args.patch_size,
        "slide_stride": args.slide_stride,
        "device": str(device),
        "arms": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
