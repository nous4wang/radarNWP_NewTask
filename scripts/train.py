#!/usr/bin/env python3
"""Train one baseline on the radar-NWP benchmark.

Usage:
    python scripts/train.py --model mocast --data-root data/BJ_packed \
        --epochs 5 --batch-size 2 --patch-size 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# cuBLAS needs a fixed workspace before the CUDA context exists, otherwise
# ``torch.use_deterministic_algorithms(True)`` raises on the first matmul.
# It is read by cuBLAS at handle creation, so setting it here -- before
# torch is imported -- is the only place that is early enough.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_nwp_bench.ablation import (
    MODE_FULL,
    MODE_NO_NWP,
    MODE_ZERO_NWP,
    apply_ablation,
    describe,
    resolve_mode,
)
from radar_nwp_bench.config import argparse_defaults, load_config, model_kwargs
from radar_nwp_bench.data import (
    NWPAlignment,
    RadarNWPDataset,
    build_dataloader,
)
from radar_nwp_bench.data.transforms import NWP_PIPELINE_VERSION
from radar_nwp_bench.metrics import PrecipitationLoss, PrecipitationVerifier
from radar_nwp_bench.models import build_model, list_models


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list_models())
    ap.add_argument(
        "--data-root", default=None,
        help="root of the dataset; may also come from the configuration file",
    )
    ap.add_argument("--packed", action="store_true", help="read .npz containers")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--in-len", type=int, default=11)
    ap.add_argument("--out-len", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--nwp-vars", nargs="*", default=None)
    ap.add_argument("--radar-vars", nargs="*", default=None)
    ap.add_argument(
        "--no-nwp", action="store_true",
        help="Ablation: train or evaluate strictly without NWP inputs (pure radar model).",
    )
    ap.add_argument(
        "--zero-nwp", action="store_true",
        help="Ablation: zero out NWP tensors during evaluation to test NWP dependency.",
    )
    ap.add_argument(
        "--alignment", default="forecast",
        choices=[a.value for a in NWPAlignment],
        help="NWP valid-time alignment policy",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--no-interpolate", action="store_true",
        help="drop the windows whose history contains a missing frame, "
             "instead of rebuilding the frame by interpolation. Used to "
             "measure what the fault tolerance is worth.",
    )
    ap.add_argument(
        "--filter-dry", action="store_true",
        help="skip the training patches that contain no rain, keeping a "
             "fraction of the dry ones so the model still sees the field's "
             "background",
    )
    ap.add_argument(
        "--dry-keep-prob", type=float, default=0.20,
        help="probability of keeping a dry patch when --filter-dry is on",
    )
    ap.add_argument(
        "--rain-threshold", type=float, default=0.05,
        help="rain rate in mm per 6-minute accumulation above which a "
             "training patch counts as wet",
    )
    ap.add_argument(
        "--max-steps", type=int, default=0, help="stop after N steps (debug)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument(
        "--split", default="train",
        choices=["train", "val", "test", "robustness_missing", "all"],
        help="case-level split; 'robustness_missing' holds out the cases "
             "whose radar series has a hole in it; 'all' uses every case "
             "(debugging)",
    )
    ap.add_argument(
        "--max-samples", type=int, default=None,
        help="cap the number of samples (smoke training on CPU)",
    )
    ap.add_argument(
        "--slide-stride", type=int, default=None,
        help="slide the patch over the full 301x301 grid with this stride "
             "(e.g. 32 -> a 9x9 grid of 32x32 patches per sample)",
    )
    ap.add_argument(
        "--config", default=None,
        help="configuration file or name (e.g. 'default', 'bj', 'mocast'). "
             "Explicit command-line flags override the file.",
    )
    ap.add_argument(
        "--model-kwargs", default=None,
        help="JSON object of extra constructor kwargs for the baseline",
    )
    ap.add_argument(
        "--loss", default="standard",
        choices=["standard", "balanced", "hierarchical", "field_csi",
                 "soft_fss", "ibcfm", "hurdle", "uot", "topo"],
        help="the training objective: 'standard' regresses the log "
             "accumulation, 'balanced' aims at the verification score with a "
             "soft CSI term over several thresholds, 'hierarchical' "
             "supervises the exceedance probabilities of a model that "
             "produces them per pixel, 'field_csi' forms the tier counts "
             "over the whole field instead, which is the form that "
             "measured better, 'hurdle' (E41) uses an occurrence + "
             "conditional-amount output head with a two-part NLL, 'uot' (E41) "
             "adds an unbalanced Sinkhorn divergence to the standard "
             "regression, and 'topo' (E44) keeps the balanced objective "
             "unchanged and adds a clDice connectivity term on top",
    )
    ap.add_argument(
        "--hurdle-occ-weight", type=float, default=1.0,
        help="E41 A1: weight of the Bernoulli occurrence term",
    )
    ap.add_argument(
        "--hurdle-amount-weight", type=float, default=1.0,
        help="E41 A1: weight of the conditional log-amount term",
    )
    ap.add_argument(
        "--uot-weight", type=float, default=1.0,
        help="E41 A2: weight of the unbalanced Sinkhorn term",
    )
    ap.add_argument(
        "--uot-grid", type=int, default=16,
        help="E41 A2: coarse grid side for the transport histogram",
    )
    ap.add_argument(
        "--uot-eps", type=float, default=0.05,
        help="E41 A2: entropic regularisation (cost normalised to [0, 1])",
    )
    ap.add_argument(
        "--topo-weight", type=float, default=0.1,
        help="E44 A3: weight of the clDice connectivity term",
    )
    ap.add_argument(
        "--topo-threshold", type=float, default=2.0,
        help="E44 A3: rain rate in mm at which the field is binarised for "
             "the connectivity measure (the strong-rain tier)",
    )
    ap.add_argument(
        "--topo-clamp-mm", type=float, default=0.0,
        help="E45 A3: truncate the forecast to this many mm before the "
             "topology term reads it; 0 disables the truncation. The "
             "truncated pixels get no gradient from the topology term, "
             "which cuts the path to the model's own mm ceiling",
    )
    ap.add_argument(
        "--topo-truth-smooth-sigma", type=float, default=0.0,
        help="E45 A3: width in pixels of the separable Gaussian blur "
             "applied to the truth before the topology term reads it; 0 "
             "disables the blur. The truth's support becomes a region "
             "instead of a few isolated pixels",
    )
    ap.add_argument(
        "--topo-area-normalize", action="store_true",
        help="E45 A3: divide the topology term's per-pixel weights by "
             "the area of the connected component they belong to, so "
             "that a one-pixel component and a hundred-pixel component "
             "contribute comparably",
    )
    ap.add_argument(
        "--tier-thresholds", nargs="*", type=float, default=None,
        help="the rain rates in mm per 6-minute accumulation for the tier "
             "terms of the hierarchical losses. 2.0 is the "
             "short-duration heavy-rain warning line (20 mm/h)",
    )
    ap.add_argument(
        "--per-pixel", action="store_true",
        help="form the tier counts per pixel instead of over the whole "
             "field. The per-pixel form is the one that measures worse, and "
             "it is kept so the comparison can be run",
    )
    ap.add_argument(
        "--alpha-csi", type=float, default=0.5,
        help="weight of the soft CSI term in the balanced loss",
    )
    ap.add_argument(
        "--qpe-weight", type=float, default=0.0,
        help="weight of the history-rainfall supervision on the model's "
             "inverted history. Zero means the objective does not use it, "
             "which is the ablation arm for route R1",
    )
    ap.add_argument(
        "--qpe-single-frame", action="store_true",
        help="supervise ONLY the t0 ground-rainfall field (y_hist[:, -1]) "
             "against the model's first output frame. For a pure neural QPE "
             "inverter that copies its t0 estimate across the forecast "
             "window: without this flag the unchanged ten-frame target "
             "would force the model to fit an average of the future frames "
             "instead of inverting the observed one. Off by default, so "
             "every existing run is bit-for-bit unaffected.",
    )
    ap.add_argument(
        "--alpha-prob", type=float, default=0.3,
        help="weight of the exceedance-probability term in the hierarchical "
             "loss",
    )
    ap.add_argument(
        "--use-focal", dest="use_focal", action="store_true", default=False,
        help="modulate the probability term by (1 - p_t)^2 in the "
             "hierarchical loss. The modulation damps the gradient where the "
             "model is already confident, which on a sparse heavy-rain field "
             "can blunt the recall the term is there to encourage",
    )
    ap.add_argument(
        "--alpha-fss", type=float, default=0.4,
        help="weight of the neighbourhood fractions term in the soft_fss "
             "loss. Zero leaves the score-aware objective untouched, which "
             "is the control that shows the term is doing something",
    )
    ap.add_argument(
        "--fss-scales", nargs="*", type=int, default=[3, 5],
        help="neighbourhood window sides in pixels for the soft_fss loss; "
             "a side of three is a three by three window",
    )
    ap.add_argument(
        "--no-focal", dest="use_focal", action="store_false",
        help="turn the focal modulation off (the default)",
    )
    ap.add_argument(
        "--head-bins", type=int, default=0,
        help="number of bins in the hierarchical rain head. Zero leaves the "
             "model's own default. Six is the aligned case, where every "
             "threshold has a bin of its own; five is the misaligned one, "
             "kept so the difference can be measured",
    )
    ap.add_argument(
        "--level-mode", default=None, choices=["midpoint", "floor"],
        help="how the hierarchical head turns the bin distribution into a "
             "rain rate. 'midpoint' represents each bin by its middle, so "
             "a pixel the head places above a threshold reconstructs to "
             "a rate above it; 'floor' uses the lower edge, which puts "
             "the reconstruction exactly on the threshold and makes the "
             "score at that threshold a matter of rounding. Left unset, "
             "the model's own configuration decides",
    )
    ap.add_argument(
        "--physics-weight", type=float, default=0.0,
        help="weight of the baseline's own physics regulariser, if it has one "
             "(for example the FURECast CAPPI self-consistency penalty)",
    )
    return ap


def _apply_config(ap: argparse.ArgumentParser, argv: list[str] | None) -> tuple[dict, dict]:
    """Parse the command line with the configuration file applied.

    The file's values become argparse *defaults*, so an explicit flag on the
    command line always wins. The configuration is read twice: once to find
    out which file was asked for, and once -- after the defaults have been
    set -- for the final parse.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)

    config = load_config(known.config)
    ap.set_defaults(**argparse_defaults(config, "train"))
    args = ap.parse_args(argv)
    if not args.data_root:
        ap.error(
            "--data-root is required (give it on the command line or set "
            "data.root in the configuration file)"
        )
    return args, config


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _BalancedExtremeCriterion:
    """Adapt :class:`BalancedExtremePrecipLoss` to the training loop.

    The training loop expects the criterion to return the loss and a
    dictionary of its components, so that the epoch summary can print
    them. The balanced loss computes a single number, so the components
    are reported as one entry under ``total``.

    The adapter is also where the history-rainfall supervision is
    attached. The loss itself takes no keyword arguments and returns a
    bare number, so the extra term has to be added here: the model
    publishes its inverted history rainfall, and this adapter compares
    it with the observed history rainfall when both are present.
    """

    def __init__(self, alpha_csi: float, qpe_weight: float = 0.0) -> None:
        from radar_nwp_bench.models.common import BalancedExtremePrecipLoss

        self.loss = BalancedExtremePrecipLoss(alpha_csi=alpha_csi)
        self.qpe_weight = qpe_weight

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model=None,
        **extra: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        value = self.loss(pred, target)
        components: dict[str, float] = {"total": float(value.detach())}

        # The inversion branch is supervised against the observed
        # history rainfall, when the model has one and the data supplies
        # the truth. The weight is read from the call, so that the
        # ablation arm (weight zero) needs no other change.
        qpe_weight = float(extra.get("qpe_weight", self.qpe_weight))
        y_hist = extra.get("y_hist")
        qpe_pred = (
            getattr(model, "last_qpe_pred", None) if model is not None else None
        )
        if (
            qpe_pred is not None
            and y_hist is not None
            and qpe_weight > 0.0
        ):
            qpe_loss = self.loss(qpe_pred, y_hist)
            value = value + qpe_weight * qpe_loss
            components["qpe_loss"] = float(qpe_loss.detach())
            components["total"] = float(value.detach())
        return value, components


class _IBCFMCriterion:
    """E39 / MASTER 4.41 -- joint anchor + conditional-flow loss.

    The flow branch lives on the model because it needs the cached
    conditioning map and anchor from ``forward``.  ``scripts/train.py``
    already passes ``model=`` into the criterion, so this adds no change to
    the training loop, the evaluator, or any other baseline.  Models without
    ``compute_flow_loss`` fail loudly instead of silently training a plain
    regression.
    """

    def __init__(self, flow_weight: float = 1.0, anchor_weight: float = 1.0):
        self.anchor = PrecipitationLoss()
        self.flow_weight = float(flow_weight)
        self.anchor_weight = float(anchor_weight)

    def __call__(self, pred, target, model=None, **extra):
        base_loss, components = self.anchor(
            pred, target, model=model, **extra
        )
        if model is None or not hasattr(model, "compute_flow_loss"):
            raise RuntimeError(
                "--loss ibcfm requires a model exposing compute_flow_loss()"
            )
        flow = model.compute_flow_loss(target)
        value = self.anchor_weight * base_loss + self.flow_weight * flow
        components = dict(components)
        components["flow_loss"] = float(flow.detach())
        components["total"] = float(value.detach())
        return value, components


def _build_criterion(args):
    """Build the training objective.

    The default is the benchmark's own loss, which regresses the log
    accumulation. The balanced variant aims at the verification score
    instead, and exists because the models trained under the default one
    predict between two and a third of the rain that actually fell. The
    hierarchical variant supervises the exceedance probabilities of a
    model that produces them, and degrades to the plain regression for a
    model that does not -- which is why it takes the model rather than
    being told which model it is looking at.
    """
    loss_name = getattr(args, "loss", "standard")
    if loss_name == "ibcfm":
        return _IBCFMCriterion(
            flow_weight=float(getattr(args, "flow_weight", 1.0)),
            anchor_weight=float(getattr(args, "anchor_weight", 1.0)),
        )
    if loss_name == "hurdle":
        # E41 A1: only the output parameterisation / training objective
        # changes.  The backbone and the evaluator are untouched.
        from radar_nwp_bench.metrics.struct_losses import HurdleCriterion

        return HurdleCriterion(
            occ_weight=float(getattr(args, "hurdle_occ_weight", 1.0)),
            amount_weight=float(getattr(args, "hurdle_amount_weight", 1.0)),
        )
    if loss_name == "uot":
        # E41 A2: only one extra loss term changes; the model is the same
        # 'rain'-head model used by the standard arm.
        from radar_nwp_bench.metrics.struct_losses import UOTCriterion

        return UOTCriterion(
            uot_weight=float(getattr(args, "uot_weight", 1.0)),
            grid=int(getattr(args, "uot_grid", 16)),
            eps=float(getattr(args, "uot_eps", 0.05)),
            physics_weight=args.physics_weight,
        )
    if loss_name == "balanced":
        return _BalancedExtremeCriterion(
            alpha_csi=args.alpha_csi,
            qpe_weight=float(getattr(args, "qpe_weight", 0.0)),
        )
    if loss_name == "topo":
        # E44 A3: the load-bearing balanced objective is used UNCHANGED
        # and one connectivity term is added on top.  The backbone is the
        # control's, so the arms differ in exactly one thing.
        #
        # E45 A3: the three flags below are LOSS-side only.  They change
        # how the topology term reads the field, not the model, so the
        # arms still differ from the control in exactly one thing and
        # the model's weights are untouched by them.
        from radar_nwp_bench.metrics.topology_losses import TopoCriterion

        base = _BalancedExtremeCriterion(
            alpha_csi=args.alpha_csi,
            qpe_weight=float(getattr(args, "qpe_weight", 0.0)),
        )
        clamp_mm = float(getattr(args, "topo_clamp_mm", 0.0))
        return TopoCriterion(
            base,
            topo_weight=float(getattr(args, "topo_weight", 0.1)),
            threshold=float(getattr(args, "topo_threshold", 2.0)),
            # A ceiling of zero means "no ceiling": the flag is off.
            clamp_mm=clamp_mm if clamp_mm > 0.0 else None,
            truth_smooth_sigma=float(
                getattr(args, "topo_truth_smooth_sigma", 0.0)
            ),
            area_normalize=bool(getattr(args, "topo_area_normalize", False)),
        )
    if loss_name == "hierarchical":
        from radar_nwp_bench.metrics.losses import HierarchicalPrecipLoss

        return HierarchicalPrecipLoss(
            alpha_prob=args.alpha_prob,
            use_focal=bool(getattr(args, "use_focal", False)),
        )
    if loss_name == "soft_fss":
        # The score-aware objective with a neighbourhood tolerance term
        # added on top. The base is the same adapter the balanced arm
        # uses, so the two arms differ by exactly one thing: the
        # fractions term. The history-rainfall supervision comes along
        # with it, which is what keeps the comparison with the other
        # arms clean.
        from radar_nwp_bench.metrics.losses import SoftFSSPrecipLoss

        base = _BalancedExtremeCriterion(
            alpha_csi=args.alpha_csi,
            qpe_weight=float(getattr(args, "qpe_weight", 0.0)),
        )
        return SoftFSSPrecipLoss(
            base,
            thresholds=tuple(args.tier_thresholds) if args.tier_thresholds
            else (0.1, 0.5, 2.0, 4.0),
            scales=tuple(getattr(args, "fss_scales", (3, 5))),
            alpha_fss=float(getattr(args, "alpha_fss", 0.4)),
        )
    if loss_name == "field_csi":
        from radar_nwp_bench.metrics.losses import HierarchicalFieldCSILoss

        return HierarchicalFieldCSILoss(
            thresholds=tuple(args.tier_thresholds) if args.tier_thresholds
            else (0.1, 0.5, 2.0, 4.0),
            alpha_prob=args.alpha_prob,
            field_ratio=not getattr(args, "per_pixel", False),
        )
    return PrecipitationLoss(physics_weight=args.physics_weight)


def _teacher_argument(model) -> str | None:
    """Find the optional teacher-forcing argument a baseline accepts.

    The baseline's ``forward`` may take a third argument -- the ground
    truth -- to teacher-force a recurrent or iterative decoder. Passing it
    switches on the mechanism the paper describes (a future-radar prior, a
    frequency memory); omitting it leaves those branches untrained.

    Returns:
        The name of the argument, or ``None`` when the baseline does not
        accept one.
    """
    import inspect

    try:
        params = inspect.signature(type(model).forward).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return None
    for candidate in ("gt_radar", "gt_spectrum", "target"):
        if candidate in params:
            return candidate
    return None


def main() -> int:
    ap = build_argparser()
    args, config = _apply_config(ap, None)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    # Make the run reproducible.
    #
    # The same configuration trained twice used to give different results:
    # two runs of ``simvp_v2_nwp_concat`` with the same seed, the same
    # learning rate and the same code differed by 1.78x on CSI@2.0, while
    # their per-epoch training losses differed by a few thousandths. The
    # divergence was in the training side -- the evaluation was shown to be
    # bit-identical when the same checkpoint is scored twice.
    #
    # Three switches turn off the sources of that drift: cuDNN's algorithm
    # autotuning picks a different convolution for different GPU loads,
    # and the atomic reductions inside a convolution backward sum in an
    # order that depends on the schedule. ``warn_only=True`` keeps the
    # operators that have no deterministic implementation running instead
    # of aborting the run -- ``F.interpolate`` backward and
    # ``ConvTranspose2d`` backward are among them on CUDA -- so this
    # removes the reproducible part of the drift, not all of it.
    #
    # What this does NOT do is shrink the spread between different seeds.
    # That spread is the variance band, and it is measured with repeated
    # seeds, not with a rerun of the same seed.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # A run that takes hours must be readable while it is running. Python
    # block-buffers stdout when it is redirected to a file, so a long run
    # would otherwise show nothing at all until it finished.
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(line_buffering=True)

    # Report the configuration up front: if anything below fails, the run
    # must still say which file it read.
    model_cfg = config
    if not model_cfg.get("model"):
        try:
            model_cfg = load_config(args.model)
        except FileNotFoundError:
            model_cfg = {}
    if model_cfg.get("_path"):
        print(f"config: {model_cfg['_path']}")

    # The ablation mode is resolved once, here, and then used by every
    # downstream decision: the dataset, the forward pass and the loss. Two
    # scripts with one behaviour each is how the harness lost --slide-stride
    # in the first place, so the switch lives in the package and both
    # scripts call the same function.
    try:
        mode = resolve_mode(args.no_nwp, args.zero_nwp)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    if mode != MODE_FULL:
        print(f"ablation: {mode} -- {describe(mode)}")

    ds_kwargs = dict(
        in_len=args.in_len,
        out_len=args.out_len,
        patch_size=args.patch_size,
        alignment=NWPAlignment(args.alignment),
        random_crop=True,
        split=None if args.split == "all" else args.split,
        max_samples=args.max_samples,
        slide_stride=args.slide_stride,
        # The "retrain without NWP" arm never reads an NWP file. The sample
        # index is untouched, so both arms are trained and scored on exactly
        # the same windows -- the difference between them is information, not
        # data.
        blank_nwp=(mode == MODE_NO_NWP),
        # The dry-patch filter only makes sense on the training split: it
        # is a statement about where the training budget should go.
        filter_dry=args.filter_dry,
        dry_keep_prob=args.dry_keep_prob,
        rain_threshold=args.rain_threshold,
        tolerate_one_missing_frame=not args.no_interpolate,
    )
    if args.nwp_vars:
        ds_kwargs["nwp_vars"] = tuple(args.nwp_vars)
    if args.radar_vars:
        ds_kwargs["radar_vars"] = tuple(args.radar_vars)

    dataset = RadarNWPDataset(args.data_root, packed=args.packed, **ds_kwargs)
    print(f"dataset: {len(dataset)} samples from {args.data_root}")
    if len(dataset) == 0:
        print("ERROR: no samples -- check --alignment or the data root")
        return 1

    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    nwp_c = len(dataset.nwp_vars)
    radar_c = len(dataset.radar_vars)
    # The model's own configuration file supplies its hyperparameters; the
    # shapes always come from the data, not from the file.
    kwargs = model_kwargs(
        model_cfg,
        shapes={
            "in_radar_c": radar_c,
            "in_nwp_c": nwp_c,
            "in_len": args.in_len,
            "out_len": args.out_len,
        },
        overrides=args.model_kwargs,
    )
    # The number of bins in the hierarchical head is a property of the
    # model, not of the data, so it comes from the command line when it is
    # given. Only the models that have the head take it.
    head_bins = getattr(args, "head_bins", 0)
    if head_bins:
        kwargs["head_bins"] = head_bins
    # The way the head reconstructs the rain rate from the bin
    # distribution is likewise a property of the head, and it is recorded
    # with the run so the evaluation reads it back.
    level_mode = getattr(args, "level_mode", None)
    if level_mode:
        kwargs["level_mode"] = level_mode
    try:
        model = build_model(args.model, **kwargs).to(device)
    except TypeError as exc:
        if "head_bins" in str(exc) or "level_mode" in str(exc):
            print(
                f"ERROR: --head-bins or --level-mode was given but "
                f"{args.model} has no hierarchical head to configure."
            )
            return 2
        raise
    # The number of NWP slices is a property of the alignment, not of the
    # adapter, so it comes from the dataset. The model needs it to build the
    # zero tensor that stands in for a missing NWP input.
    model.nwp_time_len = dataset.nwp_frames
    print(f"model: {args.model}  params: {model.num_parameters():,}")
    if kwargs:
        shown = {k: v for k, v in kwargs.items()
                 if k not in ("in_radar_c", "in_nwp_c", "in_len", "out_len")}
        if shown:
            print(f"model kwargs: {shown}")

    # An analytical baseline (advection, persistence) has no weights to fit:
    # its parameters are a declared constant, not something the data can
    # teach. Training it would burn a GPU for twenty epochs and produce the
    # same forecast it produces without any training at all, so it is
    # evaluated instead.
    if not getattr(model, "is_learnable", True):
        print(
            f"{args.model} is an analytical baseline (is_learnable = False): "
            "it is evaluated, not fitted. Nothing to train."
        )
        out_dir = Path(args.out_dir) / f"{args.model}_analytical"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "last.pt")
        run_config = dict(vars(args))
        run_config["nwp_pipeline_version"] = NWP_PIPELINE_VERSION
        (out_dir / "config.json").write_text(
            json.dumps(run_config, indent=2), encoding="utf-8"
        )
        (out_dir / "history.json").write_text("[]", encoding="utf-8")
        print(f"artifacts written to {out_dir}")
        return 0

    criterion = _build_criterion(args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    out_dir = Path(args.out_dir) / f"{args.model}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # The pipeline version travels with the checkpoint. A model trained
    # under one version of the input scaling cannot be scored under
    # another, and the evaluation refuses to pretend otherwise.
    run_config = dict(vars(args))
    run_config["nwp_pipeline_version"] = NWP_PIPELINE_VERSION
    (out_dir / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    # Some adapters accept an optional teacher-forcing signal. Passing it
    # turns on the mechanism the paper describes -- the DiffusionCast-style
    # residual prior, the FusionCast future-radar prior, the PW-FouCast
    # frequency memory. Without it those branches never receive a signal
    # and stay untrained.
    teacher_arg = _teacher_argument(model)
    if teacher_arg:
        print(f"teacher forcing: {teacher_arg}")

    step = 0
    history: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        running: list[float] = []
        epoch_start = time.time()
        for batch in loader:
            x_radar = batch["x_radar"].to(device)
            x_nwp = batch["x_nwp"].to(device)
            y = batch["y_rain"].to(device)
            # The observed history rainfall, for the models that invert
            # it. It is present only when the dataset supplies it.
            y_hist = batch.get("y_hist")
            if y_hist is not None:
                y_hist = y_hist.to(device)

            # A model that assimilates the observed history rainfall needs
            # it during training as well, not only during the evaluation.
            #
            # The evaluation publishes it on the model (``model.obs_y_hist``)
            # because the shared seam is ``model(x_radar, x_nwp)`` and a
            # third argument would make the model unevaluable. The training
            # loop has to do exactly the same thing, otherwise the model
            # spends the whole training with the observation channel empty
            # and only meets it for the first time when it is scored -- a
            # train/inference mismatch that makes the measured skill of the
            # observation meaningless.
            #
            # The lag is not applied here: the model works out which slots
            # it may trust itself (``valid_end = t_in - obs_lag_steps``) and
            # fills the rest with its own inversion. Publishing the whole
            # history here is therefore the same thing the evaluation does,
            # and the two agree by construction.
            if hasattr(model, "obs_y_hist"):
                model.obs_y_hist = y_hist

            if mode != MODE_FULL:
                x_nwp = apply_ablation(x_nwp, mode)
            if mode == MODE_NO_NWP:
                # "No NWP" is expressed as an absent modality: the model is
                # handed None and the base class resolves it to a zero tensor
                # of the right shape. The parameter count and the checkpoint
                # layout stay exactly what the multimodal run used, so the
                # two runs differ in information, not in capacity.
                x_nwp_model = None
            else:
                x_nwp_model = x_nwp

            if teacher_arg:
                pred = model(x_radar, x_nwp_model, **{teacher_arg: y})
            else:
                pred = model(x_radar, x_nwp_model)
            # The physics regulariser must see the same NWP the model saw,
            # or it would penalise the model for disagreeing with a field
            # the model was never given.
            #
            # The observed history rainfall is passed through for the
            # models that invert it (route R1). It is read with ``get``
            # rather than indexed, so a dataset that does not supply it
            # degrades to "no history supervision" instead of failing.
            if getattr(args, "qpe_single_frame", False):
                # Pure single-frame QPE. The model still emits T_out frames
                # (the fixed seam), but only its first frame carries a
                # learnable estimate; every later frame is an exact copy.
                # Slicing both sides to one frame keeps the objective on
                # the inversion itself and off the extrapolation, which is
                # the whole point of the route (MASTER §4.18).
                if y_hist is None:
                    raise RuntimeError(
                        "--qpe-single-frame requires the dataset to supply "
                        "y_hist (the observed history rainfall)"
                    )
                y = y_hist[:, -1:]
                pred = pred[:, :1]

            loss, components = criterion(
                pred, y, model=model, x_radar=x_radar,
                x_nwp=model.resolve_nwp(x_radar, x_nwp_model),
                y_hist=y_hist, qpe_weight=args.qpe_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running.append(components["total"])
            step += 1
            if step % args.log_every == 0:
                # The components differ between the objectives: the
                # standard loss reports its regression and its rain term
                # separately, the balanced one reports a single number.
                detail = " ".join(
                    f"{name}={value:.4f}"
                    for name, value in components.items()
                    if name != "total"
                )
                print(
                    f"  epoch {epoch} step {step} "
                    f"loss={components['total']:.4f} {detail}"
                )
            if args.max_steps and step >= args.max_steps:
                break

        mean_loss = float(np.mean(running)) if running else float("nan")
        history.append({"epoch": epoch, "loss": mean_loss,
                        "seconds": time.time() - epoch_start})
        print(f"epoch {epoch}: loss={mean_loss:.4f} "
              f"({time.time() - epoch_start:.1f}s)")
        torch.save(model.state_dict(), out_dir / "last.pt")
        # Written every epoch, not only at the end: a run that is killed
        # still leaves a usable record of how far it got.
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if args.max_steps and step >= args.max_steps:
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"artifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
