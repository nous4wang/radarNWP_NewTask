#!/usr/bin/env python3
"""Evaluate a trained baseline with the meteorological verification scores.

Usage:
    python scripts/evaluate.py --model mocast --checkpoint runs/mocast_x/last.pt \
        --data-root data/BJ_packed --packed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from radar_nwp_bench.metrics import PatchStitcher, PrecipitationVerifier
from radar_nwp_bench.models import build_model, list_models


def build_argparser() -> argparse.ArgumentParser:
    """Build the argument parser.

    The data-shaping flags are deliberately identical to those of
    ``scripts/train.py``: a flag that exists in one script and not the other
    is a latent crash (the parser would raise ``AttributeError`` on the
    missing namespace attribute), so the two are compared by a test.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list_models())
    ap.add_argument("--checkpoint", default=None, help="path to a state_dict")
    ap.add_argument(
        "--data-root", default=None,
        help="root of the dataset; may also come from the configuration file",
    )
    ap.add_argument("--packed", action="store_true")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--in-len", type=int, default=11)
    ap.add_argument("--out-len", type=int, default=10)
    ap.add_argument("--alignment", default="forecast",
                    choices=[a.value for a in NWPAlignment])
    ap.add_argument(
        "--nwp-vars", nargs="*", default=None,
        help="subset of NWP variables; must match the training run, or the "
             "model's NWP input has the wrong number of channels",
    )
    ap.add_argument(
        "--radar-vars", nargs="*", default=None,
        help="subset of radar variables; must match the training run",
    )
    ap.add_argument(
        "--no-nwp", action="store_true",
        help="Ablation: train or evaluate strictly without NWP inputs (pure radar model).",
    )
    ap.add_argument(
        "--zero-nwp", action="store_true",
        help="Ablation: zero out NWP tensors during evaluation to test NWP dependency.",
    )
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--no-interpolate", action="store_true",
        help="drop the windows whose history contains a missing frame, "
             "instead of rebuilding the frame by interpolation. Used to "
             "measure what the fault tolerance is worth.",
    )
    ap.add_argument(
        "--missing-frame-fill", default="linear",
        choices=["linear", "prev", "zero"],
        help="E30 counterfactual: how to fill a radar history frame that the "
             "dataset rebuilt by linear interpolation. 'linear' keeps the "
             "current behaviour bit-for-bit; 'prev' copies the preceding "
             "normal frame (zero-latency causal hold); 'zero' sets the slot "
             "to zero in the NORMALISED space, i.e. 0 dBZ, which is a weak "
             "echo and NOT a missing-data sentinel. Off by default; the "
             "main input/output contract is unchanged.",
    )
    ap.add_argument(
        "--thresholds", nargs="*", type=float, default=None,
        help="the rain rates in mm per 6-minute accumulation to score at. "
             "The default is the benchmark's set; 2.0 is the "
             "short-duration heavy-rain warning line (20 mm/h)",
    )
    ap.add_argument(
        "--out-json", default=None)
    ap.add_argument(
        "--per-window-out", default=None,
        help="also write the per-window contingency tables to this JSON "
             "file; they are what a bootstrap resamples to put an interval "
             "on the score")
    ap.add_argument(
        "--fss-radii", type=int, nargs="*", default=[],
        help="extra neighbourhood half-widths for the multiscale FSS, in "
             "pixels; a half-width of 1 is a 3x3 window, 2 is 5x5, 4 is 9x9. "
             "Off by default because the fraction field is recomputed for "
             "every half-width and the scoring is bound by the processor")
    ap.add_argument(
        "--dump-fields", default=None,
        help="optional prefix for two memory-mapped .npy files holding the "
             "stitched whole-field prediction and truth for every window "
             "(<prefix>.pred.npy / <prefix>.truth.npy, float32). Off by "
             "default; used by zero-training-compute downstream probes so "
             "the model only has to be run once.",
    )
    ap.add_argument(
        "--split", default="test",
        choices=["train", "val", "test", "robustness_missing", "all"],
        help="case-level split to evaluate on; 'robustness_missing' holds "
             "out the cases whose radar series has a hole in it; 'all' uses "
             "every case",
    )
    ap.add_argument(
        "--max-samples", type=int, default=None,
        help="cap the number of samples (quick evaluation)",
    )
    ap.add_argument(
        "--slide-stride", type=int, default=None,
        help="slide the patch over the full 301x301 grid with this stride "
             "(e.g. 32 -> a 10x10 grid of 32x32 patches per sample)",
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
        "--centre-only", action="store_true",
        help="score only the centre patch instead of the whole grid. The "
             "centre crop holds about 4.5 %% of the rainfall, so this "
             "reports a score that is almost independent of the model; it "
             "exists for debugging, not for reporting.",
    )
    return ap


def _apply_config(ap: argparse.ArgumentParser, argv: list[str] | None) -> tuple[dict, dict]:
    """Parse the command line with the configuration file applied.

    The file's values become argparse *defaults*, so an explicit flag on the
    command line always wins.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)

    config = load_config(known.config)
    ap.set_defaults(**argparse_defaults(config, "eval"))
    args = ap.parse_args(argv)
    if not args.data_root:
        ap.error(
            "--data-root is required (give it on the command line or set "
            "data.root in the configuration file)"
        )
    return args, config


def _training_run_config(checkpoint: str | None) -> dict | None:
    """Read the arguments the training run that produced this checkpoint used.

    The training script writes ``config.json`` next to the checkpoint it
    saves. It records which NWP and radar variables the model was built for,
    and those decide the *channel count* of the model's input. An
    evaluation that guesses them wrong builds a model with a different first
    layer and fails to load the weights with a shape mismatch -- the failure
    mode this indirection exists to prevent.

    Returns:
        The parsed mapping, or ``None`` when there is no training
        configuration to read.
    """
    if not checkpoint:
        return None
    path = Path(checkpoint).parent / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _trained_model_kwargs(trained: dict | None) -> dict[str, Any]:
    """The model constructor kwargs the training run was built with.

    ``train.py`` records its ``--model-kwargs`` argument verbatim in the
    ``config.json`` it writes next to the checkpoint.  A constructor
    argument that is not a weight -- an output clamp, a threshold -- does
    not appear in the ``state_dict``, so a checkpoint trained with one
    loads cleanly into a model built without it and is then scored on a
    model the training run never used.  The recorded value is what says
    which model was trained, so it is recovered here.

    Args:
        trained: the parsed ``config.json`` of the training run, or
            ``None`` when there is no training configuration to read.

    Returns:
        The recorded kwargs as a mapping.  Empty when the training run
        recorded none, when it recorded an empty string, or when what it
        recorded cannot be parsed.
    """
    if not trained:
        return {}
    raw = trained.get("model_kwargs")
    if raw is None or raw == "" or raw == {}:
        return {}
    if isinstance(raw, str):
        try:
            extra = json.loads(raw)
        except (TypeError, ValueError):
            print(
                f"the checkpoint's model_kwargs is not valid JSON ({raw!r}); "
                f"the model is built without it"
            )
            return {}
    elif isinstance(raw, dict):
        extra = dict(raw)
    else:
        return {}
    if not isinstance(extra, dict):
        return {}
    return dict(extra)


def _trained_ablation_mode(trained: dict) -> str | None:
    """Read the ablation mode the training run used, if it recorded one.

    Returns:
        One of the modes in :mod:`radar_nwp_bench.ablation`, or ``None``
        when there is no training configuration to read at all. An empty
        configuration means the run used the full input.
    """
    if trained is None:
        return None
    try:
        return resolve_mode(bool(trained.get("no_nwp")),
                           bool(trained.get("zero_nwp")))
    except ValueError:
        return None


def _build_dataset(args, nwp_vars, radar_vars, slide_stride=None):
    """Build the evaluation dataset.

    One function, because the evaluation needs the dataset twice: once to
    find out how big the grid is, and once more with a sliding stride when
    the patch is smaller than the grid. The two calls must agree about
    everything -- the variable subsets above all, since those decide the
    number of input channels the checkpoint expects.
    """
    if slide_stride is None:
        slide_stride = args.slide_stride
    kwargs = dict(
        packed=args.packed,
        in_len=args.in_len,
        out_len=args.out_len,
        patch_size=args.patch_size,
        alignment=NWPAlignment(args.alignment),
        random_crop=False,
        split=None if args.split == "all" else args.split,
        max_samples=args.max_samples,
        slide_stride=slide_stride,
        blank_nwp=(args.no_nwp),
        tolerate_one_missing_frame=not args.no_interpolate,
    )
    if nwp_vars:
        kwargs["nwp_vars"] = tuple(nwp_vars)
    if radar_vars:
        kwargs["radar_vars"] = tuple(radar_vars)
    return RadarNWPDataset(args.data_root, **kwargs)


def main() -> int:
    ap = build_argparser()
    args, config = _apply_config(ap, None)

    # --no-interpolate DROPS every window with a hole, so a counterfactual
    # fill would have nothing to act on.  Combining the two looks like a run
    # with a variable changed when in fact nothing changed.
    if args.no_interpolate and args.missing_frame_fill != "linear":
        ap.error(
            "--no-interpolate drops all gapped windows, so "
            "--missing-frame-fill would have no effect; choose one or the other"
        )

    # A long evaluation must be readable while it is running; Python
    # block-buffers stdout when it is redirected to a file.
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(line_buffering=True)

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )

    # The variable subsets decide the channel count of the model's input.
    # They come from the command line, or from the configuration file, or --
    # failing both -- from the training run that produced the checkpoint.
    # Taking them from the training run is what makes the two scripts
    # symmetric: a checkpoint trained on three NWP variables is evaluated on
    # those same three, whatever the evaluator's own defaults are.
    trained = _training_run_config(args.checkpoint)
    nwp_vars = args.nwp_vars
    radar_vars = args.radar_vars
    if nwp_vars is None and trained and trained.get("nwp_vars"):
        nwp_vars = list(trained["nwp_vars"])
        print(f"nwp variables taken from the training run: {' '.join(nwp_vars)}")
    if radar_vars is None and trained and trained.get("radar_vars"):
        radar_vars = list(trained["radar_vars"])
        print(f"radar variables taken from the training run: {' '.join(radar_vars)}")

    # The ablation mode is resolved once and then used by the dataset and
    # the forward pass alike.
    try:
        mode = resolve_mode(args.no_nwp, args.zero_nwp)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    # A checkpoint knows how it was trained, and evaluating it under a
    # different ablation is a mistake that produces a number rather than an
    # error: a model trained without the NWP, fed the real NWP at inference,
    # scores whatever it scores, and the number looks like a result. The
    # training run recorded its own mode, so the evaluation can refuse to
    # disagree with it silently.
    trained_mode = _trained_ablation_mode(trained)
    if trained_mode is not None and trained_mode != mode:
        if mode == MODE_FULL and trained_mode == MODE_NO_NWP:
            print(
                "ERROR: this checkpoint was trained without the NWP "
                "(--no-nwp) but the evaluation is feeding it the NWP. "
                "Pass --no-nwp to evaluate it as it was trained."
            )
            return 2
        if mode == MODE_FULL and trained_mode == MODE_ZERO_NWP:
            print(
                "NOTE: this checkpoint was trained with --zero-nwp; the "
                "evaluation is feeding it the NWP, which is the full "
                "multimodal input."
            )
        elif trained_mode == MODE_NO_NWP and mode != MODE_NO_NWP:
            print(
                f"NOTE: this checkpoint was trained without the NWP; the "
                f"evaluation is in '{mode}' mode."
            )

    # The same argument applies to the input scaling: a checkpoint trained
    # before the NWP was normalised expects the fields it was trained on,
    # and scoring it against normalised fields produces a number that looks
    # like a score and is not one.
    trained_version = (
        trained.get("nwp_pipeline_version") if trained else None
    )
    if trained_version and trained_version != NWP_PIPELINE_VERSION:
        print(
            f"ERROR: this checkpoint was trained under the input pipeline "
            f"'{trained_version}' but the evaluation runs "
            f"'{NWP_PIPELINE_VERSION}'. The two scales the NWP differently, "
            f"so the score would be meaningless. Retrain the model."
        )
        return 2

    if mode != MODE_FULL:
        print(f"ablation: {mode} -- {describe(mode)}")

    dataset = _build_dataset(args, nwp_vars, radar_vars, args.slide_stride)
    grid_h, grid_w = dataset.grid_shape

    # The patch has to be slid over the grid and the tiles stitched back
    # together before anything is scored. Scoring the tiles independently
    # double-counts every storm in the overlap strip and scores the halves of
    # a storm that straddles a tile boundary as misses -- the verification
    # must be done on the whole field.
    #
    # When no stride was given but the patch is smaller than the grid, the
    # evaluation slides by the patch size itself. The alternative -- scoring
    # the centre crop -- holds about 4.5 % of the rainfall, so it reports a
    # score that is almost independent of the model.
    if (args.slide_stride is None and args.patch_size < min(grid_h, grid_w)
            and not args.centre_only):
        dataset = _build_dataset(args, nwp_vars, radar_vars, args.patch_size)
        grid_h, grid_w = dataset.grid_shape

    print(
        f"evaluating {len(dataset)} samples "
        f"(split={args.split}, cases={len({s.case_path.stem for s in dataset.samples})})"
    )
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        seed=args.seed,
    )

    # The model's own configuration file supplies its hyperparameters, so the
    # checkpoint is loaded into exactly the architecture it was trained with.
    model_cfg = config
    if not model_cfg.get("model"):
        try:
            model_cfg = load_config(args.model)
        except FileNotFoundError:
            model_cfg = {}

    # A hyperparameter that was given on the command line during training
    # is not in the model's configuration file, so the training run's own
    # record is what has to be consulted. The bin count of the hierarchical
    # head is one: the head's parameters are sized by it, and a checkpoint
    # trained with five bins cannot be loaded into a six-bin head.
    #
    # The head's settings only apply to a model that has the head. A
    # baseline without it must not be given them: the constructor would
    # reject the keyword and the evaluation would fail before it read a
    # single sample.
    head_settings: dict[str, Any] = {}
    # The model constructor's kwargs are restored from the checkpoint's
    # config.json for the same reason head_bins and level_mode are: a
    # constructor argument that is not a weight -- an output clamp, a
    # threshold -- does not appear in the state_dict, so a checkpoint
    # trained with one loads cleanly into a model built without it and is
    # then scored on a model the training run never used.  The evaluation
    # has to rebuild the model the training run built, or it measures a
    # different model than the one on disk.
    restored_kwargs: dict[str, Any] = {}
    if args.checkpoint:
        trained = _training_run_config(args.checkpoint)
        if trained and trained.get("head_bins"):
            head_settings["head_bins"] = trained["head_bins"]
        if trained and trained.get("level_mode"):
            head_settings["level_mode"] = trained["level_mode"]
        restored_kwargs = _trained_model_kwargs(trained)

    if head_settings or restored_kwargs:
        import inspect

        from radar_nwp_bench.models import model_class

        model_class_ = model_class(args.model)
        accepted = (
            inspect.signature(model_class_.__init__).parameters
            if model_class_ is not None
            else {}
        )
        applicable = {k: v for k, v in head_settings.items() if k in accepted}
        applicable.update(
            {k: v for k, v in restored_kwargs.items() if k in accepted}
        )
        if applicable:
            model_cfg = dict(model_cfg)
            model_cfg["model"] = dict(model_cfg.get("model") or {})
            model_cfg["model"].update(applicable)
            for key, value in applicable.items():
                print(f"the checkpoint was trained with {key}={value}")
        elif restored_kwargs:
            print(
                f"{args.model} does not accept the constructor kwargs the "
                f"training run used ({sorted(restored_kwargs)}); they do not "
                f"apply to it"
            )
        else:
            print(
                f"{args.model} has no hierarchical head, so the head settings "
                f"of the training run do not apply to it"
            )

    kwargs = model_kwargs(
        model_cfg,
        shapes={
            "in_radar_c": len(dataset.radar_vars),
            "in_nwp_c": len(dataset.nwp_vars),
            "in_len": args.in_len,
            "out_len": args.out_len,
        },
        overrides=args.model_kwargs,
    )
    model = build_model(args.model, **kwargs).to(device)
    # The NWP slice count comes from the data, for the same reason it does
    # during training: the zero tensor that stands in for a blanked NWP has
    # to have the shape the model's NWP encoder expects.
    model.nwp_time_len = dataset.nwp_frames
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"loaded checkpoint {args.checkpoint}")
    model.eval()

    # The thresholds the scores are computed at.
    #
    # They have to be selectable, because the business thresholds are not
    # the default ones: 2.0 mm per 6-minute accumulation is 20 mm/h, which
    # is the short-duration heavy-rain warning line, and a report that
    # cannot produce that number cannot make that point. Reading a
    # threshold that is not there and reporting zero for it would be worse
    # than not reporting it.
    thresholds = tuple(args.thresholds) if args.thresholds else None
    verifier = PrecipitationVerifier(
        thresholds=thresholds,
        per_window=args.per_window_out is not None,
        fss_radii=tuple(args.fss_radii),
    )
    # Reference forecast: a dry forecast (no rain anywhere). The radar input
    # is normalised reflectivity rather than rainfall, so it cannot serve as a
    # persistence baseline without imposing a Z-R relation; the dry forecast is
    # the honest "no skill" reference for these scores.
    dry_reference = PrecipitationVerifier(thresholds=thresholds)

    # Stable identity for every scored window.  --no-interpolate DROPS the
    # windows whose history needs a rebuilt frame, so the two arms of the
    # fault-tolerance comparison do not contain the same windows (§4.16.4:
    # 962 vs 813).  A case/issue-time key lets a downstream tool restrict
    # both tables to their intersection before a PAIRED bootstrap (E28).
    window_keys: list[str] = []
    # E30: record, for every scored window, which of the eleven history
    # frames the dataset rebuilt by interpolation.  The information lives on
    # the SampleSpec, so it is identical for every tile of a window and does
    # not depend on which tile happened to complete the stitcher.
    window_missing_steps: list[list[int]] = []

    def key_for(sample_index: int) -> str:
        spec = dataset.samples[sample_index]
        return f"{Path(spec.case_path).name}:{int(spec.issue_time)}"

    def missing_for(sample_index: int) -> list[int]:
        spec = dataset.samples[sample_index]
        return [int(s) for s in spec.missing_radar_steps]

    # The patch has to be slid over the grid and the tiles stitched back
    # together before anything is scored. Scoring the tiles independently
    # double-counts every storm in the overlap strip and scores the halves of
    # a storm that straddles a tile boundary as misses -- the verification
    # must be done on the whole field.
    stitch = args.patch_size < min(grid_h, grid_w) and dataset.n_patch_positions > 1
    stitcher = None
    if stitch:
        stitcher = PatchStitcher(
            (grid_h, grid_w),
            n_frames=args.out_len,
            n_positions=dataset.n_patch_positions,
        )
        print(
            f"sliding the {args.patch_size}x{args.patch_size} patch over the "
            f"{grid_h}x{grid_w} grid and stitching "
            f"{dataset.n_patch_positions} tiles per window back together"
        )
    else:
        print(f"scoring the full {grid_h}x{grid_w} grid in one pass")

    # Optional: persist the stitched whole-field forecasts so downstream
    # zero-training-compute probes (multiscale FSS, phase relaxation, seed
    # ensembling, spectral rescans) can reuse one inference pass.  A memory
    # map keeps the RAM footprint flat.  The dtype MUST stay float32: the
    # labels are quantised at 0.1 mm per 6 min and float16 rounds 0.1 to
    # 0.0999756, which flips ~1e5 observed-rain pixels per window at the
    # 0.1 threshold. That breaks both the FSS anchor against eval.json and
    # the per-window hits+misses invariant required by paired bootstrap
    # (tools/_check_field_dtype.py). Continuous forecasts are likewise
    # rounded across the 0.5/2.0/4.0 thresholds. Disk cost is ~3.9 GB per
    # seed (prediction + truth), which the remote NVMe accommodates.
    field_pred = field_truth = field_filled = None
    if args.dump_fields:
        n_win = len(dataset.samples)
        pred_path = f"{args.dump_fields}.pred.npy"
        truth_path = f"{args.dump_fields}.truth.npy"
        Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
        field_pred = np.lib.format.open_memmap(
            pred_path, mode="w+", dtype=np.float32,
            shape=(n_win, args.out_len, grid_h, grid_w),
        )
        field_truth = np.lib.format.open_memmap(
            truth_path, mode="w+", dtype=np.float32,
            shape=(n_win, args.out_len, grid_h, grid_w),
        )
        field_filled = np.zeros(n_win, dtype=bool)
        print(f"dumping stitched fields to {pred_path} / {truth_path} "
              f"({n_win} windows)")

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x_radar = batch["x_radar"].to(device)
            x_nwp = batch["x_nwp"].to(device)
            y = batch["y_rain"].to(device)

            # E30: one-variable counterfactual on the interpolated history
            # frame.  The dataset has already written
            #   hist[step] = 0.5 * (frame_before_hole + frame_after_hole)
            # before the normalising transform.  'prev' restores the
            # preceding *normal* instant, 'zero' writes the normalised value
            # for 0 dBZ.  The source tensor is cloned first so that adjacent
            # holes all read the original, untouched frames.
            if args.missing_frame_fill != "linear":
                missing = batch["radar_missing_mask"].to(device)
                source = x_radar.clone()
                for b in range(missing.shape[0]):
                    steps = torch.nonzero(
                        missing[b], as_tuple=False
                    ).flatten().tolist()
                    for step in steps:
                        if args.missing_frame_fill == "zero":
                            x_radar[b, step].zero_()
                        elif step > 0:
                            x_radar[b, step] = source[b, step - 1]
                        # step == 0 has no preceding frame: leave the
                        # interpolated frame as-is; the subset tool reports
                        # how many windows are affected by this edge case.

            if mode != MODE_FULL:
                x_nwp = apply_ablation(x_nwp, mode)
            if mode == MODE_NO_NWP:
                # The model is handed no NWP at all; the base class resolves
                # it to a zero tensor of the right shape.
                x_nwp_model = None
            else:
                x_nwp_model = x_nwp

            # A model that assimilates the observed history rainfall
            # needs it at evaluation time too. The shared seam is
            # ``model(x_radar, x_nwp)``, so the observation is published
            # on the model rather than passed as a third argument -- that
            # way the baselines are untouched and the model still
            # evaluates.
            if hasattr(model, "obs_y_hist"):
                observed = batch.get("y_hist")
                model.obs_y_hist = (
                    observed.to(device) if observed is not None else None
                )
                # The model degrades to an empty history channel when no
                # observation is published, which keeps it evaluable on
                # its own. The evaluation is the layer that knows the
                # difference matters, so it refuses to score an arm that
                # was supposed to see observations and did not: the score
                # would be for a different task, and it would look like a
                # number rather than a mistake.
                lag = getattr(model, "obs_lag_steps", None)
                if (
                    lag is not None
                    and lag < x_radar.shape[1]
                    and model.obs_y_hist is None
                ):
                    raise RuntimeError(
                        f"the model expects observations "
                        f"(obs_lag_steps={lag} of {x_radar.shape[1]} "
                        f"frames) but the dataset supplied none"
                    )

            pred = model(x_radar, x_nwp_model)
            # the dataset yields millimetres; the model also outputs millimetres
            pred_mm = pred.clamp_min(0.0)
            truth_mm = y.clamp_min(0.0)

            if stitcher is not None:
                pred_np = pred_mm.cpu().numpy()
                truth_np = truth_mm.cpu().numpy()
                sample_idx = batch["case_index"].cpu().numpy()
                position_idx = batch["position_index"].cpu().numpy()
                origin = batch["patch_origin"].cpu().numpy()
                for b in range(pred_np.shape[0]):
                    sample_index = int(sample_idx[b])
                    stitcher.add(
                        sample_index,
                        int(position_idx[b]),
                        (int(origin[b][0]), int(origin[b][1])),
                        pred_np[b, :, 0],
                        truth_np[b, :, 0],
                    )
                    if stitcher.is_complete(sample_index):
                        s_pred, s_truth = stitcher.finalize(sample_index)
                        verifier.update(s_pred[None], s_truth[None])
                        window_keys.append(key_for(sample_index))
                        window_missing_steps.append(missing_for(sample_index))
                        dry_reference.update(
                            np.zeros_like(s_pred[None]), s_truth[None]
                        )
                        if field_pred is not None:
                            field_pred[sample_index] = s_pred.astype(np.float32)
                            field_truth[sample_index] = s_truth.astype(np.float32)
                            field_filled[sample_index] = True
                        stitcher.release(sample_index)
            else:
                pred_np = pred_mm.cpu().numpy()
                truth_np = truth_mm.cpu().numpy()
                verifier.update(pred_np, truth_np)
                dry_reference.update(np.zeros_like(pred_np), truth_np)
                idxs = batch["case_index"].cpu().numpy()
                for b in range(pred_np.shape[0]):
                    s_i = int(idxs[b])
                    window_keys.append(key_for(s_i))
                    window_missing_steps.append(missing_for(s_i))
                    if field_pred is not None:
                        field_pred[s_i] = pred_np[b, :, 0].astype(np.float32)
                        field_truth[s_i] = truth_np[b, :, 0].astype(np.float32)
                        field_filled[s_i] = True
            if args.max_batches and bi + 1 >= args.max_batches:
                break

    if stitcher is not None and stitcher.pending:
        print(
            f"WARNING: {len(stitcher.pending)} windows were never completed "
            "(the run stopped early); they are not scored"
        )

    report = verifier.report()
    report["dry_reference_csi"] = {
        k: v["csi"] for k, v in dry_reference.report()["thresholds"].items()
    }
    # The report has to say which ablation produced it. A score without that
    # cannot be placed in the comparison table, because "no NWP" and "NWP
    # blanked at inference" are different experiments that happen to involve
    # the same zero tensor.
    report["ablation"] = {
        "mode": mode,
        "description": describe(mode),
        "nwp_vars": list(dataset.nwp_vars),
        "radar_vars": list(dataset.radar_vars),
        "checkpoint": args.checkpoint,
    }
    print(f"\n{args.model} verification ({verifier.n_samples} samples)")
    print(verifier.summary())
    print("\nCSI vs the dry (no-rain) reference:")
    for thr, scores in report["thresholds"].items():
        base = report["dry_reference_csi"][thr]
        print(f"  {thr:>6} mm : CSI {scores['csi']:.3f}   dry reference {base:.3f}")

    # A report in which every critical success index is zero, and every
    # false-alarm rate is undefined, is not a score. It is the signature of
    # a model whose output never crosses a threshold -- which is exactly
    # what happens when a physical formula is fed the compressed radar
    # scale: a 45 dBZ storm comes back as 0.004 mm per 6 minutes, two
    # orders of magnitude below the weakest threshold, and the whole field
    # is scored as misses.
    #
    # Such a report is worse than no report, because it looks like a result:
    # the table prints "0.000" and the reader takes it as the baseline
    # being hopeless rather than as the baseline being broken. So the run
    # fails here instead, loudly, while the person who launched it is still
    # watching.
    csi_values = [
        scores["csi"] for scores in report["thresholds"].values()
    ]
    far_values = [
        scores["far"] for scores in report["thresholds"].values()
    ]
    degenerate = (
        len(csi_values) > 0
        and all(csi == 0.0 for csi in csi_values)
        and all(far != far for far in far_values)
    )
    if degenerate:
        print(
            "\nERROR: every CSI is zero and every false-alarm rate is undefined.\n"
            "       This is not a score -- the forecast never crossed a\n"
            "       threshold. Check the units the model is working in: the\n"
            "       radar input is log1p(dBZ)/6, and a physical relation such\n"
            "       as Z-R needs physical dBZ (expm1(x * 6)).\n"
            "       No report was written.",
            file=sys.stderr,
        )
        return 3

    if field_pred is not None:
        field_pred.flush()
        field_truth.flush()
        print(f"\nstitched fields dumped: {int(field_filled.sum())}/"
              f"{len(field_filled)} windows "
              f"({args.dump_fields}.pred.npy / .truth.npy)")
        if not field_filled.all():
            print("WARNING: some windows were never completed; downstream "
                  "probes must check the filled mask", file=sys.stderr)

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"\nreport written to {args.out_json}")
    if args.per_window_out:
        if len(window_keys) != len(verifier.window_counts):
            raise RuntimeError(
                f"window key count {len(window_keys)} != scored window count "
                f"{len(verifier.window_counts)}; refusing to write an "
                f"unalignable per-window table"
            )
        if len(window_missing_steps) != len(window_keys):
            raise RuntimeError(
                f"missing-frame list {len(window_missing_steps)} != key count "
                f"{len(window_keys)}; refusing to write an unalignable table"
            )
        Path(args.per_window_out).write_text(
            json.dumps(
                {
                    "n_windows": len(verifier.window_counts),
                    "thresholds": [f"{t}" for t in verifier.thresholds],
                    # Extra fields, additive: every existing consumer reads
                    # only "windows", so the shape contract (§20) is intact.
                    "window_keys": window_keys,
                    "missing_frame_steps": window_missing_steps,
                    "windows": verifier.per_window_table(),
                },
                indent=2,
            ),
            encoding="utf-8"
        )
        print(
            f"\nper-window table written to {args.per_window_out} "
            f"({len(verifier.window_counts)} windows)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
