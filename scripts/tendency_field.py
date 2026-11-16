#!/usr/bin/env python3
"""The Koopman tendency field of the quantum-inspired NWP plugin, as a figure.

The plugin does not feed the NWP to the radar backbone directly: it turns
the NWP into a *tendency* field first. The state projector compresses the
NWP slice into a low-rank state, the Koopman flow evolves the analysis
state with a skew-symmetric generator and contrasts the evolved state with
the forecast state, and the difference is what modulates the radar
features. The claim this figure is meant to support is that the tendency
detects convective energy release *before* the radar echo shows it, and
that it tracks the CAPE the column gives up.

Three fields are drawn per window, always on the **stitched full grid**
and never on a tile: a tile measures the patch, and the storm that
matters straddles the tiles.

    (a) the radar composite reflectivity at the issue time, with the
        tendency field drawn over it as arrows and streamlines;
    (b) the CAPE consumption between the two NWP slices the model uses
        (the analysis hour and the forecast hour), with the tendency
        magnitude in contours;
    (c) the per-pixel scatter of the tendency magnitude against the CAPE
        consumption, with the Pearson correlation coefficient.

The CAPE consumption is a difference of two *normalised* fields, so the
normalisation is inverted before the difference is taken: a difference of
two normalised values is not a CAPE change in physical units.

Usage:
    python scripts/tendency_field.py --data-root data/BJ_packed --packed \
        --split test --model simvp_v2_quantum \
        --checkpoint runs/simvp_v2_quantum/last.pt \
        --out-dir figures/tendency

    # one explicit window:
    python scripts/tendency_field.py --data-root data/BJ_packed --packed \
        --case CP_01_06230359_00001 --issue-time 06231500 \
        --model simvp_v2_quantum --checkpoint runs/.../last.pt \
        --out-dir figures/tendency

    # no checkpoint: the CAPE diagnostics only. The tendency of an
    # untrained model is not a physical quantity, so it is not plotted.
    python scripts/tendency_field.py --data-root data/BJ_packed --packed \
        --out-dir figures/tendency_cape

    # numerical self-check, no data needed:
    python scripts/tendency_field.py --self-check
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_nwp_bench.config import argparse_defaults, load_config, model_kwargs
from radar_nwp_bench.data import (
    NWPAlignment,
    RadarNWPDataset,
    build_dataloader,
)
from radar_nwp_bench.data.coordinates import (
    NWP_SHAPE,
    RADAR_SHAPE,
    bilinear_upsample_center_crop,
)
from radar_nwp_bench.data.index import minutes_between, scan_case
from radar_nwp_bench.data.pack import load_packed_mmap
from radar_nwp_bench.data.transforms import (
    NWP_NORM,
    NWP_PIPELINE_VERSION,
    clean_nwp_field,
    normalize_nwp_field,
    radar_log_transform,
)
from radar_nwp_bench.metrics import PatchStitcher
from radar_nwp_bench.models import build_model, list_models
from radar_nwp_bench.models.common import KoopmanUnitaryFlow

#: The radar/label grid: 301 x 301 cells (radar_nwp_bench/data/coordinates.py).
GRID_SIZE = 301

#: The NWP variable the CAPE diagnostic is built around.
CAPE_VAR = "CAPE"

#: The radar variable that carries the composite reflectivity.
REFLECTIVITY_VAR = "CR"

#: The reflectivity above which a pixel counts as "echo" (dBZ).
#:
#: The maximum of a reflectivity field is decided by a single pixel, so it
#: is a poor measure of *when the echo appears*: one noisy pixel can move
#: it by tens of minutes. The area of the moderate-or-heavier echo is a
#: far more stable statistic, and it is what the timing analysis uses.
ECHO_THRESHOLD_DBZ = 30.0

#: The stitched fields, in the order they are stacked for the stitcher.
#:
#: The order is fixed because the stitcher carries the frames of a window
#: in one array: the tendency is the first frame when a model is running,
#: and the CAPE consumption takes its place when it is not.
FRAME_NAMES: tuple[str, ...] = (
    "cape_consumption",
    "cape_analysis",
    "cape_forecast",
    "cr_issue",
    "cr_hist_mid",
    "cr_hist_old",
)

#: The name of the tendency frame, which only exists when a model runs.
TENDENCY_FRAME = "tendency"


# ---------------------------------------------------------------------------
# The physical conversions
# ---------------------------------------------------------------------------
def denormalize_nwp(values: np.ndarray, var_name: str) -> np.ndarray:
    """Invert :func:`~radar_nwp_bench.data.transforms.normalize_nwp_field`.

    The dataset hands the NWP over already centred and scaled, so the
    physical field has to be recovered before any difference of two NWP
    slices means anything. The normalisation is affine and per variable,
    so the inverse is exact.

    Args:
        values: the normalised field, any shape.
        var_name: the NWP variable the field belongs to.

    Returns:
        The field in the variable's own physical unit (J/kg for CAPE).
    """
    out = np.asarray(values, dtype=np.float32)
    stats = NWP_NORM.get(var_name)
    if stats is None:
        return out
    mean, std = stats
    return (out * np.float32(std) + np.float32(mean)).astype(np.float32)


def cape_consumption(analysis: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """The CAPE the column gives up between the analysis and the forecast hour.

    Both arguments are the **normalised** fields the dataset returns, and
    the result is in J/kg. Negative means CAPE was released, which is
    what a convective column does when it rains.

    The normalisation is inverted before the difference is taken. The
    offset cancels in the difference, so only the scale matters -- but
    taking the difference of the normalised fields and calling it J/kg
    would be wrong by a factor of the CAPE standard deviation (761.5).

    Args:
        analysis: the normalised CAPE of the analysis slice.
        forecast: the normalised CAPE of the forecast slice.

    Returns:
        ``forecast - analysis`` in J/kg.
    """
    return (
        denormalize_nwp(forecast, CAPE_VAR)
        - denormalize_nwp(analysis, CAPE_VAR)
    ).astype(np.float32)


def dbz_from_network(values: np.ndarray) -> np.ndarray:
    """Invert :func:`~radar_nwp_bench.data.transforms.radar_log_transform`.

    The radar fields reach the network as ``log1p(dBZ) / 6``, so the dBZ
    of the composite reflectivity is recovered with ``expm1``.

    Args:
        values: the transformed field.

    Returns:
        The field in dBZ.
    """
    return np.expm1(
        np.asarray(values, dtype=np.float32) * np.float32(6.0)
    ).astype(np.float32)


def tendency_magnitude(d_psi: torch.Tensor) -> torch.Tensor:
    """The magnitude of the Koopman tendency field.

    The tendency is a *multi-channel* feature field: the plugin projects
    the NWP into an ``embed_dim``-wide state and evolves it, so there is
    one tendency value per channel per pixel and no single channel has a
    physical unit. The magnitude is the Euclidean norm over the channels,
    which is the one number that says "how strongly does the plugin think
    something is happening here".

    Args:
        d_psi: ``(B, C, H, W)`` -- the tendency the Koopman flow produced.

    Returns:
        ``(B, H, W)`` -- the per-pixel magnitude.
    """
    return torch.linalg.vector_norm(d_psi, dim=1)


def tendency_flow(magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A planar flow that points at the growing tendency.

    The plugin's tendency is a multi-channel feature field, not a planar
    vector field, so there is no vector to draw and the figure does not
    pretend otherwise. What is drawn is the *magnitude* (the arrow length)
    and the direction in which that magnitude grows (the arrow direction):
    the arrows point towards the region where the tendency is building
    up, and the streamlines converge on it.

    The gradient is regularised by a fraction of its own maximum, so a
    plateau (where the gradient vanishes but the magnitude does not) gives
    a short arrow rather than an arbitrary direction.

    Args:
        magnitude: the per-pixel tendency magnitude, ``(H, W)``.

    Returns:
        The ``(u, v)`` flow, in the same shape as the input.
    """
    field_values = np.asarray(magnitude, dtype=np.float64)
    if field_values.ndim != 2:
        raise ValueError(f"expected a 2-D field, got {field_values.shape}")
    gy, gx = np.gradient(field_values)
    norm = np.hypot(gx, gy)
    peak = float(norm.max()) if norm.size else 0.0
    delta = 0.05 * peak if peak > 0 else 1.0
    u = field_values * gx / (norm + delta)
    v = field_values * gy / (norm + delta)
    return u.astype(np.float32), v.astype(np.float32)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """The Pearson correlation of two arrays.

    Non-finite pairs are dropped, because one NaN would otherwise take
    the whole coefficient with it.

    Args:
        x: the first variable.
        y: the second variable.

    Returns:
        The correlation, or ``nan`` when there is nothing to correlate.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"size mismatch: {a.size} vs {b.size}")
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denominator <= 0.0:
        return float("nan")
    return float((a * b).sum() / denominator)


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """The Spearman rank correlation of two arrays.

    The rank correlation is reported next to the Pearson one because the
    tendency magnitude is heavy-tailed: a handful of pixels carry a large
    share of the total, and a rank statistic says whether the ordering
    survives that.

    Args:
        x: the first variable.
        y: the second variable.

    Returns:
        The rank correlation, or ``nan`` when there is nothing to correlate.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if a.size < 2:
        return float("nan")
    return pearson_r(_rank(a), _rank(b))


def _rank(values: np.ndarray) -> np.ndarray:
    """The ranks of an array, with ties given their average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # Average the ranks inside a group of equal values.
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


# ---------------------------------------------------------------------------
# Capturing the tendency
# ---------------------------------------------------------------------------
class TendencyCapture:
    """Captures the tendency field the Koopman flow actually produced.

    A **forward hook** is used rather than recomputing the tendency from
    the model's own submodules. The hook reads the tensor the model used,
    so the figure cannot drift away from the quantity that actually
    modulates the radar features: a recomputation would silently disagree
    with the model the moment the plugin's internals changed (a different
    state dimension, a resampling step, a new term in the generator), and
    the disagreement would not be visible -- it would just be a figure of
    something the model never saw.

    Args:
        model: the baseline to watch.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._handles: list[Any] = []
        self._last: torch.Tensor | None = None
        self.n_calls = 0
        self.n_modules = 0
        for module in model.modules():
            if _is_koopman_flow(module):
                self._handles.append(module.register_forward_hook(self._hook))
                self.n_modules += 1

    def _hook(self, module, inputs, output) -> None:
        """Record the tendency the module produced."""
        self._last = output
        self.n_calls += 1

    @property
    def last(self) -> torch.Tensor | None:
        """The tendency of the most recent forward pass, or ``None``."""
        return self._last

    def clear(self) -> None:
        """Forget the recorded tendency, so a stale one cannot be used."""
        self._last = None

    def close(self) -> None:
        """Remove the hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _is_koopman_flow(module: torch.nn.Module) -> bool:
    """Whether a module is a :class:`KoopmanUnitaryFlow`.

    The class is matched by identity *and* by name: the registry loads a
    baseline by file path, so a model file that was imported twice would
    hold a second, different class object with the same name.
    """
    if isinstance(module, KoopmanUnitaryFlow):
        return True
    return type(module).__name__ == "KoopmanUnitaryFlow"


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    """Build the argument parser.

    The data-shaping flags deliberately mirror those of
    ``scripts/evaluate.py``: a flag that exists in one script and not the
    other is a latent crash, so the two are compared by a test.
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-root", default=None,
        help="root of the dataset; may also come from the configuration file",
    )
    ap.add_argument("--packed", action="store_true", help="read .npz containers")
    ap.add_argument(
        "--split", default="test", choices=["train", "val", "test", "all"],
        help="case-level split to analyse; 'all' uses every case",
    )
    ap.add_argument(
        "--model", default=None, choices=list_models(),
        help="baseline to run; required when --checkpoint is given",
    )
    ap.add_argument(
        "--checkpoint", default=None,
        help="path to a state_dict. Without it only the NWP diagnostics are "
             "produced: the tendency of an untrained model is not a "
             "physical quantity.",
    )
    ap.add_argument(
        "--out-dir", default=None,
        help="directory for tendency_field.json and the figures",
    )
    ap.add_argument(
        "--case", default=None,
        help="case id (the file stem of the case) to analyse. Without it "
             "the windows with the largest CAPE consumption are picked.",
    )
    ap.add_argument(
        "--issue-time", default=None,
        help="issue time of the window to analyse, as MMDDHHMM "
             "(e.g. 06231500). Without it the window with the largest CAPE "
             "consumption is picked.",
    )
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument(
        "--slide-stride", type=int, default=64,
        help="slide the patch over the full 301x301 grid with this stride",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--device", default="auto",
        help="'auto' uses CUDA when it is available, otherwise the CPU",
    )
    ap.add_argument(
        "--max-cases", type=int, default=8,
        help="how many windows to select (at most one per case, so the "
             "figure set shows distinct storms)",
    )
    ap.add_argument(
        "--time-scan", type=int, default=12,
        help="how many issue times of the selected case to scan when "
             "looking for the time the tendency peaks (0 disables the scan)",
    )
    ap.add_argument(
        "--time-stride", type=int, default=3,
        help="the scan steps this many sample windows at a time (one sample "
             "is one 6-minute radar step)",
    )
    ap.add_argument("--in-len", type=int, default=11)
    ap.add_argument("--out-len", type=int, default=10)
    ap.add_argument(
        "--alignment", default="forecast",
        choices=[a.value for a in NWPAlignment],
    )
    ap.add_argument("--nwp-vars", nargs="*", default=None)
    ap.add_argument("--radar-vars", nargs="*", default=None)
    ap.add_argument(
        "--arrow-stride", type=int, default=8,
        help="draw one arrow every N pixels of the field",
    )
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--config", default=None,
        help="configuration file or name (e.g. 'default', 'bj'). "
             "Explicit command-line flags override the file.",
    )
    ap.add_argument(
        "--model-kwargs", default=None,
        help="JSON object of extra constructor kwargs for the baseline",
    )
    ap.add_argument(
        "--self-check", action="store_true",
        help="run the numerical self-check on synthetic fields and exit; "
             "no dataset is needed",
    )
    return ap


def _apply_config(
    ap: argparse.ArgumentParser, argv: list[str] | None
) -> tuple[dict, dict]:
    """Parse the command line with the configuration file applied.

    The file's values become argparse *defaults*, so an explicit flag on
    the command line always wins.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)

    config = load_config(known.config)
    ap.set_defaults(**argparse_defaults(config, "eval"))
    args = ap.parse_args(argv)
    return args, config


def _training_run_config(checkpoint: str | None) -> dict | None:
    """Read the arguments of the training run that produced a checkpoint.

    The training script writes ``config.json`` next to the checkpoint it
    saves. It records which NWP and radar variables the model was built
    for, and those decide the *channel count* of the model's input. A
    diagnostic that guesses them wrong builds a model with a different
    first layer and fails to load the weights with a shape mismatch.

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


def _build_dataset(args, nwp_vars, radar_vars, slide_stride=None):
    """Build the diagnostic dataset.

    One function, because the analysis needs the dataset twice: once to
    find out how big the grid is, and once more with a sliding stride
    when the patch is smaller than the grid. The two calls must agree
    about everything -- the variable subsets above all, since those decide
    the number of input channels the checkpoint expects.
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
        max_samples=None,
        slide_stride=slide_stride,
    )
    if nwp_vars:
        kwargs["nwp_vars"] = tuple(nwp_vars)
    if radar_vars:
        kwargs["radar_vars"] = tuple(radar_vars)
    return RadarNWPDataset(args.data_root, **kwargs)


# ---------------------------------------------------------------------------
# The CAPE field of a case
# ---------------------------------------------------------------------------
class CaseCape:
    """The CAPE field of one case, on the radar grid, in J/kg.

    The ranking statistic has to be the same quantity the figure draws, so
    the CAPE is brought onto the radar grid the way the dataset brings it
    (a bilinear upsample of the NWP's native grid, then the centre crop)
    rather than being read on the native grid, where the corners of the
    NWP domain lie outside the radar's field of view.

    Args:
        dataset: the dataset the case belongs to; it says whether the
            archive is packed and where the case lives.
        case_path: the path of the case.
    """

    def __init__(self, dataset: RadarNWPDataset, case_path: Path) -> None:
        self._cube = self._load(dataset, case_path)
        self._grid: dict[int, np.ndarray] = {}

    def _load(self, dataset: RadarNWPDataset, case_path: Path) -> np.ndarray:
        """Read the CAPE of every NWP time of the case, in J/kg."""
        if dataset.packed:
            if case_path.is_dir():
                loaded = load_packed_mmap(case_path)
                nwp = loaded["nwp"]
                names = [str(v) for v in loaded["nwp_vars"]]
                index = names.index(CAPE_VAR)
                return np.asarray(nwp[:, index], dtype=np.float32)
            with np.load(case_path, allow_pickle=False) as data:
                nwp = data["nwp"]
                names = [str(v) for v in data["nwp_vars"]]
                index = names.index(CAPE_VAR)
                return np.asarray(nwp[:, index], dtype=np.float32)

        index = scan_case(case_path)
        frames = []
        for stamp in index.nwp_times:
            path = index.nwp_file(CAPE_VAR, stamp)
            if path is None:
                frames.append(np.zeros(NWP_SHAPE, dtype=np.float32))
            else:
                frames.append(clean_nwp_field(np.load(path)))
        return np.stack(frames).astype(np.float32)

    def grid(self, time_index: int) -> np.ndarray:
        """The CAPE of one NWP time, on the radar grid, in J/kg."""
        if time_index not in self._grid:
            self._grid[time_index] = bilinear_upsample_center_crop(
                self._cube[time_index], radar_shape=RADAR_SHAPE
            )
        return self._grid[time_index]


def window_consumption(cape: CaseCape, spec) -> np.ndarray:
    """The CAPE consumption of one window, on the radar grid, in J/kg.

    The two slices are the ones the plugin itself uses: the first and the
    last of the window's NWP window, which under the default ``forecast``
    alignment are the analysis hour and the forecast hour.

    Args:
        cape: the CAPE field of the case.
        spec: the sample specification of the window.

    Returns:
        The consumption field, ``(H, W)``, in J/kg. Negative means CAPE
        was released.
    """
    analysis = cape.grid(spec.nwp_idx[0])
    forecast = cape.grid(spec.nwp_idx[-1])
    return (forecast - analysis).astype(np.float32)


# ---------------------------------------------------------------------------
# The window choice
# ---------------------------------------------------------------------------
@dataclass
class WindowChoice:
    """One candidate window and its CAPE consumption."""

    sample_index: int
    case_id: str
    issue_time: int
    consumption_mean: float
    consumption_min: float

    @property
    def issue_time_str(self) -> str:
        """The issue time as an ``MMDDHHMM`` string."""
        return f"{int(self.issue_time):08d}"


def rank_windows(dataset: RadarNWPDataset) -> list[WindowChoice]:
    """Rank every window of the split by its CAPE consumption.

    The rule is explicit, because a figure of "a" case invites the
    question "why that one?". The windows are ranked by the **mean CAPE
    consumption over the 301x301 radar grid** -- the forecast hour minus
    the analysis hour, in J/kg -- and the most negative (the largest
    release) comes first.

    Args:
        dataset: the dataset to rank.

    Returns:
        The windows, most negative consumption first.
    """
    by_case: dict[str, list[int]] = {}
    for index, spec in enumerate(dataset.samples):
        by_case.setdefault(spec.case_path.stem, []).append(index)

    choices: list[WindowChoice] = []
    for case_id in sorted(by_case):
        indices = by_case[case_id]
        cape = CaseCape(dataset, dataset.samples[indices[0]].case_path)
        for index in indices:
            spec = dataset.samples[index]
            consumption = window_consumption(cape, spec)
            choices.append(WindowChoice(
                sample_index=index,
                case_id=case_id,
                issue_time=spec.issue_time,
                consumption_mean=float(consumption.mean()),
                consumption_min=float(consumption.min()),
            ))
    choices.sort(key=lambda c: (c.consumption_mean, c.issue_time))
    return choices


def select_windows(
    choices: list[WindowChoice],
    args,
    case_ids: set[str],
    wanted_issue: int | None,
) -> tuple[list[WindowChoice], str]:
    """Choose the windows the figure is drawn for.

    With ``--case`` and ``--issue-time`` the choice is explicit and
    reproducible. Without them the windows with the largest CAPE
    consumption are taken, at most one per case so that the figure set
    shows distinct storms.

    Args:
        choices: the ranked windows.
        args: the parsed arguments.
        case_ids: every case id the split holds.
        wanted_issue: the requested issue time, or ``None``.

    Returns:
        The chosen windows and the rule that chose them.

    Raises:
        ValueError: when the requested case or issue time does not exist.
    """
    if args.case is not None and args.case not in case_ids:
        listed = (
            ", ".join(sorted(case_ids))
            if len(case_ids) <= 12
            else f"{len(case_ids)} cases"
        )
        raise ValueError(
            f"unknown case {args.case!r}; the split {args.split} holds {listed}"
        )

    candidates = choices
    if args.case is not None:
        candidates = [c for c in candidates if c.case_id == args.case]
    if wanted_issue is not None:
        candidates = [c for c in candidates if c.issue_time == wanted_issue]

    if not candidates:
        if wanted_issue is not None:
            raise ValueError(
                f"no window in the split {args.split} has the issue time "
                f"{wanted_issue:08d}"
                + (f" in the case {args.case}" if args.case else "")
            )
        raise ValueError(
            f"no window matches the request (case={args.case}, "
            f"issue_time={args.issue_time})"
        )

    if args.case is not None and wanted_issue is not None:
        rule = (
            f"the window was requested explicitly: case {args.case}, issue "
            f"time {wanted_issue:08d}."
        )
        return [candidates[0]], rule

    if args.case is not None:
        chosen = candidates[0]
        rule = (
            f"the case {args.case} was requested; inside it the window with "
            f"the largest CAPE consumption was taken (issue time "
            f"{chosen.issue_time_str}, mean consumption "
            f"{chosen.consumption_mean:+.1f} J/kg)."
        )
        return [chosen], rule

    # At most one window per case: the best one of each case, then the
    # cases whose best window releases the most CAPE.
    best_per_case: dict[str, WindowChoice] = {}
    for choice in candidates:
        best_per_case.setdefault(choice.case_id, choice)
    chosen = sorted(
        best_per_case.values(), key=lambda c: (c.consumption_mean, c.issue_time)
    )[: max(1, int(args.max_cases))]
    rule = (
        "the windows of the split are ranked by the mean CAPE consumption "
        f"over the {GRID_SIZE}x{GRID_SIZE} radar grid (the forecast hour "
        "minus the analysis hour, in J/kg), the most negative -- the largest "
        "release -- first; "
        f"at most one window per case is used and the first {len(chosen)} "
        "cases of that ranking are taken."
    )
    return chosen, rule


def scan_windows(
    dataset: RadarNWPDataset,
    chosen: WindowChoice,
    n_scan: int,
    stride: int,
) -> list[int]:
    """The issue times of the case that the timing analysis walks over.

    The scan walks forward from the chosen window first (the claim is
    that the tendency leads the echo, so the interesting times are the
    ones after it) and backwards afterwards if there is room.

    Args:
        dataset: the dataset the window belongs to.
        chosen: the window the figure is drawn for.
        n_scan: how many issue times to visit.
        stride: how many sample windows to step at a time.

    Returns:
        The indices (into ``dataset.samples``) of the scanned windows.
    """
    if n_scan <= 0:
        return [chosen.sample_index]

    same_case = [
        index for index, spec in enumerate(dataset.samples)
        if spec.case_path.stem == chosen.case_id
    ]
    same_case.sort(key=lambda i: dataset.samples[i].issue_time)
    if chosen.sample_index not in same_case:
        return [chosen.sample_index]
    position = same_case.index(chosen.sample_index)

    step = max(1, int(stride))
    picked: list[int] = []
    offset = 0
    while len(picked) < n_scan and offset < len(same_case):
        index = position + offset
        if 0 <= index < len(same_case):
            picked.append(same_case[index])
        offset += step
    offset = -step
    while len(picked) < n_scan and offset > -len(same_case):
        index = position + offset
        if 0 <= index < len(same_case):
            picked.insert(0, same_case[index])
        offset -= step
    return picked


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------
@dataclass
class WindowFields:
    """The stitched fields of one window."""

    case_id: str
    issue_time: int
    nwp_lead: list[int] = field(default_factory=list)
    frames: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def issue_time_str(self) -> str:
        """The issue time as an ``MMDDHHMM`` string."""
        return f"{int(self.issue_time):08d}"

    def get(self, name: str) -> np.ndarray | None:
        """One of the stitched fields, or ``None`` when it was not produced."""
        return self.frames.get(name)


def frame_names(in_len: int, has_tendency: bool) -> list[str]:
    """The stitched frames, in the order the stitcher carries them.

    The radar history frames only exist when the history is long enough
    to hold them: the frame 30 minutes before the issue time needs six
    frames of history and the one 60 minutes before needs eleven.

    Args:
        in_len: the number of radar history frames.
        has_tendency: whether a model is running.

    Returns:
        The names of the frames.
    """
    names = []
    if has_tendency:
        names.append(TENDENCY_FRAME)
    names.extend(["cape_consumption", "cape_analysis", "cape_forecast",
                  "cr_issue"])
    if in_len >= 6:
        names.append("cr_hist_mid")
    if in_len >= 11:
        names.append("cr_hist_old")
    return names


def analyse_windows(
    dataset: RadarNWPDataset,
    loader,
    model: torch.nn.Module | None,
    capture: TendencyCapture | None,
    wanted: set[int],
    grid_shape: tuple[int, int],
    n_positions: int,
    cr_index: int,
    cape_index: int,
    in_len: int,
    device: torch.device,
) -> dict[int, WindowFields]:
    """Run the selected windows and stitch their fields back together.

    The tendency is captured per patch and stitched with the same
    :class:`~radar_nwp_bench.metrics.stitching.PatchStitcher` the
    verification uses, so the figure shows the same full grid the scores
    are computed on.

    Args:
        dataset: the dataset the windows come from.
        loader: the data loader over the selected windows.
        model: the baseline, or ``None`` when no checkpoint was given.
        capture: the hook that records the tendency.
        wanted: the indices of the windows to run.
        grid_shape: the full grid, ``(H, W)``.
        n_positions: the number of tiles each window is cut into.
        cr_index: the channel of the composite reflectivity.
        cape_index: the channel of CAPE.
        in_len: the number of radar history frames.
        device: where to run the model.

    Returns:
        The stitched fields, keyed by the window index.

    Raises:
        RuntimeError: when the plugin produced no tendency at all.
    """
    names = frame_names(in_len, model is not None)
    stitcher = PatchStitcher(
        grid_shape, n_frames=len(names), n_positions=n_positions
    )
    results: dict[int, WindowFields] = {}
    leads: dict[int, list[int]] = {}

    with torch.no_grad():
        for batch in loader:
            x_radar = batch["x_radar"].to(device)
            x_nwp = batch["x_nwp"].to(device)

            tiles: dict[str, np.ndarray] = {}

            if model is not None:
                assert capture is not None
                capture.clear()
                model(x_radar, x_nwp)
                d_psi = capture.last
                if d_psi is None:
                    raise RuntimeError(
                        "the plugin produced no tendency field: the Koopman "
                        "flow was not called. This happens when the NWP is "
                        "blanked (the coupler passes through untouched)."
                    )
                tiles[TENDENCY_FRAME] = (
                    tendency_magnitude(d_psi).detach().cpu().numpy()
                )

            # The CAPE slices the plugin itself uses: the first and the
            # last of the window's NWP window.
            cape_norm = x_nwp[:, :, cape_index]
            analysis = _denormalize_tensor(cape_norm[:, 0], CAPE_VAR)
            forecast = _denormalize_tensor(cape_norm[:, -1], CAPE_VAR)
            tiles["cape_analysis"] = analysis.cpu().numpy()
            tiles["cape_forecast"] = forecast.cpu().numpy()
            tiles["cape_consumption"] = (forecast - analysis).cpu().numpy()

            # The composite reflectivity at the issue time and, when the
            # history is long enough, a couple of frames before it.
            tiles["cr_issue"] = _dbz_tensor(x_radar[:, -1, cr_index])
            if "cr_hist_mid" in names:
                tiles["cr_hist_mid"] = _dbz_tensor(x_radar[:, -6, cr_index])
            if "cr_hist_old" in names:
                tiles["cr_hist_old"] = _dbz_tensor(x_radar[:, -11, cr_index])

            stack = np.stack(
                [tiles[name] for name in names], axis=1
            ).astype(np.float32)

            sample_idx = batch["case_index"].cpu().numpy()
            position_idx = batch["position_index"].cpu().numpy()
            origin = batch["patch_origin"].cpu().numpy()
            lead = batch["nwp_lead"].cpu().numpy()
            for b in range(stack.shape[0]):
                index = int(sample_idx[b])
                if index not in wanted:
                    continue
                leads.setdefault(index, [int(v) for v in lead[b]])
                stitcher.add(
                    index,
                    int(position_idx[b]),
                    (int(origin[b][0]), int(origin[b][1])),
                    stack[b],
                    stack[b],
                )
                if stitcher.is_complete(index):
                    stitched, _ = stitcher.finalize(index)
                    spec = dataset.samples[index]
                    results[index] = WindowFields(
                        case_id=spec.case_path.stem,
                        issue_time=spec.issue_time,
                        nwp_lead=leads.get(index, []),
                        frames={
                            name: stitched[i].astype(np.float32)
                            for i, name in enumerate(names)
                        },
                    )
                    stitcher.release(index)
    return results


def _denormalize_tensor(values: torch.Tensor, var_name: str) -> torch.Tensor:
    """Invert the NWP normalisation on a torch tensor."""
    stats = NWP_NORM.get(var_name)
    if stats is None:
        return values
    mean, std = stats
    return values * std + mean


def _dbz_tensor(values: torch.Tensor) -> np.ndarray:
    """Invert the radar log transform on a torch tensor.

    The result is a numpy array: the caller stacks it with fields that
    have already been moved to the host, and numpy cannot mix a CUDA
    tensor with a numpy array.
    """
    return torch.expm1(values * 6.0).cpu().numpy()


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------
def masked_pixels(fields: WindowFields) -> np.ndarray:
    """The pixels whose CAPE is a measurement rather than a fill.

    The archive's fill code becomes a zero CAPE, so a pixel where either
    slice is zero carries no information about the CAPE change. Such
    pixels are left out of the correlation rather than allowed to
    dominate it.

    Args:
        fields: the stitched fields of the window.

    Returns:
        A boolean mask, ``True`` where the pixel is usable.
    """
    consumption = fields.get("cape_consumption")
    analysis = fields.get("cape_analysis")
    forecast = fields.get("cape_forecast")
    if consumption is None:
        return np.ones((0, 0), dtype=bool)
    if analysis is None or forecast is None:
        return np.ones(consumption.shape, dtype=bool)
    return (analysis > 0.0) & (forecast > 0.0)


def _field_stats(values: np.ndarray) -> dict:
    """The summary statistics of one field."""
    array = np.asarray(values, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "mean": None, "std": None, "min": None, "max": None,
            "p05": None, "p95": None,
        }
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def _argmax_location(values: np.ndarray) -> list[int] | None:
    """The ``[row, column]`` of the largest value of a field."""
    if values.size == 0:
        return None
    flat = int(np.argmax(values))
    return [int(flat // values.shape[1]), int(flat % values.shape[1])]


def echo_area(values: np.ndarray) -> float:
    """The share of the field that carries echo above the threshold."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float((array >= ECHO_THRESHOLD_DBZ).mean())


def _safe_max(values: np.ndarray | None) -> float | None:
    """The largest value of a field, or ``None``."""
    if values is None or values.size == 0:
        return None
    return float(values.max())


def case_statistics(
    fields: WindowFields,
    scan: list[tuple[int, WindowFields]],
) -> dict:
    """The numbers one case contributes to the report.

    Args:
        fields: the stitched fields of the figure's window.
        scan: the scanned windows of the same case, as
            ``(issue time, fields)`` pairs.

    Returns:
        The statistics, ready for the JSON report.
    """
    consumption = fields.get("cape_consumption")
    tendency = fields.get(TENDENCY_FRAME)
    mask = masked_pixels(fields)

    stats: dict[str, Any] = {
        "case_id": fields.case_id,
        "issue_time": int(fields.issue_time),
        "issue_time_str": fields.issue_time_str,
        "nwp_lead_minutes": list(fields.nwp_lead),
        "cape_consumption": _field_stats(consumption),
        "cape_consumption_units": "J/kg; negative means CAPE was released",
    }
    if consumption is not None and mask.any():
        stats["cape_consumption"]["fraction_released"] = float(
            (consumption[mask] < 0).mean()
        )
    else:
        stats["cape_consumption"]["fraction_released"] = None

    cr_issue = fields.get("cr_issue")
    if cr_issue is not None:
        stats["radar"] = {
            "cr_max_dbz": float(cr_issue.max()),
            "cr_p99_dbz": float(np.percentile(cr_issue, 99)),
            "cr_mean_dbz": float(cr_issue.mean()),
            "echo_area_fraction": echo_area(cr_issue),
            "echo_threshold_dbz": ECHO_THRESHOLD_DBZ,
            "history_max_dbz": {
                "t-30min": _safe_max(fields.get("cr_hist_mid")),
                "t-60min": _safe_max(fields.get("cr_hist_old")),
            },
        }

    if tendency is not None and consumption is not None:
        stats["tendency_magnitude"] = _field_stats(tendency)
        stats["tendency_magnitude"]["argmax_row_col"] = _argmax_location(tendency)
        stats["tendency_magnitude"]["units"] = (
            "dimensionless: the Euclidean norm of the plugin's tendency over "
            "its feature channels"
        )
        stats["correlation"] = {
            "pearson_r": pearson_r(tendency[mask], consumption[mask]),
            "spearman_r": spearman_r(tendency[mask], consumption[mask]),
            "n_pixels": int(mask.sum()),
            "n_pixels_total": int(mask.size),
            "masked_fraction": float(1.0 - mask.mean()) if mask.size else 0.0,
            "definition": (
                "the Pearson correlation between the per-pixel tendency "
                "magnitude and the per-pixel CAPE consumption of the same "
                "pixel of the stitched full grid. A negative coefficient "
                "means the tendency is large where the CAPE is being "
                "released."
            ),
        }

    stats["timing"] = _timing_statistics(fields, scan, tendency is not None)
    return stats


def _timing_statistics(
    fields: WindowFields,
    scan: list[tuple[int, WindowFields]],
    has_tendency: bool,
) -> dict:
    """Where the tendency peaks relative to the radar echo.

    The scan visits a series of issue times of the same case. For each of
    them the tendency magnitude and the echo area are measured on the
    stitched full grid, and the two peaks are compared.

    Args:
        fields: the fields of the figure's window.
        scan: the scanned windows, as ``(issue time, fields)`` pairs.
        has_tendency: whether a tendency was produced at all.

    Returns:
        The timing statistics.
    """
    entries = []
    for issue_time, window in scan:
        magnitude = window.get(TENDENCY_FRAME)
        cr = window.get("cr_issue")
        consumption = window.get("cape_consumption")
        entries.append({
            "issue_time": int(issue_time),
            "issue_time_str": f"{int(issue_time):08d}",
            "tendency_max": _safe_max(magnitude),
            "tendency_mean": (
                float(np.asarray(magnitude, dtype=np.float64).mean())
                if magnitude is not None else None
            ),
            "echo_area_fraction": echo_area(cr) if cr is not None else None,
            "cr_max_dbz": _safe_max(cr),
            "cape_consumption_min": (
                float(np.asarray(consumption, dtype=np.float64).min())
                if consumption is not None else None
            ),
        })

    result: dict[str, Any] = {
        "scan": entries,
        "echo_metric": (
            "the share of the 301x301 grid whose composite reflectivity is "
            f"at least {ECHO_THRESHOLD_DBZ:.0f} dBZ: a far more stable "
            "measure of 'the echo is there' than the maximum, which one "
            "noisy pixel can move"
        ),
    }
    if not entries:
        return result

    times = sorted(int(e["issue_time"]) for e in entries)
    gaps = [minutes_between(b, a) for a, b in zip(times, times[1:])]
    result["scan_resolution_minutes"] = (
        int(np.median(gaps)) if gaps else None
    )

    if has_tendency:
        key = "tendency_max"
        label = "tendency"
    else:
        key = "cape_consumption_min"
        label = "CAPE consumption"
    values = [(e[key], e) for e in entries if e.get(key) is not None]
    if not values:
        return result
    # The key is explicit: two windows can share the same value, and the
    # default tuple comparison would then try to order the dictionaries.
    peak_value, peak = (max(values, key=lambda item: item[0]) if has_tendency
                        else min(values, key=lambda item: item[0]))

    echo_values = [
        (e["echo_area_fraction"], e) for e in entries
        if e.get("echo_area_fraction") is not None
    ]
    if not echo_values:
        return result
    _, echo_peak = max(echo_values, key=lambda item: item[0])

    lead = minutes_between(echo_peak["issue_time"], peak["issue_time"])
    result.update({
        "peak_variable": label,
        "peak_issue_time": peak["issue_time"],
        "peak_issue_time_str": peak["issue_time_str"],
        "peak_value": float(peak_value),
        "echo_peak_issue_time": echo_peak["issue_time"],
        "echo_peak_issue_time_str": echo_peak["issue_time_str"],
        "echo_peak_area_fraction": echo_peak["echo_area_fraction"],
        "lead_minutes": int(lead),
        "lead_meaning": (
            "the number of minutes between the peak of the " + label
            + " and the peak of the radar echo. A positive value means the "
            + label + " came first, i.e. the signal is there before the "
            "echo is. The value is quantised by the resolution of the "
            "scan (the spacing of the scanned issue times)."
        ),
    })
    return result


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def plot_tendency(
    path: Path,
    fields: WindowFields,
    stats: dict,
    title: str,
    arrow_stride: int = 8,
    dpi: int = 200,
) -> None:
    """Write the diagnostic figure.

    Without a tendency there is a single panel: the CAPE consumption on
    its own. A tendency from an untrained model is not a physical
    quantity, and drawing it would put a picture of noise next to a
    measurement.

    Args:
        path: where to write the PNG.
        fields: the stitched fields of the window.
        stats: the statistics printed on the figure.
        title: the figure title.
        arrow_stride: draw one arrow every N pixels.
        dpi: the resolution of the PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    consumption = fields.get("cape_consumption")
    tendency = fields.get(TENDENCY_FRAME)
    cr_issue = fields.get("cr_issue")

    if tendency is None:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0), constrained_layout=True)
        _draw_cape(ax, consumption, None)
        ax.set_title("(b) CAPE consumption (analysis hour -> forecast hour)")
        fig.suptitle(
            title
            + "\nno checkpoint: the tendency of an untrained model is not a\n"
            "physical quantity, so only the CAPE panel is shown",
            fontsize=10,
        )
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    # -- (a) the radar echo with the tendency over it ----------------------
    vmax_dbz = max(60.0, float(np.ceil(cr_issue.max() / 10.0) * 10.0))
    image = ax_a.imshow(
        cr_issue, cmap="turbo", vmin=0.0, vmax=vmax_dbz, origin="upper",
        interpolation="nearest",
    )
    colour_bar = fig.colorbar(image, ax=ax_a, fraction=0.046, pad=0.02)
    colour_bar.set_label("composite reflectivity (dBZ)")

    u, v = tendency_flow(tendency)
    peak = float(np.abs(u).max()) or float(np.abs(v).max()) or 1.0
    u = u / peak
    v = v / peak
    # The arrows are drawn only where the tendency is worth looking at.
    #
    # The tendency magnitude is a *dense* field: nearly every pixel has
    # one, so drawing all of them turns the panel into a speckle of
    # arrows and hides the radar underneath. The threshold is a quantile
    # of the field's own magnitudes, so it adapts to whatever the model
    # produced.
    threshold = _magnitude_threshold(tendency)
    step = max(1, int(arrow_stride))
    ys = np.arange(0, u.shape[0], step)
    xs = np.arange(0, u.shape[1], step)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    keep = tendency[::step, ::step] > threshold
    ax_a.quiver(
        grid_x[keep], grid_y[keep],
        u[::step, ::step][keep], v[::step, ::step][keep],
        color="white", alpha=0.8, scale_units="xy", scale=1.0 / step,
        angles="xy", width=0.0035, headwidth=3.5,
    )
    if float(np.abs(u).max()) > 0:
        streamlines = ax_a.streamplot(
            np.arange(u.shape[1]), np.arange(u.shape[0]), u, v,
            color="white", density=1.1, linewidth=0.7, arrowsize=0.8,
        )
        # ``streamplot`` has no ``alpha`` keyword, so the transparency is
        # set on the collection it returns.
        streamlines.lines.set_alpha(0.55)
    _draw_contours(ax_a, tendency, threshold)
    ax_a.set_title(
        "(a) radar composite reflectivity at the issue time\n"
        "arrows: tendency magnitude (length), up-gradient direction"
    )
    ax_a.set_xlabel("grid column (0.01 deg)")
    ax_a.set_ylabel("grid row (0.01 deg)")

    # -- (b) the CAPE consumption with the tendency in contours ------------
    _draw_cape(ax_b, consumption, tendency)
    ax_b.set_title(
        "(b) CAPE consumption: blue = released, red = built up\n"
        "contours: tendency magnitude"
    )
    ax_b.set_xlabel("grid column (0.01 deg)")
    ax_b.set_ylabel("grid row (0.01 deg)")

    # -- (c) the scatter ----------------------------------------------------
    _draw_scatter(ax_c, fields, stats)

    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _contour_levels(values: np.ndarray) -> list[float]:
    """A few sensible contour levels of the tendency magnitude."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return []
    positive = finite[finite > 0]
    if positive.size == 0:
        return []
    levels = np.percentile(positive, [60.0, 80.0, 94.0])
    return sorted({float(level) for level in levels if level > 0})


def _magnitude_threshold(values: np.ndarray) -> float:
    """The magnitude below which the tendency is not worth drawing.

    The threshold is a quantile of the field's own magnitudes, so it
    adapts to whatever the model produced: a field that is flat almost
    everywhere still gets a readable set of contours, and a field with a
    sharp peak does not get its peak buried under the contours of the
    noise around it.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, 75.0))


def _draw_contours(ax, values: np.ndarray, threshold: float) -> None:
    """Draw the tendency magnitude as contours, above the threshold.

    The field is masked below the threshold: a contour line is drawn
    through the *unmasked* pixels only, so the panel does not turn into a
    mesh of loops around the noise.
    """
    levels = _contour_levels(values)
    if len(levels) < 2:
        return
    masked = np.ma.masked_where(
        np.asarray(values, dtype=np.float64) < threshold, values
    )
    ax.contour(masked, levels=levels, colors="black", linewidths=0.7,
               alpha=0.7)


def _draw_cape(ax, consumption: np.ndarray | None, tendency: np.ndarray | None):
    """Draw the CAPE consumption with a diverging colour scale."""
    if consumption is None:
        ax.text(0.5, 0.5, "no CAPE field", ha="center", va="center",
                transform=ax.transAxes)
        return
    limit = float(np.nanpercentile(np.abs(consumption), 99))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(np.abs(consumption))) or 1.0
    image = ax.imshow(
        consumption, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper",
        interpolation="nearest",
    )
    colour_bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colour_bar.set_label("CAPE consumption (J/kg)")
    if tendency is not None:
        _draw_contours(ax, tendency, _magnitude_threshold(tendency))


def _draw_scatter(ax, fields: WindowFields, stats: dict):
    """Draw the per-pixel scatter and the correlation coefficient."""
    tendency = fields.get(TENDENCY_FRAME)
    consumption = fields.get("cape_consumption")
    if tendency is None or consumption is None:
        ax.text(0.5, 0.5, "no tendency field", ha="center", va="center",
                transform=ax.transAxes)
        return
    mask = masked_pixels(fields)
    x = consumption[mask]
    y = tendency[mask]
    if x.size:
        hexbin = ax.hexbin(
            x, y, gridsize=60, cmap="viridis", mincnt=1, linewidths=0.0,
            bins="log",
        )
        ax.figure.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.02,
                           label="pixels")
        if x.size > 2:
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            # A subsample keeps the fit cheap on a 90k-pixel field.
            take = max(2, x_sorted.size // 20000)
            coefficients = np.polyfit(x_sorted[::take], y_sorted[::take], 1)
            span = np.linspace(float(x.min()), float(x.max()), 50)
            ax.plot(span, np.polyval(coefficients, span), color="crimson",
                    lw=1.6, label="least squares fit")
            ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("CAPE consumption (J/kg)")
    ax.set_ylabel("tendency magnitude")
    correlation = stats.get("correlation") or {}
    pearson = correlation.get("pearson_r")
    spearman = correlation.get("spearman_r")
    text = "Pearson r = n/a"
    if pearson is not None and np.isfinite(pearson):
        text = f"Pearson r = {pearson:+.3f}"
        if spearman is not None and np.isfinite(spearman):
            text += f"\nSpearman rho = {spearman:+.3f}"
        text += f"\n{int(correlation.get('n_pixels', 0))} pixels"
    ax.set_title("(c) tendency magnitude vs CAPE consumption\n" + text,
                 fontsize=11)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# The self-check
# ---------------------------------------------------------------------------
def _self_check() -> int:
    """Check the numerical core against fields with known answers.

    The CAPE consumption, the normalisation inversion, the correlation
    coefficient, the tendency magnitude, the forward hook and the
    stitching are all checked against values that are known in advance,
    so the tool can be trusted on an archive nobody can see.

    Returns:
        0 when every check passes, 1 otherwise.
    """
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    # -- the normalisation inversion ---------------------------------------
    physical = np.array([[1000.0, 2000.0, 441.075]], dtype=np.float32)
    normalised = normalize_nwp_field(physical, CAPE_VAR)
    recovered = denormalize_nwp(normalised, CAPE_VAR)
    check(
        "the CAPE normalisation round-trips to the physical field",
        bool(np.allclose(recovered, physical, atol=1e-2)),
        f"recovered {recovered.ravel().tolist()} J/kg",
    )

    # -- the CAPE consumption is in physical units -------------------------
    analysis = normalize_nwp_field(np.array([[2000.0]], dtype=np.float32), CAPE_VAR)
    forecast = normalize_nwp_field(np.array([[1500.0]], dtype=np.float32), CAPE_VAR)
    consumption = cape_consumption(analysis, forecast)
    check(
        "the CAPE consumption is the physical difference in J/kg",
        abs(float(consumption[0, 0]) + 500.0) < 1e-2,
        f"{float(consumption[0, 0]):+.3f} J/kg (expected -500)",
    )
    naive = float(forecast[0, 0] - analysis[0, 0])
    check(
        "a difference of normalised values is NOT a CAPE change in J/kg",
        abs(naive - float(consumption[0, 0])) > 100.0,
        f"the normalised difference is {naive:+.4f}, the physical one is "
        f"{float(consumption[0, 0]):+.1f} J/kg",
    )
    unchanged = cape_consumption(
        normalize_nwp_field(np.array([[2500.0]], dtype=np.float32), CAPE_VAR),
        normalize_nwp_field(np.array([[2500.0]], dtype=np.float32), CAPE_VAR),
    )
    check(
        "an unchanged CAPE column consumes nothing",
        abs(float(unchanged[0, 0])) < 1e-2,
        f"{float(unchanged[0, 0]):+.6f} J/kg",
    )

    # -- the reflectivity inversion ----------------------------------------
    dbz = np.array([[0.0, 25.5, 65.0]], dtype=np.float32)
    recovered_dbz = dbz_from_network(radar_log_transform(dbz, REFLECTIVITY_VAR))
    check(
        "the radar log transform round-trips to dBZ",
        bool(np.allclose(recovered_dbz, dbz, atol=1e-3)),
        f"recovered {recovered_dbz.ravel().tolist()} dBZ",
    )

    # -- the correlation coefficient ----------------------------------------
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 3.0, 2.0, 4.0])
    check(
        "the Pearson correlation of a known pair is 0.8",
        abs(pearson_r(x, y) - 0.8) < 1e-12,
        f"r = {pearson_r(x, y):.12f}",
    )
    check(
        "a variable correlates perfectly with itself",
        abs(pearson_r(x, x) - 1.0) < 1e-12,
        f"r = {pearson_r(x, x):.12f}",
    )
    check(
        "a variable correlates perfectly with its negation",
        abs(pearson_r(x, -x) + 1.0) < 1e-12,
        f"r = {pearson_r(x, -x):.12f}",
    )
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(4)
    check(
        "an unrelated field does not correlate",
        abs(pearson_r(x, noise)) < 0.99,
        f"r = {pearson_r(x, noise):+.4f}",
    )
    check(
        "the rank correlation survives a monotone but non-linear relation",
        abs(spearman_r(x, x ** 3) - 1.0) < 1e-12,
        f"rho = {spearman_r(x, x ** 3):.12f}",
    )
    check(
        "a constant field has no correlation",
        not np.isfinite(pearson_r(x, np.ones_like(x))),
        "r = nan, as it should be",
    )

    # -- the tendency magnitude --------------------------------------------
    d_psi = torch.zeros(1, 3, 4, 4)
    d_psi[0, 0] = 3.0
    d_psi[0, 1] = 4.0
    magnitude = tendency_magnitude(d_psi)
    check(
        "the tendency magnitude is the norm over the channels",
        bool(torch.allclose(magnitude, torch.full((1, 4, 4), 5.0))),
        f"magnitude {float(magnitude[0, 0, 0]):.4f} (expected 5)",
    )

    # -- the flow points up the gradient ------------------------------------
    ramp = np.tile(np.linspace(0.5, 1.5, 16, dtype=np.float32), (16, 1))
    u, v = tendency_flow(ramp)
    check(
        "the tendency flow points uphill, towards the growing tendency",
        float(u[8, 0]) > 0.0 and float(u[8, 15]) > 0.0 and float(v[8, 8]) == 0.0,
        f"u = {float(u[8, 0]):+.4f} at the left edge and "
        f"{float(u[8, 15]):+.4f} at the right edge of a rising ramp",
    )
    zero_flow = tendency_flow(np.zeros((8, 8), dtype=np.float32))
    check(
        "the tendency flow vanishes where the tendency does",
        float(np.abs(zero_flow[0]).max()) == 0.0,
        "a zero tendency field draws no arrows",
    )
    coordinates = np.arange(16, dtype=np.float32) - 8.0
    gy, gx = np.meshgrid(coordinates, coordinates, indexing="ij")
    bump = np.exp(-(gx ** 2 + gy ** 2) / 8.0).astype(np.float32)
    u, v = tendency_flow(bump)
    check(
        "the tendency flow converges on a maximum",
        float(u[8, 2]) > 0.0 and float(u[8, 14]) < 0.0,
        f"u = {float(u[8, 2]):+.4f} on the left flank, "
        f"{float(u[8, 14]):+.4f} on the right flank of a bump",
    )

    # -- the forward hook fires on a real model ----------------------------
    hook_ok, hook_detail = _check_forward_hook()
    check("the forward hook on the Koopman flow fires", hook_ok, hook_detail)

    # -- the stitcher reassembles the field --------------------------------
    stitch_ok, stitch_detail = _check_stitching()
    check("the stitched field reproduces the field that was cut up",
          stitch_ok, stitch_detail)

    # -- the selection rule picks the largest release ----------------------
    rank_ok, rank_detail = _check_selection_rule()
    check("the selection rule picks the window with the largest CAPE release",
          rank_ok, rank_detail)

    # -- report -------------------------------------------------------------
    width = max(len(name) for name, _, _ in results)
    print("tendency_field self-check")
    print("=" * (width + 30))
    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name.ljust(width)}  {detail}")
        failed += 0 if ok else 1
    print("=" * (width + 30))
    print(f"{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


def _check_forward_hook() -> tuple[bool, str]:
    """Check that the hook records the tendency the model actually used.

    A randomly initialised model on synthetic tensors is enough: the hook
    has to fire, and the tensor it records has to be the one the Koopman
    flow returned.
    """
    try:
        from radar_nwp_bench.models import build_model

        torch.manual_seed(0)
        model = build_model(
            "simvp_v2_quantum", in_radar_c=10, in_nwp_c=27,
            in_len=4, out_len=2, hid_c=8, num_blocks=1,
        )
        model.eval()
        capture = TendencyCapture(model)
        if capture.n_modules == 0:
            return False, "the model holds no Koopman flow"

        x_radar = torch.rand(1, 4, 10, 16, 16)
        x_nwp = torch.rand(1, 2, 27, 16, 16)
        with torch.no_grad():
            model(x_radar, x_nwp)
        captured = capture.last
        if captured is None:
            capture.close()
            return False, "the hook did not fire"

        # The captured tensor has to be the one the flow produced.
        with torch.no_grad():
            reference = model.qnwp_coupler.koopman_flow(
                model.qnwp_coupler.state_proj(x_nwp[:, 0]),
                model.qnwp_coupler.state_proj(x_nwp[:, -1]),
            )
        same = bool(torch.allclose(captured, reference, atol=1e-5))
        detail = (
            f"{capture.n_calls} call(s), captured {tuple(captured.shape)}, "
            f"matches the module's own output: {same}"
        )
        capture.close()
        return same, detail
    except Exception as exc:  # noqa: BLE001 - a check reports everything
        return False, f"{type(exc).__name__}: {exc}"


def _check_stitching() -> tuple[bool, str]:
    """Check that the stitched field is the field that was cut up.

    The figure shows the stitched grid, so a stitch that shifted or
    weighted the field would put the tendency and the radar in the wrong
    places relative to each other -- which is exactly the mistake a
    reader would not see.
    """
    try:
        from radar_nwp_bench.data.coordinates import symmetric_patch_origins

        size = 32
        grid = 64
        field_values = np.zeros((grid, grid), dtype=np.float32)
        field_values[10:20, 30:45] = 7.0
        origins = symmetric_patch_origins(grid, size, size)
        per_axis = len(origins)
        stitcher = PatchStitcher(
            (grid, grid), n_frames=1, n_positions=per_axis ** 2
        )
        position = 0
        for y in origins:
            for x in origins:
                tile = field_values[y:y + size, x:x + size]
                stitcher.add(0, position, (int(y), int(x)),
                             tile[None], tile[None])
                position += 1
        stitched, _ = stitcher.finalize(0)
        error = float(np.abs(stitched[0] - field_values).max())
        return error < 0.2, f"the largest error after stitching is {error:.4f}"
    except Exception as exc:  # noqa: BLE001 - a check reports everything
        return False, f"{type(exc).__name__}: {exc}"


def _check_selection_rule() -> tuple[bool, str]:
    """Check that the ranking puts the largest CAPE release first."""
    try:
        choices = [
            WindowChoice(0, "A", 100, -10.0, -50.0),
            WindowChoice(1, "B", 200, -900.0, -1500.0),
            WindowChoice(2, "C", 300, -10.0, -50.0),
        ]
        choices.sort(key=lambda c: (c.consumption_mean, c.issue_time))
        ok = choices[0].case_id == "B"
        return ok, (
            f"the ranking starts with {choices[0].case_id} "
            f"({choices[0].consumption_mean:+.1f} J/kg)"
        )
    except Exception as exc:  # noqa: BLE001 - a check reports everything
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def _json_safe(value):
    """Convert a report into something strict JSON can hold.

    ``json.dump`` writes ``NaN`` and ``Infinity`` for a non-finite float,
    which are not valid JSON and are rejected by most parsers. A missing
    number is written as ``null`` instead.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _parse_issue_time(text: str | None) -> int | None:
    """Parse an ``MMDDHHMM`` issue time into the archive's integer stamp."""
    if text is None:
        return None
    cleaned = str(text).strip()
    try:
        value = int(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"the issue time {text!r} is not a number; it must be written as "
            "MMDDHHMM (e.g. 06231500)"
        ) from exc
    if not 0 <= value <= 99999999:
        raise ValueError(
            f"the issue time {text!r} is out of range; it must be written as "
            "MMDDHHMM (e.g. 06231500)"
        )
    return value


def main() -> int:
    ap = build_argparser()
    args, config = _apply_config(ap, None)

    if args.self_check:
        return _self_check()

    if not args.data_root:
        print("ERROR: --data-root is required (give it on the command line or "
              "set data.root in the configuration file)")
        return 2
    if not args.out_dir:
        print("ERROR: --out-dir is required (it receives tendency_field.json "
              "and the figures)")
        return 2
    if args.checkpoint and not args.model:
        print("ERROR: --checkpoint was given without --model; the baseline "
              "to load it into cannot be guessed")
        return 2
    if args.max_cases < 1:
        print("ERROR: --max-cases must be at least 1")
        return 2
    if args.time_stride < 1:
        print("ERROR: --time-stride must be at least 1")
        return 2
    if args.arrow_stride < 1:
        print("ERROR: --arrow-stride must be at least 1")
        return 2

    try:
        wanted_issue = _parse_issue_time(args.issue_time)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    if not sys.stdout.isatty():
        sys.stdout.reconfigure(line_buffering=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("ERROR: --device cuda was requested but no CUDA device is "
                  "available; use --device cpu")
            return 2

    # The variable subsets decide the channel count of the model's input.
    # They come from the command line, or from the configuration file, or
    # from the training run that produced the checkpoint.
    trained = _training_run_config(args.checkpoint)
    nwp_vars = args.nwp_vars
    radar_vars = args.radar_vars
    if nwp_vars is None and trained and trained.get("nwp_vars"):
        nwp_vars = list(trained["nwp_vars"])
        print(f"nwp variables taken from the training run: {' '.join(nwp_vars)}")
    if radar_vars is None and trained and trained.get("radar_vars"):
        radar_vars = list(trained["radar_vars"])
        print(f"radar variables taken from the training run: {' '.join(radar_vars)}")

    trained_version = trained.get("nwp_pipeline_version") if trained else None
    if trained_version and trained_version != NWP_PIPELINE_VERSION:
        print(
            f"ERROR: this checkpoint was trained under the input pipeline "
            f"'{trained_version}' but the diagnostic runs "
            f"'{NWP_PIPELINE_VERSION}'. The two scale the NWP differently, so "
            f"the CAPE consumption would be meaningless. Retrain the model."
        )
        return 2

    # -- the dataset --------------------------------------------------------
    try:
        dataset = _build_dataset(args, nwp_vars, radar_vars, args.slide_stride)
        grid_h, grid_w = dataset.grid_shape
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: cannot read the dataset from {args.data_root}: {exc}")
        return 2

    if args.slide_stride is None and args.patch_size < min(grid_h, grid_w):
        dataset = _build_dataset(args, nwp_vars, radar_vars, args.patch_size)
        grid_h, grid_w = dataset.grid_shape

    if len(dataset) == 0:
        print("ERROR: no samples in the split -- check --split, --alignment "
              "or the data root")
        return 1

    if CAPE_VAR not in dataset.nwp_vars:
        print(f"ERROR: the NWP subset does not carry {CAPE_VAR}; the CAPE "
              "consumption cannot be computed")
        return 2
    if REFLECTIVITY_VAR not in dataset.radar_vars:
        print(f"ERROR: the radar subset does not carry {REFLECTIVITY_VAR}; "
              "the composite reflectivity cannot be drawn")
        return 2

    cape_index = list(dataset.nwp_vars).index(CAPE_VAR)
    cr_index = list(dataset.radar_vars).index(REFLECTIVITY_VAR)

    print(
        f"the split {args.split} holds {len(dataset)} windows in "
        f"{len({s.case_path.stem for s in dataset.samples})} cases "
        f"(grid {grid_h}x{grid_w})"
    )

    # -- the selection ------------------------------------------------------
    try:
        choices = rank_windows(dataset)
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        print(f"ERROR: cannot read the CAPE fields of the split: {exc}")
        return 2

    case_ids = {s.case_path.stem for s in dataset.samples}
    try:
        chosen, rule = select_windows(choices, args, case_ids, wanted_issue)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    print("\nselection rule")
    print(f"  {rule}")
    print("  the windows that were picked:")
    for choice in chosen:
        print(
            f"    {choice.case_id}  issue {choice.issue_time_str}  "
            f"mean CAPE consumption {choice.consumption_mean:+.1f} J/kg  "
            f"minimum {choice.consumption_min:+.1f} J/kg"
        )
    print()

    # -- the model ----------------------------------------------------------
    model = None
    capture = None
    if args.checkpoint:
        model_cfg = config
        if not model_cfg.get("model"):
            try:
                model_cfg = load_config(args.model)
            except FileNotFoundError:
                model_cfg = {}
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
        try:
            model = build_model(args.model, **kwargs).to(device)
        except Exception as exc:  # noqa: BLE001 - the reason is reported
            print(f"ERROR: cannot build the baseline {args.model}: {exc}")
            return 2
        model.nwp_time_len = dataset.nwp_frames
        try:
            state = torch.load(args.checkpoint, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        except Exception as exc:  # noqa: BLE001 - the reason is reported
            print(f"ERROR: cannot load the checkpoint {args.checkpoint}: {exc}")
            return 2
        print(f"loaded checkpoint {args.checkpoint} into {args.model}")
        model.eval()
        capture = TendencyCapture(model)
        if capture.n_modules == 0:
            print(f"ERROR: the model {args.model} holds no Koopman flow, so "
                  "there is no tendency field to draw")
            capture.close()
            return 2
    else:
        print("no checkpoint given: the NWP diagnostics only. The tendency of "
              "an untrained model is not a physical quantity, so no tendency "
              "field is plotted and no correlation is reported.")

    # -- the windows to run -------------------------------------------------
    wanted_old: list[int] = []
    scan_map: dict[str, list[int]] = {}
    for choice in chosen:
        indices = scan_windows(dataset, choice, args.time_scan, args.time_stride)
        scan_map[choice.case_id] = indices
        wanted_old.extend(indices)
    wanted_old = sorted(set(wanted_old))
    position_of = {old: new for new, old in enumerate(wanted_old)}

    dataset.samples = [dataset.samples[index] for index in wanted_old]
    n_windows = len(dataset.samples)
    print(f"running {n_windows} window{'' if n_windows == 1 else 's'} "
          f"({dataset.n_patch_positions} tiles each)")

    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        seed=args.seed,
    )

    try:
        results = analyse_windows(
            dataset, loader, model, capture,
            set(range(len(dataset.samples))),
            (grid_h, grid_w), dataset.n_patch_positions,
            cr_index, cape_index, args.in_len, device,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        if capture is not None:
            capture.close()
        return 2
    except (ValueError, OSError) as exc:
        print(f"ERROR: the analysis failed: {exc}")
        if capture is not None:
            capture.close()
        return 2
    finally:
        if capture is not None:
            capture.close()

    if not results:
        print("ERROR: no window was completed; the sliding walk does not "
              "tile the field")
        return 1

    # -- the outputs --------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for choice in chosen:
        position = position_of.get(choice.sample_index)
        if position is None or position not in results:
            print(f"WARNING: the window {choice.case_id} "
                  f"{choice.issue_time_str} was not completed; it is skipped")
            continue
        fields = results[position]
        # The scan holds the *original* indices of the windows; the
        # dataset now holds the filtered list, so they are remapped
        # through ``position_of`` before they are looked up.
        scan = [
            (
                dataset.samples[position_of[index]].issue_time,
                results[position_of[index]],
            )
            for index in scan_map[choice.case_id]
            if position_of.get(index) in results
        ]
        stats = case_statistics(fields, scan)
        reports.append(stats)

        figure_path = out_dir / (
            f"tendency_{choice.case_id}_{choice.issue_time_str}.png"
        )
        title = (
            f"{choice.case_id}  issue {choice.issue_time_str}  "
            f"({'with model' if model is not None else 'no model'})"
        )
        plot_tendency(
            figure_path, fields, stats, title,
            arrow_stride=args.arrow_stride, dpi=args.dpi,
        )
        print(f"  wrote {figure_path.name}")

    report = {
        "tool": "scripts/tendency_field.py",
        "model": args.model,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "data_root": str(args.data_root),
        "packed": bool(args.packed),
        "patch_size": args.patch_size,
        "slide_stride": args.slide_stride,
        "grid": [int(grid_h), int(grid_w)],
        "has_tendency": model is not None,
        "nwp_pipeline_version": NWP_PIPELINE_VERSION,
        "run": {
            "device": str(device),
            "batch_size": args.batch_size,
            "in_len": args.in_len,
            "out_len": args.out_len,
            "alignment": args.alignment,
            "nwp_vars": list(dataset.nwp_vars),
            "radar_vars": list(dataset.radar_vars),
            "time_scan": args.time_scan,
            "time_stride": args.time_stride,
            "arrow_stride": args.arrow_stride,
            "n_tiles_per_window": int(dataset.n_patch_positions),
            "n_windows_run": int(len(dataset.samples)),
        },
        "selection": {
            "rule": rule,
            "requested_case": args.case,
            "requested_issue_time": (
                f"{wanted_issue:08d}" if wanted_issue is not None else None
            ),
            "n_candidates": len(choices),
            "chosen": [
                {
                    "case_id": c.case_id,
                    "issue_time": c.issue_time,
                    "issue_time_str": c.issue_time_str,
                    "mean_cape_consumption_j_per_kg": c.consumption_mean,
                    "min_cape_consumption_j_per_kg": c.consumption_min,
                }
                for c in chosen
            ],
        },
        "units": {
            "cape_consumption": "J/kg; negative means CAPE was released. It "
                                "is the difference of the two NWP slices the "
                                "plugin uses (the analysis hour and the "
                                "forecast hour), with the dataset's "
                                "normalisation inverted before the "
                                "difference is taken.",
            "reflectivity": "dBZ",
            "tendency_magnitude": "dimensionless; the Euclidean norm of the "
                                   "plugin's tendency over its feature "
                                   "channels",
            "time": "MMDDHHMM",
        },
        "definitions": {
            "tendency_capture": (
                "a forward hook on the KoopmanUnitaryFlow module: the "
                "figure shows the tensor the model itself used, not a "
                "recomputation of it"
            ),
            "quiver": (
                "the plugin's tendency is a multi-channel feature field, "
                "not a planar vector field, so the arrows carry the "
                "magnitude as their length and the spatial gradient of the "
                "magnitude as their direction: they point towards the "
                "region where the tendency is growing"
            ),
            "correlation": (
                "the Pearson correlation between the per-pixel tendency "
                "magnitude and the per-pixel CAPE consumption on the "
                "stitched 301x301 grid, with the pixels whose CAPE is a "
                "fill left out"
            ),
            "reading_the_sign": (
                "the hypothesis the figure exists to test says the sign "
                "should be NEGATIVE: where the tendency is large, CAPE "
                "should have been consumed, and consumption is negative. A "
                "positive coefficient says the opposite -- the tendency is "
                "largest where CAPE was built up -- and it is not a subtle "
                "difference in strength but a contradiction of the "
                "mechanism. Two things can produce it. The plugin may have "
                "learned the NWP field's static structure (the humidity "
                "gradient, the echo shape) rather than its tendency, in "
                "which case the tendency magnitude simply marks where the "
                "atmosphere is most interesting and carries no information "
                "about the energy budget. Or the correlation may be "
                "carried by the large-scale pattern -- both fields are "
                "smooth, and a shared synoptic structure is enough to "
                "produce a coefficient that means nothing. The report "
                "gives the fraction of the grid where CAPE was released, "
                "so the reader can tell whether there was any release to "
                "correlate with at all."
            ),
        },
        "cases": reports,
    }

    (out_dir / "tendency_field.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # -- the console summary ------------------------------------------------
    header = (
        f"{'case':<26} {'issue':<9} {'r':>8}  {'cape mean':>10} "
        f"{'tendency max':>13}  lead"
    )
    print("\n" + header)
    print("-" * len(header))
    for stats in reports:
        correlation = stats.get("correlation") or {}
        pearson = correlation.get("pearson_r")
        consumption = stats.get("cape_consumption") or {}
        tendency = stats.get("tendency_magnitude") or {}
        timing = stats.get("timing") or {}
        lead = timing.get("lead_minutes")

        r_text = f"{pearson:+.3f}" if pearson is not None else "n/a"
        cape_text = (
            f"{consumption['mean']:+.1f}"
            if consumption.get("mean") is not None else "n/a"
        )
        tendency_text = (
            f"{tendency['max']:.3f}"
            if tendency.get("max") is not None else "n/a"
        )
        lead_text = f"{lead:+d} min" if lead is not None else "n/a"
        print(
            f"{stats['case_id']:<26} {stats['issue_time_str']:<9} "
            f"{r_text:>8}  {cape_text:>10} {tendency_text:>13}  {lead_text}"
        )

    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
