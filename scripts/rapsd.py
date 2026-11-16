#!/usr/bin/env python3
"""Radially averaged power spectral density (RAPSD) of predicted vs observed rain.

The tool answers two spectral questions about a nowcast:

* does the forecast keep the small-scale energy of the observed field
  (convective cores a few kilometres across), or has it been blurred?
* does it *invent* small-scale energy the observation does not have
  (a broadband noise floor: "diffusion hallucination")?

Both are answered on the **stitched full field**, never on a tile. A tile-wise
spectrum measures the patch, not the storm: the tiles overlap, each storm is
split across several of them, and the Hann feathering that stitches them
together is itself a spectral filter. The tiles are reassembled here with the
same stitcher the verification uses
(:class:`radar_nwp_bench.metrics.stitching.PatchStitcher`), so the spectrum
describes the same field the scores describe.

Usage:
    python scripts/rapsd.py --data-root data/BJ_packed --packed --split test \
        --checkpoint runs/mocast_x/last.pt --model mocast --out-dir figures/rapsd_mocast

    # truth field only, no checkpoint:
    python scripts/rapsd.py --data-root data/BJ_packed --packed --split test \
        --out-dir figures/rapsd_truth

    # numerical self-check, no data needed:
    python scripts/rapsd.py --self-check
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_nwp_bench.config import argparse_defaults, load_config, model_kwargs
from radar_nwp_bench.data import (
    NWPAlignment,
    RadarNWPDataset,
    build_dataloader,
)
from radar_nwp_bench.metrics import PatchStitcher
from radar_nwp_bench.models import build_model, list_models

#: The radar/label grid: 301 x 301 cells (radar_nwp_bench/data/coordinates.py).
GRID_SIZE = 301

#: The grid resolution in degrees per cell (configs/data/bj.yaml).
GRID_RESOLUTION_DEG = 0.01

#: Mean Earth radius (km), the IUGG value the repository's georeferencing uses.
EARTH_RADIUS_KM = 6371.009

#: Kilometres per degree of latitude on a mean sphere.
KM_PER_DEGREE = 2.0 * np.pi * EARTH_RADIUS_KM / 360.0

#: The default cell size: 0.01 deg of latitude on a mean sphere.
#:
#: This is the *meridional* scale. At the archive's latitude (~40 N) the
#: zonal scale is smaller by cos(latitude) (0.85 km/px rather than 1.11),
#: so the physical domain is ~334.7 km north-south by ~256 km east-west.
#: The spectrum is computed in index space, where the grid *is* isotropic;
#: the kilometre conversion below therefore uses the meridional scale, and
#: the wavelength of an east-west feature is overestimated by ~1/cos(40).
#: The predicted/truth *ratio* is unaffected by this: both fields are
#: measured on the same grid with the same conversion.
DEFAULT_KM_PER_PIXEL = GRID_RESOLUTION_DEG * KM_PER_DEGREE

#: Wavelength boundaries of the two bands the paper table quotes, in km.
CONVECTIVE_MAX_KM = 10.0
CONVECTIVE_MIN_KM = 2.0
SYNCOPTIC_MIN_KM = 30.0


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
        help="path to a state_dict. Without it only the observed field is "
             "analysed and no predicted spectrum is produced.",
    )
    ap.add_argument(
        "--out-dir", default=None,
        help="directory for rapsd.json, rapsd.png and rapsd.csv",
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
        "--max-cases", type=int, default=None,
        help="cap the number of cases taken from the split",
    )
    ap.add_argument(
        "--max-samples", type=int, default=None,
        help="cap the number of sample windows analysed",
    )
    ap.add_argument(
        "--sample-every", type=int, default=1,
        help="analyse every N-th sample window, which spreads the analysis "
             "over the archive instead of taking a contiguous prefix",
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
        "--taper", default="none", choices=["none", "hann"],
        help="'hann' applies a separable 2-D raised-cosine (Hann) window to "
             "the whole field before the transform, which suppresses the "
             "spectral leakage of the periodic-boundary assumption. The "
             "leakage is measured and reported either way.",
    )
    ap.add_argument(
        "--km-per-pixel", type=float, default=DEFAULT_KM_PER_PIXEL,
        help="cell size in kilometres, used to convert the radial wavenumber "
             "into a wavelength (the default is the meridional scale of the "
             "0.01 deg grid on a mean sphere)",
    )
    ap.add_argument("--seed", type=int, default=0)
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
        "--self-check", action="store_true",
        help="run the numerical self-check on synthetic fields and exit; "
             "no dataset is needed",
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
    return args, config


def _training_run_config(checkpoint: str | None) -> dict | None:
    """Read the arguments of the training run that produced a checkpoint.

    The training script writes ``config.json`` next to the checkpoint it
    saves. It records which NWP and radar variables the model was built for,
    and those decide the *channel count* of the model's input. An analysis
    that guesses them wrong builds a model with a different first layer and
    fails to load the weights with a shape mismatch.

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


# ---------------------------------------------------------------------------
# The spectral core
# ---------------------------------------------------------------------------
def mode_wavenumbers(shape: tuple[int, int]) -> np.ndarray:
    """The radial wavenumber of every mode of the 2-D DFT, in cycles per field.

    The DFT of an ``ny x nx`` field has one mode per pixel. The mode at
    index ``(iy, ix)`` carries the integer number of cycles
    ``kx = rint(fftfreq(nx) * nx)`` across the field in x and the analogous
    ``ky`` in y, so its radial wavenumber is ``sqrt(kx**2 + ky**2)`` cycles
    per field. The returned grid is *fftshifted*, so it lines up with
    ``fftshift(fft2(field))``.

    Args:
        shape: the ``(ny, nx)`` shape of the field.

    Returns:
        An array of shape ``(ny, nx)`` holding the radial wavenumber of each
        mode, in cycles per field.
    """
    ny, nx = shape
    kx = np.rint(np.fft.fftfreq(nx) * nx)
    ky = np.rint(np.fft.fftfreq(ny) * ny)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    return np.fft.fftshift(np.hypot(kx_grid, ky_grid))


def radial_bins(k_grid: np.ndarray, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign every Fourier mode to a radial bin.

    The bins are one cycle per field wide and centred on the integers: bin
    ``j`` holds the modes whose radial wavenumber lies in ``[j - 0.5, j + 0.5)``.

    Args:
        k_grid: the radial wavenumber of every mode (see
            :func:`mode_wavenumbers`).
        k_max: the largest bin to keep. Modes beyond it lie in the corners of
            the transformed square, where the annulus is only partly sampled
            and the radial average would be biased; they are dropped.

    Returns:
        The bin index of every mode, and the number of modes per bin.
    """
    index = np.floor(k_grid + 0.5).astype(np.int64)
    counts = np.bincount(index.ravel(), minlength=int(k_max) + 1)
    return index, counts[: int(k_max) + 1]


def radial_average(
    power: np.ndarray,
    index: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Average a 2-D power spectrum over the annuli of each radial bin.

    Args:
        power: the 2-D power spectrum, ``abs(FFT)**2 / n_pixels**2``.
        index: the bin index of every mode.
        counts: the number of modes per bin.

    Returns:
        The mean power per mode in each bin; ``nan`` for empty bins.
    """
    totals = np.bincount(
        index.ravel(), weights=power.ravel(), minlength=counts.size
    )[: counts.size]
    out = np.full(counts.size, np.nan)
    occupied = counts > 0
    out[occupied] = totals[occupied] / counts[occupied]
    return out


def frame_spectrum(
    field: np.ndarray,
    index: np.ndarray,
    counts: np.ndarray,
    taper: np.ndarray | None = None,
) -> np.ndarray:
    """The radially averaged power spectrum of one 2-D field.

    The mean is removed first: an offset is not spectral structure, and
    leaving it in would put the field's total in the DC bin.

    Args:
        field: the 2-D field, in millimetres.
        index: the bin index of every mode.
        counts: the number of modes per bin.
        taper: an optional window of the field's shape, multiplied into the
            field before the transform.

    Returns:
        The mean power per mode in each radial bin.
    """
    x = np.asarray(field, dtype=np.float64)
    x = x - x.mean()
    if taper is not None:
        x = x * taper
    spectrum = np.fft.fftshift(np.fft.fft2(x))
    power = (spectrum.real ** 2 + spectrum.imag ** 2) / (x.size ** 2)
    return radial_average(power, index, counts)


def hann_taper(shape: tuple[int, int]) -> np.ndarray:
    """A separable 2-D raised-cosine (Hann) window of the given shape.

    The window goes to zero at the edges, so the windowed field is
    continuous across the periodic wrap and the transform of it carries far
    less leakage than the transform of the bare field. It also attenuates
    the field's own high-wavenumber content, which is why the leakage is
    measured and reported rather than assumed.

    Args:
        shape: the ``(ny, nx)`` shape of the field.

    Returns:
        The window, with values in ``[0, 1]``.
    """
    ny, nx = shape
    return np.outer(np.hanning(ny), np.hanning(nx))


def seam_step_ratio(field: np.ndarray) -> tuple[float, float]:
    """How violently the field is discontinuous across the periodic wrap.

    The transform treats the field as periodic, so the difference between
    the last and the first column (and row) is folded into the spectrum as
    if it were structure. Comparing that step with the field's own interior
    pixel-to-pixel variation gives a dimensionless number that is ~1 for a
    field that is genuinely continuous across the wrap and large for a field
    whose storms are cut off at the domain edge.

    The interior scale is an *RMS*, not a mean: a precipitation field is
    mostly flat with a few storms in it, and a mean absolute difference
    would be diluted by the flat majority, which would make the ratio
    depend on how sparse the field is rather than on how badly the
    boundary is treated.

    Args:
        field: the 2-D field, in millimetres.

    Returns:
        The step ratio across the x wrap and across the y wrap.
    """
    g = np.asarray(field, dtype=np.float64)
    g = g - g.mean()
    ny, nx = g.shape
    seam_x = float(np.sqrt(np.mean((g[:, nx - 1] - g[:, 0]) ** 2)))
    seam_y = float(np.sqrt(np.mean((g[ny - 1, :] - g[0, :]) ** 2)))
    inner_x = float(np.sqrt(np.mean(np.diff(g, axis=1) ** 2)))
    inner_y = float(np.sqrt(np.mean(np.diff(g, axis=0) ** 2)))
    ratio_x = seam_x / inner_x if inner_x > 0 else 0.0
    ratio_y = seam_y / inner_y if inner_y > 0 else 0.0
    return ratio_x, ratio_y


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------
class SpectrumAccumulator:
    """Accumulates the radially averaged spectra of many stitched samples.

    Three quantities are kept per radial bin: the mean power of the
    observed field, the mean power of the predicted field, and the mean of
    the per-sample ratios. The per-sample ratio is the quantity the paper
    figure quotes; it is kept separately from the ratio of the accumulated
    means because the two differ when the samples disagree about where the
    power is, and the difference is worth seeing.
    """

    def __init__(self, n_bins: int) -> None:
        self.n_bins = n_bins
        self.truth_sum = np.zeros(n_bins)
        self.pred_sum = np.zeros(n_bins)
        self.ratio_sum = np.zeros(n_bins)
        self.ratio_count = np.zeros(n_bins)
        self.n_samples = 0
        self.n_frames = 0
        self.n_pred_frames = 0
        self.leakage_truth: list[float] = []
        self.leakage_pred: list[float] = []

    def add_sample(
        self,
        truth: np.ndarray,
        pred: np.ndarray | None,
        index: np.ndarray,
        counts: np.ndarray,
        taper: np.ndarray | None,
    ) -> None:
        """Accumulate one stitched sample.

        Args:
            truth: the observed field, ``(T, ny, nx)`` in millimetres.
            pred: the predicted field, ``(T, ny, nx)`` in millimetres, or
                ``None`` when only the observation is available.
            index: the bin index of every mode.
            counts: the number of modes per bin.
            taper: the optional window applied before the transform.
        """
        t_spec = np.stack([
            frame_spectrum(frame, index, counts, taper) for frame in truth
        ])
        self.truth_sum += np.nansum(t_spec, axis=0)
        self.n_frames += truth.shape[0]

        if pred is not None:
            p_spec = np.stack([
                frame_spectrum(frame, index, counts, taper) for frame in pred
            ])
            self.pred_sum += np.nansum(p_spec, axis=0)
            self.n_pred_frames += pred.shape[0]
            t_mean = np.nanmean(t_spec, axis=0)
            p_mean = np.nanmean(p_spec, axis=0)
            valid = np.isfinite(t_mean) & np.isfinite(p_mean) & (t_mean > 0)
            self.ratio_sum[valid] += p_mean[valid] / t_mean[valid]
            self.ratio_count[valid] += 1

        self.n_samples += 1

    def spectra(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The accumulated spectra and the accumulated ratio.

        Returns:
            The mean observed spectrum, the mean predicted spectrum (all
            ``nan`` when there is no prediction) and the mean of the
            per-sample ratios.
        """
        if self.n_frames == 0:
            nan = np.full(self.n_bins, np.nan)
            return nan, nan.copy(), nan.copy()
        truth = self.truth_sum / self.n_frames
        pred = (
            self.pred_sum / self.n_pred_frames
            if self.n_pred_frames > 0 else np.full(self.n_bins, np.nan)
        )
        ratio = np.full(self.n_bins, np.nan)
        seen = self.ratio_count > 0
        ratio[seen] = self.ratio_sum[seen] / self.ratio_count[seen]
        return truth, pred, ratio


def band_power(spectrum: np.ndarray, counts: np.ndarray, k_lo: int, k_hi: int) -> float:
    """The power of a spectrum inside a band of radial wavenumbers.

    The radially averaged spectrum is a *mean* per mode, so the power of a
    band is the sum over the band of the bin means weighted by the number of
    modes each bin holds. Summed over every bin this reproduces the variance
    of the analysed field.

    Args:
        spectrum: the radially averaged spectrum.
        counts: the number of modes per bin.
        k_lo: the first bin of the band (inclusive).
        k_hi: the last bin of the band (inclusive).

    Returns:
        The band power, in the units of the analysed field.
    """
    band = slice(int(k_lo), int(k_hi) + 1)
    values = spectrum[band]
    weights = counts[band]
    good = np.isfinite(values) & np.isfinite(weights)
    return float(np.sum(values[good] * weights[good]))


#: The smallest share of the observed field's total power that a bin may
#: hold for the predicted/observed ratio of that bin to be meaningful.
#:
#: Below it the observed spectrum is at the numerical noise floor, and the
#: ratio would be the model's own noise divided by nothing. Such bins are
#: reported as missing rather than as a huge number, which would otherwise
#: dominate the ratio panel of the figure.
MIN_TRUTH_POWER_FRACTION = 1e-12


def ratio_spectrum(
    truth: np.ndarray,
    pred: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """The predicted/observed ratio of the accumulated spectra, per bin.

    The ratio is taken between the two accumulated mean spectra rather than
    as the mean of the per-sample ratios, because it is far less sensitive
    to a single sample whose observed spectrum happens to be near zero in
    one bin.

    Bins in which the observed field holds a negligible share of its own
    total power are left undefined: the observed spectrum is at the
    numerical noise floor there, and the ratio would be a number divided by
    nothing.

    Args:
        truth: the accumulated observed spectrum.
        pred: the accumulated predicted spectrum.
        counts: the number of modes per bin.

    Returns:
        The ratio in each bin, ``nan`` where it is undefined.
    """
    out = np.full(truth.size, np.nan)
    total = band_power(truth, counts, 0, truth.size - 1)
    if total <= 0:
        return out
    floor = MIN_TRUTH_POWER_FRACTION * total
    # The power of each bin on its own, which is what the floor applies to.
    per_bin = np.where(
        np.isfinite(truth) & (counts > 0), truth * counts, 0.0
    )
    good = (
        np.isfinite(truth) & np.isfinite(pred) & (truth > 0)
        & (per_bin > floor)
    )
    out[good] = pred[good] / truth[good]
    return out


def band_fraction(spectrum: np.ndarray, counts: np.ndarray, k_lo: int, k_hi: int) -> float:
    """The share of a spectrum's total power that lies in a band.

    Args:
        spectrum: the radially averaged spectrum.
        counts: the number of modes per bin.
        k_lo: the first bin of the band (inclusive).
        k_hi: the last bin of the band (inclusive).

    Returns:
        The band's share of the total power, in ``[0, 1]``, or ``nan`` when
        the spectrum carries no power at all.
    """
    total = band_power(spectrum, counts, 0, spectrum.size - 1)
    if total <= 0:
        return float("nan")
    k_hi = int(min(k_hi, spectrum.size - 1))
    return band_power(spectrum, counts, k_lo, k_hi) / total


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def plot_rapsd(
    path: Path,
    k: np.ndarray,
    field_km: float,
    truth: np.ndarray,
    pred: np.ndarray,
    ratio: np.ndarray,
    bands: dict[str, tuple[int, int]],
    summary: dict,
    title: str,
) -> None:
    """Write the two-panel RAPSD figure.

    Panel (i) shows the observed and the predicted spectrum on log-log
    axes; panel (ii) shows the predicted/observed ratio against the
    wavenumber with the line of equality at 1.0. The convective and the
    synoptic bands are shaded in both panels.

    Args:
        path: where to write the PNG.
        k: the radial wavenumber of each bin, in cycles per field.
        field_km: the width of the field in kilometres, used for the
            wavelength axis.
        truth: the accumulated observed spectrum.
        pred: the accumulated predicted spectrum.
        ratio: the accumulated predicted/observed ratio.
        bands: the band boundaries, keyed by band name.
        summary: the scalar summary, printed on the figure.
        title: the figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    convective = bands["convective"]
    synoptic = bands["synoptic"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    ax, axr = axes

    def shade(a):
        a.axvspan(convective[0], convective[1], color="#f4a261", alpha=0.18,
                  label=f"convective core ({CONVECTIVE_MIN_KM:.0f}-"
                        f"{CONVECTIVE_MAX_KM:.0f} km)")
        a.axvspan(synoptic[0], synoptic[1], color="#457b9d", alpha=0.18,
                  label=f"synoptic (>{SYNCOPTIC_MIN_KM:.0f} km)")

    ax.loglog(k, truth, color="#1d3557", lw=1.8, label="observed")
    if np.any(np.isfinite(pred)):
        ax.loglog(k, pred, color="#e76f51", lw=1.8, label="predicted")
    shade(ax)
    ax.set_xlabel("radial wavenumber k (cycles per field)")
    ax.set_ylabel("RAPSD  (mm$^2$ per pixel$^2$ per mode)")
    ax.set_title("radially averaged power spectrum")
    ax.grid(True, which="both", alpha=0.3)
    # The DC bin is the field's mean, which was removed, so it is a row of
    # numerical noise that would stretch the axis over a dozen empty decades.
    ax.set_xlim(1.0, k[-1])
    ax.legend(loc="lower left", fontsize=9)

    top = ax.secondary_xaxis(
        "top",
        functions=(
            lambda kk: np.where(kk > 0, field_km / np.where(kk > 0, kk, 1.0), np.nan),
            lambda lam: np.where(
                lam > 0, field_km / np.where(lam > 0, lam, 1.0), np.nan
            ),
        ),
    )
    top.set_xlabel("wavelength (km)")

    if np.any(np.isfinite(ratio)):
        axr.semilogx(k, ratio, color="#2a9d8f", lw=1.8,
                     label="predicted / observed")
        axr.axhline(1.0, color="0.35", ls="--", lw=1.2, label="perfect agreement")
        # The ratio spans decades, and a linear axis would flatten every
        # value below 1 into the axis. The scale is logarithmic, with the
        # symmetric range around 1 making "too little" and "too much" energy
        # equally visible.
        finite = ratio[np.isfinite(ratio) & (ratio > 0)]
        if finite.size:
            lo = max(float(finite.min()), 1e-3)
            hi = min(float(finite.max()), 1e3)
            axr.set_yscale("log")
            axr.set_ylim(lo, hi)
        shade(axr)
        axr.legend(loc="upper left", fontsize=9)
    else:
        axr.text(0.5, 0.5, "no prediction\n(truth-only mode)",
                 ha="center", va="center", transform=axr.transAxes)
    axr.set_xlabel("radial wavenumber k (cycles per field)")
    axr.set_ylabel("predicted / observed")
    axr.set_title("spectral ratio")
    axr.set_xlim(1.0, k[-1])
    axr.grid(True, which="both", alpha=0.3)

    if summary.get("high_k_ratio") is not None:
        text = (
            f"high-k band (lambda < {CONVECTIVE_MAX_KM:.0f} km): "
            f"ratio {summary['high_k_ratio']:.3f}\n"
            f"low-k band (lambda > {SYNCOPTIC_MIN_KM:.0f} km): "
            f"ratio {summary['low_k_ratio']:.3f}"
        )
    elif summary.get("predicted") is False:
        text = "observed field only"
    else:
        text = "the observed field has no power in the quoted bands"
    fig.suptitle(f"{title}\n{text}", fontsize=11)

    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The self-check
# ---------------------------------------------------------------------------
def _self_check() -> int:
    """Check the spectral core against fields with a known spectrum.

    A synthetic field is built from two exact Fourier modes of the 301-point
    DFT, at radial wavenumbers 10 and 40 cycles per field. The radially
    averaged spectrum must put its peaks back on those wavenumbers. The
    Parseval identity (the bin-count-weighted sum of the spectrum equals the
    variance of the field), the band ratios and the leakage diagnostics are
    checked on the same field, and the leakage diagnostics are calibrated on
    a field with a deliberately torn edge.

    Returns:
        0 when every check passes, 1 otherwise.
    """
    n = GRID_SIZE
    shape = (n, n)
    k_grid = mode_wavenumbers(shape)
    k_max = n // 2
    index, counts = radial_bins(k_grid, k_max)

    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    yy, xx = np.mgrid[0:n, 0:n]
    field = (
        1.0 * np.cos(2.0 * np.pi * (6 * xx + 8 * yy) / n)
        + 0.5 * np.cos(2.0 * np.pi * (24 * xx + 32 * yy) / n)
    )
    expected = (10.0, 40.0)

    # -- the peaks come back where they were put ---------------------------
    spec = frame_spectrum(field, index, counts)
    peak_bins = _local_peaks(spec, min_value=1e-12)
    for want in expected:
        got = min(peak_bins, key=lambda b: abs(b - want)) if peak_bins else None
        check(
            f"peak at k={want:.0f} recovered",
            got is not None and abs(got - want) <= 1,
            f"nearest local peak at k={got}",
        )

    # -- Parseval ----------------------------------------------------------
    kk = np.arange(counts.size)
    total = float(np.nansum(spec * counts))
    check(
        "Parseval: the spectrum integrates to the variance",
        abs(total - field.var()) <= 1e-9 * max(field.var(), 1e-12),
        f"sum {total:.12f} vs var {field.var():.12f}",
    )

    # -- the tapered spectrum keeps the peaks ------------------------------
    taper = hann_taper(shape)
    spec_t = frame_spectrum(field, index, counts, taper)
    peaks_t = _local_peaks(spec_t, min_value=1e-12)
    for want in expected:
        got = min(peaks_t, key=lambda b: abs(b - want)) if peaks_t else None
        check(
            f"tapered peak at k={want:.0f} within one bin",
            got is not None and abs(got - want) <= 1,
            f"nearest local peak at k={got}",
        )

    # -- the band ratios are right -----------------------------------------
    boosted = (
        1.0 * np.cos(2.0 * np.pi * (6 * xx + 8 * yy) / n)
        + 1.0 * np.cos(2.0 * np.pi * (24 * xx + 32 * yy) / n)
    )
    acc = SpectrumAccumulator(counts.size)
    acc.add_sample(field[None], boosted[None], index, counts, None)
    truth, pred, ratio = acc.spectra()
    low = (1, 11)
    high = (34, 150)
    low_ratio = band_power(pred, counts, *low) / band_power(truth, counts, *low)
    high_ratio = band_power(pred, counts, *high) / band_power(truth, counts, *high)
    check(
        "low-k band ratio is 1.0 (the k=10 mode is untouched)",
        abs(low_ratio - 1.0) <= 1e-9,
        f"ratio {low_ratio:.12f}",
    )
    check(
        "high-k band ratio is 4.0 (the k=40 mode is doubled in amplitude)",
        abs(high_ratio - 4.0) <= 1e-6,
        f"ratio {high_ratio:.12f}",
    )

    # -- the band ratios mean what they are supposed to mean ---------------
    # A prediction that is the observation must give a ratio of 1; one that
    # has been blurred must lose high-k energy; one that carries extra
    # broadband noise must gain it. Without this the numbers are just two
    # quotients that happen to be printed.
    blurred = _low_pass(field, 34)
    noisy = field + 0.05 * field.std() * np.random.default_rng(0).standard_normal(
        field.shape
    )
    identical = _band_ratios(field, field, counts, low, high)
    blurred_r = _band_ratios(field, blurred, counts, low, high)
    noisy_r = _band_ratios(field, noisy, counts, low, high)
    check(
        "an identical prediction gives a band ratio of 1",
        abs(identical[0] - 1.0) <= 1e-9 and abs(identical[1] - 1.0) <= 1e-9,
        f"high-k {identical[0]:.9f}, low-k {identical[1]:.9f}",
    )
    check(
        "a blurred prediction loses high-k energy",
        blurred_r[0] < 1.0,
        f"high-k ratio {blurred_r[0]:.6f}",
    )
    check(
        "a noisy prediction gains high-k energy",
        noisy_r[0] > 1.0,
        f"high-k ratio {noisy_r[0]:.6f}",
    )

    # -- the leakage diagnostics are calibrated -----------------------------
    ratio_x, ratio_y = seam_step_ratio(field)
    check(
        "seam step ratio is ~1 for a continuous field",
        abs(0.5 * (ratio_x + ratio_y) - 1.0) <= 0.05,
        f"x {ratio_x:.4f}, y {ratio_y:.4f}",
    )
    torn = field.copy()
    torn[:, :40] += 3.0
    torn_x, torn_y = seam_step_ratio(torn)
    check(
        "seam step ratio is large for a field torn at the edge",
        0.5 * (torn_x + torn_y) > 5.0,
        f"x {torn_x:.4f}, y {torn_y:.4f}",
    )

    # The taper makes the field continuous across the wrap, so the high-k
    # share of a torn field must move when it is applied. A field whose
    # small-scale structure is genuine must move much less.
    hi = high
    clean_frac = band_fraction(
        frame_spectrum(field, index, counts), counts, *hi
    )
    clean_tap = band_fraction(
        frame_spectrum(field, index, counts, taper), counts, *hi
    )
    torn_frac = band_fraction(
        frame_spectrum(torn, index, counts), counts, *hi
    )
    torn_tap = band_fraction(
        frame_spectrum(torn, index, counts, taper), counts, *hi
    )
    check(
        "the taper moves the high-k share of a torn field more than a clean one",
        abs(torn_frac - torn_tap) > abs(clean_frac - clean_tap),
        f"torn {torn_frac - torn_tap:+.5f}, clean {clean_frac - clean_tap:+.5f}",
    )

    # -- report -------------------------------------------------------------
    width = max(len(name) for name, _, _ in results)
    print("rapsd self-check")
    print("=" * (width + 30))
    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name.ljust(width)}  {detail}")
        failed += 0 if ok else 1
    print("=" * (width + 30))
    print(f"{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


def _local_peaks(spectrum: np.ndarray, min_value: float) -> list[int]:
    """The bin indices of the local maxima of a spectrum."""
    peaks: list[int] = []
    for b in range(1, spectrum.size - 1):
        v = spectrum[b]
        if not np.isfinite(v) or v < min_value:
            continue
        if v >= spectrum[b - 1] and v > spectrum[b + 1]:
            peaks.append(b)
    return peaks


def _low_pass(field: np.ndarray, k_cut: int) -> np.ndarray:
    """A field with everything above a radial wavenumber removed.

    Args:
        field: the 2-D field.
        k_cut: the wavenumber above which the content is removed.

    Returns:
        The low-passed field.
    """
    ny, nx = field.shape
    kx = np.rint(np.fft.fftfreq(nx) * nx)
    ky = np.rint(np.fft.fftfreq(ny) * ny)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    k = np.hypot(kx_grid, ky_grid)
    spectrum = np.fft.fft2(field)
    return np.real(np.fft.ifft2(np.where(k < k_cut, spectrum, 0)))


def _band_ratios(
    truth: np.ndarray,
    pred: np.ndarray,
    counts: np.ndarray,
    low: tuple[int, int],
    high: tuple[int, int],
) -> tuple[float, float]:
    """The high-k and low-k band power ratios of two fields.

    Args:
        truth: the observed field.
        pred: the predicted field.
        counts: the number of modes per bin.
        low: the low-k band.
        high: the high-k band.

    Returns:
        The high-k ratio and the low-k ratio.
    """
    shape = truth.shape
    k_grid = mode_wavenumbers(shape)
    index, _ = radial_bins(k_grid, min(shape) // 2)
    t_spec = frame_spectrum(truth, index, counts)
    p_spec = frame_spectrum(pred, index, counts)
    high_ratio = (
        band_power(p_spec, counts, *high) / band_power(t_spec, counts, *high)
    )
    low_ratio = (
        band_power(p_spec, counts, *low) / band_power(t_spec, counts, *low)
    )
    return high_ratio, low_ratio


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def _build_dataset(args, nwp_vars, radar_vars, slide_stride=None):
    """Build the analysis dataset.

    One function, because the analysis needs the dataset twice: once to find
    out how big the grid is, and once more with a sliding stride when the
    patch is smaller than the grid. The two calls must agree about
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
        max_samples=None,
        slide_stride=slide_stride,
        max_cases=args.max_cases,
    )
    if nwp_vars:
        kwargs["nwp_vars"] = tuple(nwp_vars)
    if radar_vars:
        kwargs["radar_vars"] = tuple(radar_vars)
    return RadarNWPDataset(args.data_root, **kwargs)


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
        print("ERROR: --out-dir is required (it receives rapsd.json, "
              "rapsd.png and rapsd.csv)")
        return 2
    if args.checkpoint and not args.model:
        print("ERROR: --checkpoint was given without --model; the baseline "
              "to load it into cannot be guessed")
        return 2
    if args.sample_every < 1:
        print("ERROR: --sample-every must be at least 1")
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
    # They come from the command line, or from the configuration file, or --
    # failing both -- from the training run that produced the checkpoint.
    trained = _training_run_config(args.checkpoint)
    nwp_vars = args.nwp_vars
    radar_vars = args.radar_vars
    if nwp_vars is None and trained and trained.get("nwp_vars"):
        nwp_vars = list(trained["nwp_vars"])
        print(f"nwp variables taken from the training run: {' '.join(nwp_vars)}")
    if radar_vars is None and trained and trained.get("radar_vars"):
        radar_vars = list(trained["radar_vars"])
        print(f"radar variables taken from the training run: {' '.join(radar_vars)}")

    try:
        dataset = _build_dataset(args, nwp_vars, radar_vars, args.slide_stride)
        grid_h, grid_w = dataset.grid_shape
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: cannot read the dataset from {args.data_root}: {exc}")
        return 2

    # The patch has to be slid over the grid and the tiles stitched back
    # together before anything is analysed. Scoring the tiles independently
    # would double-count every storm in the overlap strip and measure half a
    # storm as if it were a whole one.
    if args.slide_stride is None and args.patch_size < min(grid_h, grid_w):
        dataset = _build_dataset(args, nwp_vars, radar_vars, args.patch_size)
        grid_h, grid_w = dataset.grid_shape

    # --sample-every thins the windows so the analysis is spread over the
    # archive rather than taken from its first few hours.
    if args.sample_every > 1:
        dataset.samples = dataset.samples[:: args.sample_every]
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: args.max_samples]

    if len(dataset) == 0:
        print("ERROR: no samples in the split -- check --split, --alignment "
              "or the data root")
        return 1

    print(
        f"analysing {len(dataset)} windows "
        f"(split={args.split}, cases={len({s.case_path.stem for s in dataset.samples})})"
    )

    # -- the model ---------------------------------------------------------
    model = None
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
        except Exception as exc:
            print(f"ERROR: cannot build the baseline {args.model}: {exc}")
            return 2
        # The NWP slice count comes from the data, for the same reason it does
        # during training.
        model.nwp_time_len = dataset.nwp_frames
        try:
            state = torch.load(args.checkpoint, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        except Exception as exc:
            print(f"ERROR: cannot load the checkpoint {args.checkpoint}: {exc}")
            return 2
        print(f"loaded checkpoint {args.checkpoint} into {args.model}")
        model.eval()
    else:
        print("no checkpoint given: analysing the observed field only")

    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        seed=args.seed,
    )

    # -- the spectral grid -------------------------------------------------
    shape = (grid_h, grid_w)
    k_grid = mode_wavenumbers(shape)
    k_max = min(grid_h, grid_w) // 2
    index, counts = radial_bins(k_grid, k_max)
    k = np.arange(counts.size, dtype=np.float64)

    km_per_pixel = float(args.km_per_pixel)
    field_km = shape[1] * km_per_pixel
    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength_km = np.where(k > 0, field_km / k, np.inf)

    # The bands follow from the kilometre conversion, so they move with it.
    convective_lo = int(np.ceil(field_km / CONVECTIVE_MAX_KM))
    convective_hi = int(min(k_max, np.floor(field_km / CONVECTIVE_MIN_KM)))
    synoptic_lo = 1
    synoptic_hi = int(np.floor(field_km / SYNCOPTIC_MIN_KM))
    bands = {
        "convective": (convective_lo, convective_hi),
        "synoptic": (synoptic_lo, synoptic_hi),
    }

    taper = hann_taper(shape) if args.taper == "hann" else None
    acc = SpectrumAccumulator(counts.size)

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
        print(f"analysing the full {grid_h}x{grid_w} grid in one pass")

    # The leakage of the periodic-boundary assumption is measured on every
    # frame whatever the taper setting, so the reader can judge the numbers
    # rather than trust them.
    leak_truth: list[float] = []
    leak_pred: list[float] = []
    taper_truth: list[float] = []
    taper_pred: list[float] = []

    with torch.no_grad():
        for batch in loader:
            x_radar = batch["x_radar"].to(device)
            x_nwp = batch["x_nwp"].to(device)
            y = batch["y_rain"].to(device)

            if model is not None:
                pred = model(x_radar, x_nwp)
                pred_mm = pred.clamp_min(0.0)
            else:
                pred_mm = None
            truth_mm = y.clamp_min(0.0)

            if stitcher is not None:
                pred_np = (
                    pred_mm.cpu().numpy() if pred_mm is not None else None
                )
                truth_np = truth_mm.cpu().numpy()
                sample_idx = batch["case_index"].cpu().numpy()
                position_idx = batch["position_index"].cpu().numpy()
                origin = batch["patch_origin"].cpu().numpy()
                for b in range(truth_np.shape[0]):
                    sample_index = int(sample_idx[b])
                    truth_tile = truth_np[b, :, 0]
                    pred_tile = (
                        pred_np[b, :, 0] if pred_np is not None else truth_tile
                    )
                    stitcher.add(
                        sample_index,
                        int(position_idx[b]),
                        (int(origin[b][0]), int(origin[b][1])),
                        pred_tile,
                        truth_tile,
                    )
                    if stitcher.is_complete(sample_index):
                        s_pred, s_truth = stitcher.finalize(sample_index)
                        # Without a checkpoint there is no prediction. The
                        # stitcher needs a placeholder for it, but the
                        # placeholder must not be reported as one: doing so
                        # would print a ratio of 1.0 and read as a perfect
                        # model.
                        acc.add_sample(
                            s_truth, s_pred if model is not None else None,
                            index, counts, taper,
                        )
                        _measure_leakage(
                            s_truth,
                            s_pred if model is not None else None,
                            index, counts, bands,
                            leak_truth, leak_pred, taper_truth, taper_pred,
                        )
                        stitcher.release(sample_index)
            else:
                truth_np = truth_mm.cpu().numpy()[:, :, 0]
                pred_np = (
                    pred_mm.cpu().numpy()[:, :, 0] if pred_mm is not None else None
                )
                acc.add_sample(truth_np, pred_np, index, counts, taper)
                _measure_leakage(
                    truth_np, pred_np, index, counts, bands,
                    leak_truth, leak_pred, taper_truth, taper_pred,
                )

    if stitcher is not None and stitcher.pending:
        print(
            f"WARNING: {len(stitcher.pending)} windows were never completed "
            "(the run stopped early); they are not analysed"
        )

    truth, pred, ratio = acc.spectra()
    ratio_of_means = ratio_spectrum(truth, pred, counts)

    summary: dict = {
        "high_k_ratio": None,
        "low_k_ratio": None,
        "predicted": bool(acc.n_pred_frames > 0),
    }
    if acc.n_pred_frames > 0:
        hi_lo, hi_hi = bands["convective"]
        lo_lo, lo_hi = bands["synoptic"]
        t_hi = band_power(truth, counts, hi_lo, hi_hi)
        p_hi = band_power(pred, counts, hi_lo, hi_hi)
        t_lo = band_power(truth, counts, lo_lo, lo_hi)
        p_lo = band_power(pred, counts, lo_lo, lo_hi)
        summary["high_k_ratio"] = p_hi / t_hi if t_hi > 0 else None
        summary["low_k_ratio"] = p_lo / t_lo if t_lo > 0 else None
        summary["high_k_truth_power"] = t_hi
        summary["high_k_pred_power"] = p_hi
        summary["low_k_truth_power"] = t_lo
        summary["low_k_pred_power"] = p_lo
        # The same ratios taken from the mean of the per-sample ratios.
        band_ratio_hi = _band_ratio_of_ratio(ratio, counts, hi_lo, hi_hi)
        band_ratio_lo = _band_ratio_of_ratio(ratio, counts, lo_lo, lo_hi)
        summary["high_k_ratio_per_sample"] = band_ratio_hi
        summary["low_k_ratio_per_sample"] = band_ratio_lo

    leakage = _leakage_report(
        leak_truth, leak_pred, taper_truth, taper_pred
    )

    # -- the outputs -------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "tool": "scripts/rapsd.py",
        "model": args.model,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "data_root": str(args.data_root),
        "packed": bool(args.packed),
        "patch_size": args.patch_size,
        "slide_stride": args.slide_stride,
        "taper": args.taper,
        "n_samples": acc.n_samples,
        "n_frames": acc.n_frames,
        "has_prediction": bool(acc.n_pred_frames > 0),
        "grid": {
            "shape": [int(grid_h), int(grid_w)],
            "k_max": int(k_max),
            "km_per_pixel": km_per_pixel,
            "field_length_km": field_km,
            "pixel_size_source": (
                "0.01 deg of latitude on a mean sphere "
                f"(R = {EARTH_RADIUS_KM} km): {DEFAULT_KM_PER_PIXEL:.6f} km/px. "
                "The zonal scale at 40 N is smaller by cos(latitude) "
                "(0.85 km/px), so the kilometre wavelengths are north-south "
                "equivalents; the predicted/observed ratio does not depend on "
                "the conversion."
            ),
        },
        "units": {
            "wavenumber": "cycles per field (k = sqrt(kx^2 + ky^2), kx and ky "
                          "the integer cycle counts of the 2-D DFT of the "
                          "field)",
            "wavelength": "km",
            "spectrum": "mm^2 per pixel^2 per mode: the mean of "
                        "|FFT(x)|^2 / n_pixels^2 over the modes of the bin. "
                        "The bin-count-weighted sum over all bins equals the "
                        "variance of the analysed field.",
            "ratio": "dimensionless",
        },
        "k": k.tolist(),
        "wavelength_km": wavelength_km.tolist(),
        "n_modes": counts.tolist(),
        "truth": truth.tolist(),
        "predicted": pred.tolist(),
        "ratio": ratio.tolist(),
        "ratio_of_means": ratio_of_means.tolist(),
        "bands": {
            "convective": {
                "wavelength_km": [CONVECTIVE_MIN_KM, CONVECTIVE_MAX_KM],
                "k": [int(convective_lo), int(convective_hi)],
            },
            "synoptic": {
                "wavelength_km": [SYNCOPTIC_MIN_KM, None],
                "k": [int(synoptic_lo), int(synoptic_hi)],
            },
        },
        "summary": summary,
        "leakage": leakage,
    }

    (out_dir / "rapsd.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    _write_csv(out_dir / "rapsd.csv", k, wavelength_km, counts, truth, pred,
               ratio, ratio_of_means, bands)

    title = (
        f"{args.model} ({args.split})" if args.model else f"observed field ({args.split})"
    )
    plot_rapsd(
        out_dir / "rapsd.png",
        k, field_km, truth, pred, ratio_of_means, bands, summary, title,
    )

    # -- the console summary ------------------------------------------------
    print(f"\nanalysed {acc.n_samples} stitched windows "
          f"({acc.n_frames} frames, {grid_h}x{grid_w} each)")
    if summary["high_k_ratio"] is not None:
        print(f"high-k band (lambda < {CONVECTIVE_MAX_KM:.0f} km, "
              f"k = {convective_lo}-{convective_hi}): "
              f"ratio {summary['high_k_ratio']:.4f}")
        print(f"low-k  band (lambda > {SYNCOPTIC_MIN_KM:.0f} km, "
              f"k = {synoptic_lo}-{synoptic_hi}): "
              f"ratio {summary['low_k_ratio']:.4f}")
        print("a ratio below 1 means the forecast lost small-scale energy "
              "(blur); above 1 means it added some (noise)")
    elif acc.n_pred_frames > 0:
        print("the observed field has no power in the quoted bands: the "
              "analysed windows are dry, so there is nothing to compare")
    else:
        print("no prediction: the observed field only")
    seam = leakage["seam_step_ratio_median"]
    seam_pred = leakage["seam_step_ratio_pred_median"]
    seam_text = f"{seam:.3f}" if seam is not None else "n/a"
    seam_pred_text = f"{seam_pred:.3f}" if seam_pred is not None else "n/a"
    print(f"leakage: seam step ratio {seam_text} (observed), "
          f"{seam_pred_text} (predicted); 1.0 is a seamless periodic field")
    sensitivity = leakage["high_k_taper_sensitivity"]
    if sensitivity is not None:
        print(f"leakage: the high-k band power moves by {sensitivity * 100:+.2f}% "
              "of the total when the field is tapered")
    print(f"\nwritten to {out_dir}")
    return 0


def _band_ratio_of_ratio(
    ratio: np.ndarray, counts: np.ndarray, k_lo: int, k_hi: int
) -> float | None:
    """The band mean of the accumulated per-sample ratio spectrum."""
    band = slice(int(k_lo), int(k_hi) + 1)
    values = ratio[band]
    good = np.isfinite(values)
    if not np.any(good):
        return None
    return float(values[good].mean())


def _measure_leakage(
    truth: np.ndarray,
    pred: np.ndarray | None,
    index: np.ndarray,
    counts: np.ndarray,
    bands: dict,
    leak_truth: list[float],
    leak_pred: list[float],
    taper_truth: list[float],
    taper_pred: list[float],
) -> None:
    """Measure the leakage of the periodic-boundary assumption on one sample.

    Two numbers are collected per frame.

    The *seam step ratio* is the size of the field's discontinuity across
    the periodic wrap relative to its own interior pixel-to-pixel
    variation. It is ~1 for a field that is genuinely continuous across the
    wrap and large for one whose storms are cut off at the domain edge. It
    needs no reference field and is the honest measure of how badly the
    periodic assumption is violated.

    The *taper sensitivity* is the change in the share of the total power
    that sits in the high-wavenumber band when the field is tapered. The
    taper makes the field continuous across the wrap, so the difference
    between the two is the part of the reported high-k power that depends
    on the boundary treatment rather than on the field itself.

    The sensitivity is a *difference of fractions*, not a ratio of powers:
    a ratio is unbounded when the untapered high-k power is near zero,
    which is exactly the case for a field with no small-scale structure.
    """
    hi_lo, hi_hi = bands["convective"]
    taper = hann_taper(truth.shape[-2:])
    for field, leak_list, taper_list in (
        (truth, leak_truth, taper_truth),
        (pred, leak_pred, taper_pred),
    ):
        if field is None:
            continue
        for frame in field:
            ratio_x, ratio_y = seam_step_ratio(frame)
            leak_list.append(0.5 * (ratio_x + ratio_y))
            raw = frame_spectrum(frame, index, counts)
            tapered = frame_spectrum(frame, index, counts, taper)
            raw_frac = band_fraction(raw, counts, hi_lo, hi_hi)
            tap_frac = band_fraction(tapered, counts, hi_lo, hi_hi)
            if np.isfinite(raw_frac) and np.isfinite(tap_frac):
                taper_list.append(raw_frac - tap_frac)


def _leakage_report(
    leak_truth: list[float],
    leak_pred: list[float],
    taper_truth: list[float],
    taper_pred: list[float],
) -> dict:
    """Assemble the leakage numbers into the report."""
    def _median(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    def _mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "explanation": (
            "The transform treats the field as periodic, so the difference "
            "between opposite edges is folded into the spectrum as "
            "structure. 'seam_step_ratio' is that discontinuity divided by "
            "the field's own interior pixel-to-pixel variation: 1.0 is a "
            "field that is continuous across the wrap, a large value means "
            "the domain cuts a storm and the periodic assumption is badly "
            "violated there. 'high_k_taper_sensitivity' is the change in "
            "the share of the total power that sits in the high-wavenumber "
            "band when the field is tapered (which makes it continuous "
            "across the wrap): it is the part of the reported high-k power "
            "that depends on the boundary treatment rather than on the "
            "field. Both are medians over the analysed frames."
        ),
        "seam_step_ratio_median": _median(leak_truth),
        "seam_step_ratio_mean": _mean(leak_truth),
        "seam_step_ratio_pred_median": _median(leak_pred),
        "seam_step_ratio_pred_mean": _mean(leak_pred),
        "high_k_taper_sensitivity": _median(taper_truth),
        "high_k_taper_sensitivity_pred": _median(taper_pred),
        "high_k_taper_sensitivity_mean": _mean(taper_truth),
        "high_k_taper_sensitivity_pred_mean": _mean(taper_pred),
        "n_frames_measured": len(leak_truth),
    }


def _json_safe(value):
    """Convert a report into something strict JSON can hold.

    ``json.dump`` writes ``NaN`` and ``Infinity`` for a non-finite float,
    which are not valid JSON and are rejected by most parsers. A missing
    number is written as ``null`` instead, which every parser accepts and
    which says "there is no value here" rather than "the value is not a
    number".

    Args:
        value: any nesting of dicts, lists, tuples, numpy arrays and floats.

    Returns:
        The same structure with every non-finite float replaced by ``None``.
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


def _write_csv(
    path: Path,
    k: np.ndarray,
    wavelength_km: np.ndarray,
    counts: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    ratio: np.ndarray,
    ratio_of_means: np.ndarray,
    bands: dict,
) -> None:
    """Write the per-bin numbers as a CSV for plotting elsewhere.

    The columns are the same numbers the JSON carries: the radial
    wavenumber, the wavelength it corresponds to, the number of Fourier
    modes in the bin, the two spectra, the two ratio estimates, and the
    band the bin falls in. A missing value is written as an empty field,
    which every CSV reader reads as missing.
    """
    convective_lo, convective_hi = bands["convective"]
    synoptic_lo, synoptic_hi = bands["synoptic"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "k_cycles_per_field",
            "wavelength_km",
            "n_modes",
            "truth_rapsd",
            "predicted_rapsd",
            "ratio",
            "ratio_of_means",
            "band",
        ])
        for b in range(k.size):
            if b == 0:
                band = "dc"
            elif synoptic_lo <= b <= synoptic_hi:
                band = "synoptic"
            elif convective_lo <= b <= convective_hi:
                band = "convective"
            else:
                band = "mid"
            writer.writerow([
                f"{k[b]:g}",
                f"{wavelength_km[b]:.6g}" if np.isfinite(wavelength_km[b]) else "",
                int(counts[b]),
                f"{truth[b]:.10g}" if np.isfinite(truth[b]) else "",
                f"{pred[b]:.10g}" if np.isfinite(pred[b]) else "",
                f"{ratio[b]:.10g}" if np.isfinite(ratio[b]) else "",
                f"{ratio_of_means[b]:.10g}" if np.isfinite(ratio_of_means[b]) else "",
                band,
            ])


if __name__ == "__main__":
    raise SystemExit(main())
