"""Meteorological verification scores for precipitation nowcasting.

All scores operate on accumulation in millimetres over the evaluation window
and are computed per threshold, aggregated over samples.

    CSI  Critical Success Index        (hits + misses + false alarms)
    ETS  Equitable Threat Score        (skill relative to chance)
    HSS  Heidke Skill Score            (a rescaled CSI)
    BIAS Frequency Bias                (hits + false alarms) / (hits + misses)
    FSS  Fractions Skill Score         (scale-sensitive neighbourhood check)

The FSS needs a neighbourhood radius to be meaningful; the default of 3 pixels
matches the 3 km grid of the public precipitation benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Thresholds in mm/6min used by the public radar benchmarks.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


@dataclass
class VerificationCounts:
    """Contingency-table accumulators for one threshold."""

    hits: float = 0.0
    misses: float = 0.0
    false_alarms: float = 0.0
    correct_negatives: float = 0.0

    def add(self, other: "VerificationCounts") -> None:
        self.hits += other.hits
        self.misses += other.misses
        self.false_alarms += other.false_alarms
        self.correct_negatives += other.correct_negatives

    @property
    def total(self) -> float:
        return self.hits + self.misses + self.false_alarms + self.correct_negatives

    def csi(self) -> float:
        denom = self.hits + self.misses + self.false_alarms
        return self.hits / denom if denom > 0 else float("nan")

    def bias(self) -> float:
        denom = self.hits + self.misses
        return (self.hits + self.false_alarms) / denom if denom > 0 else float("nan")

    def ets(self) -> float:
        denom = self.hits + self.misses + self.false_alarms
        if denom <= 0:
            return float("nan")
        chance = (
            (self.hits + self.misses) * (self.hits + self.false_alarms) / self.total
        )
        return (self.hits - chance) / (denom - chance) if denom != chance else float("nan")

    def hss(self) -> float:
        """Heidke Skill Score.

        ``HSS = 2 (H*CN - M*FA) / ((H+M)(FA+CN) + (H+FA)(M+CN))``

        Properties that pin the formula down:
          * a perfect forecast scores exactly 1, whatever the base rate;
          * a climatological forecast (one that knows only the rain
            frequency) scores exactly 0, because ``H*CN == M*FA`` there;
          * a completely wrong forecast scores -1.

        The denominator is the sum of the two skill-vs-chance terms; using the
        raw count of hits/misses/false alarms instead would leave the score
        unbounded, which is how a first implementation of this goes wrong.
        """
        if self.total <= 0:
            return float("nan")
        num = 2.0 * (
            self.hits * self.correct_negatives - self.misses * self.false_alarms
        )
        den = (
            (self.hits + self.misses) * (self.false_alarms + self.correct_negatives)
            + (self.hits + self.false_alarms) * (self.misses + self.correct_negatives)
        )
        if den == 0:
            return float("nan")
        return num / den

    def pod(self) -> float:
        denom = self.hits + self.misses
        return self.hits / denom if denom > 0 else float("nan")

    def far(self) -> float:
        denom = self.hits + self.false_alarms
        return self.false_alarms / denom if denom > 0 else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "csi": self.csi(),
            "ets": self.ets(),
            "hss": self.hss(),
            "bias": self.bias(),
            "pod": self.pod(),
            "far": self.far(),
            "hits": self.hits,
            "misses": self.misses,
            "false_alarms": self.false_alarms,
        }


def contingency_counts(
    pred: np.ndarray, truth: np.ndarray, threshold: float
) -> VerificationCounts:
    """Build the contingency table for one threshold.

    Args:
        pred: predicted accumulation (mm).
        truth: observed accumulation (mm).
        threshold: rain rate above which a pixel counts as "rain".
    """
    p = np.asarray(pred) >= threshold
    t = np.asarray(truth) >= threshold
    hits = float(np.count_nonzero(p & t))
    misses = float(np.count_nonzero(~p & t))
    false_alarms = float(np.count_nonzero(p & ~t))
    correct_negatives = float(np.count_nonzero(~p & ~t))
    return VerificationCounts(hits, misses, false_alarms, correct_negatives)


def _box_sum(field: np.ndarray, radius: int) -> np.ndarray:
    """Sum of a field over a square neighbourhood of half-width ``radius``.

    Works on a 2D field and on a batch of 2D fields; the last two axes are
    treated as the spatial ones. Computed with an integral image so the cost
    is independent of the radius.
    """
    lead = field.shape[:-2]
    h, w = field.shape[-2:]
    flat = field.reshape(-1, h, w)
    padded = np.zeros((flat.shape[0], h + 1, w + 1), dtype=np.int64)
    padded[:, 1:, 1:] = np.cumsum(np.cumsum(flat.astype(np.int64), axis=1), axis=2)
    r = radius
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    total = (
        padded[:, y1[:, None], x1[None, :]]
        - padded[:, y0[:, None], x1[None, :]]
        - padded[:, y1[:, None], x0[None, :]]
        + padded[:, y0[:, None], x0[None, :]]
    )
    return total.reshape(*lead, h, w)


def fractions_skill_score(
    pred: np.ndarray,
    truth: np.ndarray,
    threshold: float,
    radius: int = 3,
) -> float:
    """Fractions Skill Score for one threshold (Roberts & Lean, 2008).

    For every grid point the *fraction* of its neighbourhood that is rain is
    computed for the forecast and for the observation. The score compares those
    fraction fields:

    ``FSS = 1 - sum((a - b)^2) / sum(a^2 + b^2)``

    where ``a`` is the observed fraction field and ``b`` the forecast one.

    The denominator ``sum(a^2 + b^2)`` is the largest possible squared error
    for the given fields, so the score runs from 0 (no skill -- the forecast
    carries no spatial information) to 1 (perfect). Because the comparison is
    point-wise on *fractions*, a spatially displaced forecast is penalised
    gently rather than catastrophically, which is the whole point of the
    metric; a plain contingency-table score cannot distinguish a displaced
    storm from a wrong one.

    The neighbourhood is a square (Chebyshev) window, the usual approximation
    to the circular one; at the domain edge the window is truncated and the
    fraction is normalised by the number of points actually inside it.

    Args:
        pred: forecast accumulation (mm). The last two axes are spatial.
        truth: observed accumulation (mm), same shape.
        threshold: rain rate above which a pixel counts as rain.
        radius: neighbourhood half-width in pixels.
    """
    p = (np.asarray(pred) >= threshold).astype(np.float64)
    t = (np.asarray(truth) >= threshold).astype(np.float64)

    rate_obs = _box_sum(t, radius) / _box_sum(np.ones_like(t), radius)
    rate_fcst = _box_sum(p, radius) / _box_sum(np.ones_like(p), radius)

    numer = ((rate_obs - rate_fcst) ** 2).sum()
    denom = (rate_obs**2 + rate_fcst**2).sum()
    if denom == 0:
        # both fields are empty: no error and no skill to measure
        return 1.0
    return float(1.0 - numer / denom)


@dataclass
class PrecipitationVerifier:
    """Accumulate verification statistics across a validation run.

    Example:
        >>> v = PrecipitationVerifier()
        >>> v.update(pred_mm, truth_mm)
        >>> report = v.report()
    """

    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    fss_radius: int = 3
    #: Extra neighbourhood half-widths for the multiscale FSS, in pixels.
    #:
    #: Empty by default, because the fraction field is recomputed for every
    #: half-width and the verification is bound by the processor rather than
    #: by the card. A caller that wants the multiscale table asks for it
    #: explicitly. Note that the half-width is not the window size: a
    #: half-width of one is a three by three window, two is five by five,
    #: four is nine by nine.
    fss_radii: tuple[int, ...] = ()
    counts: dict[float, VerificationCounts] = field(default_factory=dict)
    fss_sum: dict[float, float] = field(default_factory=dict)
    fss_n: dict[float, int] = field(default_factory=dict)
    #: The multiscale FSS, keyed by half-width then threshold.
    fss_multi_sum: dict[int, dict[float, float]] = field(default_factory=dict)
    fss_multi_n: dict[int, dict[float, int]] = field(default_factory=dict)
    #: Per-lead-time CSI, keyed by lead index then threshold.
    lead_counts: dict[int, dict[float, VerificationCounts]] = field(default_factory=dict)
    n_samples: int = 0
    #: Per-window contingency counts, kept only when asked for.
    #:
    #: The accumulated totals above are enough to compute a score but not
    #: to put an interval on it: a bootstrap resamples the independent
    #: units, and the unit here is the evaluation window, not the pixel.
    #: Without the per-window tables the interval can only be guessed from
    #: the spread of repeated runs, which measures the training noise and
    #: says nothing about the sampling uncertainty of the score.
    per_window: bool = False
    window_counts: list[dict[float, VerificationCounts]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # A caller that has no thresholds in mind passes None, and the
        # default set is the right answer for it. Without this the
        # verifier would iterate over None and fail at the first sample.
        if self.thresholds is None:
            self.thresholds = DEFAULT_THRESHOLDS
        for thr in self.thresholds:
            self.counts.setdefault(thr, VerificationCounts())
            self.fss_sum.setdefault(thr, 0.0)
            self.fss_n.setdefault(thr, 0)
        for radius in self.fss_radii:
            self.fss_multi_sum.setdefault(radius, {})
            self.fss_multi_n.setdefault(radius, {})
            for thr in self.thresholds:
                self.fss_multi_sum[radius].setdefault(thr, 0.0)
                self.fss_multi_n[radius].setdefault(thr, 0)

    def update(self, pred: np.ndarray, truth: np.ndarray) -> None:
        """Accumulate one batch.

        Args:
            pred: ``(B, T, H, W)`` or ``(B, T, 1, H, W)`` predictions in mm.
            truth: matching observations in mm.
        """
        pred = np.asarray(pred)
        truth = np.asarray(truth)
        if pred.ndim == 5:
            pred = pred[:, :, 0]
        if truth.ndim == 5:
            truth = truth[:, :, 0]
        if pred.shape != truth.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {truth.shape}")

        self.n_samples += pred.shape[0]
        for thr in self.thresholds:
            c = contingency_counts(pred, truth, thr)
            self.counts[thr].add(c)
            self.fss_sum[thr] += fractions_skill_score(
                pred, truth, thr, self.fss_radius
            )
            self.fss_n[thr] += 1
            for lead in range(pred.shape[1]):
                lc = self.lead_counts.setdefault(lead, {})
                acc = lc.setdefault(thr, VerificationCounts())
                acc.add(contingency_counts(pred[:, lead], truth[:, lead], thr))

        for radius in self.fss_radii:
            for thr in self.thresholds:
                self.fss_multi_sum[radius][thr] += fractions_skill_score(
                    pred, truth, thr, radius
                )
                self.fss_multi_n[radius][thr] += 1

        if self.per_window:
            self._record_windows(pred, truth)

    def _record_windows(self, pred: np.ndarray, truth: np.ndarray) -> None:
        """Store the contingency table of every window in the batch.

        The windows are the independent units of the evaluation: they come
        from different positions and different cases, and they overlap in
        time only inside a case. A bootstrap therefore resamples whole
        windows, and it needs the table of each one separately.
        """
        for b in range(pred.shape[0]):
            entry: dict[float, VerificationCounts] = {}
            for thr in self.thresholds:
                entry[thr] = contingency_counts(
                    pred[b], truth[b], thr
                )
            self.window_counts.append(entry)

    def per_window_table(self) -> list[dict[str, dict[str, float]]]:
        """The per-window tables, in a form that survives JSON.

        Returns one list entry per evaluated window, each holding the four
        contingency numbers for every threshold.
        """
        return [
            {
                f"{thr}": {
                    "hits": c.hits,
                    "misses": c.misses,
                    "false_alarms": c.false_alarms,
                    "correct_negatives": c.correct_negatives,
                }
                for thr, c in entry.items()
            }
            for entry in self.window_counts
        ]

    def report(self) -> dict:
        """Aggregate scores into a nested dict."""
        out: dict = {"n_samples": self.n_samples, "thresholds": {}}
        for thr in self.thresholds:
            c = self.counts[thr]
            entry = c.as_dict()
            n = self.fss_n[thr]
            entry["fss"] = self.fss_sum[thr] / n if n else float("nan")
            if self.fss_radii:
                # The window size is what a reader cares about, so the
                # half-width is reported as the side of the square it
                # stands for: a half-width of one is a three by three
                # window, two is five by five, four is nine by nine.
                entry["fss_by_window"] = {
                    f"{2 * radius + 1}x{2 * radius + 1}": (
                        self.fss_multi_sum[radius][thr]
                        / self.fss_multi_n[radius][thr]
                        if self.fss_multi_n[radius][thr]
                        else float("nan")
                    )
                    for radius in self.fss_radii
                }
            out["thresholds"][f"{thr}"] = entry
        # The lead-time breakdown carries the whole score set, not only the
        # CSI: a paper that claims "the false-alarm rate is lower at long
        # lead times" needs the false-alarm rate per lead time, and the CSI
        # alone does not distinguish a model that misses rain from one that
        # invents it.
        #
        # The lead is keyed by its string form, so the report survives a
        # round trip through JSON with the same shape as the threshold
        # keys -- a reader of the file should not have to know that one
        # level is an integer and the other is not.
        out["lead_scores"] = {
            str(lead): {
                f"{thr}": {
                    "csi": cnt.csi(),
                    "ets": cnt.ets(),
                    "hss": cnt.hss(),
                    "far": cnt.far(),
                    "pod": cnt.pod(),
                    "bias": cnt.bias(),
                }
                for thr, cnt in per_thr.items()
            }
            for lead, per_thr in sorted(self.lead_counts.items())
        }
        # The plain CSI-by-lead table is kept as well, for the scripts that
        # read it.
        out["lead_csi"] = {
            lead: {f"{thr}": scores["csi"] for thr, scores in per_thr.items()}
            for lead, per_thr in out["lead_scores"].items()
        }
        return out

    def summary(self) -> str:
        """Human-readable score table."""
        rows = ["threshold |  CSI   ETS    HSS    BIAS   FSS   POD    FAR"]
        for thr in self.thresholds:
            c = self.counts[thr]
            n = self.fss_n[thr]
            fss = self.fss_sum[thr] / n if n else float("nan")
            rows.append(
                f"{thr:8.2f} | {c.csi():.3f} {c.ets():.3f} {c.hss():.3f} "
                f"{c.bias():.3f} {fss:.3f} {c.pod():.3f} {c.far():.3f}"
            )
        return "\n".join(rows)


def verify(
    pred: np.ndarray,
    truth: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    fss_radius: int = 3,
) -> dict:
    """One-shot verification of a prediction batch."""
    v = PrecipitationVerifier(thresholds=thresholds, fss_radius=fss_radius)
    v.update(pred, truth)
    return v.report()
