"""Reassemble sliding patches into a full-grid forecast with smooth Hann feathering."""
from __future__ import annotations

import numpy as np


class PatchStitcher:
    """Accumulate patch forecasts into full-grid fields using 2D Hann-window weighting.

    Args:
        grid_shape: the full grid, (height, width).
        n_frames: the number of forecast frames per sample.
        n_positions: the number of spatial positions each sample is cut into.
        use_hann: whether to apply 2D Hann feathering to suppress tile boundary seams.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        n_frames: int,
        n_positions: int,
        use_hann: bool = True,
    ):
        self.grid_shape = tuple(grid_shape)
        self.n_frames = n_frames
        self.n_positions = n_positions
        self.use_hann = use_hann

        self._sums: dict[int, np.ndarray] = {}
        self._weights: dict[int, np.ndarray] = {}
        self._truths: dict[int, np.ndarray] = {}
        self._seen: dict[int, set[int]] = {}
        self._order: list[int] = []
        self._window_cache: dict[tuple[int, int], np.ndarray] = {}

    def _get_window_weight(self, th: int, tw: int) -> np.ndarray:
        key = (th, tw)
        if key not in self._window_cache:
            if not self.use_hann or th <= 4 or tw <= 4:
                self._window_cache[key] = np.ones((th, tw), dtype=np.float64)
            else:
                wy = np.hanning(th).astype(np.float64)
                wx = np.hanning(tw).astype(np.float64)
                w2d = np.outer(wy, wx)
                self._window_cache[key] = np.clip(w2d, 0.05, 1.0)
        return self._window_cache[key]

    def add(
        self,
        sample_index: int,
        position_index: int,
        origin: tuple[int, int],
        pred: np.ndarray,
        truth: np.ndarray,
    ) -> None:
        y0, x0 = origin
        height, width = self.grid_shape
        th, tw = pred.shape[-2], pred.shape[-1]
        if y0 + th > height or x0 + tw > width:
            raise ValueError(
                f"tile at {origin} of size {th}x{tw} does not fit in {self.grid_shape}"
            )

        if sample_index not in self._sums:
            self._sums[sample_index] = np.zeros(
                (self.n_frames, height, width), dtype=np.float64
            )
            self._weights[sample_index] = np.zeros((height, width), dtype=np.float64)
            self._truths[sample_index] = np.full(
                (self.n_frames, height, width), np.nan, dtype=np.float64
            )
            self._seen[sample_index] = set()
            self._order.append(sample_index)

        w = self._get_window_weight(th, tw)
        self._sums[sample_index][:, y0:y0 + th, x0:x0 + tw] += pred * w[None, :, :]
        self._weights[sample_index][y0:y0 + th, x0:x0 + tw] += w

        window = self._truths[sample_index][:, y0:y0 + th, x0:x0 + tw]
        np.copyto(window, truth, where=np.isnan(window))
        self._seen[sample_index].add(position_index)

    def is_complete(self, sample_index: int) -> bool:
        return len(self._seen.get(sample_index, ())) == self.n_positions

    def finalize(self, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
        sums = self._sums[sample_index]
        weights = self._weights[sample_index]
        truth = self._truths[sample_index]

        uncovered = weights < 1e-6
        if uncovered.any():
            raise ValueError(
                f"{int(uncovered.sum())} pixels of the grid were never covered "
                f"by a tile; the sliding walk does not tile the field"
            )

        pred = sums / weights[None, :, :]
        if np.isnan(truth).any():
            raise ValueError("some pixels have no observation")
        return pred.astype(np.float32), truth.astype(np.float32)

    def release(self, sample_index: int) -> None:
        self._sums.pop(sample_index, None)
        self._weights.pop(sample_index, None)
        self._truths.pop(sample_index, None)
        self._seen.pop(sample_index, None)
        if sample_index in self._order:
            self._order.remove(sample_index)

    @property
    def pending(self) -> list[int]:
        return list(self._order)


__all__ = ["PatchStitcher"]