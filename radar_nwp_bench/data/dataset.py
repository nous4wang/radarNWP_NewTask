"""The unified radar + NWP precipitation dataset.

Supports:
    - Symmetric patch sliding
    - Activity-aware dynamic dry patch filtering
    - Intermediate missing frame interpolation & tolerance
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .alignment import NWPAlignment, select_nwp_window
from .coordinates import (
    NWP_NATIVE_SHAPE,
    NWP_TO_RADAR_SCALE,
    RADAR_SHAPE,
    _bilinear_upsample,
    symmetric_patch_origins,
)
from .index import (
    IRREGULAR_CASES,
    NWP_VARS,
    RADAR_STEP_MIN,
    RADAR_VARS,
    add_minutes,
)
from .pack import load_packed, load_packed_mmap
from .transforms import normalize_nwp_field, normalize_rain, radar_log_transform


@dataclass
class SampleSpec:
    """One training sample resolved to array indices with missing frame markers."""

    case_path: Path
    issue_time: int
    radar_idx: tuple[int, ...]
    label_idx: tuple[int, ...]
    nwp_idx: tuple[int, ...]
    nwp_lead_minutes: tuple[int, ...]
    missing_radar_steps: tuple[int, ...] = ()  # 标记内部插值帧位置
    # The label frames that share a timestamp with the history frames.
    #
    # The archive's RADAR and LABEL series carry the same timestamps --
    # 98 of 98 frames agree -- so the ground rainfall at each of the
    # eleven history instants exists. The entry is the index into the
    # case's label array, or -1 when that instant was never recorded.
    #
    # A -1 does not mean "no observation": the same hole in the radar
    # series is repaired by interpolating its two neighbours in time, and
    # the label is repaired the same way, so the two stay consistent.
    hist_label_idx: tuple[int, ...] = ()

    @property
    def has_forecast_nwp(self) -> bool:
        return any(l > 0 for l in self.nwp_lead_minutes)


class RadarNWPDataset(Dataset):
    """Radar + NWP -> 6-minute accumulated rainfall dataset."""

    def __init__(
        self,
        data_root: str | Path,
        packed: bool = False,
        in_len: int = 11,
        out_len: int = 10,
        patch_size: int | None = 128,
        alignment: NWPAlignment | str = NWPAlignment.FORECAST,
        nwp_vars: tuple[str, ...] = NWP_VARS,
        radar_vars: tuple[str, ...] = RADAR_VARS,
        random_crop: bool = True,
        normalize_target: bool = False,
        max_cases: int | None = None,
        case_ids: list[str] | None = None,
        split: str | None = None,
        max_samples: int | None = None,
        slide_stride: int | None = None,
        require_forecast_nwp: bool = True,
        cache_bytes: int = 2 << 30,
        blank_nwp: bool = False,
        filter_dry: bool = False,
        dry_keep_prob: float = 0.20,
        rain_threshold: float = 0.05,
        tolerate_one_missing_frame: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(data_root)
        self.packed = packed
        self.in_len = in_len
        self.out_len = out_len
        self.patch_size = patch_size
        self.alignment = NWPAlignment(alignment)
        self.nwp_vars = tuple(nwp_vars)
        self.radar_vars = tuple(radar_vars)
        self.random_crop = random_crop
        self.normalize_target = normalize_target
        self.require_forecast_nwp = require_forecast_nwp
        self.slide_stride = slide_stride
        self.blank_nwp = blank_nwp
        self.split = split
        self.filter_dry = filter_dry
        self.dry_keep_prob = dry_keep_prob
        self.rain_threshold = rain_threshold
        self.tolerate_one_missing_frame = tolerate_one_missing_frame

        self._patch_origins: tuple[tuple[int, int], ...] = ()
        self._grid_shape: tuple[int, int] = ()

        self.samples: list[SampleSpec] = []
        self._item_map: list[tuple[int, int]] | None = None
        self._cache: dict[Path, dict] = {}
        self._cache_order: list[Path] = []
        self._cache_bytes = 0
        self.cache_bytes = cache_bytes

        self._build_index(max_cases, case_ids, split)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.set_grid_shape(*RADAR_SHAPE)

        # 训练集启动活跃度自适应过滤
        if self.filter_dry and self.split == "train" and self.slide_stride is not None:
            self._build_active_item_map()

    def _case_files(self) -> list[Path]:
        if self.packed:
            files = sorted(p for p in self.root.glob("*.npz") if p.is_file())
            if files:
                return files
            return sorted(p for p in self.root.iterdir() if (p / "radar.npy").is_file())
        return sorted(p for p in self.root.glob("CP_*") if p.is_dir())

    def _build_index(
        self,
        max_cases: int | None,
        case_ids: list[str] | None = None,
        split: str | None = None,
    ) -> None:
        from .index import scan_case, split_cases

        files = self._case_files()
        if case_ids is not None:
            wanted = set(case_ids)
            files = [f for f in files if f.stem in wanted]

        # 专门处理缺失帧独立测试集与标准划分
        if split == "robustness_missing":
            files = [f for f in files if f.stem in IRREGULAR_CASES or f.name in IRREGULAR_CASES]
        elif split is not None:
            # split_cases subtracts the irregular cases itself, so the clean
            # partition cannot be contaminated by a case with a hole in it.
            names = [f.stem for f in files]
            parts = split_cases(names)
            if split not in parts:
                raise ValueError(
                    f"unknown split {split!r}; expected one of "
                    f"{sorted(parts)} or 'robustness_missing'"
                )
            wanted = set(parts[split])
            files = [f for f in files if f.stem in wanted]

        if max_cases is not None:
            files = files[:max_cases]
        if not files:
            raise FileNotFoundError(f"no cases found under {self.root} for split {split}")

        for path in files:
            if self.packed:
                if path.is_dir():
                    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
                    radar_times = list(meta["radar_times"])
                    label_times = list(meta["label_times"])
                    nwp_times = list(meta["nwp_times"])
                else:
                    with np.load(path, allow_pickle=False) as data:
                        radar_times = data["radar_times"].tolist()
                        label_times = data["label_times"].tolist()
                        nwp_times = data["nwp_times"].tolist()
            else:
                index = scan_case(path)
                radar_times = index.radar_times
                label_times = index.label_times
                nwp_times = index.nwp_times

            if len(radar_times) < self.in_len + self.out_len:
                continue
            self._index_case(path, radar_times, label_times, nwp_times)

    def _index_case(
        self,
        path: Path,
        radar_times: list[int],
        label_times: list[int],
        nwp_times: list[int],
    ) -> None:
        label_pos = {t: i for i, t in enumerate(label_times)}
        nwp_pos = {t: i for i, t in enumerate(nwp_times)}
        radar_pos = {t: i for i, t in enumerate(radar_times)}

        for r in range(self.in_len - 1, len(radar_times) - self.out_len):
            issue_time = radar_times[r]
            # The target window is the 10 frames after the issue time, spaced
            # exactly 6 minutes apart -- the same rule as the history. Taking
            # the next 10 *positions* would let a skipped step push the first
            # target to T+12, and the lead-time curve would then report a
            # "30-minute" forecast that is really 36 minutes out.
            future = [
                add_minutes(issue_time, k * RADAR_STEP_MIN)
                for k in range(1, self.out_len + 1)
            ]
            if not all(t in label_pos for t in future):
                continue

            window = select_nwp_window(
                self.alignment, issue_time, radar_times, nwp_times, self.out_len
            )
            if window is None or (self.require_forecast_nwp and not window.has_forecast):
                continue
            if not all(t in nwp_pos for t in window.valid_times):
                continue

            # The history window is the 11 frames that end at the issue time,
            # spaced exactly 6 minutes apart.
            #
            # Taking the 11 *positions* before r would silently accept a
            # window whose frames span 66 minutes because one step was
            # skipped: the frames would still be 11 in number, but the
            # interval between the last two would be 12 minutes, and the
            # model would read a history that is not evenly sampled while
            # believing it is. Every lead time in the benchmark is defined
            # against the issue time, so an unevenly sampled history makes
            # the lead-time curve meaningless.
            #
            # So the window is built from *timestamps*: the frame at
            # issue_time - k*6 minutes for k = 0 .. in_len-1. A frame that
            # was never recorded leaves a hole, and the hole is repaired by
            # interpolating its two neighbours in time.
            hist_indices = []
            missing_steps = []
            # The label frames that share a timestamp with the history.
            # The archive's RADAR and LABEL series carry the same
            # timestamps, so the ground rainfall at each history instant
            # exists -- except where the same hole in the record took it
            # away, which is marked -1 and repaired on read in the same
            # way as the radar.
            hist_label_indices: list[int] = []
            for k in range(self.in_len - 1, -1, -1):
                target = add_minutes(issue_time, -k * RADAR_STEP_MIN)
                hist_label_indices.append(label_pos.get(target, -1))
                if target in radar_pos:
                    hist_indices.append(radar_pos[target])
                    continue
                # The frame is missing. It is repairable only if the frames
                # on either side of it in time are both present; otherwise
                # the window is not a 6-minute history and is dropped.
                before = add_minutes(target, -RADAR_STEP_MIN)
                after = add_minutes(target, RADAR_STEP_MIN)
                if before in radar_pos and after in radar_pos:
                    if not self.tolerate_one_missing_frame:
                        # The window cannot be built without the frame, and
                        # the tolerance has been switched off: drop it
                        # rather than quietly feed a different history.
                        hist_indices = []
                        break
                    # The slot holds the earlier neighbour; the value is
                    # rebuilt from both when the sample is read.
                    hist_indices.append(radar_pos[before])
                    missing_steps.append(len(hist_indices) - 1)
                else:
                    hist_indices = []
                    break

            if not hist_indices:
                continue

            self.samples.append(
                SampleSpec(
                    case_path=path,
                    issue_time=issue_time,
                    radar_idx=tuple(hist_indices),
                    label_idx=tuple(label_pos[t] for t in future),
                    nwp_idx=tuple(nwp_pos[t] for t in window.valid_times),
                    nwp_lead_minutes=window.lead_minutes,
                    missing_radar_steps=tuple(missing_steps),
                    hist_label_idx=tuple(hist_label_indices),
                )
            )

    def _build_active_item_map(self) -> None:
        """Build the list of patches that carry rain, keeping a fraction of
        the dry ones.

        The purpose is to spend the training budget on patches that contain
        something. Most of the 301x301 field is dry at any given moment, and
        a patch that never sees rain teaches the model nothing about rain.

        Two details make the filter mean what it says.

        The rainfall is measured **per target frame**, not summed over the
        whole target window. The threshold is a rain rate in millimetres per
        6-minute accumulation -- the same quantity the verification uses --
        so summing ten frames and comparing the total against a per-frame
        threshold would call a patch wet that carries only a tenth of the
        threshold in every one of its frames. A patch counts as active when
        *any* of its target frames reaches the threshold.

        The origins are the ones the reader will actually use: with a slide
        stride the patch walks the grid on a fixed lattice, and
        :meth:`_crop_origin` is what :meth:`__getitem__` calls. Using the
        lattice here and the reader there would filter patches that are
        never read and keep patches that are.

        The dry patches that are kept are chosen by a seeded generator, so
        the selection is reproducible: the same archive gives the same
        training set.
        """
        n_pos = self.n_patch_positions
        active_items = []
        rng = np.random.default_rng(42)

        for s_idx, spec in enumerate(self.samples):
            case = self._load_case(spec.case_path)
            label = case["label"]
            # (n_target_frames, H, W) -- the rainfall of each target frame.
            frames = label[list(spec.label_idx), 0]

            for p_idx in range(n_pos):
                y0, x0 = self._crop_origin(
                    label.shape[-2], label.shape[-1], p_idx
                )
                sz = self.patch_size
                window = frames[:, y0:y0 + sz, x0:x0 + sz]
                if window.size == 0:
                    continue
                patch_max_rain = float(window.max())
                if (
                    patch_max_rain >= self.rain_threshold
                    or rng.random() < self.dry_keep_prob
                ):
                    active_items.append((s_idx, p_idx))

        self._item_map = active_items

    def _load_case(self, path: Path) -> dict:
        cached = self._cache.get(path)
        if cached is not None:
            if path in self._cache_order:
                self._cache_order.remove(path)
            self._cache_order.append(path)
            return cached

        if self.packed:
            if path.is_dir():
                loaded = load_packed_mmap(path)
            else:
                loaded = load_packed(path)
        else:
            loaded = _load_raw_case(path)

        size = sum(int(v.nbytes) for v in loaded.values() if isinstance(v, np.ndarray))
        self._cache[path] = loaded
        self._cache_bytes += size
        self._cache_order.append(path)

        while self._cache_order and self._cache_bytes > self.cache_bytes:
            oldest = self._cache_order.pop(0)
            evicted = self._cache.pop(oldest, None)
            if evicted is not None:
                self._cache_bytes -= sum(int(v.nbytes) for v in evicted.values() if isinstance(v, np.ndarray))
        return loaded

    def __len__(self) -> int:
        if self._item_map is not None:
            return len(self._item_map)
        return len(self.samples) * self.n_patch_positions

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._item_map is not None:
            sample_idx, position_idx = self._item_map[idx]
        else:
            n_pos = self.n_patch_positions
            sample_idx, position_idx = divmod(idx, n_pos)

        spec = self.samples[sample_idx]
        case = self._load_case(spec.case_path)

        radar = case["radar"]
        label = case["label"]

        radar_vars = list(case["radar_vars"])
        nwp_vars = list(case["nwp_vars"])
        r_sel = [radar_vars.index(v) for v in self.radar_vars]
        n_sel = [nwp_vars.index(v) for v in self.nwp_vars]

        sz = self.patch_size
        y0, x0 = self._crop_origin(radar.shape[-2], radar.shape[-1], position_idx)

        hist = radar[np.ix_(spec.radar_idx, r_sel, np.arange(y0, y0 + sz), np.arange(x0, x0 + sz))].copy()

        # 对内部缺失帧执行前后线性插值容忍修复
        #
        # The window is built from timestamps, so a hole is a slot whose
        # 6-minute frame was never recorded. The slot holds the frame from
        # *before* the hole (that is the nearest one that exists), and the
        # frame after the hole sits in the next slot -- the window is
        # evenly spaced, so the neighbour of a missing frame is always the
        # slot next to it.
        #
        # The value written back is the linear interpolation in time: half
        # the frame before the hole plus half the frame after it. The mask
        # marks the slot that was rebuilt, so a model can tell an
        # interpolated frame from an observed one.
        missing_mask = np.zeros(self.in_len, dtype=bool)
        if spec.missing_radar_steps:
            for step in spec.missing_radar_steps:
                if 0 <= step < self.in_len - 1:
                    hist[step] = 0.5 * (hist[step] + hist[step + 1])
                    missing_mask[step] = True

        hist = self._radar_transform(hist)
        tgt = label[np.ix_(spec.label_idx, [0], np.arange(y0, y0 + sz), np.arange(x0, x0 + sz))]

        # The ground rainfall at the eleven history instants.
        #
        # The archive's RADAR and LABEL series carry the same timestamps
        # (98 of 98 frames agree), so this is the observed rainfall at the
        # moments the radar history covers -- the same times, with no
        # displacement between them. A model that learns to invert
        # reflectivity into rainfall can be supervised on it directly,
        # which is the whole point of exposing it.
        #
        # A slot whose label was never recorded is repaired the way the
        # radar hole is repaired: by the two neighbours in time. The
        # search goes outward rather than to the adjacent slots, because
        # two holes can be adjacent and the neighbour of a hole is then
        # itself a hole -- taking the adjacent slot would silently
        # produce a row of zeros.
        y_hist, hist_missing_mask = self._read_hist_label(
            label, spec.hist_label_idx, y0, x0, sz
        )

        if self.blank_nwp:
            nwp_res = np.zeros((len(spec.nwp_idx), len(n_sel), sz, sz), dtype=np.float32)
        else:
            nwp = case["nwp"]
            nwp_res = np.stack(
                [self._resample_nwp_patch(nwp[t], n_sel, y0, x0, sz) for t in spec.nwp_idx],
                axis=0,
            )
            # Each NWP variable is centred and scaled by its own measured
            # statistics. The variables are physical quantities in different
            # units and their ranges span six orders of magnitude (the
            # divergence fields around 1e-3, the LFC level around 1e4), so
            # fed raw the large fields dominate the first convolution and
            # the small ones -- which carry the dynamical signal -- vanish.
            # The radar fields are normalised the same way.
            #
            # The tensor is (T, C, H, W): the variable is the *second* axis,
            # so every slice of it is normalised at once.
            for channel, name in enumerate(self.nwp_vars):
                nwp_res[:, channel] = normalize_nwp_field(nwp_res[:, channel], name)

        if self.normalize_target:
            tgt = normalize_rain(tgt)

        return {
            "x_radar": torch.from_numpy(np.ascontiguousarray(hist)),
            "x_nwp": torch.from_numpy(np.ascontiguousarray(nwp_res)),
            "y_rain": torch.from_numpy(np.ascontiguousarray(tgt)),
            "y_hist": torch.from_numpy(np.ascontiguousarray(y_hist)),
            "radar_missing_mask": torch.from_numpy(missing_mask),
            "hist_missing_mask": torch.from_numpy(hist_missing_mask),
            "issue_time": torch.tensor(spec.issue_time, dtype=torch.long),
            "nwp_lead": torch.tensor(spec.nwp_lead_minutes, dtype=torch.long),
            "patch_origin": torch.tensor([y0, x0], dtype=torch.long),
            "case_index": torch.tensor(sample_idx, dtype=torch.long),
            "position_index": torch.tensor(position_idx, dtype=torch.long),
        }

    def _read_hist_label(
        self,
        label: np.ndarray,
        hist_label_idx: tuple[int, ...],
        y0: int,
        x0: int,
        sz: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read the ground rainfall at the history instants.

        Args:
            label: the case's label array, ``(T, 1, H, W)`` in mm per
                6-minute accumulation.
            hist_label_idx: the label index of each history slot, or -1
                where that instant was never recorded.
            y0: the crop origin along the first spatial axis.
            x0: the crop origin along the second.
            sz: the crop size.

        Returns:
            The rainfall of shape ``(T_in, 1, sz, sz)`` and a boolean mask
            marking the slots that were rebuilt rather than observed.
        """
        n_hist = len(hist_label_idx)
        frames: list[np.ndarray] = []
        rebuilt = np.zeros(n_hist, dtype=bool)

        for slot, index in enumerate(hist_label_idx):
            if index >= 0:
                frames.append(label[index, 0:1, y0:y0 + sz, x0:x0 + sz])
                continue

            # The instant was not recorded. Take the nearest neighbour on
            # either side that was, and interpolate between them in time.
            before = next(
                (
                    hist_label_idx[slot - back]
                    for back in range(1, slot + 1)
                    if hist_label_idx[slot - back] >= 0
                ),
                -1,
            )
            after = next(
                (
                    hist_label_idx[slot + fwd]
                    for fwd in range(1, n_hist - slot)
                    if hist_label_idx[slot + fwd] >= 0
                ),
                -1,
            )
            if before >= 0 and after >= 0:
                value = 0.5 * (
                    label[before, 0:1, y0:y0 + sz, x0:x0 + sz]
                    + label[after, 0:1, y0:y0 + sz, x0:x0 + sz]
                )
            elif before >= 0:
                value = label[before, 0:1, y0:y0 + sz, x0:x0 + sz]
            elif after >= 0:
                value = label[after, 0:1, y0:y0 + sz, x0:x0 + sz]
            else:
                # No recorded neighbour anywhere: the slot has no
                # information, and saying so is better than inventing it.
                value = np.zeros((1, sz, sz), dtype=label.dtype)
            frames.append(value)
            rebuilt[slot] = True

        stacked = np.stack(frames, axis=0).astype(np.float32)
        return np.clip(stacked, 0.0, None), rebuilt

    def _resample_nwp_patch(
        self,
        field: np.ndarray,
        n_sel: list[int],
        y0: int,
        x0: int,
        size: int,
    ) -> np.ndarray:
        sub = field[n_sel]
        scale = NWP_TO_RADAR_SCALE
        crop = (NWP_NATIVE_SHAPE[0] * scale - RADAR_SHAPE[0]) // 2

        def source_window(origin: int, axis_len: int) -> tuple[int, int]:
            first = origin + crop
            last = origin + crop + size - 1
            s_first = (first + 0.5) / scale - 0.5
            s_last = (last + 0.5) / scale - 0.5
            lo = int(np.floor(s_first))
            hi = int(np.floor(s_last)) + 2
            return max(0, lo), min(axis_len, hi)

        src_y0, src_y1 = source_window(y0, sub.shape[-2])
        src_x0, src_x1 = source_window(x0, sub.shape[-1])

        window = sub[:, src_y0:src_y1, src_x0:src_x1]
        up = _bilinear_upsample(window, scale)
        off_y = (y0 + crop) - src_y0 * scale
        off_x = (x0 + crop) - src_x0 * scale
        return up[:, off_y:off_y + size, off_x:off_x + size]

    def _radar_transform(self, hist: np.ndarray) -> np.ndarray:
        out = np.empty_like(hist)
        for c, var in enumerate(self.radar_vars):
            out[:, c] = radar_log_transform(hist[:, c], var)
        return out

    def set_grid_shape(self, height: int, width: int) -> None:
        self._grid_shape = (int(height), int(width))
        self._patch_origins = self._enumerate_origins(height, width)

    def _enumerate_origins(
        self, height: int, width: int
    ) -> tuple[tuple[int, int], ...]:
        size = self.patch_size
        if size is None or size >= min(height, width):
            return ((0, 0),)
        max_y = height - size
        max_x = width - size
        if self.slide_stride is None:
            return ((max_y // 2, max_x // 2),)
        stride = max(1, int(self.slide_stride))
        ys = symmetric_patch_origins(height, size, stride)
        xs = symmetric_patch_origins(width, size, stride)
        return tuple((y, x) for y in ys for x in xs)

    @property
    def patch_origins(self) -> tuple[tuple[int, int], ...]:
        if not self._patch_origins:
            self.set_grid_shape(RADAR_SHAPE[0], RADAR_SHAPE[1])
        return self._patch_origins

    @property
    def grid_shape(self) -> tuple[int, int]:
        if not self._grid_shape:
            self.set_grid_shape(*RADAR_SHAPE)
        return self._grid_shape

    @property
    def n_patch_positions(self) -> int:
        return len(self.patch_origins)

    @property
    def nwp_frames(self) -> int:
        if not self.samples:
            return 2
        return len(self.samples[0].nwp_idx)

    def _crop_origin(
        self, h: int, w: int, position_index: int = 0
    ) -> tuple[int, int]:
        size = self.patch_size
        if size is None or size >= min(h, w):
            return 0, 0
        if self.slide_stride is not None:
            origins = self.patch_origins
            return origins[position_index % len(origins)]
        max_y, max_x = h - size, w - size
        if self.random_crop:
            rng = np.random.default_rng()
            return int(rng.integers(0, max_y + 1)), int(rng.integers(0, max_x + 1))
        return max_y // 2, max_x // 2


def _load_raw_case(path: Path) -> dict:
    from .index import scan_case
    from .transforms import clean_label_rain, clean_nwp_field, clean_radar_field

    index = scan_case(path)
    radar_times = index.radar_times
    label_times = index.label_times
    nwp_times = index.nwp_times
    radar_vars = list(RADAR_VARS)
    nwp_vars = list(NWP_VARS)

    radar = np.zeros((len(radar_times), len(radar_vars), *RADAR_SHAPE), dtype=np.float32)
    for ti, stamp in enumerate(radar_times):
        for vi, var in enumerate(radar_vars):
            p = index.radar_file(var, stamp)
            if p is not None:
                radar[ti, vi] = clean_radar_field(np.load(p), var)

    label = np.zeros((len(label_times), 1, *RADAR_SHAPE), dtype=np.float32)
    for ti, stamp in enumerate(label_times):
        p = index.label_file(stamp)
        if p is not None:
            label[ti, 0] = clean_label_rain(np.load(p))

    nwp = np.zeros((len(nwp_times), len(nwp_vars), *index_nwp_shape()), dtype=np.float32)
    for ti, stamp in enumerate(nwp_times):
        for vi, var in enumerate(nwp_vars):
            p = index.nwp_file(var, stamp)
            if p is not None:
                nwp[ti, vi] = clean_nwp_field(np.load(p))

    return {
        "radar": radar,
        "label": label,
        "nwp": nwp,
        "radar_times": np.asarray(radar_times, dtype=np.int64),
        "label_times": np.asarray(label_times, dtype=np.int64),
        "nwp_times": np.asarray(nwp_times, dtype=np.int64),
        "radar_vars": np.asarray(radar_vars),
        "nwp_vars": np.asarray(nwp_vars),
        "meta": {"case_id": path.name},
    }


def index_nwp_shape() -> tuple[int, int]:
    from .coordinates import NWP_SHAPE

    return NWP_SHAPE


def _seed_worker(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    base = getattr(info, "seed", None) if info is not None else None
    if base is not None:
        np.random.seed(base % (2**32))
        # The worker inherits the same base seed, but torch's own generator
        # inside the worker was only seeded for numpy. Anything that draws
        # from torch inside a worker -- and the sampler state that the
        # persistent workers carry across epochs -- was therefore free to
        # differ between two runs of the same configuration. Seeding it
        # here is what makes the epoch-0 loss of two identical runs agree.
        torch.manual_seed(base % (2**63))


def build_dataloader(
    dataset: RadarNWPDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=_seed_worker if seed is not None else None,
        persistent_workers=num_workers > 0,
    )