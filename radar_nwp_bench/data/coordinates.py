"""Grid registration between the RADAR/LABEL grid and the NWP grid.

The two modalities live on different lattices:
    RADAR / LABEL : 301 x 301 @ 0.01 deg   (3.00 deg span)
    NWP           : 167 x 167 @ 0.03 deg   (5.01 deg span)

The NWP window contains the RADAR window, so the registration is a scale-up of
the NWP field followed by a centre crop. This module is the single place that
knows about that relationship.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RADAR_SHAPE = (301, 301)
NWP_SHAPE = (167, 167)
NWP_NATIVE_SHAPE = NWP_SHAPE
NWP_TO_RADAR_SCALE = 3

_GRID_KEYS = (
    "StartLon", "StartLat", "EndLon", "EndLat",
    "resolution_X", "resolution_Y", "nx", "ny",
)


@dataclass(frozen=True)
class GridSpec:
    """Georeferencing of one raster grid."""

    start_lon: float
    start_lat: float
    end_lon: float
    end_lat: float
    res_x: float
    res_y: float
    nx: int
    ny: int

    @property
    def center_lon(self) -> float:
        return 0.5 * (self.start_lon + self.end_lon)

    @property
    def center_lat(self) -> float:
        return 0.5 * (self.start_lat + self.end_lat)

    def lonlat_at(self, row: int, col: int) -> tuple[float, float]:
        """Longitude/latitude of a pixel centre (row increases southward)."""
        lon = self.start_lon + (col + 0.5) * self.res_x
        lat = self.start_lat - (row + 0.5) * self.res_y
        return lon, lat

    def contains(self, other: "GridSpec") -> bool:
        return (
            self.start_lon <= other.start_lon
            and self.end_lon >= other.end_lon
            and self.start_lat >= other.start_lat
            and self.end_lat <= other.end_lat
        )


def parse_grid_file(path: str | Path) -> GridSpec:
    """Parse a FORMAT header file into a GridSpec."""
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if "=" in line:
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip()
    missing = [k for k in _GRID_KEYS if k not in values]
    if missing:
        raise ValueError(f"{path}: missing grid keys {missing}")
    return GridSpec(
        start_lon=float(values["StartLon"]),
        start_lat=float(values["StartLat"]),
        end_lon=float(values["EndLon"]),
        end_lat=float(values["EndLat"]),
        res_x=float(values["resolution_X"]),
        res_y=float(values["resolution_Y"]),
        nx=int(values["nx"]),
        ny=int(values["ny"]),
    )


def symmetric_patch_origins(grid_dim: int, patch_size: int, nominal_stride: int) -> tuple[int, ...]:
    """生成全视场对称均匀的切片起点，彻底消除末端局部密集重叠。

    Args:
        grid_dim: 全网格边长（例如 301）
        patch_size: 切片尺寸（例如 64）
        nominal_stride: 名义标称步长（例如 64）
    Returns:
        对称均匀分布的起点整数元组。
    """
    if patch_size >= grid_dim:
        return (0,)
    total_span = grid_dim - patch_size
    n_steps = max(1, math.ceil(total_span / nominal_stride))
    origins = tuple(round(i * total_span / n_steps) for i in range(n_steps + 1))
    return origins


def bilinear_upsample_center_crop(
    field: np.ndarray,
    radar_shape: tuple[int, int] = RADAR_SHAPE,
    scale: int = NWP_TO_RADAR_SCALE,
) -> np.ndarray:
    """Register an NWP field onto the RADAR grid."""
    arr = np.asarray(field, dtype=np.float32)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[None]
    if arr.shape[-2:] != NWP_SHAPE:
        raise ValueError(
            f"expected NWP field of shape {NWP_SHAPE}, got {arr.shape[-2:]}"
        )
    up = _bilinear_upsample(arr, scale)
    out = _center_crop_to(out=up, target=radar_shape)
    return out[0] if squeeze else out


def _reflect_index(index: np.ndarray, n: int) -> np.ndarray:
    if n == 1:
        return np.zeros_like(index)
    period = 2 * (n - 1)
    idx = np.mod(index, period)
    return np.where(idx >= n, period - idx, idx)


def _upsample_coordinates(
    in_size: int, scale: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dst = np.arange(in_size * scale, dtype=np.float64)
    src = (dst + 0.5) / scale - 0.5
    i0 = np.floor(src).astype(np.int64)
    weight = (src - i0).astype(np.float32)
    return _reflect_index(i0, in_size), _reflect_index(i0 + 1, in_size), weight


def _bilinear_upsample(arr: np.ndarray, scale: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    r0, r1, rw = _upsample_coordinates(h, scale)
    c0, c1, cw = _upsample_coordinates(w, scale)

    top = arr[..., r0[:, None], c0[None, :]] * (1.0 - cw)
    top = top + arr[..., r0[:, None], c1[None, :]] * cw
    bot = arr[..., r1[:, None], c0[None, :]] * (1.0 - cw)
    bot = bot + arr[..., r1[:, None], c1[None, :]] * cw
    out = top * (1.0 - rw)[..., :, None] + bot * rw[..., :, None]
    return out.astype(np.float32)


def _center_crop_to(out: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    h, w = out.shape[-2:]
    th, tw = target
    if h < th or w < tw:
        raise ValueError(f"cannot crop {target} out of {out.shape[-2:]}")
    r0 = (h - th) // 2
    c0 = (w - tw) // 2
    return out[..., r0:r0 + th, c0:c0 + tw]


def resample_nwp_to_radar_torch(
    field: "torch.Tensor",
    radar_shape: tuple[int, int] = RADAR_SHAPE,
) -> "torch.Tensor":
    """Torch counterpart of bilinear_upsample_center_crop."""
    import torch

    if field.dim() == 2:
        x = field[None, None]
        squeeze_to = 2
    elif field.dim() == 3:
        x = field[None]
        squeeze_to = 3
    elif field.dim() == 4:
        x = field
        squeeze_to = 0
    else:
        raise ValueError(f"expected a 2D/3D/4D tensor, got {field.dim()}D")

    if x.shape[-2:] != NWP_SHAPE:
        raise ValueError(
            f"expected NWP field of shape {NWP_SHAPE}, got {tuple(x.shape[-2:])}"
        )

    h, w = NWP_SHAPE
    scale = NWP_TO_RADAR_SCALE
    dev = x.device

    def coords(n: int) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        dst = torch.arange(n * scale, device=dev, dtype=torch.float64)
        src = (dst + 0.5) / scale - 0.5
        i0 = torch.floor(src)
        weight = (src - i0).to(torch.float32)
        period = 2 * (n - 1)
        i0r = torch.remainder(i0, period)
        i0r = torch.where(i0r >= n, period - i0r, i0r)
        i1r = torch.remainder(i0 + 1, period)
        i1r = torch.where(i1r >= n, period - i1r, i1r)
        return i0r.long(), i1r.long(), weight

    r0, r1, rw = coords(h)
    c0, c1, cw = coords(w)

    xf = x.float()
    top = xf[:, :, r0, :][:, :, :, c0] * (1 - cw)[None, None, None, :]
    top = top + xf[:, :, r0, :][:, :, :, c1] * cw[None, None, None, :]
    bot = xf[:, :, r1, :][:, :, :, c0] * (1 - cw)[None, None, None, :]
    bot = bot + xf[:, :, r1, :][:, :, :, c1] * cw[None, None, None, :]
    up = top * (1 - rw)[None, None, :, None] + bot * rw[None, None, :, None]

    _, _, oh, ow = up.shape
    th, tw = radar_shape
    r_start, c_start = (oh - th) // 2, (ow - tw) // 2
    out = up[:, :, r_start:r_start + th, c_start:c_start + tw]

    if squeeze_to == 2:
        return out[0, 0]
    if squeeze_to == 3:
        return out[0]
    return out


_TIMESTAMP_RE = re.compile(r"(\d{4})-(\d{4})")


def parse_timestamp(name: str) -> tuple[int, int]:
    m = _TIMESTAMP_RE.search(name)
    if not m:
        raise ValueError(f"no timestamp in filename: {name}")
    return int(m.group(1)), int(m.group(2))