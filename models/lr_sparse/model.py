"""E43 / MASTER 4.45 -- low-rank + sparse decomposition INSIDE the model.

The label is a zero-inflated, heavy-tailed, spatially coherent sparse
measure: 91 % zeros, a smooth background, a connected block of strong
rain, a support displaced from the radar's.  All 18 reproduced models
express that label the same way -- one ``expm1`` head regressed per pixel
in ``log1p(mm)`` space -- and a per-pixel regression has no cross-pixel
structure with which to express "background plus a sparse rain region".

This module writes that structure into the model, and it does so with
TWO architectural mechanisms and NO new loss term:

* the **low-rank background**.  The decoder's output map is projected
  onto the ``r_low`` lowest-frequency DCT modes of the patch, so the
  background is confined to an ``r_low``-dimensional subspace of the
  spatial domain.  This is a genuine rank constraint, not a decoration:
  the projection is onto a FIXED basis, so the network cannot express an
  arbitrary smooth field, only a mixture of a few known smooth modes.
  It is also exactly the label's measured shape -- the background is
  3.7x smoother than the radar's (``tv_z`` 0.0312 against 0.1165).

* the **sparse rain region**.  The amplitude of the rain branch is
  modulated by an explicit per-pixel support mask, so "where it rains"
  and "how much it rains" are two separate quantities the network can
  express separately.  The mask uses a LEARNED THRESHOLD (the L0-style
  gate of Luo et al., ICLR 2018): ``sigmoid((g - tau) / T)`` with ``tau``
  predicted per sample and a fixed temperature ``T``.  The model
  therefore chooses both the shape of its support and how much area to
  claim, with no hand-tuned area penalty and no new loss term.

Why no new loss term
---------------------
E42's hard lesson was that a loss term whose optimum points in the same
direction as the regression drives the field to zero, and that a second
term can turn the arm into a two-variable experiment.  Here the whole
mechanism is in the MODEL BODY: the承重 ``balanced`` loss is used
unchanged, so the arms differ in exactly one thing.

S0 gate
-------
With ``use_low_rank=False`` and ``use_sparse_gate=False`` the
computation graph is identical to ``models/simvp_v2/model.py`` and the
weights load under the SAME names (``enc``, ``in_time_proj``,
``translator``, ``out_time_proj``, ``dec.0`` ... ``dec.6``), which is what
``tools/_e43_s0_equivalence.py`` checks for bit equality.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import ConvNeXtBlock, floor_at_zero

#: Cache of the fixed DCT bases, keyed by (height, width, rank).
_DCT_CACHE: Dict[Tuple[int, int, int], torch.Tensor] = {}


def _lowest_dct_modes(h: int, w: int, rank: int) -> torch.Tensor:
    """The ``rank`` lowest-frequency orthonormal 2-D cosine modes.

    The modes are enumerated by increasing ``ny**2 + nx**2``, so the
    first ones are the constant mode and the first vertical and
    horizontal gradients, and the last one is still a low frequency.

    Args:
        h: patch height.
        w: patch width.
        rank: how many modes to keep.

    Returns:
        A ``(rank, h, w)`` float32 tensor whose rows are orthonormal.
    """
    key = (h, w, rank)
    cached = _DCT_CACHE.get(key)
    if cached is not None:
        return cached
    pairs = sorted(
        ((ny, nx) for ny in range(rank) for nx in range(rank)),
        key=lambda p: (p[0] ** 2 + p[1] ** 2, p[0], p[1]),
    )[:rank]
    y = torch.arange(h, dtype=torch.float32).unsqueeze(1)
    x = torch.arange(w, dtype=torch.float32).unsqueeze(0)
    modes = []
    for ny, nx in pairs:
        cy = torch.cos(math.pi * (y + 0.5) * ny / h)
        cx = torch.cos(math.pi * (x + 0.5) * nx / w)
        mode = (cy * cx).squeeze(1).squeeze(0)
        mode = mode / mode.norm().clamp_min(1e-8)
        modes.append(mode)
    basis = torch.stack(modes, dim=0)
    _DCT_CACHE[key] = basis
    return basis


@register_model("lr_sparse")
class LowRankSparseModel(BasePrecipitationModel):
    """SimVP-v2 whose head is split into a low-rank background and a
    sparse rain region.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        hid_c: channel width.
        num_blocks: translator depth.
        r_low: dimension of the spatial subspace the background is
            confined to.
        gate_temp: temperature of the support gate.  A small value makes
            the support nearly binary; a large one makes it smooth.
        low_rank: when False the background is left unconstrained.
        sparse_gate: when False the rain branch carries no support mask.
            Both switches off reproduce ``simvp_v2`` exactly, which is
            what the S0 gate checks.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
        r_low: int = 8,
        gate_temp: float = 1.0,
        low_rank: bool = True,
        sparse_gate: bool = True,
    ) -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.r_low = int(r_low)
        self.gate_temp = float(gate_temp)
        self.use_low_rank = bool(low_rank)
        self.use_sparse_gate = bool(sparse_gate)

        total_in = in_radar_c + in_nwp_c * 2

        # --- identical to simvp_v2 (module names matter for load_state_dict) ---
        self.enc = nn.Sequential(
            nn.Conv2d(total_in, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hid_c * 2),
            nn.GELU(),
        )

        self.in_time_proj = nn.Conv2d(in_len * hid_c * 2, hid_c * 4, kernel_size=1)
        self.translator = nn.Sequential(
            *[ConvNeXtBlock(hid_c * 4) for _ in range(num_blocks)]
        )
        self.out_time_proj = nn.Conv2d(hid_c * 4, out_len * hid_c * 2, kernel_size=1)

        # The decoder is simvp_v2's decoder verbatim, so the state dict
        # loads under the same names and the S0 check is a bit equality.
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hid_c * 2, hid_c, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, 1, kernel_size=1),
        )

        # The support gate and its learned threshold.
        self.sparse_gate = nn.Conv2d(hid_c, 1, kernel_size=1)
        self.gate_threshold = nn.Conv2d(hid_c, 1, kernel_size=1)

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        B, T_in, _, H, W = x_radar.shape
        nwp_flat = x_nwp.view(B, 1, -1, H, W).expand(-1, T_in, -1, -1, -1)
        x_concat = torch.cat([x_radar, nwp_flat], dim=2)

        feats = self.enc(x_concat.view(B * T_in, -1, H, W))
        _, C_f, H_f, W_f = feats.shape

        feats_time = feats.view(B, T_in * C_f, H_f, W_f)
        h = self.in_time_proj(feats_time)
        h = self.translator(h)
        future_feats = self.out_time_proj(h).view(B * self.out_len, C_f, H_f, W_f)

        body = self.dec[:-1](future_feats)
        amp = self.dec[-1](body)                      # (B*T, 1, H, W)

        if not self.use_low_rank and not self.use_sparse_gate:
            # The S0 path: exactly simvp_v2's decoder on exactly
            # simvp_v2's features.
            log_mm = amp.view(B, self.out_len, 1, H, W)
            return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))

        # --- the low-rank background ---
        # Project the raw output map onto the r_low lowest DCT modes.
        # The projection is onto a FIXED basis, so the background cannot
        # take an arbitrary shape: it is a mixture of a few known smooth
        # modes, which is the label's measured background structure.
        if self.use_low_rank:
            basis = _lowest_dct_modes(H, W, self.r_low).to(
                device=amp.device, dtype=amp.dtype
            )
            flat = amp.reshape(amp.shape[0], -1)                 # (B*T, H*W)
            coeff = flat @ basis.reshape(self.r_low, -1).T      # (B*T, r_low)
            low = (coeff @ basis.reshape(self.r_low, -1)).reshape(
                amp.shape[0], 1, H, W
            )
        else:
            low = torch.zeros_like(amp)

        # --- the sparse rain region ---
        # The mask says WHERE, the amplitude says HOW MUCH.  The
        # threshold is predicted per sample, so the model chooses how
        # much area to claim; there is no hand-tuned area penalty.
        if self.use_sparse_gate:
            gate = self.sparse_gate(body)                       # (B*T,1,H,W)
            thr = self.gate_threshold(body).mean(dim=(2, 3))     # (B*T,1)
            thr = thr.view(-1, 1, 1, 1)
            mask = torch.sigmoid((gate - thr) / self.gate_temp)
            rain = mask * amp
        else:
            rain = amp

        out = low + rain
        log_mm = out.view(B, self.out_len, 1, H, W)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))
