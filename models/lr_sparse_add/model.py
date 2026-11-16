"""E44 / MASTER 4.46 -- A2: an explicit additive decomposition, low-rank plus sparse.

E43's arm composed the output multiplicatively::

    out = low + mask * amp        # the sparse part is taken FROM the projected amp

That is the reason its amplitude tail collapsed: the sparse term was
derived from ``amp``, and ``amp`` had already been through the
projection, so the tail was shaved before the sparse branch ever saw
it.  Moving the gate did not help because the gate multiplies a signal
that is already damaged.

This module changes the FORM of the decomposition, not a switch.  Two
tensors are added::

    out = low + sparse

* ``low`` is the low-rank component: the decoder's output map projected
  onto the eight lowest-frequency DCT modes, unchanged from E43.
* ``sparse`` is an INDEPENDENTLY PREDICTED head.  It never passes
  through the projection, so it carries the heavy tail on its own and
  the low-rank term has no way to flatten it.

This is the additive form of Robust PCA (``X = L + S``), and it is a
different mathematical object from E43's single tensor under two
operations: here the two components are separate predictions that are
summed, there one tensor is projected and then gated.

What is NOT changed: the backbone, the loss, the output contract, the
number of loss terms.  There is no area penalty and no sparsity
regulariser, because E42's lesson was that a second term whose optimum
points the same way as the regression drives the field to zero, and a
sparsity penalty on the sparse component would point exactly that way.
The sparsity of the sparse component comes from its own structure: a
one-by-one convolution followed by ``floor_at_zero``.

The S0 gate checks that with the switch off this module is bit-equal to
``simvp_v2``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import ConvNeXtBlock, floor_at_zero

from models.lr_sparse.model import _lowest_dct_modes


@register_model("lr_sparse_add")
class LowRankPlusSparseModel(BasePrecipitationModel):
    """SimVP-v2 split into an additive low-rank and sparse decomposition.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        hid_c: channel width.
        num_blocks: translator depth.
        r_low: dimension of the spatial subspace the background is
            confined to.
        additive: when False the sparse head is switched off and the
            module reduces to the low-rank projection, which still is not
            ``simvp_v2``; with ``low_rank`` off as well it reproduces
            ``simvp_v2`` exactly, which is what the S0 gate checks.
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
        low_rank: bool = True,
        additive: bool = True,
    ) -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.r_low = int(r_low)
        self.use_low_rank = bool(low_rank)
        self.use_additive = bool(additive)

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

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hid_c * 2, hid_c, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, 1, kernel_size=1),
        )

        # The sparse component: an independent prediction that never sees
        # the projection.
        self.sparse_head = nn.Conv2d(hid_c, 1, kernel_size=1)

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

        if not self.use_low_rank and not self.use_additive:
            # The S0 path: exactly simvp_v2's decoder on exactly
            # simvp_v2's features.
            log_mm = amp.view(B, self.out_len, 1, H, W)
            return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))

        # --- the low-rank component ---
        if self.use_low_rank:
            basis = _lowest_dct_modes(H, W, self.r_low).to(
                device=amp.device, dtype=amp.dtype
            )
            flat = amp.reshape(amp.shape[0], -1)
            coeff = flat @ basis.reshape(self.r_low, -1).T
            low = (coeff @ basis.reshape(self.r_low, -1)).reshape(
                amp.shape[0], 1, H, W
            )
        else:
            low = torch.zeros_like(amp)

        # --- the sparse component: predicted independently ---
        if self.use_additive:
            sparse = self.sparse_head(body)
        else:
            sparse = torch.zeros_like(amp)

        out = low + sparse
        log_mm = out.view(B, self.out_len, 1, H, W)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))
