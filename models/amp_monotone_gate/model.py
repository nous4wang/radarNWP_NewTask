"""E45 / MASTER 4.47 -- A1: the amplitude-monotone gate.

E44 measured the gate's own failure on its trained checkpoint
(134500 frames, ``tools/_e44_diag_probe.py``):::

    corr(mask, amp)  = -0.858320
    mask_at_peak     =  0.195087

The gate and the amplitude were strongly ANTI-correlated, and the gate
was nearly closed exactly AT the peak, so the composition
``low + mask * (amp - low)`` kept only 15 % of the peak
(``maxratio_as_trained = 0.152377``).  The cause was the gate's FORM:
a free convolution gate, multiplied into the residual under a
truth-weighted regression loss, has no reason to point at the rain.

This module changes only that form.  The gate becomes a MONOTONE
FUNCTION OF THE RESIDUAL::

    amp   = dec[-1](body)               # (B*T, 1, H, W), log-mm
    low   = P_low(amp)                  # r_low lowest-frequency DCT modes
    resid = amp - low
    thr   = <one learned scalar per (window, lead time)>
    mask  = sigmoid((resid - thr) / T)  # monotone in resid, and in nothing else
    out   = low + mask * resid          # == E44-A1's ``low + mask * (amp - low)``

Because ``mask`` is an increasing function of ``resid`` alone,
``corr(mask, resid) >= 0`` holds BY CONSTRUCTION and not by training:
the peak pixel has the largest residual and therefore the largest
mask, so the composition ``low + mask * resid`` applies the largest
factor to the largest residual.  The threshold is one number per
(window, lead time), so the statement is made WITHIN a frame -- which
is the unit the read-out probe measures it in.

What is NOT changed: the backbone, the loss, the number of loss terms,
the output contract, the projection basis and rank, and the composition
``low + mask * resid`` itself.  No new dependency is introduced -- only
``torch``.  The ONLY thing that changes is the OUTPUT COMPOSITION: where
the gate's value comes from.

The S0 gate checks that with both switches off this module is bit-equal
to ``simvp_v2``.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import ConvNeXtBlock, floor_at_zero

from models.lr_sparse.model import _lowest_dct_modes


@register_model("amp_monotone_gate")
class AmpMonotoneGateModel(BasePrecipitationModel):
    """SimVP-v2 whose gate is a monotone function of the residual.

    The gate's threshold is predicted from the decoder body's global
    pooling, which gives ONE number per (window, lead time): the
    body's leading axis is the (window, frame) axis, so the pooling is
    over the spatial axes only.  It is calibrated PER FRAME rather than
    per window because the amplitude scale changes with the lead time,
    and a per-frame calibration fits that mechanism better than a
    single number per window would.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        hid_c: channel width.
        num_blocks: translator depth.
        r_low: dimension of the spatial subspace the background is
            confined to.
        gate_temp: temperature of the gate.  A small value makes the
            gate nearly a step function of the residual, a large one
            makes it smooth.
        low_rank: when False the background is left unconstrained.
        monotone_gate: when False the rain branch carries no gate.
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
        monotone_gate: bool = True,
    ) -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.r_low = int(r_low)
        self.gate_temp = float(gate_temp)
        self.use_low_rank = bool(low_rank)
        self.use_monotone_gate = bool(monotone_gate)

        #: The diagnostic tensors of the last forward pass.  They are
        #: detached: the training loop calls this model over and over,
        #: so a live graph would be kept alive for nothing.
        self.last_debug: Dict[str, torch.Tensor] = {}

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
            nn.ConvTranspose2d(hid_c * 2, hid_c, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, 1, kernel_size=1),
        )

        # The threshold of the monotone gate.  Both the weight and the
        # bias start at zero, so the threshold is exactly 0 when the
        # training begins and the gate starts out as a plain monotone
        # function of the residual.
        self.gate_threshold = nn.Conv2d(hid_c, 1, kernel_size=1)
        nn.init.zeros_(self.gate_threshold.weight)
        nn.init.zeros_(self.gate_threshold.bias)

    # ------------------------------------------------------------------
    # the gate, isolated so that its monotonicity can be unit-tested
    # ------------------------------------------------------------------

    def _gate_mask(self, resid: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
        """The amplitude-monotone gate.

        The mask is a function of the residual and of the threshold
        ONLY: for a fixed threshold it is strictly increasing in the
        residual, which is what makes ``corr(mask, resid) >= 0`` a
        property of the construction rather than of the training.

        Args:
            resid: the residual amplitude, ``(B*T, 1, H, W)``.
            thr: the per-frame threshold, ``(B*T,)``.

        Returns:
            The mask, ``(B*T, 1, H, W)``.
        """
        return torch.sigmoid((resid - thr.view(-1, 1, 1, 1)) / self.gate_temp)

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

        if not self.use_low_rank and not self.use_monotone_gate:
            # The S0 path: exactly simvp_v2's decoder on exactly
            # simvp_v2's features.  The new head is not read and the
            # projection is not computed, so nothing but the shared
            # trunk can reach the output.
            self.last_debug = {
                "amp": amp.detach(),
                "low": torch.zeros_like(amp),
                "resid": amp.detach(),
                "mask": torch.ones_like(amp),
                "thr": torch.zeros(amp.shape[0], device=amp.device,
                                   dtype=amp.dtype),
            }
            log_mm = amp.view(B, self.out_len, 1, H, W)
            return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))

        # --- the low-rank background ---
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

        # The residual: what the projection removed, i.e. the part of
        # the amplitude that the background does not account for.
        resid = amp - low

        # --- the amplitude-monotone gate ---
        # The threshold is one number per (window, lead time); see the
        # class docstring for why it is calibrated per frame.
        if self.use_monotone_gate:
            thr = self.gate_threshold(body).mean(dim=(2, 3)).reshape(-1)
            mask = self._gate_mask(resid, thr)
        else:
            thr = torch.zeros(amp.shape[0], device=amp.device, dtype=amp.dtype)
            mask = torch.ones_like(amp)

        out = low + mask * resid

        self.last_debug = {
            "amp": amp.detach(),
            "low": low.detach(),
            "resid": resid.detach(),
            "mask": mask.detach(),
            "thr": thr.detach(),
        }

        log_mm = out.view(B, self.out_len, 1, H, W)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))
