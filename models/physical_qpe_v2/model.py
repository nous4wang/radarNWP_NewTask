"""v1 QPE inverter with an explicit NWP sub-cloud evaporation gate.

Gate 1 positioning (MASTER section 4.20.1): this is NOT a new physical
inductive bias.  ``R = R_raw(radar) * T(NWP column)`` is the same multiplicative
humidity correction that section 4.17 already measured as uniformly negative,
and section 4.19's v1 already learns an unconstrained pixelwise function of all
27 NWP channels (including RH1000/RH700/RH500/PWAT/LCL/LFC) concatenated with
the radar.  A learned gate bounded in [0, 1] is a SUBSET of that hypothesis
space, not an addition to it.

So the only legitimate experiment is a one-variable paired comparison:

  * ``physical_qpe_v2(use_gate=True)``   -- the gate arm
  * ``physical_qpe_v2(use_gate=False)`` -- T == 1, architecture otherwise
    byte-identical to v1; its state_dict keys and values are identical to
    ``physical_qpe`` for the same seed, which is the zero-experiment that
    proves the ablation arm is not a different model.

The architecture below is a deliberate COPY of v1, not an import:
``registry.py`` loads every ``models/<name>/model.py`` under the module name
``_rnb_model_<name>``.  Importing v1 from here would execute v1's
``@register_model("physical_qpe")`` a second time under a different module
name and collide with the registry (load-bearing convention 14).  Keeping the
parameter names (``stem`` / ``blocks`` / ``head``) identical is what makes the
``use_gate=False`` state_dict equal to v1's.

D3-A: gate arm must beat the v1 ep00 reference (CSI@2.0 = 0.1226) with a
paired bootstrap CI excluding zero.  D3-B: the difference must be attributable
to the gate, not to an architecture rewrite -- which this file guarantees by
construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import floor_at_zero


class _ResidualBlock(nn.Module):
    """A small conv block that keeps the grid resolution unchanged (v1 copy)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.body(x))


class PhysicalEvaporationGate(nn.Module):
    """Learned transmissivity T in [0, 1] from the t0 NWP column.

    The NWP channels are the norm-v2 normalized fields, which is correct for
    a LEARNED gate (no physical formula is being inverted here).  Section
    4.17's analytical formula, by contrast, needed an explicit inverse
    normalization -- that is a different construction, not this one.
    """

    def __init__(self, in_nwp_c: int = 27, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_nwp_c, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x_nwp_t0: torch.Tensor) -> torch.Tensor:
        return self.net(x_nwp_t0)  # (B, 1, H, W)


@register_model("physical_qpe_v2")
class PhysicalQPEModelV2(BasePrecipitationModel):
    """v1 single-frame QPE inverter, optionally gated by NWP transmissivity.

    Args:
        use_gate: ``False`` gives T == 1 (the D3-B one-variable ablation).
            Its parameters are exactly v1's, so the same seed reproduces
            section 4.19 bit-for-bit.
        gate_hidden: width of the gate MLP.
        All other args mirror ``models/physical_qpe/model.py`` exactly.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
        nwp_slice: str = "analysis",
        use_gate: bool = True,
        gate_hidden: int = 32,
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.hid_c = hid_c
        self.nwp_slice = nwp_slice
        self.use_gate = bool(use_gate)
        if nwp_slice not in ("analysis", "forecast"):
            raise ValueError("nwp_slice must be 'analysis' or 'forecast'")

        total_in = in_radar_c + in_nwp_c
        groups = 8 if hid_c % 8 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(total_in, hid_c, 3, padding=1),
            nn.GroupNorm(groups, hid_c),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(hid_c) for _ in range(num_blocks)]
        )
        self.head = nn.Conv2d(hid_c, 1, 1)
        if self.use_gate:
            self.evaporation_gate = PhysicalEvaporationGate(
                in_nwp_c, gate_hidden
            )

    def forward(
        self, x_radar: torch.Tensor, x_nwp: torch.Tensor | None
    ) -> torch.Tensor:
        """Return the (optionally gated) t0 rainfall copied across T_out."""
        radar_t0 = x_radar[:, -1]  # (B, C_radar, H, W)
        if x_nwp is None:
            nwp_t0 = radar_t0.new_zeros(
                (radar_t0.shape[0], self.in_nwp_c, *radar_t0.shape[-2:])
            )
        else:
            slice_index = 0 if self.nwp_slice == "analysis" else -1
            nwp_t0 = x_nwp[:, slice_index]

        x = torch.cat([radar_t0, nwp_t0], dim=1)
        h = self.blocks(self.stem(x))
        log_mm = self.head(h)
        # Head predicts log1p(mm); expm1 returns millimetres. The
        # straight-through floor keeps dry pixels in the gradient path (§12).
        r0 = floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))

        if self.use_gate:
            r0 = r0 * self.evaporation_gate(nwp_t0)

        return r0.unsqueeze(1).expand(-1, self.out_len, -1, -1, -1)
