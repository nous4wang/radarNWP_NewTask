"""E39 / MASTER 4.41 -- IBCFM: Intensity-Banded Conditional Flow Matching.

Pre-registered design, NOT a validated method
---------------------------------------------
This adapter is the implementation half of
``docs/experiments/E39-generative-route-reexamination-and-ibcfm.md``.  It has
**never been trained or evaluated on the dataset**; no CSI / BIAS number in
this repository belongs to it.  Criteria C0-C5 in E39 are frozen before any
compute was spent, and C2 (MASTER 11.9) must return ``VERDICT: PASS`` before
this model is allowed into the performance layer.

What is actually new (Gate 1, by mechanism not by name)
-------------------------------------------------------
The two "diffusion" baselines already in the benchmark are deterministic
single-sample models:

* ``models/diffcast/model.py:344-357`` starts its DDIM loop from
  ``torch.zeros`` ("no residual" hypothesis);
* ``models/cascast_nwp/model.py`` uses a *fixed* ``noise_seed`` for one
  realisation.

Neither samples from ``p(Y | X)``.  IBCFM keeps the same deterministic anchor
idea but transports **random Gaussian noise** to the residual field with a
conditional flow-matching (straight-line ODE) objective, draws ``K`` genuine
members at inference, and turns them into a threshold-exceedance probability
field.  The A1 (zero start) vs A2 (random start) ablation differs by exactly
one boolean switch, so it is a controlled single-variable comparison.

Design constraints taken from the external review (E39 section 2)
-----------------------------------------------------------------
* Explicit deterministic anchor + probabilistic residual, trained jointly
  (MW-Nowcast ablation, arXiv:2609.34836).
* **Pixel space**, no VAE / latent compression: the same ablation reports that
  latent variants lose skill at high thresholds because compression smooths
  compact precipitation cores.  This is consistent with the benchmark's own
  ``cascast_nwp`` (latent DiT) 0.0282 vs ``diffcast`` (pixel residual) 0.0717.
* Conditional Flow Matching rather than DDPM: far fewer sampling steps, so a
  K-member ensemble is affordable (FlowCast, ICLR 2026, arXiv:2511.09731).
* Intensity-banded supervision (MTLDM, IEEE TGRS 2025, 10.1109/TGRS.2025.3564916)
  and a threshold-exceedance ensemble decision (S3NN, IEEE TGRS 2026,
  10.1109/TGRS.2026.3684546; Guilloteau et al., IEEE TGRS 2025).

Interface contract (unchanged, MASTER 1.2)
------------------------------------------
``forward(x_radar, x_nwp) -> (B, 10, 1, H, W)`` millimetres per 6 minutes.
The evaluator is not modified: the decision field is still a non-negative
rainfall field in mm.

Deliberate limitations
----------------------
* ``p_star`` and ``exceed_mm`` below are **placeholders**.  They MUST be
  calibrated on the TRAIN split (E39 C0 / section 4.1).  No validation fields
  exist (E26), and tuning them on test repeats the E18 information leak.
* Training returns the deterministic anchor ``mu``; the flow loss is supplied
  separately by ``compute_flow_loss(target)`` through the ``--loss ibcfm``
  criterion.  This keeps the training loop and evaluator untouched.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from radar_nwp_bench.models import BasePrecipitationModel, register_model
from radar_nwp_bench.models.common import ConvBlock, RainHead, floor_at_zero

#: Rainfall intensity bands, matching the benchmark's verification tiers
#: (0.1 / 0.5 / 2.0 / 4.0 mm per 6 min).  Weights are normalised to mean 1
#: inside ``_band_weights`` so the overall loss scale never changes.
BAND_EDGES: tuple[float, ...] = (0.5, 2.0, 4.0)
BAND_WEIGHTS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)


class ResidualVelocityNet(nn.Module):
    """Fully-convolutional velocity field, **no spatial downsampling**.

    ``z_t`` (1 channel), the conditioning map, the anchor in log space and a
    2-channel sinusoidal flow-time embedding are concatenated on the full
    grid.  Keeping the full resolution is a deliberate choice: E39 section 2
    records that latent-space compression attenuates local extremes.
    """

    def __init__(self, cond_channels: int, base: int = 16, n_blocks: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(cond_channels + 4, base, 1)
        self.body = nn.Sequential(
            *[ConvBlock(base, base, 3, residual=True) for _ in range(n_blocks)]
        )
        self.out = nn.Conv2d(base, 1, 1)
        # Start at the identity transport (zero velocity), so an untrained
        # model reproduces the deterministic anchor instead of blowing up.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        mu_log: torch.Tensor,
    ) -> torch.Tensor:
        """Return the velocity with the same shape as ``z_t``.

        All arguments are flattened over the batch and lead-time axes.
        ``t`` is ``(BT, 1)`` in ``[0, 1]``.
        """
        bt, _, h, w = z_t.shape
        tt = t.reshape(bt, 1).float()
        phase = 2.0 * math.pi * tt
        temb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        temb = temb[:, :, None, None].expand(-1, -1, h, w)
        x = torch.cat([z_t, cond, mu_log, temb], dim=1)
        return self.out(self.body(self.in_proj(x)))


@register_model("ibcfm")
class IBCFMModel(BasePrecipitationModel):
    """Deterministic anchor + pixel-space CFM residual + ensemble decision."""

    nwp_time_len: int = 2

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 32,
        unet_base: int = 16,
        n_velocity_blocks: int = 4,
        n_members: int = 8,
        sample_steps: int = 4,
        random_start: bool = True,
        intensity_bands: bool = True,
        decision_mode: str = "decision",
        p_star: float = 0.5,
        exceed_mm: float = 2.5,
        tier: float = 2.0,
        noise_scale: float = 1.0,
    ) -> None:
        super().__init__(
            in_radar_c=in_radar_c,
            in_nwp_c=in_nwp_c,
            in_len=in_len,
            out_len=out_len,
        )
        if decision_mode not in ("member", "mean", "decision"):
            raise ValueError(f"unknown decision_mode: {decision_mode!r}")
        if sample_steps < 1 or n_members < 1:
            raise ValueError("sample_steps and n_members must be >= 1")

        self.hid_c = int(hid_c)
        self.n_members = int(n_members)
        self.sample_steps = int(sample_steps)
        self.random_start = bool(random_start)
        self.intensity_bands = bool(intensity_bands)
        self.decision_mode = decision_mode
        # PLACEHOLDERS -- calibrate on TRAIN before any dataset run (E39 4.1).
        self.p_star = float(p_star)
        self.exceed_mm = float(exceed_mm)
        self.tier = float(tier)
        self.noise_scale = float(noise_scale)

        self.radar_enc = nn.Sequential(
            nn.Conv2d(in_len * in_radar_c, hid_c, 1),
            ConvBlock(hid_c, hid_c, 3),
        )
        self.nwp_enc = nn.Sequential(
            nn.Conv2d(self.nwp_time_len * in_nwp_c, hid_c, 1),
            ConvBlock(hid_c, hid_c, 3),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * hid_c, hid_c, 3, padding=1),
            ConvBlock(hid_c, hid_c, 3, residual=True),
        )
        self.time_embed = nn.Parameter(torch.zeros(1, out_len, hid_c, 1, 1))
        self.anchor_head = RainHead(hid_c, hidden=hid_c)
        self.velocity = ResidualVelocityNet(
            cond_channels=hid_c, base=unet_base, n_blocks=n_velocity_blocks
        )

        # Cached for the training criterion; refreshed on every forward.
        self._cond: Optional[torch.Tensor] = None
        self._mu: Optional[torch.Tensor] = None
        self.last_members: Optional[torch.Tensor] = None

    # -- encoding ------------------------------------------------------------
    def _encode(self, x_radar: torch.Tensor, x_nwp: Optional[torch.Tensor]):
        b = x_radar.shape[0]
        h, w = x_radar.shape[-2:]
        radar = x_radar.reshape(b, self.in_len * self.in_radar_c, h, w)
        r = self.radar_enc(radar)

        if x_nwp is None:
            # BasePrecipitationModel.__call__ normally already resolved this,
            # but keep the adapter safe when called directly.
            n_flat = x_radar.new_zeros(
                b, self.nwp_time_len * self.in_nwp_c, h, w
            )
        else:
            nt = x_nwp.shape[1]
            if nt < self.nwp_time_len:
                x_nwp = F.pad(
                    x_nwp, (0, 0, 0, 0, 0, self.nwp_time_len - nt)
                )
            elif nt > self.nwp_time_len:
                x_nwp = x_nwp[:, : self.nwp_time_len]
            n_flat = x_nwp.reshape(b, -1, h, w)
        n = self.nwp_enc(n_flat)
        cond = self.fusion(torch.cat([r, n], dim=1))
        return cond

    def _anchor(self, cond: torch.Tensor) -> torch.Tensor:
        b, c, h, w = cond.shape
        t = self.out_len
        feats = cond.unsqueeze(1).expand(-1, t, -1, -1, -1) + self.time_embed
        mu = self.anchor_head(feats.reshape(b * t, c, h, w))
        return mu.reshape(b, t, 1, h, w)

    def _flatten_condition(
        self, cond: torch.Tensor, mu_log: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = mu_log.shape[:2]
        h, w = cond.shape[-2:]
        cond_bt = (
            cond.unsqueeze(1)
            .expand(-1, t, -1, -1, -1)
            .reshape(b * t, self.hid_c, h, w)
        )
        mu_bt = mu_log.reshape(b * t, 1, h, w)
        return cond_bt, mu_bt

    # -- conditional flow matching ------------------------------------------
    def _noise_like(self, x1: torch.Tensor) -> torch.Tensor:
        if self.random_start:
            return torch.randn_like(x1) * self.noise_scale
        # A1 control arm: the exact zero-start mechanism of ``diffcast``.
        return torch.zeros_like(x1)

    def _band_weights(self, target: torch.Tensor) -> torch.Tensor:
        if not self.intensity_bands:
            return torch.ones_like(target)
        edges = torch.tensor(BAND_EDGES, device=target.device, dtype=target.dtype)
        idx = torch.bucketize(target.detach(), edges, right=False)
        idx = idx.clamp(0, len(BAND_WEIGHTS) - 1)
        table = torch.tensor(
            BAND_WEIGHTS, device=target.device, dtype=target.dtype
        )
        w = table[idx]
        return w / w.mean().clamp_min(1e-6)

    def compute_flow_loss(self, target: torch.Tensor) -> torch.Tensor:
        """CFM regression loss, called by ``--loss ibcfm`` after ``forward``.

        ``target`` is ``(B, T_out, 1, H, W)`` in mm/6min.  Training uses
        ``log1p`` space throughout, consistent with ``common.RainHead``.
        """
        if self._cond is None or self._mu is None:
            raise RuntimeError(
                "compute_flow_loss() requires a preceding forward() call"
            )
        mu_log = torch.log1p(self._mu.clamp_min(0))
        x1 = torch.log1p(target.clamp_min(0)) - mu_log
        x0 = self._noise_like(x1)

        cond_bt, mu_bt = self._flatten_condition(self._cond, mu_log)
        x1_bt = x1.reshape_as(mu_bt)
        x0_bt = x0.reshape_as(mu_bt)
        bt = x1_bt.shape[0]

        t = torch.rand(bt, 1, device=x1.device, dtype=x1.dtype)
        z_t = (1.0 - t.reshape(bt, 1, 1, 1)) * x0_bt + t.reshape(
            bt, 1, 1, 1
        ) * x1_bt
        v = self.velocity(z_t, t, cond_bt, mu_bt)
        u = x1_bt - x0_bt
        w = self._band_weights(target).reshape_as(mu_bt)
        return ((v - u) ** 2 * w).mean()

    @torch.no_grad()
    def sample_members(self, n: Optional[int] = None) -> torch.Tensor:
        """Draw ``K`` genuine members; returns ``(B, K, T, 1, H, W)`` in mm."""
        if self._cond is None or self._mu is None:
            raise RuntimeError("sample_members() requires a preceding forward()")
        k = int(n or self.n_members)
        mu_log = torch.log1p(self._mu.clamp_min(0))
        cond_bt, mu_bt = self._flatten_condition(self._cond, mu_log)
        bt = mu_bt.shape[0]
        b, t, _, h, w = self._mu.shape

        members = []
        for _ in range(k):
            z = self._noise_like(mu_bt)
            for step in range(self.sample_steps):
                tt = torch.full(
                    (bt, 1), step / self.sample_steps,
                    device=mu_bt.device, dtype=mu_bt.dtype,
                )
                z = z + self.velocity(z, tt, cond_bt, mu_bt) / self.sample_steps
            y = torch.expm1((mu_bt + z).clamp(-10.0, 6.0))
            members.append(
                floor_at_zero(y).reshape(b, t, 1, h, w)
            )
        return torch.stack(members, dim=1)

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        x_radar: torch.Tensor,
        x_nwp: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Predict 6-minute accumulated rainfall in millimetres.

        Training returns the deterministic anchor (the flow loss is computed
        separately); evaluation returns the configured decision field, still
        a plain mm field so the evaluator needs no change.
        """
        if x_radar.dim() != 5:
            raise ValueError(
                f"x_radar must be (B,T,C,H,W), got {tuple(x_radar.shape)}"
            )
        cond = self._encode(x_radar, x_nwp)
        mu = self._anchor(cond)
        # Keep the graph alive: the criterion backprops through both branches.
        self._cond = cond
        self._mu = mu

        if self.training:
            return mu

        members = self.sample_members()
        self.last_members = members
        if self.decision_mode == "member":
            return members[:, 0]
        if self.decision_mode == "mean":
            return members.mean(dim=1)
        p = (members >= self.tier).to(mu.dtype).mean(dim=1)
        return torch.where(
            p >= self.p_star,
            torch.full_like(p, self.exceed_mm),
            torch.zeros_like(p),
        )
