"""MoCast adapter: turbulent motion modelling under physical guidance.

Paper: "MoCast: Learning Turbulent Motions Under Physical Guidance for
Precipitation Nowcasting" (AAAI 2026).

The paper splits nowcasting into a *motion* branch and a *source/sink* branch
that are fused by advection.  This module implements both mechanisms:

* **PMM -- physics-guided motion module.**  The learned motion field is
  decomposed with a Helmholtz-type split into its curl-free (potential) and
  divergence-free (rotational) parts, using a learned scalar potential (whose
  gradient is curl-free by construction), a learned vector potential (whose
  rotated gradient is divergence-free by construction) and a Fourier (Leray)
  projector on the field itself.  A low-pass spectral split then separates the
  coherent *mean* motion from the *fluctuating* (turbulent) motion, and a
  fixed Haar wavelet transform extracts the high-frequency detail that scales
  the turbulent component.
* **MSM -- mixture-of-experts source/sink module.**  Experts are multi-scale
  depthwise convolutions whose outputs are AdaIN-normalised with the motion
  statistics and mixed by a spatial softmax gate.  The mixture is the source
  field; its opposite is the sink field (mass conserving).
* **Fusion.**  ``X_{t+1} = Adv(X_t; M) + S``, where ``Adv`` is the
  differentiable advection warp from :mod:`..common` and ``S`` is the
  source/sink field.  The latent state is rolled out autoregressively for
  ``out_len`` frames and decoded to millimetres with :class:`..common.RainHead`.

Only torch primitives are used: the Haar wavelet transform is implemented by
hand (no PyWavelets) and the spectral split uses ``torch.fft``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import (
    ConvBlock,
    NWPEncoder,
    RainHead,
    floor_at_zero,
    warp,
)



def init_rain_head(head: RainHead) -> None:
    """Start the rain head close to "no rain".

    The head predicts in log space and exponentiates at the end, so a small
    random final layer with a zero bias makes the initial prediction a small
    rainfall instead of an arbitrary value.
    """
    final = head.net[-1]
    if isinstance(final, nn.Conv2d):
        if final.bias is not None:
            nn.init.zeros_(final.bias)
        nn.init.normal_(final.weight, std=0.01)


# ---------------------------------------------------------------------------
# Fixed Haar wavelet transform (hand written, no external dependency).
# ---------------------------------------------------------------------------
def haar_dwt2(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One level of a fixed 2-D Haar wavelet transform.

    Args:
        x: ``(B, C, H, W)`` real tensor.

    Returns:
        ``(ll, lh, hl, hh)``, each ``(B, C, H // 2, W // 2)``.  Odd sizes are
        replicate-padded first.
    """
    b, c, h, w = x.shape
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        h, w = x.shape[-2:]
    a = x[:, :, 0::2, 0::2]
    b_ = x[:, :, 0::2, 1::2]
    c_ = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = 0.5 * (a + b_ + c_ + d)
    lh = 0.5 * (a + b_ - c_ - d)  # horizontal detail
    hl = 0.5 * (a - b_ + c_ - d)  # vertical detail
    hh = 0.5 * (a - b_ - c_ + d)  # diagonal detail
    return ll, lh, hl, hh


def haar_idwt2(
    ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`haar_dwt2` for a single level."""
    b, c, h2, w2 = ll.shape
    a = 0.5 * (ll + lh + hl + hh)
    b_ = 0.5 * (ll + lh - hl - hh)
    c_ = 0.5 * (ll - lh + hl - hh)
    d = 0.5 * (ll - lh - hl + hh)
    top = torch.stack([a, b_], dim=-1).reshape(b, c, h2, w2 * 2)
    bottom = torch.stack([c_, d], dim=-1).reshape(b, c, h2, w2 * 2)
    return torch.stack([top, bottom], dim=-2).reshape(b, c, h2 * 2, w2 * 2)


def _gradients(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central differences of a ``(B, 1, H, W)`` field, returned as ``(dy, dx)``."""
    kernel = field.new_tensor([[-0.5, 0.0, 0.5], [-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]])
    kernel_x = kernel.transpose(-1, -2).contiguous()
    c = field.shape[1]
    padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
    dy = F.conv2d(padded, kernel.expand(c, 1, 3, 3), groups=c)
    dx = F.conv2d(padded, kernel_x.expand(c, 1, 3, 3), groups=c)
    return dy, dx


def _radial_mask(h: int, w: int, device: torch.device, cutoff: float) -> torch.Tensor:
    """Isotropic Gaussian low-pass mask in the half spectrum layout ``(h, w // 2 + 1)``."""
    fy = torch.fft.fftfreq(h, device=device).reshape(h, 1)
    fx = torch.fft.rfftfreq(w, device=device).reshape(1, -1)
    radius = torch.sqrt(fy**2 + fx**2)
    return torch.exp(-(radius / max(cutoff, 1e-6)) ** 2)


def helmholtz_split(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 2-D flow into its curl-free and divergence-free parts.

    This is the Fourier (Leray) projector: the curl-free part is parallel to
    the wave vector, the divergence-free part is orthogonal to it.

    Args:
        flow: ``(B, 2, H, W)`` displacement field ordered ``(dy, dx)``.

    Returns:
        ``(curl_free, divergence_free)``, each ``(B, 2, H, W)``.
    """
    h, w = flow.shape[-2:]
    spec = torch.fft.rfft2(flow, norm="ortho")
    fy = torch.fft.fftfreq(h, device=flow.device).reshape(h, 1)
    fx = torch.fft.rfftfreq(w, device=flow.device).reshape(1, -1)
    k2 = (fx**2 + fy**2).clamp_min(1e-8)
    # component 0 is dy, component 1 is dx
    dot = (fy * spec[:, 0] + fx * spec[:, 1]) / k2
    curl_free = torch.stack([fy * dot, fx * dot], dim=1)
    # The zero wave number carries no direction: keep the mean flow in the
    # curl-free branch so that the two parts still sum to the original field.
    curl_free = curl_free + spec[:, :, :1, :1]
    divergence_free = spec - curl_free
    size = (h, w)
    return (
        torch.fft.irfft2(curl_free, s=size, norm="ortho"),
        torch.fft.irfft2(divergence_free, s=size, norm="ortho"),
    )


def spectral_lowpass(field: torch.Tensor, cutoff: float = 0.25) -> torch.Tensor:
    """Keep only the low frequencies of a real field."""
    h, w = field.shape[-2:]
    spec = torch.fft.rfft2(field, norm="ortho")
    spec = spec * _radial_mask(h, w, field.device, cutoff)
    return torch.fft.irfft2(spec, s=(h, w), norm="ortho")


# ---------------------------------------------------------------------------
# Motion branch: physics-guided motion module (PMM).
# ---------------------------------------------------------------------------
class PhysicsGuidedMotion(nn.Module):
    """PMM of MoCast.

    The module predicts a motion field from the latent state, its temporal
    difference and the physical (NWP) guidance.  The field is assembled from
    a learned scalar potential (curl-free by construction), a learned vector
    potential (divergence-free by construction) and a free residual flow,
    then decomposed with the Fourier Helmholtz projector.  The coherent mean
    motion is the low-pass curl-free part, while the fluctuating motion is the
    high-pass divergence-free part, scaled by a Haar wavelet turbulence
    indicator.
    """

    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(channels * 3, hidden, 3),
            ConvBlock(hidden, hidden, 3, residual=True),
        )
        self.flow_head = nn.Conv2d(hidden, 2, 3, padding=1)
        self.potential_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.vector_potential_head = nn.Conv2d(hidden, 2, 3, padding=1)
        self.turbulence_gain = nn.Parameter(torch.tensor(1.0))
        self.flow_gain = nn.Parameter(torch.tensor(1.0))
        self.max_shift = 4.0  # pixels per 6-minute step

    def forward(
        self,
        state: torch.Tensor,
        previous_state: torch.Tensor,
        guidance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(flow, turbulence)`` with shapes ``(B, 2, H, W)`` and ``(B, 1, H, W)``."""
        feat = self.body(torch.cat([state, state - previous_state, guidance], dim=1))

        free_flow = self.flow_head(feat)

        # Learned scalar potential -> gradient, a curl-free field.
        potential = self.potential_head(feat)
        dphi_dy, dphi_dx = _gradients(potential)
        curl_free_field = torch.cat([dphi_dy, dphi_dx], dim=1)

        # Learned vector potential -> rotated gradient, a divergence-free field.
        vector_potential = self.vector_potential_head(feat)
        psi = vector_potential[:, :1]
        dpsi_dy, dpsi_dx = _gradients(psi)
        divergence_free_field = torch.cat([-dpsi_dx, dpsi_dy], dim=1)

        flow = free_flow + curl_free_field + divergence_free_field

        # Fourier split into coherent and turbulent parts.
        curl_free, divergence_free = helmholtz_split(flow)
        mean_motion = spectral_lowpass(curl_free)
        fluctuating = divergence_free - spectral_lowpass(divergence_free)

        turbulence = self._turbulence(state)
        flow = mean_motion + self.flow_gain * fluctuating * (
            1.0 + self.turbulence_gain * turbulence
        )
        # Keep the displacement within a physically plausible range.
        flow = self.max_shift * torch.tanh(flow / self.max_shift)
        return flow, turbulence

    @staticmethod
    def _turbulence(state: torch.Tensor) -> torch.Tensor:
        """High-frequency energy of the state, from a Haar wavelet decomposition."""
        ll, lh, hl, hh = haar_dwt2(state)
        detail = haar_idwt2(torch.zeros_like(ll), lh, hl, hh)
        energy = detail.abs().mean(dim=1, keepdim=True)
        return F.interpolate(
            energy, size=state.shape[-2:], mode="bilinear", align_corners=False
        )


# ---------------------------------------------------------------------------
# Source/sink branch: mixture-of-experts (MSM).
# ---------------------------------------------------------------------------
class _Expert(nn.Module):
    """Multi-scale depthwise expert."""

    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dw7 = nn.Conv2d(
            channels, channels, 7, padding=6, dilation=2, groups=channels
        )
        self.pointwise = nn.Conv2d(channels * 3, hidden, 1)
        self.out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1)
        return self.out(F.silu(self.pointwise(y)))


class MixtureOfExpertsSourceSink(nn.Module):
    """MSM of MoCast: AdaIN-normalised expert mixture with a spatial softmax gate."""

    def __init__(self, channels: int, hidden: int = 32, num_experts: int = 4) -> None:
        super().__init__()
        self.experts = nn.ModuleList([_Expert(channels, hidden) for _ in range(num_experts)])
        # Gate sees the advected state, the guidance and the turbulence map.
        self.gate = nn.Conv2d(channels * 2 + 1, num_experts, 1)
        # AdaIN affine parameters predicted from the motion statistics.
        self.style = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 2 * channels))
        self.source_head = nn.Conv2d(channels, channels, 1)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        advected: torch.Tensor,
        guidance: torch.Tensor,
        flow: torch.Tensor,
        turbulence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(source, sink)`` fields, each ``(B, C, H, W)``."""
        # Motion statistics: mean and standard deviation of the flow field.
        stats = torch.stack(
            [
                flow.mean(),
                flow.std() if flow.numel() > 1 else flow.mean(),
                turbulence.mean(),
                turbulence.std() if turbulence.numel() > 1 else turbulence.mean(),
            ]
        )
        gamma, beta = self.style(stats).chunk(2, dim=0)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

        gate_in = torch.cat([advected, guidance, turbulence], dim=1)
        weights = torch.softmax(self.gate(gate_in), dim=1)  # spatial softmax over experts

        mixed = torch.zeros_like(advected)
        for index, expert in enumerate(self.experts):
            out = expert(advected)
            # AdaIN: normalise the expert output, then apply the motion style.
            mean = out.mean(dim=(2, 3), keepdim=True)
            std = out.std(dim=(2, 3), keepdim=True) + 1e-5
            out = (out - mean) / std
            out = out * (1.0 + gamma) + beta
            mixed = mixed + weights[:, index : index + 1] * out

        source = self.gain * self.source_head(mixed)
        sink = -source  # mass conserving counterpart
        return source, sink


# ---------------------------------------------------------------------------
# Radar encoding.
# ---------------------------------------------------------------------------
class _RadarEncoder(nn.Module):
    """Encode every radar frame into a latent map and fuse them temporally."""

    def __init__(self, in_channels: int, in_len: int, channels: int = 32) -> None:
        super().__init__()
        self.frame = nn.Sequential(
            ConvBlock(in_channels, channels, 3),
            ConvBlock(channels, channels, 3, residual=True),
        )
        self.temporal = nn.Conv2d(channels * in_len, channels, 1)

    def forward(self, x_radar: torch.Tensor) -> torch.Tensor:
        """Return ``(B, T, C, H, W)`` latent maps for the whole history."""
        b, t = x_radar.shape[:2]
        feats = self.frame(x_radar.reshape(b * t, *x_radar.shape[2:]))
        feats = feats.reshape(b, t, *feats.shape[1:])
        # Temporal fusion over the whole history (learned, per pixel).
        fused = self.temporal(feats.reshape(b, -1, *feats.shape[-2:]))
        return feats + fused.unsqueeze(1)


# ---------------------------------------------------------------------------
# Adapter.
# ---------------------------------------------------------------------------
@register_model("mocast")
class MoCastAdapter(BasePrecipitationModel):
    """MoCast (AAAI 2026) adapter for the radar + NWP precipitation benchmark.

    Args:
        in_radar_c: number of radar channels per history frame.
        in_nwp_c: number of NWP channels per NWP frame.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        channels: width of the latent state.
        hidden: width of the internal convolutions.
        num_experts: number of experts in the MSM mixture.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        channels: int = 32,
        hidden: int = 32,
        num_experts: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.channels = channels
        self.encoder = _RadarEncoder(in_radar_c, in_len, channels)
        self.nwp_encoder = NWPEncoder(in_nwp_c, channels, hidden=hidden)
        self.motion = PhysicsGuidedMotion(channels, hidden=hidden)
        self.source_sink = MixtureOfExpertsSourceSink(
            channels, hidden=hidden, num_experts=num_experts
        )
        self.rain_head = RainHead(channels, hidden=hidden)
        init_rain_head(self.rain_head)

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        """Predict future accumulated rainfall.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)``.

        Returns:
            ``(B, T_out, 1, H, W)`` non-negative rainfall in millimetres.
        """
        self._check_shapes(x_radar, x_nwp)
        b, t = x_radar.shape[:2]

        latents = self.encoder(x_radar)  # (B, T, C, H, W)
        guidance = self.nwp_encoder(x_nwp)
        if guidance.shape[-2:] != latents.shape[-2:]:
            guidance = F.interpolate(
                guidance, size=latents.shape[-2:], mode="bilinear", align_corners=False
            )

        state = latents[:, -1]
        previous_state = latents[:, -2] if t > 1 else latents[:, -1]

        frames: list[torch.Tensor] = []
        for _ in range(self.out_len):
            flow, turbulence = self.motion(state, previous_state, guidance)
            advected = warp(state, flow)
            source, _sink = self.source_sink(advected, guidance, flow, turbulence)
            # Fusion: X_{t+1} = Adv(X_t; M) + S
            old_state = state
            state = advected + source
            previous_state = old_state
            frames.append(self.rain_head(state))

        out = torch.stack(frames, dim=1)
        # RainHead exponentiates, so the output is non-negative by construction.
        # A hard floor would cut the gradient wherever the prediction is
        # still negative, so those pixels could never be taught.
        return floor_at_zero(out)
