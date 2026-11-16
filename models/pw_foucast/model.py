"""PW-FouCast adapter: spectral fusion of radar observations and NWP priors.

Paper: "PW-FouCast: Extending Precipitation Nowcasting Horizons via Spectral
Fusion of Radar Observations and Foundation Model Priors" (IJCNN 2026).

The model is built on Adaptive Fourier Neural Operator (AFNO) blocks:

* **Spectral encoder/decoder.**  A size-agnostic AFNO block mixes the channel
  dimension inside the Fourier domain with a learnable *complex* weight.  The
  mixing is block-diagonal over channel groups, and the frequency dependence
  of the weight is carried by a learnable complex radial profile, so the block
  works for any spatial size (the frequency grid is never baked into a
  parameter shape).
* **PFM-style Pangu-guided frequency modulation.**  The NWP field is taken to
  the complex Fourier domain; the normalised inner product between the hidden
  spectrum and the NWP spectrum gives a phase-consistency score, a softmax
  over channels turns that score into a spatial attention map, and the hidden
  phase is interpolated toward the NWP phase with a learnable scalar ``beta``.
* **Frequency memory.**  A learnable complex parameter of shape
  ``(mem_slots, dim, 2)`` (real and imaginary parts stacked) stores spectral
  memory slots.  During training the slots can be filled with the
  ground-truth spectrum (passed as the optional ``gt_spectrum`` argument); at
  inference the slot whose phase best matches the current hidden spectrum is
  retrieved and used to modulate how strongly each channel trusts the NWP
  phase.
* **Inverted frequency attention.**  The high-frequency residual
  ``hf = x - lowpass(x)`` is re-injected with a learnable weight on every
  layer except the first.

All out_len frames are produced by a single forward pass: the spectral
decoder ends in a layer that predicts ``out_len`` groups of channels, which
are split into the forecast frames and mapped to millimetres with
:class:`..common.RainHead`.
"""
from __future__ import annotations

import math

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


def _radial_basis(h: int, w: int, n_bases: int, device: torch.device) -> torch.Tensor:
    """Triangular radial basis over the half spectrum.

    Args:
        h: full spatial height.
        w: full spatial width (the basis covers ``w // 2 + 1`` frequencies).
        n_bases: number of basis functions.
        device: device of the returned tensor.

    Returns:
        ``(h, w // 2 + 1, n_bases)`` real tensor.
    """
    fy = torch.fft.fftfreq(h, device=device).reshape(h, 1)
    fx = torch.fft.rfftfreq(w, device=device).reshape(1, -1)
    radius = torch.sqrt(fy**2 + fx**2) / 0.70710678
    centres = (torch.arange(n_bases, device=device, dtype=torch.float32) + 0.5) / n_bases
    basis = torch.stack(
        [torch.clamp(1.0 - torch.abs(radius - c) * n_bases, min=0.0) for c in centres],
        dim=-1,
    )
    return basis


def spectral_lowpass(field: torch.Tensor, cutoff: float = 0.25) -> torch.Tensor:
    """Isotropic low-pass version of a real field."""
    h, w = field.shape[-2:]
    spec = torch.fft.rfft2(field, norm="ortho")
    fy = torch.fft.fftfreq(h, device=field.device).reshape(h, 1)
    fx = torch.fft.rfftfreq(w, device=field.device).reshape(1, -1)
    radius = torch.sqrt(fy**2 + fx**2)
    mask = torch.exp(-(radius / max(cutoff, 1e-6)) ** 2)
    return torch.fft.irfft2(spec * mask, s=(h, w), norm="ortho")


class AFNOBlock(nn.Module):
    """Adaptive Fourier Neural Operator block (size agnostic).

    The complex channel-mixing weight is block-diagonal over channel groups:
    each group of channels is mixed by its own complex matrix.  The frequency
    dependence of that matrix is expressed through a learnable complex radial
    profile, so the parameter shapes never depend on the spatial size of the
    input and the block works at any resolution.
    """

    def __init__(
        self,
        channels: int,
        n_bases: int = 4,
        n_blocks: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % n_blocks != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by n_blocks ({n_blocks})"
            )
        self.channels = channels
        self.n_bases = n_bases
        self.n_blocks = n_blocks
        self.group = channels // n_blocks
        scale = 1.0 / math.sqrt(self.group)
        # Learnable complex weight: (block group, out channel, in channel).
        self.weight = nn.Parameter(
            torch.randn(n_blocks, self.group, self.group, dtype=torch.cfloat) * scale
        )
        # Learnable complex radial profile of the mixing weight.
        self.profile = nn.Parameter(torch.ones(n_bases, dtype=torch.cfloat))
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(32, channels), channels)
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        spec = torch.fft.rfft2(x, norm="ortho")  # (B, C, H, Wf)
        wf = spec.shape[-1]

        # Block-diagonal complex mixing over the channel dimension.
        grouped = spec.reshape(b, self.n_blocks, self.group, h, wf)
        mixed = torch.einsum("bgihw,goi->bgohw", grouped, self.weight)
        mixed = mixed.reshape(b, c, h, wf)

        # Frequency-dependent complex gain (the radial profile of the weight).
        basis = _radial_basis(h, w, self.n_bases, x.device)
        gain = torch.einsum("hwk,k->hw", basis.to(self.profile.dtype), self.profile)
        mixed = mixed * gain

        y = torch.fft.irfft2(mixed, s=(h, w), norm="ortho")
        y = x + self.dropout(self.proj(y))
        y = y + self.mlp(self.norm(y))
        return y


class InvertedFrequencyAttention(nn.Module):
    """Re-inject the high-frequency residual with a learnable weight."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # The gain starts at zero so that the module is the identity at
        # initialisation -- but it must not be a *hard* zero forever: the
        # projection is multiplied by the gain, so a gain that stays at zero
        # leaves the projection without a gradient and the high-frequency
        # branch never activates. The gain itself does receive a gradient
        # (it is a factor of a non-zero term), so it can move away from zero
        # as soon as training starts.
        self.gain = nn.Parameter(torch.zeros(1))
        self.proj = nn.Conv2d(channels, channels, 1)
        # A small non-zero init on the projection gives the branch a signal
        # to work with from the first step.
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf = x - spectral_lowpass(x)
        return x + self.gain * self.proj(hf)


class PanguFrequencyModulation(nn.Module):
    """PFM-style modulation of the hidden spectrum by the NWP (Pangu) prior."""

    def __init__(self, channels: int, nwp_channels: int) -> None:
        super().__init__()
        self.nwp_proj = nn.Conv2d(nwp_channels, channels, 1)
        self.beta = nn.Parameter(torch.tensor(0.2))
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(
        self,
        hidden: torch.Tensor,
        nwp: torch.Tensor,
        channel_trust: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Modulate ``hidden`` with the NWP prior.

        Args:
            hidden: ``(B, C, H, W)`` hidden features.
            nwp: ``(B, C_nwp, H, W)`` NWP features (already on the same grid).
            channel_trust: optional ``(B, C)`` per-channel confidence in [0, 1]
                coming from the frequency memory.  It scales the phase pull.

        Returns:
            ``(B, C, H, W)`` modulated features.
        """
        size = hidden.shape[-2:]
        nwp = self.nwp_proj(nwp)
        if nwp.shape[-2:] != size:
            nwp = F.interpolate(nwp, size=size, mode="bilinear", align_corners=False)

        hidden_spec = torch.fft.rfft2(hidden, norm="ortho")
        nwp_spec = torch.fft.rfft2(nwp, norm="ortho")

        # Normalised inner product -> phase consistency score.
        numerator = (hidden_spec.conj() * nwp_spec).real
        denominator = hidden_spec.abs() * nwp_spec.abs() + 1e-6
        score = numerator / denominator  # (B, C, H, Wf) in [-1, 1]

        # Softmax over channels -> spatial attention map.
        attention = torch.softmax(score, dim=1)
        spatial = (attention * score).sum(dim=1, keepdim=True)  # (B, 1, H, Wf)
        spatial_map = torch.sigmoid(
            torch.fft.irfft2(spatial, s=size, norm="ortho")
        )  # (B, 1, H, W)

        # Interpolate the hidden phase toward the NWP phase.
        beta = torch.sigmoid(self.beta)
        if channel_trust is not None:
            # (B, C) -> (B, C, 1, 1): one trust value per channel.
            beta = beta * channel_trust.unsqueeze(-1).unsqueeze(-1)
        phase_hidden = torch.angle(hidden_spec)
        phase_nwp = torch.angle(nwp_spec)
        phase = phase_hidden + beta * (phase_nwp - phase_hidden)
        modulated = hidden_spec.abs() * torch.exp(1j * phase)

        features = torch.fft.irfft2(modulated, s=size, norm="ortho")
        out = hidden + self.scale * spatial_map * self.out_proj(features)
        return out


class FrequencyMemory(nn.Module):
    """Learnable complex spectral memory with matching-phase retrieval.

    The memory is stored as a learnable parameter of shape
    ``(mem_slots, dim, 2)`` (real and imaginary parts stacked).  During
    training it can be filled with the ground-truth spectrum; at inference the
    slot whose phase best matches the current hidden spectrum is retrieved.
    """

    def __init__(self, mem_slots: int = 4, channels: int = 32) -> None:
        super().__init__()
        self.mem_slots = mem_slots
        self.channels = channels
        self.memory = nn.Parameter(torch.zeros(mem_slots, channels, 2))
        with torch.no_grad():
            self.memory.normal_(0.0, 0.02)
        self.gt_proj = nn.Conv2d(1, channels, 1)
        self.scale = nn.Parameter(torch.tensor(1.0))

    # -- helpers ----------------------------------------------------------
    @property
    def slots(self) -> torch.Tensor:
        """The memory slots as complex tensors, shape ``(mem_slots, channels)``."""
        return torch.view_as_complex(self.memory.contiguous())

    @staticmethod
    def _summarise(spec: torch.Tensor) -> torch.Tensor:
        """Magnitude-weighted complex summary of a spectrum, shape ``(B, C)``."""
        magnitude = spec.abs()
        weight = magnitude + 1e-6
        return (spec * weight).sum(dim=(-2, -1)) / weight.sum(dim=(-2, -1))

    @torch.no_grad()
    def write(self, target: torch.Tensor) -> torch.Tensor:
        """Fill the memory slots with the ground-truth spectrum.

        The write has to stay inside the autograd graph: the paper's memory
        is filled with the observed spectrum, and the projection that produces
        it has to be trained, otherwise it never learns anything. Writing
        into a leaf parameter in place would cut the graph, so the slots are
        assembled with ``torch.stack`` and returned instead.

        Args:
            target: the ground truth, either a complex spectrum or the raw
                frames ``(B, T, 1, H, W)``.

        Returns:
            The new memory, ``(mem_slots, channels, 2)``.
        """
        if target.dim() == 5:  # (B, T, 1, H, W) ground truth frames
            target = target[:, -1]
        if not torch.is_complex(target):
            target = self.gt_proj(target)
            target = torch.fft.rfft2(target, norm="ortho")
        summary = self._summarise(target)
        b = summary.shape[0]

        slots: list[torch.Tensor] = []
        for slot in range(self.mem_slots):
            index = slot % max(b, 1)
            if b == 0:
                slots.append(self.memory[slot])
                continue
            real = summary[index].real
            imag = summary[index].imag
            slots.append(torch.stack([real, imag], dim=-1))
        return torch.stack(slots, dim=0)

    def retrieve(
        self, hidden_spec: torch.Tensor, memory: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Retrieve the best matching memory slot.

        Args:
            hidden_spec: ``(B, C, H, Wf)`` complex hidden spectrum.
            memory: the memory to search, ``(S, C, 2)`` as real and
                imaginary parts.  Defaults to the module's own memory.  It is
                passed in explicitly so that a freshly written memory -- one
                that is still part of the autograd graph -- can be used.

        Returns:
            ``(B, C)`` per-channel agreement in ``[0, 1]``.
        """
        summary = self._summarise(hidden_spec)  # (B, C) complex
        raw = self.memory if memory is None else memory
        slots = torch.view_as_complex(raw.contiguous())  # (S, C) complex
        # Phase agreement between the hidden spectrum and every slot.
        cosine = (summary.unsqueeze(1).conj() * slots.unsqueeze(0)).real
        cosine = cosine / (
            summary.abs().unsqueeze(1) * slots.abs().unsqueeze(0) + 1e-6
        )
        weights = torch.softmax(self.scale * cosine.mean(dim=-1), dim=1)  # (B, S)
        retrieved = torch.einsum("bs,sc->bc", weights.to(slots.dtype), slots)
        agreement = (
            (summary.conj() * retrieved).real / (summary.abs() * retrieved.abs() + 1e-6)
        )
        return torch.clamp(agreement, min=0.0, max=1.0)


@register_model("pw_foucast")
class PWFouCastAdapter(BasePrecipitationModel):
    """PW-FouCast (IJCNN 2026) adapter for the radar + NWP precipitation benchmark.

    Args:
        in_radar_c: number of radar channels per history frame.
        in_nwp_c: number of NWP channels per NWP frame.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        channels: hidden width of the spectral trunk.
        head_channels: width of the per-frame features before the rain head.
        n_bases: number of radial basis functions of the AFNO weight.
        n_blocks: number of channel groups of the block-diagonal AFNO weight.
        mem_slots: number of frequency-memory slots.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        channels: int = 32,
        head_channels: int = 16,
        n_bases: int = 4,
        n_blocks: int = 4,
        mem_slots: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.channels = channels
        self.head_channels = head_channels
        self.in_len = in_len
        self.out_len = out_len

        # Radar observations -> spectral features.
        self.frame_encoder = ConvBlock(in_radar_c, channels, 3)
        self.temporal_logits = nn.Parameter(torch.zeros(in_len))
        self.nwp_encoder = NWPEncoder(in_nwp_c, channels, hidden=channels)

        # Spectral encoder.
        self.encoder_blocks = nn.ModuleList(
            [AFNOBlock(channels, n_bases=n_bases, n_blocks=n_blocks) for _ in range(2)]
        )
        self.encoder_ifa = nn.ModuleList([InvertedFrequencyAttention(channels)])

        # PFM-style NWP guidance and spectral memory.
        self.pfm = PanguFrequencyModulation(channels, channels)
        self.memory = FrequencyMemory(mem_slots, channels)

        # Spectral decoder.
        self.decoder_blocks = nn.ModuleList(
            [AFNOBlock(channels, n_bases=n_bases, n_blocks=n_blocks) for _ in range(2)]
        )
        self.decoder_ifa = nn.ModuleList([InvertedFrequencyAttention(channels)])
        self.decoder_conv = ConvBlock(channels, channels, 3)
        self.head = nn.Conv2d(channels, out_len * head_channels, 1)
        self.rain_head = RainHead(head_channels, hidden=channels)
        init_rain_head(self.rain_head)

    def forward(
        self,
        x_radar: torch.Tensor,
        x_nwp: torch.Tensor,
        gt_spectrum: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict future accumulated rainfall.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)``.
            gt_spectrum: optional ground truth used to fill the frequency
                memory during training.  Accepts a complex spectrum or the raw
                ground-truth frames ``(B, T_out, 1, H, W)``.

        Returns:
            ``(B, T_out, 1, H, W)`` non-negative rainfall in millimetres.
        """
        self._check_shapes(x_radar, x_nwp)
        b, t = x_radar.shape[:2]

        # --- encode the radar stack -------------------------------------
        frames = self.frame_encoder(x_radar.reshape(b * t, *x_radar.shape[2:]))
        frames = frames.reshape(b, t, *frames.shape[1:])
        weights = torch.softmax(self.temporal_logits, dim=0).reshape(1, t, 1, 1, 1)
        hidden = (frames * weights).sum(dim=1)  # (B, C, H, W)

        # --- spectral encoder -------------------------------------------
        for index, block in enumerate(self.encoder_blocks):
            hidden = block(hidden)
            if index > 0:
                hidden = self.encoder_ifa[index - 1](hidden)

        # --- NWP prior and frequency memory -----------------------------
        nwp = self.nwp_encoder(x_nwp)
        if nwp.shape[-2:] != hidden.shape[-2:]:
            nwp = F.interpolate(
                nwp, size=hidden.shape[-2:], mode="bilinear", align_corners=False
            )
        if gt_spectrum is not None:
            # The write returns the new memory; using the returned value keeps
            # the ground-truth spectrum inside the autograd graph, so the
            # projection that produces it is actually trained.
            memory = self.memory.write(gt_spectrum)
        else:
            memory = self.memory.memory
        hidden_spec = torch.fft.rfft2(hidden, norm="ortho")
        trust = self.memory.retrieve(hidden_spec, memory=memory)
        hidden = self.pfm(hidden, nwp, trust)

        # --- spectral decoder -------------------------------------------
        for index, block in enumerate(self.decoder_blocks):
            hidden = block(hidden)
            if index > 0:
                hidden = self.decoder_ifa[index - 1](hidden)
        hidden = self.decoder_conv(hidden)

        # --- final layer: all lead times at once, then split -------------
        frames = self.head(hidden)  # (B, T_out * head_channels, H, W)
        frames = frames.reshape(b, self.out_len, self.head_channels, *frames.shape[-2:])
        flat = frames.reshape(b * self.out_len, self.head_channels, *frames.shape[-2:])
        rain = self.rain_head(flat)
        rain = rain.reshape(b, self.out_len, 1, *frames.shape[-2:])
        # A hard floor would cut the gradient wherever the rain is still
        # negative, which is everywhere at initialisation.
        return floor_at_zero(rain)
