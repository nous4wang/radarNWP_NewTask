"""Shared building blocks for the baseline adapters.

These are deliberately generic: the *architecture* of each baseline lives in its
own module, and only the mechanical pieces (conv blocks, attention, output
heads, NWP encoders, quantum tendency couplers) are shared. Sharing them keeps
the adapters short enough to be read, and keeps one convention for how a model
turns a log-space target into millimetres.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.index import NWP_VARS


def _group_count(channels: int, max_groups: int = 32) -> int:
    """Largest divisor of ``channels`` that is at most ``max_groups``.

    ``nn.GroupNorm`` requires the group count to divide the channel count
    exactly, so a naive ``min(32, channels)`` breaks for widths like 48.
    """
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """Conv -> norm -> activation, with an optional residual connection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm: bool = True,
        act: bool = True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=not norm)
        ]
        if norm:
            layers.append(nn.GroupNorm(_group_count(out_ch), out_ch))
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.body = nn.Sequential(*layers)
        self.residual = residual and in_ch == out_ch and stride == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        return x + out if self.residual else out


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel gating."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        pooled = x.mean(dim=(2, 3))
        return x * self.fc(pooled).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial gating produced from a channel-pooled feature map."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * self.gate(self.conv(pooled))


class SelfAttention2d(nn.Module):
    """Single-head spatial self-attention over an ``(B, C, H, W)`` feature map."""

    def __init__(self, channels: int, max_tokens: int = 4096) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = h * w
        if tokens > self.max_tokens:
            step = int(math.ceil(tokens / self.max_tokens))
            x = x[:, :, ::step, ::step]
            h, w = x.shape[-2:]
            tokens = h * w
        y = self.norm(x)
        qkv = self.qkv(y).reshape(b, 3, c, tokens)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(attn)


class CrossAttention2d(nn.Module):
    """Spatial cross-attention from one feature map into another."""

    def __init__(self, q_channels: int, kv_channels: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(q_channels, q_channels, 1)
        self.k = nn.Conv2d(kv_channels, q_channels, 1)
        self.v = nn.Conv2d(kv_channels, q_channels, 1)
        self.proj = nn.Conv2d(q_channels, q_channels, 1)
        self.norm = nn.GroupNorm(min(32, q_channels), q_channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ctx = context
        if ctx.shape[-2:] != (h, w):
            ctx = F.interpolate(ctx, size=(h, w), mode="bilinear", align_corners=False)
        tokens = h * w
        q = self.q(self.norm(x)).reshape(b, c, tokens).permute(0, 2, 1)
        k = self.k(ctx).reshape(b, c, tokens).permute(0, 2, 1)
        v = self.v(ctx).reshape(b, c, tokens).permute(0, 2, 1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(attn)


class MLP(nn.Module):
    """Channel-wise feed-forward block."""

    def __init__(self, channels: int, ratio: int = 2) -> None:
        super().__init__()
        hidden = channels * ratio
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SpatialCrossAttention(nn.Module):
    """Spatial cross-attention: one feature map queries another.

    The query comes from the radar features and the keys and the values from
    the NWP features, so every position of the radar can pull the NWP
    information that belongs to it.

    This block is deliberately the *conventional* one -- multi-head
    attention, a residual connection and a channel MLP -- and carries no
    Koopman or quantum-inspired machinery anywhere. That is what makes it
    usable as the control row of an ablation: the point of such a row is
    to separate "the NWP helps" from "this particular coupling of the NWP
    helps", so the alternative has to be the strongest ordinary alternative
    rather than a straw man.

    Args:
        q_channels: channel count of the query feature map.
        kv_channels: channel count of the key/value feature map.
        num_heads: number of attention heads.
        mlp_ratio: expansion factor of the channel MLP.
        max_tokens: the largest number of spatial tokens the attention is
            computed over. Beyond it both grids are strided down, the
            attention runs at the lower resolution and its result is
            upsampled again. The cost of the attention is quadratic in the
            token count, and the production grid is 301x301.
    """

    def __init__(
        self,
        q_channels: int,
        kv_channels: int,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if q_channels % num_heads != 0:
            raise ValueError(
                f"q_channels must be divisible by num_heads: "
                f"{q_channels} / {num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = q_channels // num_heads
        self.max_tokens = max_tokens
        self.norm = nn.GroupNorm(_group_count(q_channels), q_channels)
        self.q_proj = nn.Conv2d(q_channels, q_channels, 1)
        self.k_proj = nn.Conv2d(kv_channels, q_channels, 1)
        self.v_proj = nn.Conv2d(kv_channels, q_channels, 1)
        self.proj = nn.Conv2d(q_channels, q_channels, 1)
        self.ffn_norm = nn.GroupNorm(_group_count(q_channels), q_channels)
        hidden = q_channels * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Conv2d(q_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, q_channels, 1),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Attend the query map to the context map.

        Args:
            x: ``(B, C, H, W)`` -- the query feature map (the radar side).
            context: ``(B, C_kv, H2, W2)`` -- the key/value feature map
                (the NWP side). It is resampled onto the query grid when the
                two live at different resolutions.

        Returns:
            ``(B, C, H, W)`` -- the query map with the context attended in.
        """
        b, c, h, w = x.shape
        ctx = context
        if ctx.shape[-2:] != (h, w):
            ctx = F.interpolate(ctx, size=(h, w), mode="bilinear", align_corners=False)

        # Both grids are strided down together when they carry too many
        # tokens, and the attention result is upsampled back, so the
        # residual connection still applies to the full feature map.
        step = 1
        if h * w > self.max_tokens:
            step = int(math.ceil(math.sqrt((h * w) / self.max_tokens)))
        query = x[:, :, ::step, ::step]
        keys = ctx[:, :, ::step, ::step]
        tokens = query.shape[-2] * query.shape[-1]

        # (B, C, H, W) -> (B, heads, tokens, head_dim). The channel axis is
        # the outer one, so splitting it into (heads, head_dim) and folding
        # the spatial axes into the token axis is a plain reshape.
        q = self.q_proj(self.norm(query)).reshape(b, self.num_heads, self.head_dim, tokens)
        q = q.permute(0, 1, 3, 2)
        k = self.k_proj(keys).reshape(b, self.num_heads, self.head_dim, tokens)
        k = k.permute(0, 1, 3, 2)
        v = self.v_proj(keys).reshape(b, self.num_heads, self.head_dim, tokens)
        v = v.permute(0, 1, 3, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        # Back to (B, C, H, W): the head axis has to be folded into the
        # channel axis in the same order it was split in.
        attn = attn.permute(0, 1, 3, 2).reshape(b, c, *query.shape[-2:])
        if step > 1:
            attn = F.interpolate(
                attn, size=(h, w), mode="bilinear", align_corners=False
            )

        x = x + self.proj(attn)
        return x + self.ffn(self.ffn_norm(x))


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: depthwise convolution, then a channel MLP.

    It lives here rather than in one baseline's file because a baseline
    that another baseline is derived from needs to reach it, and the model
    files are loaded by path (they may not import the framework at module
    level, or the registry would find it half-initialised).
    """

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return res + x


class NWPEncoder(nn.Module):
    """Encode the NWP cube into a radar-grid feature map.

    The NWP input is ``(B, T_nwp, C_nwp, H, W)``. The two slices differ in
    meaning -- one is the analysis hour, the other the forecast hour -- so the
    encoder keeps them separate and fuses them by concatenation plus a learned
    gate. The gate is what lets a model notice that only the forecast slice
    carries information about the target window.

    Args:
        in_nwp_c: number of NWP channels in the input (the true count).
        out_channels: width of the produced feature map.
        hidden: internal width of the per-slice backbone.
    """

    def __init__(self, in_nwp_c: int, out_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.in_nwp_c = in_nwp_c
        self.hidden = hidden
        self.temporal_embed = nn.Embedding(8, hidden)
        self.backbone = nn.Sequential(
            ConvBlock(in_nwp_c + hidden, hidden, 3),
            ConvBlock(hidden, hidden, 3, residual=True),
            ConvBlock(hidden, out_channels, 3),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x_nwp: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_nwp.shape
        if c != self.in_nwp_c:
            raise ValueError(
                f"NWPEncoder expects {self.in_nwp_c} channels, got {c}"
            )
        feats = []
        for i in range(t):
            idx = torch.full((b,), i, dtype=torch.long, device=x_nwp.device)
            emb = self.temporal_embed(idx).view(b, -1, 1, 1).expand(b, -1, h, w)
            feats.append(self.backbone(torch.cat([x_nwp[:, i], emb], dim=1)))
        if len(feats) == 1:
            return feats[0]
        stacked = torch.stack(feats, dim=1)
        prior = stacked[:, :-1].mean(dim=1) if t > 1 else stacked[:, 0]
        latest = stacked[:, -1]
        g = self.gate(torch.cat([prior, latest], dim=1))
        return g * latest + (1 - g) * prior


# ===========================================================================
# Quantum-Inspired NWP Modules (MPS Projector, Koopman Unitary Flow, Coupler)
# ===========================================================================


class QuantumStateProjector(nn.Module):
    """Matrix Product State (MPS) inspired low-rank quantum state compressor for NWP fields.

    Compresses high-dimensional multi-variable NWP slices (27 channels by default)
    into compact, entangled quantum-inspired representations while preserving cross-variable
    physical correlations (thermodynamic stability vs. dynamic kinematics).

    The two groups are defined by what the variable *is*, not by where it
    happens to sit in the channel list. Splitting the list in half would put
    the divergence fields (DVG200/850/925, which describe the kinematic
    field) on the thermodynamic side and the wind speeds (WS500/700/925,
    also kinematic) on the other, which is the opposite of what the two
    branches are for.
    """

    #: NWP variables that describe the thermodynamic state of the column.
    THERMODYNAMIC_VARS = frozenset({
        "CAPE", "HT0", "HT10", "HTw0", "KI", "LCL", "LFC",
        "LI300", "LI500", "muLCL", "PE", "PWAT",
        "Q1000", "Q700", "Q850", "RH1000", "RH500", "RH700",
        "TdSfc850", "TTdMax74", "TTdMean74",
    })

    #: NWP variables that describe the kinematic state of the column.
    KINEMATIC_VARS = frozenset({
        "DVG200", "DVG850", "DVG925", "WS500", "WS700", "WS925",
    })

    def __init__(
        self,
        in_vars: int | tuple[str, ...] = 27,
        bond_dim: int = 16,
        embed_dim: int = 64,
        nwp_vars: Optional[tuple[str, ...]] = None,
    ) -> None:
        super().__init__()
        # The number of variables may be given as a count or as the list of
        # names, which is the more useful of the two: it says which physical
        # fields the projector is about to be handed.
        if isinstance(in_vars, int):
            names = tuple(nwp_vars) if nwp_vars else tuple(NWP_VARS)[:in_vars]
            in_vars = len(names)
        else:
            names = tuple(in_vars)
            in_vars = len(names)
            if nwp_vars:
                names = tuple(nwp_vars)
        self.in_vars = in_vars
        self.bond_dim = bond_dim
        self.embed_dim = embed_dim

        # Split the variables into their two physical groups. The split is by
        # name, so it survives a reordering of the channel list; when the
        # names are not known the list is split in half, which is the best
        # that can be done without them.
        thermo_idx = [i for i, n in enumerate(names)
                      if n in self.THERMODYNAMIC_VARS]
        dynam_idx = [i for i, n in enumerate(names)
                     if n in self.KINEMATIC_VARS]
        if not thermo_idx or not dynam_idx:
            # Unknown or unclassifiable variables: fall back to a positional
            # split, which at least keeps the two branches the same size.
            half = in_vars // 2
            thermo_idx = list(range(half))
            dynam_idx = list(range(half, in_vars))
        self.register_buffer(
            "thermo_index", torch.tensor(thermo_idx, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "dynam_index", torch.tensor(dynam_idx, dtype=torch.long),
            persistent=False,
        )

        thermo_c = max(len(thermo_idx), 1)
        dynam_c = max(len(dynam_idx), 1)

        self.thermo_core = nn.Conv2d(thermo_c, bond_dim, kernel_size=1, bias=False)
        self.dynam_core = nn.Conv2d(dynam_c, bond_dim, kernel_size=1, bias=False)

        # Variational entanglement simulating quantum CNOT/rotational gate interactions
        self.entangle_conv = nn.Conv2d(bond_dim * 2, embed_dim, kernel_size=3, padding=1, bias=False)
        # A norm over the *whole* group, not a per-position one.
        #
        # A GroupNorm with one group normalises over the channel axis and
        # the spatial axes together, which leaves the spatial structure of
        # the field intact: measured on a real NWP field, the correlation
        # between neighbouring pixels is 0.9996 in the input and 0.9907
        # after the projection. A per-sample norm (InstanceNorm2d) instead
        # drops it to 0.8662 -- it divides away the very structure the
        # tendency is supposed to describe.
        self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, x_slice: torch.Tensor) -> torch.Tensor:
        """Project one NWP slice into a Hilbert feature space.

        Args:
            x_slice: ``(B, C_nwp, H, W)`` -- the NWP variables of one time
                slice, for a batch of ``B``. The variables are split into
                their two physical groups here, so the caller hands over
                all of them and does not have to know the grouping.

        Returns:
            ``(B, embed_dim, H, W)``.
        """
        x_thermo = x_slice.index_select(1, self.thermo_index)
        x_dynam = x_slice.index_select(1, self.dynam_index)

        phi_t = self.thermo_core(x_thermo)
        phi_d = self.dynam_core(x_dynam)

        state = torch.cat([phi_t, phi_d], dim=1)
        return F.gelu(self.norm(self.entangle_conv(state)))


class KoopmanUnitaryFlow(nn.Module):
    """Koopman-inspired generator with skew-Hermitian weights for unitary physical flow.

    Computes an energy-conserving operator U = exp(A - A^T) to project the analysis
    state psi_t0 forward, then estimates non-adiabatic dynamical and thermodynamic
    tendencies relative to the forecast state psi_t1.
    """

    def __init__(self, state_dim: int = 64) -> None:
        super().__init__()
        self.state_dim = state_dim
        # Skew-symmetric weight parameter matrix (A - A^T) guarantees strict unitarity
        self.skew_weight = nn.Parameter(torch.randn(state_dim, state_dim) * 0.02)
        self.tendency_conv = nn.Sequential(
            nn.Conv2d(state_dim * 2, state_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(state_dim, state_dim, kernel_size=1),
        )

    def forward(self, psi_t0: torch.Tensor, psi_t1: torch.Tensor) -> torch.Tensor:
        """Evolve state psi_t0 unitarily and contrast with psi_t1 to produce physical tendencies."""
        skew_hamiltonian = self.skew_weight - self.skew_weight.t()
        u_matrix = torch.matrix_exp(skew_hamiltonian)

        b, c, h, w = psi_t0.shape
        psi_t0_flat = psi_t0.permute(0, 2, 3, 1).reshape(b * h * w, c)
        evolved_t0 = (psi_t0_flat @ u_matrix).view(b, h, w, c).permute(0, 3, 1, 2)

        return self.tendency_conv(torch.cat([evolved_t0, psi_t1], dim=1))


class QuantumInspiredTendencyCoupler(nn.Module):
    """Universal Plugin: Modulates baseline radar/spatiotemporal features with NWP quantum tendencies.

    Seamlessly attaches into any of the 15 baselines before the rainfall output head.
    Supports both 4D (B, C, H, W) and 5D (B, T, C, H, W) radar features, and safely
    bypasses modulation (identity pass-through) when NWP is withheld under ablation.

    Args:
        in_nwp_c: Number of NWP variables (default 27).
        radar_feat_dim: Feature channels of the radar backbone output.
        out_len: Forecast horizon frames (default 10).
    """

    def __init__(
        self,
        in_nwp_c: int | tuple[str, ...] = 27,
        radar_feat_dim: int = 64,
        out_len: int = 10,
        nwp_vars: Optional[tuple[str, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_nwp_c = in_nwp_c if isinstance(in_nwp_c, int) else len(in_nwp_c)
        self.radar_feat_dim = radar_feat_dim
        self.out_len = out_len

        self.state_proj = QuantumStateProjector(
            in_vars=in_nwp_c, bond_dim=16, embed_dim=radar_feat_dim,
            nwp_vars=nwp_vars,
        )
        self.koopman_flow = KoopmanUnitaryFlow(state_dim=radar_feat_dim)

        # Spatial FiLM modulation: generates scale (gamma) and shift (beta) per time frame
        self.film_modulator = nn.Sequential(
            nn.Conv2d(radar_feat_dim, radar_feat_dim * 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(radar_feat_dim * 2, radar_feat_dim * 2 * out_len, kernel_size=1),
        )

        # Convective trigger gate: amplifies extreme precipitation updraft cores
        self.extreme_trigger_gate = nn.Sequential(
            nn.Conv2d(radar_feat_dim, out_len, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        radar_features: torch.Tensor,
        x_nwp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply quantum-inspired physical tendency modulation to radar features.

        Args:
            radar_features: ``(B, T_out, C, H, W)`` or ``(B, C, H, W)`` feature map.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)`` or ``None`` (ablation mode).

        Returns:
            Modulated feature map with identical shape to ``radar_features``.
        """
        # Ablation / pure-radar safety pass-through
        if x_nwp is None or (x_nwp == 0).all():
            return radar_features

        b = radar_features.shape[0]
        c = self.radar_feat_dim

        # 1. Quantum state projection (low-rank tensor train mapping)
        #
        # The two slices are the analysis hour and the forecast hour. An
        # alignment that yields a different number of slices would index out
        # of range, so the last slice stands in for the forecast hour when
        # there is only one.
        psi_0 = self.state_proj(x_nwp[:, 0])
        psi_1 = self.state_proj(x_nwp[:, -1])

        # 2. Koopman unitary flow tendency estimation
        d_psi = self.koopman_flow(psi_0, psi_1)

        # 3. Predict per-forecast-frame FiLM scale & shift factors
        #
        # The tendency lives on the NWP grid, but the radar features are
        # usually at a different resolution -- an encoder that downsamples
        # twice puts the features at a quarter of the field size. The
        # modulation has to be resampled onto the features, or the
        # multiplication broadcasts against the wrong shape.
        if d_psi.shape[-2:] != radar_features.shape[-2:]:
            d_psi = F.interpolate(
                d_psi, size=radar_features.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        params = self.film_modulator(d_psi).view(b, self.out_len, 2 * c, *d_psi.shape[-2:])
        gamma = torch.tanh(params[:, :, :c])
        beta = params[:, :, c:]

        # 4. Trigger gate for extreme rainfall updrafts
        trigger = self.extreme_trigger_gate(d_psi).unsqueeze(2)  # (B, T_out, 1, H, W)

        # 5. Modulate radar features
        if radar_features.dim() == 5:
            # Matches (B, T_out, C, H, W)
            return radar_features * (1.0 + gamma * (1.0 + trigger)) + beta
        else:
            # Collapsed / single frame (B, C, H, W): average over forecast horizons
            g_mean = gamma.mean(dim=1)
            b_mean = beta.mean(dim=1)
            t_mean = trigger.mean(dim=1)
            return radar_features * (1.0 + g_mean * (1.0 + t_mean)) + b_mean


# ===========================================================================
# Output Heads and Advection Utilities
# ===========================================================================


def floor_at_zero(x: torch.Tensor) -> torch.Tensor:
    """Clamp a tensor at zero while keeping the gradient path alive.

    ``torch.clamp_min(x, 0)`` has an exactly zero derivative below the
    threshold, so every value that sits on the wrong side of the floor is
    cut off from the loss. For a physically floored quantity -- rainfall
    cannot be negative -- that means those pixels are frozen forever: a
    prediction that starts slightly negative can never be taught to become
    positive.

    The straight-through form returns the same *value* as the clamp while
    routing the gradient of the unclamped tensor, so the model is still told
    which way to move.

    Args:
        x: the tensor to floor.

    Returns:
        ``max(x, 0)`` in value, with the gradient of ``x``.
    """
    return x + (x.clamp_min(0.0) - x).detach()


class RainHead(nn.Module):
    """Turn a feature map into non-negative rainfall in millimetres.

    The training target is ``log1p(mm)`` (see :func:`..data.transforms.normalize_rain`),
    so the head predicts in log space and exponentiates at the end. The
    exponentiation makes the output non-negative by construction, which matters
    for CSI-type scores that compare against a physical threshold.

    The final floor at zero is applied with :func:`floor_at_zero`. A hard
    ``clamp_min(0)`` has an exactly zero derivative below the threshold, so
    every pixel whose logit starts slightly negative would be frozen at "no
    rain" for the whole run: at initialisation 42.8 % of the output pixels
    sit on that side of the boundary. The straight-through form returns the
    same values while letting the gradient through, so those pixels can still
    be taught to predict rain.
    """

    def __init__(self, in_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, hidden, 3),
            ConvBlock(hidden, hidden, 3, residual=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = torch.expm1(self.net(x).clamp(-10.0, 6.0))
        return floor_at_zero(raw)


class BalancedExtremePrecipLoss(nn.Module):
    """A loss that aims at the verification score instead of at the mean.

    The benchmark's own loss regresses the log accumulation. That is a
    reasonable thing to minimise and a poor thing to verify: the measured
    behaviour of every model trained under it is that it predicts between
    two and a third of the rain that actually fell. The frequency bias of
    the trained models sits at 0.08 to 0.33, which means the strong-rain
    scores collapse for a reason that has nothing to do with where the rain
    is -- the models simply do not predict enough of it.

    This loss addresses that directly. It keeps a weighted log-space
    regression, which is good at placing the rain, and adds a differentiable
    soft CSI term over several thresholds, which is what the score is
    actually made of. The soft CSI is the fraction of the rain that the
    forecast gets right, computed with a sigmoid in place of the step, so
    the gradient tells the model which way to move.

    Args:
        thresholds: the rain rates in mm per 6-minute accumulation, in
            increasing order.
        alpha_csi: the weight of the soft CSI term against the regression.
    """

    def __init__(
        self,
        thresholds: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 4.0),
        alpha_csi: float = 0.5,
    ) -> None:
        super().__init__()
        self.thresholds = tuple(thresholds)
        self.alpha_csi = alpha_csi

    def forward(
        self,
        pred_rain: torch.Tensor,
        target_rain: torch.Tensor,
        cum_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            pred_rain: ``(B, T, ...)`` predicted rainfall in millimetres.
            target_rain: matching observed rainfall in millimetres.
            cum_probs: the nested exceedance probabilities from
                :class:`QuantumHierarchicalRainHead`, when the head
                provides them. They are not used by the score itself, but
                the argument keeps the two components interchangeable.

        Returns:
            The scalar loss.
        """
        pred = pred_rain.clamp_min(0.0)
        target = target_rain.clamp_min(0.0)

        # The regression is in log space, which is where the magnitudes
        # live, and it is weighted by the observed rain so that the pixels
        # that carry the signal get the gradient.
        pred_log = torch.log1p(pred)
        target_log = torch.log1p(target)
        weight = 1.0 + 2.0 * torch.tanh(target / 2.0)
        reg_loss = torch.mean(
            weight * F.smooth_l1_loss(pred_log, target_log, reduction="none")
        )

        # The soft CSI. The sigmoid sharpens the prediction around the
        # threshold, so the counts below behave like the contingency table
        # the verification would build.
        soft_csi = pred.new_zeros(())
        for i, threshold in enumerate(self.thresholds):
            p_mask = torch.sigmoid((pred - threshold) * 5.0)
            t_mask = (target >= threshold).to(pred.dtype)

            hits = torch.sum(p_mask * t_mask)
            misses = torch.sum((1.0 - p_mask) * t_mask)
            false_alarms = torch.sum(p_mask * (1.0 - t_mask))

            csi = (hits + 1e-5) / (hits + misses + false_alarms + 1e-5)
            # The higher thresholds count more: the extreme rain is where
            # the scores collapse, and a loss that weights every threshold
            # equally spends its gradient on the rain that is already
            # predicted well.
            soft_csi = soft_csi + (1.0 + 0.5 * i) * (1.0 - csi)

        total = reg_loss + self.alpha_csi * soft_csi / len(self.thresholds)
        return total


class QuantumHierarchicalRainHead(nn.Module):
    """A rain head that predicts a nested set of rain rates instead of one.

    The single-channel head regresses a number, and the number it learns
    is systematically too small: the measured frequency bias of the models
    trained with it is 0.08 to 0.33, so the extreme-rain scores collapse.
    The reason is structural. A regression on the log accumulation has no
    way to express "this pixel is very likely to be in the heavy bin" and
    "this pixel is very likely to be dry" at the same time, and the
    squared error between the two pushes it towards the middle.

    This head makes the prediction a distribution over ordered rain bins
    instead. The features are read as a state vector, measured against a
    set of learned measurement directions, and the amplitudes are turned
    into probabilities by Born's rule. Because the exceedance probability
    of a bin is the tail sum of the bin probabilities, the result is
    nested by construction:

        P(R >= 0.1) >= P(R >= 0.5) >= P(R >= 1.0) >= P(R >= 2.0) >= P(R >= 4.0)

    which is a property of the arithmetic rather than of the training, so
    the model cannot learn an inconsistent set of probabilities.

    The rain rate is then the expectation over that distribution, so the
    heavy bin pulls the mean up and the head is free to commit to it
    instead of hedging towards the middle of the range.

    .. note::
       **The number of bins has to be one more than the number of
       thresholds.** The tail sum over ``num_bins`` probabilities gives
       ``num_bins`` exceedance probabilities, and the first of them is
       ``P(R >= 0)``, which is identically one -- it cannot move, so a
       loss that supervises it contributes a constant and no gradient.
       The last threshold needs its own bin to be represented at all.

       With ``num_bins=6`` and the thresholds ``(0.1, 0.5, 1.0, 2.0,
       4.0)`` the bins are the intervals ``[0,0.1) [0.1,0.5) [0.5,1.0)
       [1.0,2.0) [2.0,4.0) [4.0,inf)``, the tail sums are exactly the
       five exceedance probabilities, and dropping the first leaves them
       aligned one-to-one with the thresholds.

    Args:
        in_channels: the channels of the feature map.
        hidden_dim: the width of the state vector.
        num_bins: the number of rain bins. One more than the number of
            thresholds is what makes the alignment work.
        thresholds: the bin boundaries in mm per 6-minute accumulation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_bins: int = 6,
        thresholds: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 4.0),
        level_mode: str = "floor",
    ) -> None:
        super().__init__()
        if num_bins < len(thresholds):
            raise ValueError(
                f"{len(thresholds)} thresholds need at least "
                f"{len(thresholds)} bins, got {num_bins}"
            )
        if level_mode not in ("midpoint", "floor"):
            raise ValueError(
                f"level_mode must be 'midpoint' or 'floor', got {level_mode!r}"
            )
        self.num_bins = num_bins
        self.thresholds = tuple(thresholds)
        self.level_mode = level_mode

        # The thresholds the probability channels actually answer.
        #
        # With one bin per threshold plus the open-ended one, the tail
        # sums are exactly the exceedance probabilities of the thresholds
        # and the first of them -- P(R >= 0), which is identically one --
        # is the one to drop. The channels then answer the thresholds
        # one to one.
        #
        # With exactly as many bins as thresholds there is no room for
        # that extra bin: the first tail sum is P(R >= 0) again, so the
        # channels answer (0, 0.1, 0.5, 1.0, 2.0) and every threshold
        # is off by one against its probability. The head publishes what
        # it answers rather than what it was asked for, so a loss that
        # looks the thresholds up by value cannot be misled into reading
        # the wrong channel.
        if self.num_bins > len(self.thresholds):
            self.channel_thresholds = tuple(self.thresholds)
        else:
            self.channel_thresholds = (0.0,) + tuple(self.thresholds[:-1])

        # The state preparation: the feature map becomes a vector in a
        # Hilbert space, one vector per pixel.
        self.state_prep = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # The measurement directions: one per rain bin. They are the
        # projectors of the measurement, and their being learned rather
        # than fixed is what lets the head choose where the bin
        # boundaries fall in the feature space.
        self.measurement_bases = nn.Parameter(
            torch.randn(num_bins, hidden_dim) * 0.02
        )

        # The intensity of each bin. The rain rate is the expectation of
        # the distribution over these, so the head can commit to the heavy
        # bins rather than averaging them away.
        self.intensity_offsets = nn.Sequential(
            nn.Conv2d(hidden_dim, num_bins, kernel_size=3, padding=1),
            nn.Softplus(),
        )

        # The exceedance probabilities of the last forward pass, aligned
        # with the thresholds. The head is the only thing that knows them,
        # so it keeps them; the loss reads them from here rather than
        # from a second forward pass.
        self.last_cum_probs: Optional[torch.Tensor] = None

    def _bin_levels(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """The representative rain rate of each bin.

        The bins are the intervals between the thresholds, with a dry bin
        below the first one and an open-ended bin above the last. The
        levels are the midpoints of those intervals, so a pixel whose
        rain the head places in a bin reconstructs to a rate inside that
        bin, and the score at a threshold is decided by which bin the
        head chose rather than by the rounding.

        The dry bin is the exception: the label's smallest positive value
        is a tenth of a millimetre, so the bin holds only zeros and its
        level is zero.

        Args:
            device: the device the tensor is built on.
            dtype: the dtype of the tensor.

        Returns:
            ``(1, num_bins, 1, 1)`` the level of each bin.
        """
        # The edges of the bins. The thresholds are the interior edges,
        # and the open-ended top bin is taken to span the same width as
        # the bin below it, so that it has a midpoint as well.
        edges = [0.0] + list(self.thresholds)
        if len(edges) > self.num_bins:
            edges = edges[: self.num_bins]
        while len(edges) < self.num_bins + 1:
            width = edges[-1] - edges[-2] if len(edges) >= 2 else 1.0
            edges.append(edges[-1] + width)

        if self.level_mode == "floor":
            levels = list(edges[: self.num_bins])
        else:
            levels = [
                0.5 * (edges[k] + edges[k + 1]) for k in range(self.num_bins)
            ]
        levels[0] = 0.0

        return torch.tensor(
            levels, device=device, dtype=dtype
        ).view(1, -1, 1, 1)

    def forward(
        self, feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the rain and the nested exceedance probabilities.

        Args:
            feat: ``(B, C, H, W)`` the backbone's feature map.

        Returns:
            ``pred_rain`` of shape ``(B, 1, H, W)`` in millimetres, and
            ``cum_probs`` of shape ``(B, len(thresholds), H, W)`` holding
            ``P(R >= threshold_k)``, which decreases with k.
        """
        # The state vector. Normalising it keeps the measurement a
        # probability: the amplitudes of a unit state against orthonormal
        # directions sum to one.
        psi = self.state_prep(feat)
        psi_norm = F.normalize(psi, p=2, dim=1)

        bases_norm = F.normalize(self.measurement_bases, p=2, dim=1)
        inner_prod = torch.einsum("bchw,kc->bkhw", psi_norm, bases_norm)
        probs = F.softmax(inner_prod.square() * 10.0, dim=1)

        # The exceedance probability of bin k is the tail sum of the bin
        # probabilities from k upwards, so it is nested without anything
        # having to enforce it.
        cum_all = torch.flip(
            torch.cumsum(torch.flip(probs, dims=[1]), dim=1), dims=[1]
        )
        # The alignment with the thresholds depends on how many bins there
        # are.
        #
        # With one bin per threshold plus the open-ended top one, the
        # tail sums are exactly the exceedance probabilities of the
        # thresholds, and the first of them -- P(R >= 0), which is
        # identically one -- is the one to drop. That is the aligned
        # case, and it is the default.
        #
        # With exactly as many bins as thresholds there is no room for
        # that extra bin: the first tail sum is P(R >= 0) again, and the
        # thresholds are off by one against the probabilities. The
        # probabilities are still nested and still usable, they just
        # answer a different question, and the head is built to allow
        # that so the difference can be measured rather than argued
        # about.
        if self.num_bins > len(self.thresholds):
            cum_probs = cum_all[:, 1:]
        else:
            cum_probs = cum_all
        self.last_cum_probs = cum_probs

        # The rain rate is the expectation over the distribution.
        #
        # The representative value of each bin decides whether the rate
        # can cross a threshold at all, and the verification asks exactly
        # that: whether the prediction is at or above a threshold.
        #
        # With the lower edge of the bin as the representative, the bin
        # that begins at 2.0 mm is represented by 2.0 mm -- the threshold
        # itself. A pixel the head is certain about therefore
        # reconstructs to exactly the threshold, and any leakage of
        # probability into a lower bin pulls it under: the expectation of
        # a distribution whose mass sits on a bin edge and leaks slightly
        # below it is below the edge. Measured on the archive, the
        # field-wide tier loss produced a maximum prediction of
        # 1.999797 mm, with 1.51 per cent of the pixels within a
        # hundredth of a millimetre below the 2.0 mm threshold and none
        # above it, so the score at 2.0 mm was exactly zero however much
        # probability mass the head put on the heavy bins.
        #
        # The midpoint of the bin is the representative that makes the
        # reconstruction agree with the measurement. A pixel whose rain
        # falls in the bin above a threshold reconstructs to a rate above
        # it, and one that falls below reconstructs to a rate below it,
        # so the score at a threshold is decided by which bin the head
        # chose rather than by the rounding. It is also the minimum
        # squared-error estimate of a value known only to lie in the bin.
        #
        # The open-ended top bin has no midpoint, so the bin above it is
        # taken to span the same width as the bin below it.
        offsets = self.intensity_offsets(psi)
        bin_levels = self._bin_levels(feat.device, feat.dtype)
        rain_levels = bin_levels + offsets
        pred_rain = torch.sum(probs * rain_levels, dim=1, keepdim=True)

        return floor_at_zero(pred_rain), cum_probs


def make_warp_grid(
    flow: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    """Build the sampling grid for a differentiable advection warp.

    Args:
        flow: ``(B, 2, H, W)`` displacement in pixels, ordered ``(dy, dx)``.
        size: output spatial size ``(H, W)``.

    Returns:
        ``(B, H, W, 2)`` grid for :func:`torch.nn.functional.grid_sample`.
    """
    b = flow.shape[0]
    h, w = size
    ys, xs = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    base = torch.stack([ys, xs], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
    warped = base + flow
    grid = torch.stack([warped[:, 1], warped[:, 0]], dim=-1)
    max_y = h - 1
    max_x = w - 1
    grid[..., 0] = 2.0 * grid[..., 0] / max(max_x, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(max_y, 1) - 1.0
    return grid


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Advect ``x`` ``(B, C, H, W)`` by ``flow`` ``(B, 2, H, W)``."""
    grid = make_warp_grid(flow, x.shape[-2:])
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="border", align_corners=True
    )

class RadarNWPToRainQPE(nn.Module):
    """The inversion branch: radar plus NWP state to a rain rate.

    The NWP's moisture and thermodynamic fields correct the reflectivity
    to rain rate, which is a much easier task for them than predicting
    where a convective cell will be an hour from now.
    """

    def __init__(self, in_radar_c: int = 10, in_nwp_c: int = 27, hid_c: int = 64):
        super().__init__()
        self.qpe_enc = nn.Sequential(
            ConvBlock(in_radar_c + in_nwp_c, hid_c, 3),
            ConvBlock(hid_c, hid_c, 3),
        )

    def forward(
        self, x_radar: torch.Tensor, nwp_analysis: torch.Tensor
    ) -> torch.Tensor:
        """Invert one batch of radar histories.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            nwp_analysis: ``(B, C_nwp, H, W)`` the analysis-hour state.

        Returns:
            ``(B * T_in, hid_c, H, W)`` the inversion features.
        """
        b, t_in, c_r, h, w = x_radar.shape
        nwp_exp = nwp_analysis.unsqueeze(1).expand(-1, t_in, -1, -1, -1)
        x_in = torch.cat([x_radar, nwp_exp], dim=2).reshape(b * t_in, -1, h, w)
        return self.qpe_enc(x_in)


class QPEQPFBackbone(nn.Module):
    """The shared skeleton of the QPE-QPF split.

    Two stages. The first inverts each of the history frames into a rain
    rate and publishes it; the second extrapolates from the inverted
    history. The two stages share one rainfall head.

    The shared head is the mechanism. The head that produces the
    history estimate can be supervised against observations that carry
    no displacement error, so it stays anchored to the true rain
    distribution and keeps its sharpness. The head that produces the
    forecast is the same head, so it inherits that anchoring instead of
    learning to hedge against the future frames' position error.

    This lives here rather than in one of the model packages because
    two models share it, and the model packages are loaded by path --
    importing across them would register the same class twice.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
    ):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.hid_c = hid_c

        self.qpe_net = RadarNWPToRainQPE(in_radar_c, in_nwp_c, hid_c)
        # One head for both stages.
        self.shared_rain_head = RainHead(hid_c)

        # The forecast stage works in the space of the inverted rainfall:
        # the inversion features, the inverted rain, the original radar,
        # and the forecast-hour NWP state.
        qpf_in_c = hid_c + 1 + in_radar_c + in_nwp_c
        self.spatial_down = ConvBlock(qpf_in_c, hid_c * 2, 3, stride=2)

        self.translator = nn.Sequential(
            nn.Conv2d(in_len * hid_c * 2, hid_c * 4, 1),
            nn.GroupNorm(_group_count(hid_c * 4), hid_c * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid_c * 4, hid_c * 4, 3, padding=1, groups=hid_c * 4),
            nn.GroupNorm(_group_count(hid_c * 4), hid_c * 4),
            nn.SiLU(inplace=True),
        )

        # The time axis is projected out here, the way the backbone does
        # it: the convolution produces ``out_len * C_f`` channels, which
        # reshape into one feature map per forecast frame.
        self.out_time_proj = nn.Conv2d(hid_c * 4, out_len * hid_c * 2, 1)

        self.spatial_up = nn.Sequential(
            nn.ConvTranspose2d(hid_c * 2, hid_c, 3, stride=2, padding=1,
                               output_padding=1),
            nn.GroupNorm(_group_count(hid_c), hid_c),
            nn.SiLU(inplace=True),
        )

        #: The inverted history rainfall, for the objective to read.
        self.last_qpe_pred: Optional[torch.Tensor] = None

    def invert(
        self, x_radar: torch.Tensor, nwp_analysis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert the history into a rainfall field.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            nwp_analysis: ``(B, C_nwp, H, W)``.

        Returns:
            The inversion features ``(B * T_in, hid_c, H, W)`` and the
            inverted rainfall ``(B, T_in, 1, H, W)``.
        """
        qpe_feat_flat = self.qpe_net(x_radar, nwp_analysis)
        qpe_rain_flat = self.shared_rain_head(qpe_feat_flat)
        b, t_in = x_radar.shape[:2]
        h, w = x_radar.shape[-2:]
        self.last_qpe_pred = qpe_rain_flat.view(b, t_in, 1, h, w)
        return qpe_feat_flat, self.last_qpe_pred

    def extrapolate(
        self,
        x_radar: torch.Tensor,
        qpe_feat_flat: torch.Tensor,
        history_rain: torch.Tensor,
        nwp_forecast: torch.Tensor,
    ) -> torch.Tensor:
        """Extrapolate the inverted history into the future.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            qpe_feat_flat: the inversion features.
            history_rain: ``(B, T_in, 1, H, W)`` the rainfall the forecast
                branch starts from. It may be the inverted estimate or an
                observed field, or a mixture of the two.
            nwp_forecast: ``(B, C_nwp, H, W)`` the forecast-hour state.

        Returns:
            ``(B, T_out, 1, H, W)`` in millimetres.
        """
        b, t_in, _, h, w = x_radar.shape
        qpe_feat = qpe_feat_flat.view(b, t_in, self.hid_c, h, w)
        rain_log = torch.log1p(history_rain.clamp_min(0.0))
        nwp_exp = nwp_forecast.unsqueeze(1).expand(-1, t_in, -1, -1, -1)

        qpf_in = torch.cat([qpe_feat, rain_log, x_radar, nwp_exp], dim=2)
        down_feat = self.spatial_down(qpf_in.reshape(b * t_in, -1, h, w))
        _, c_down, h_d, w_d = down_feat.shape

        fut_down = self.translator(down_feat.view(b, t_in * c_down, h_d, w_d))
        fut_down = self.out_time_proj(fut_down).view(
            b * self.out_len, self.hid_c * 2, h_d, w_d
        )
        fut_up = self.spatial_up(fut_down)
        return floor_at_zero(
            self.shared_rain_head(fut_up).view(b, self.out_len, 1, h, w)
        )
