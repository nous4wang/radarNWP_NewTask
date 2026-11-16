"""Hybrid Quantum Manifold Architecture: shallow fusion, deep tendency,
hierarchical projection head.

This model exists to answer one measurement. The comparison matrix showed
that every deep model on this benchmark predicts between two and a third
of the rain that actually fell -- the frequency bias sat at 0.08 to 0.33
across the family -- and that the extreme-rain scores collapse as a
result. The single-channel regression head cannot fix that on its own: it
is asked for one number per pixel, and the squared error between "no
rain" and "very heavy rain" pulls that number towards the middle.

So this architecture changes two things at once, and keeps the third
thing that already worked.

What it keeps: the shallow concatenation. M1 -- the plain concatenation
of the radar and the NWP channels -- was the only NWP variant that was
not behind its backbone, so the spatial gradient of the NWP reaches the
first convolution directly. That is preserved here: the radar and the NWP
are concatenated at the input, before any bottleneck.

What it adds: a deep dynamical tendency. The Koopman flow reads the two
NWP slices -- the analysis hour and the forecast hour -- and produces a
tendency field that modulates the latent sequence, so the forecast is
told what the atmosphere is doing rather than only what it looks like
right now.

What it replaces: the output head. The single-channel regression becomes
a measurement over five ordered rain bins
(:class:`~radar_nwp_bench.models.common.QuantumHierarchicalRainHead`).
The prediction is a distribution, the exceedance probabilities are
nested by construction, and the rain rate is the expectation over that
distribution -- so the heavy bin can pull the mean up instead of being
averaged away.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import (
    ConvBlock,
    KoopmanUnitaryFlow,
    QuantumHierarchicalRainHead,
    QuantumStateProjector,
)


@register_model("simvp_v2_hqma")
class SimVPv2HybridQuantumManifold(BasePrecipitationModel):
    """SimVP-v2 with a hierarchical quantum rain head and a deep tendency.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: radar history frames.
        out_len: forecast frames.
        hid_c: width of the latent feature maps.
        bond_dim: the bond dimension of the quantum state projector.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        bond_dim: int = 32,
        nwp_slices: int = 2,
        head_bins: int = 6,
        level_mode: str = "midpoint",
    ):
        super().__init__(in_radar_c, in_nwp_c, in_len, out_len)
        self.in_len = in_len
        self.out_len = out_len

        # 1. 浅层双干路混合输入：雷达与 NWP 直接拼接，不设信息瓶颈
        # The NWP contributes as many channels as it has slices times its
        # variables. The width is fixed at construction, so the number of
        # slices has to be known then: the alignment policies differ in how
        # many they return, and the concatenation cannot be resized after
        # the fact. The default of two is the forecast policy's -- the
        # analysis hour and the forecast hour -- which is what the
        # benchmark uses everywhere.
        self.nwp_slices = nwp_slices
        total_in = in_radar_c + in_nwp_c * nwp_slices
        self.spatial_enc = nn.Sequential(
            ConvBlock(total_in, hid_c, 3),
            ConvBlock(hid_c, hid_c * 2, 3, stride=2),
        )

        # 2. 隐藏时空流形转换器
        #
        # The depthwise convolution keeps the translation local -- each
        # channel is mixed with its own neighbourhood and nothing else --
        # which is what a tendency field needs, because the tendency is a
        # local dynamical signal and a fully connected projection over the
        # whole history would smear it.
        self.in_time_proj = nn.Conv2d(in_len * hid_c * 2, hid_c * 4, 1)
        self.translator = nn.Sequential(
            nn.Conv2d(hid_c * 4, hid_c * 4, 3, padding=1, groups=hid_c * 4),
            nn.GroupNorm(min(32, hid_c * 4), hid_c * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid_c * 4, out_len * hid_c * 2, 1),
        )

        # 3. 动力倾向量子演化
        self.state_proj = QuantumStateProjector(
            in_vars=in_nwp_c, bond_dim=bond_dim, embed_dim=hid_c * 2
        )
        self.koopman_flow = KoopmanUnitaryFlow(state_dim=hid_c * 2)
        self.tendency_film = nn.Conv2d(
            hid_c * 2, hid_c * 2 * out_len, kernel_size=3, padding=1
        )

        # 4. 空间解码器
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hid_c * 2, hid_c, 3, stride=2, padding=1,
                               output_padding=1),
            nn.GroupNorm(min(32, hid_c), hid_c),
            nn.SiLU(inplace=True),
        )

        # 5. 极端降水层级量子 POVM 预测头
        #
        # The number of bins is one more than the number of thresholds, so
        # that every threshold has a bin of its own and the exceedance
        # probabilities line up with the thresholds one to one. The
        # five-bin version is kept for the comparison that measures what
        # the alignment is worth.
        self.head_bins = head_bins
        self.quantum_povm_head = QuantumHierarchicalRainHead(
            in_channels=hid_c, hidden_dim=64, num_bins=head_bins,
            level_mode=level_mode,
        )

        # The exceedance probabilities of the last forward pass, reshaped
        # to (B, T_out, K, H, W). The loss reads them from here.
        self.last_cum_probs: Optional[torch.Tensor] = None

    def forward(
        self, x_radar: torch.Tensor, x_nwp: torch.Tensor
    ) -> torch.Tensor:
        """Forecast the rainfall.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)``, or ``None``.

        Returns:
            ``(B, T_out, 1, H, W)`` in millimetres.
        """
        b, t_in, _, h, w = x_radar.shape
        # Whether the NWP was actually supplied has to be recorded before
        # resolve_nwp turns a missing one into an all-zero tensor.
        has_nwp = x_nwp is not None
        x_nwp = self.resolve_nwp(x_radar, x_nwp)

        if x_nwp.shape[1] != self.nwp_slices:
            raise ValueError(
                f"this model was built for {self.nwp_slices} NWP slices but "
                f"the data carries {x_nwp.shape[1]}. The shallow "
                f"concatenation fixes the input width, so the two have to "
                f"agree; build the model with nwp_slices="
                f"{x_nwp.shape[1]} or use the 'forecast' alignment."
            )

        # A. 浅层直连特征提取
        # (B, T_nwp, C, H, W) -> (B, 1, T_nwp*C, H, W) -> broadcast over the
        # history frames. The NWP is a single state per sample, so it is
        # repeated rather than indexed per frame.
        nwp_flat = x_nwp.reshape(b, 1, -1, h, w).expand(-1, t_in, -1, -1, -1)
        x_cat = torch.cat([x_radar, nwp_flat], dim=2)
        feats = self.spatial_enc(x_cat.reshape(b * t_in, -1, h, w))
        _, c_f, h_f, w_f = feats.shape

        # B. 时空演变基础潜态
        latent = self.in_time_proj(feats.reshape(b, t_in * c_f, h_f, w_f))
        latent = self.translator(latent)
        latent_frames = latent.reshape(b, self.out_len, c_f, h_f, w_f)

        # C. 深度量子倾向调制
        # The tendency is read from the two NWP slices -- the analysis
        # hour and the forecast hour -- and applied per forecast frame.
        #
        # It is skipped when the NWP is withheld, which the ablation does
        # by handing the model no NWP at all. The check has to happen
        # before resolve_nwp, because resolve_nwp turns a missing NWP into
        # an all-zero tensor, and a zero tensor is a state whose evolution
        # is zero -- reading it would apply a tendency that says nothing
        # about the atmosphere.
        if has_nwp and x_nwp.shape[1] >= 2:
            psi_0 = self.state_proj(
                F.interpolate(x_nwp[:, 0], size=(h_f, w_f), mode="bilinear",
                              align_corners=False)
            )
            psi_1 = self.state_proj(
                F.interpolate(x_nwp[:, 1], size=(h_f, w_f), mode="bilinear",
                              align_corners=False)
            )
            d_psi = self.koopman_flow(psi_0, psi_1)
            dyn_bias = self.tendency_film(d_psi).reshape(
                b, self.out_len, c_f, h_f, w_f
            )
            latent_frames = latent_frames * (1.0 + torch.tanh(dyn_bias))

        # D. 空间解码上采样
        dec_feats = self.dec(latent_frames.reshape(b * self.out_len, c_f, h_f, w_f))

        # E. 量子投影测度头输出
        #
        # The head returns the rainfall and the exceedance probabilities.
        # The forward pass returns the rainfall alone, so that the
        # evaluation script, the tests and the distributed path all keep
        # the single-tensor contract they already have; the
        # probabilities are published on the model for the loss to read.
        pred_rain, cum_probs = self.quantum_povm_head(dec_feats)
        self.last_cum_probs = cum_probs.reshape(
            b, self.out_len, cum_probs.shape[1], *cum_probs.shape[-2:]
        )
        return pred_rain.reshape(b, self.out_len, 1, h, w)
