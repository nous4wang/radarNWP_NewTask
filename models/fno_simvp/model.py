"""E44 / MASTER 4.46 -- A4: the operator-learning branch (an FNO decoder).

The encoder, the time projection and the translator are ``simvp_v2``'s.
What changes is the OPERATOR that turns the latent state into the
forecast: the convolutional decoder is replaced by a Fourier Neural
Operator layer.

The reason this branch exists is the one quantity no mechanism has
touched yet.  Across the whole project the radar field and the label
field correlate at r = +0.17 per pixel, which is the cross-modal gap;
E33 measured that the spatial ARRANGEMENT of the small scales carries
61 % of the recoverable information.  A convolutional decoder has a
spatially local receptive field, so it cannot express a global phase
offset: moving a rain region by tens of pixels is invisible to a stack
of 3x3 kernels that have to be slid across it.  An FNO layer applies a
learned operator in the spectral domain, and the lowest frequency modes
are exactly what carries a large-scale displacement.

The layer keeps only the lowest ``mode_keep`` frequency modes of each
axis.  That is what makes the parameter count comparable with the
convolutional decoder it replaces: a dense spectral MLP over every mode
of a 64x64 grid would have a hundred times the parameters, and the
point of the branch is the operator, not the capacity.

What is NOT changed: the backbone, the loss, the output contract, the
number of loss terms.

The S0 gate checks that with ``use_fno`` off this module is bit-equal to
``simvp_v2``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import ConvNeXtBlock, floor_at_zero


class SpectralMLP(nn.Module):
    """The learnable operator of the FNO layer.

    Two mixing steps, both of them in the spectral domain: the weights
    are mixed across the retained modes, and the values are mixed
    across the channels.

    Args:
        modes: number of retained modes per axis.
        channels: number of channels.
        hidden: width of the channel mixing.
    """

    def __init__(self, modes: int, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.modes = int(modes)
        self.channels = int(channels)
        self.weight = nn.Parameter(
            torch.randn(self.modes, self.modes, self.channels, self.channels)
            / (self.channels ** 0.5)
        )
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(self.channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, self.channels, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the operator to a spectral tensor.

        Args:
            z: ``(B, C, modes, modes)`` complex tensor.

        Returns:
            The same shape, complex.
        """
        mh, mw = z.shape[-2], z.shape[-1]
        # The weight is defined for the nominal mode count; a smaller
        # field keeps the corresponding corner of it.
        #
        # The weight is indexed [out_mode_y, out_mode_x, out_channel,
        # in_channel] and the contraction runs over the input mode axes
        # and the input channel axis:
        #
        #     out[b, l, i, j] = sum_{c, k} w[i, j, l, c] * z[b, c, k, j]
        #
        # The input's second mode axis is contracted with the weight's
        # second mode axis, and the first mode axis of the output is the
        # weight's first mode axis.
        w = self.weight[:mh, :mw].to(z.dtype)
        # z: (B, C, k, j) -> transpose to (B, k, j, C) so the mode axes
        # and the channel axis are adjacent.
        zt = z.permute(0, 2, 3, 1)                       # (B, k, j, C)
        out = torch.einsum("ijlc,bkjc->bikl", w, zt)      # (B, i, k, l)
        out = out.permute(0, 3, 2, 1)                     # (B, l, k, i)
        # Mix the channels.
        out = out + self.channel_mlp(out.real) + 1j * self.channel_mlp(out.imag)
        return out


@register_model("fno_simvp")
class FNOSimVPModel(BasePrecipitationModel):
    """``simvp_v2`` with a Fourier Neural Operator decoder.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        hid_c: channel width.
        num_blocks: translator depth.
        mode_keep: number of frequency modes retained per axis.
        use_fno: when False the convolutional decoder is used and the
            module reproduces ``simvp_v2`` exactly.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
        mode_keep: int = 16,
        use_fno: bool = True,
    ) -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.hid_c = int(hid_c)
        self.mode_keep = int(mode_keep)
        self.use_fno = bool(use_fno)

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

        # The convolutional decoder, kept for the S0 path and as the
        # upsampling stem of the FNO path.
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

        # The FNO operator, applied at the decoder's output resolution.
        self.spectral = SpectralMLP(self.mode_keep, hid_c, hidden=hid_c)

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

        if not self.use_fno:
            # The S0 path: exactly simvp_v2's decoder on exactly
            # simvp_v2's features.
            out = self.dec(future_feats)
            log_mm = out.view(B, self.out_len, 1, H, W)
            return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))

        # --- the FNO decoder ---
        # The convolutional stem still does the upsampling; the operator
        # then acts in the spectral domain on the upsampled field, which
        # is where a global phase shift can be expressed.
        body = self.dec[:-1](future_feats)            # (B*T, hid_c, H, W)
        amp = self.dec[-1](body)                      # (B*T, 1, H, W)

        z = torch.fft.rfft2(body.float(), norm="ortho")       # (B*T, C, H, W//2+1)
        mh = min(self.mode_keep, z.shape[-2])
        mw = min(self.mode_keep, z.shape[-1])
        z = z[..., :mh, :mw]
        z = self.spectral(z)
        # Restore the discarded modes by zero padding them back.
        full = torch.zeros(z.shape[0], z.shape[1], z.shape[-2], body.shape[-1],
                           dtype=torch.float32, device=z.device)
        full[..., :mh, :mw] = z.real
        full_imag = torch.zeros_like(full)
        full_imag[..., :mh, :mw] = z.imag
        recon = torch.fft.irfft2(torch.complex(full, full_imag), s=body.shape[-2:],
                                 norm="ortho")

        # The spectral reconstruction carries the operator's contribution;
        # the convolutional head keeps the direct path so the model is
        # not asked to rebuild everything from the spectrum.
        out = amp + recon.mean(dim=1, keepdim=True)
        log_mm = out.view(B, self.out_len, 1, H, W)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))
