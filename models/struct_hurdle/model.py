"""E41 / MASTER 4.43 -- one backbone, two output parameterisations.

``head_mode="rain"`` is **line-for-line identical** to
``models/simvp_v2/model.py::SimVPv2Model``: same module names, same parameter
shapes, same forward arithmetic.  Loading a ``simvp_v2`` state dict into this
model and feeding the same input must give bit-identical output -- that is
E41's S0 gate, and it is what makes A2 a genuine single-variable ablation.

``head_mode="hurdle"`` changes exactly one thing: the decoder's final
convolution emits two channels instead of one,

    occurrence logit -> p = sigmoid(logit)          P(mm >= threshold)
    conditional amount -> a = expm1(log_amount)     E[mm | rain]
    prediction = p * a                              E[mm | X]

The output contract stays ``(B, 10, 1, H, W)`` millimetres, so the evaluator,
the threshold contingency tables and ``--dump-fields`` are untouched.  The
two extra tensors are cached on the model for ``--loss hurdle`` to consume.

This is a CHANGE OF PARAMETERISATION, not a claim of extra information: a
hurdle model still predicts a conditional expectation.  What it changes is
that the 91 % dry mass and the heavy amplitude tail are no longer forced
through one dense Gaussian regression per pixel.  Whether that improves
CSI@2.0 is what E41 S1/S3 measures; nothing here asserts that it does.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import ConvNeXtBlock, floor_at_zero


@register_model("struct_hurdle")
class StructHurdleModel(BasePrecipitationModel):
    """SimVP-v2 backbone with a switchable rain / hurdle output head."""

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
        head_mode: str = "rain",
        occ_prior_logit: float = -2.0,
    ) -> None:
        super().__init__()
        if head_mode not in ("rain", "hurdle"):
            raise ValueError(f"head_mode must be 'rain' or 'hurdle', got {head_mode!r}")
        self.in_len = in_len
        self.out_len = out_len
        self.head_mode = head_mode

        total_in = in_radar_c + in_nwp_c * 2
        out_c = 1 if head_mode == "rain" else 2

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
            nn.ConvTranspose2d(
                hid_c * 2, hid_c, kernel_size=3, stride=2,
                padding=1, output_padding=1,
            ),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, out_c, kernel_size=1),
        )

        if head_mode == "hurdle":
            # Wet-pixel prior: sigmoid(-2) = 0.119, close to the observed
            # 1 - zero_frac = 0.088 on the strongest frames.  This is a
            # structural prior, not a tuned hyper-parameter.
            with torch.no_grad():
                self.dec[-1].bias[0] = float(occ_prior_logit)
                self.dec[-1].bias[1] = 0.0

        self.last_occ_logit: torch.Tensor | None = None
        self.last_amount: torch.Tensor | None = None

    def forward(
        self, x_radar: torch.Tensor, x_nwp: torch.Tensor
    ) -> torch.Tensor:
        b, t_in, _, h, w = x_radar.shape
        nwp_flat = x_nwp.view(b, 1, -1, h, w).expand(-1, t_in, -1, -1, -1)
        x_concat = torch.cat([x_radar, nwp_flat], dim=2)

        feats = self.enc(x_concat.view(b * t_in, -1, h, w))
        _, c_f, h_f, w_f = feats.shape
        feats_time = feats.view(b, t_in * c_f, h_f, w_f)
        hid = self.in_time_proj(feats_time)
        hid = self.translator(hid)
        future_feats = self.out_time_proj(hid).view(
            b * self.out_len, c_f, h_f, w_f
        )
        out = self.dec(future_feats).view(
            b, self.out_len, -1, h, w
        )

        if self.head_mode == "rain":
            # Bit-identical to simvp_v2.
            self.last_occ_logit = None
            self.last_amount = None
            return floor_at_zero(torch.expm1(out.clamp(-10.0, 6.0)))

        occ_logit = out[:, :, :1]
        amount = floor_at_zero(torch.expm1(out[:, :, 1:2].clamp(-10.0, 6.0)))
        self.last_occ_logit = occ_logit
        self.last_amount = amount
        return torch.sigmoid(occ_logit) * amount
