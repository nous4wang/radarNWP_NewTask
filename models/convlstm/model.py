"""ConvLSTM Baseline for Radar-NWP precipitation nowcasting."""
import torch
import torch.nn as nn
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.common import floor_at_zero


class ConvLSTMCell(nn.Module):
    def __init__(self, in_c: int, hid_c: int, kernel_size: int = 3):
        super().__init__()
        self.hid_c = hid_c
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_c + hid_c, 4 * hid_c, kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hid_c, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


@register_model("convlstm")
class ConvLSTMModel(BasePrecipitationModel):
    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.num_layers = num_layers
        self.hid_c = hid_c

        # 多模态早期融合投射: Radar (10) + NWP 2个时效展平 (2 * 27 = 54)
        total_in = in_radar_c + in_nwp_c * 2
        self.in_proj = nn.Conv2d(total_in, hid_c, kernel_size=1)

        self.cell_list = nn.ModuleList([
            ConvLSTMCell(hid_c if i == 0 else hid_c, hid_c) for i in range(num_layers)
        ])
        self.out_head = nn.Sequential(
            nn.Conv2d(hid_c, hid_c // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid_c // 2, 1, kernel_size=1),
        )
        # The head predicts in log space and is exponentiated back to
        # millimetres at the end, with a straight-through floor.
        #
        # Two things were wrong here before, and either one alone is enough
        # to stop the model learning:
        #
        # * A hard nn.ReLU() ended the head. The target is log1p(mm), so
        #   nearly every pixel starts below the threshold, ReLU returns zero
        #   and its derivative is zero there too -- the output head never
        #   received a gradient and the model predicted one constant field
        #   for the whole run. Measured on the real archive: the training
        #   loss sat at 0.1502 while a constant "no rain" forecast scores
        #   0.1543, so the model was *worse* than saying nothing, and its
        #   CSI@0.5 was exactly 0.
        #
        # * The head predicted millimetres directly. The target ranges over
        #   three orders of magnitude (0.001 mm typical, 2.5 mm in a storm
        #   core), and the loss compares them in log space, so a prediction
        #   made in millimetres spends its capacity on the few heavy pixels
        #   and predicts a flat light drizzle everywhere else. The models in
        #   this repository that do learn predict in log space for exactly
        #   this reason.
        nn.init.zeros_(self.out_head[2].bias)

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        B, T_in, _, H, W = x_radar.shape
        # NWP 沿通道广播
        nwp_flat = x_nwp.view(B, 1, -1, H, W).expand(-1, T_in, -1, -1, -1)
        x_concat = torch.cat([x_radar, nwp_flat], dim=2)

        h = [torch.zeros(B, self.hid_c, H, W, device=x_radar.device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hid_c, H, W, device=x_radar.device) for _ in range(self.num_layers)]

        # 编码阶段
        for t in range(self.in_len):
            cur = self.in_proj(x_concat[:, t])
            for l in range(self.num_layers):
                h[l], c[l] = self.cell_list[l](cur, h[l], c[l])
                cur = h[l]

        # 自回归解码阶段
        preds = []
        cur_input = h[-1]
        for _ in range(self.out_len):
            cur = cur_input
            for l in range(self.num_layers):
                h[l], c[l] = self.cell_list[l](cur, h[l], c[l])
                cur = h[l]
            out_frame = self.out_head(cur)
            preds.append(out_frame)
            cur_input = cur

        # The floor is applied outside the head, with a straight-through
        # gradient, so a pixel that starts below zero can still be taught to
        # rain. See the note on the output head above.
        log_mm = torch.stack(preds, dim=1)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))