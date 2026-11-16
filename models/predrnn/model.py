"""PredRNN Baseline with Spatiotemporal LSTM (ST-LSTM) cells."""
import torch
import torch.nn as nn
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.common import floor_at_zero


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_c: int, hid_c: int, kernel_size: int = 3):
        super().__init__()
        self.hid_c = hid_c
        padding = kernel_size // 2
        self.conv_x = nn.Conv2d(in_c, 7 * hid_c, kernel_size, padding=padding, bias=False)
        self.conv_h = nn.Conv2d(hid_c, 4 * hid_c, kernel_size, padding=padding, bias=False)
        self.conv_m = nn.Conv2d(hid_c, 3 * hid_c, kernel_size, padding=padding, bias=False)
        self.conv_o = nn.Conv2d(2 * hid_c, hid_c, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor, m: torch.Tensor):
        gx = self.conv_x(x)
        gh = self.conv_h(h)
        gm = self.conv_m(m)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(gx, self.hid_c, dim=1)
        i_h, f_h, g_h, o_h = torch.split(gh, self.hid_c, dim=1)
        i_m, f_m, g_m = torch.split(gm, self.hid_c, dim=1)

        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h)
        g = torch.tanh(g_x + g_h)
        c_new = f * c + i * g

        i_prime = torch.sigmoid(i_x_prime + i_m)
        f_prime = torch.sigmoid(f_x_prime + f_m)
        g_prime = torch.tanh(g_x_prime + g_m)
        m_new = f_prime * m + i_prime * g_prime

        mem = torch.cat([c_new, m_new], dim=1)
        o = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o * torch.tanh(self.conv_o(mem))
        return h_new, c_new, m_new


@register_model("predrnn")
class PredRNNModel(BasePrecipitationModel):
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

        total_in = in_radar_c + in_nwp_c * 2
        self.in_proj = nn.Conv2d(total_in, hid_c, kernel_size=1)
        self.cell_list = nn.ModuleList([
            SpatioTemporalLSTMCell(hid_c, hid_c) for _ in range(num_layers)
        ])
        self.out_head = nn.Sequential(
            nn.Conv2d(hid_c, hid_c // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid_c // 2, 1, kernel_size=1),
        )
        # 输出头以普通卷积收尾，零点由外部的直通 floor 施加。
        #
        # 原来这里用 nn.ReLU() 收尾：目标量是 log1p(mm)，几乎所有像素初始
        # 都落在阈值以下，ReLU 返回 0，而它的导数在负半轴也是 0 —— 梯度
        # 全部消失，模型整轮都在预测同一个常数场。实测：训练 loss 停在
        # 0.1502，而"永远说没雨"的常数预报是 0.1543，即比什么都不说还差，
        # CSI@0.5 恰好为 0。
        #
        # 之前给 bias 一个 0.1 的正值来强行打开梯度通路，但那只把输出推成
        # 恒定正值，学不到降水的空间结构。改成在输出外部施加直通 floor：
        # 值与 clamp 相同，梯度却能回到每一个像素。
        nn.init.zeros_(self.out_head[2].bias)

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        B, T_in, _, H, W = x_radar.shape
        nwp_flat = x_nwp.view(B, 1, -1, H, W).expand(-1, T_in, -1, -1, -1)
        x_concat = torch.cat([x_radar, nwp_flat], dim=2)

        h = [torch.zeros(B, self.hid_c, H, W, device=x_radar.device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hid_c, H, W, device=x_radar.device) for _ in range(self.num_layers)]
        m = torch.zeros(B, self.hid_c, H, W, device=x_radar.device)

        # 编码阶段 (时空记忆 M 在层与时步间流转)
        for t in range(self.in_len):
            cur = self.in_proj(x_concat[:, t])
            for l in range(self.num_layers):
                h[l], c[l], m = self.cell_list[l](cur, h[l], c[l], m)
                cur = h[l]

        # 解码预测未来
        preds = []
        cur_input = h[-1]
        for _ in range(self.out_len):
            cur = cur_input
            for l in range(self.num_layers):
                h[l], c[l], m = self.cell_list[l](cur, h[l], c[l], m)
                cur = h[l]
            out = self.out_head(cur)
            preds.append(out)
            cur_input = cur

        # 直通零点：值与 clamp(max(x,0)) 相同，梯度却能到达负值像素。
        # 输出头在 log 空间预测，再指数还原成毫米（见上面的说明）。
        log_mm = torch.stack(preds, dim=1)
        return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))