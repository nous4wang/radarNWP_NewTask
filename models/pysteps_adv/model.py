import torch
import torch.nn.functional as F
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models import register_model

#: The radar input reaches the model through ``radar_log_transform``, which
#: maps a dBZ field to ``log1p(dBZ) / 6``.  The Z-R relation below needs
#: physical dBZ, so the transform has to be undone before the inversion.
#: The inversion of ``log1p(v) / 6`` is ``expm1(x * 6)``.
RADAR_LOG_SCALE = 6.0

#: The ceiling of the reflectivity the Z-R relation is applied to.
#:
#: The archive holds up to 71 dBZ, and the relation Z = 200 * R^1.6 is
#: meant for that range.  Beyond it the inversion runs away -- at 400 dBZ
#: the rain rate overflows a float32 -- so the field is clipped here.  On
#: real data this never bites; it keeps the model finite on inputs the
#: dataset would never produce.
MAX_DBZ = 75.0

#: The regulariser of the displacement estimate, in dBZ per pixel.
#:
#: The estimate divides the cross term by the squared gradient, so in flat
#: regions the denominator goes to zero and the noise is amplified without
#: bound.  The constant has to live in the same units as the gradient, and
#: the gradient is in dBZ per pixel once the input is denormalised -- the
#: previous value of 1e-4 belonged to the compressed scale and was roughly
#: twenty times too small here, which let the noise through.
GRADIENT_FLOOR_DBZ = 2.8


@register_model("pysteps_adv")
class PySTEPSAdvection(BasePrecipitationModel):
    """气象业务平流基线：利用最末两帧反射率估计光流，半拉格朗日变形外推并由 Z-R 反演降水。"""

    #: 这是一个解析基线，没有可训练参数；只评测、不训练。
    is_learnable = False

    def __init__(self, in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10):
        super().__init__()
        self.out_len = out_len
        # 无可训参数，保留 dummy 参数配合框架梯度流程
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    @staticmethod
    def _to_dbz(x: torch.Tensor) -> torch.Tensor:
        """Undo the radar input transform, returning physical dBZ.

        The dataset normalises every radar channel with
        ``log1p(clip(dBZ, 0)) / 6`` (see
        :func:`radar_nwp_bench.data.transforms.radar_log_transform`).  The
        Z-R inversion is a relation in dBZ, not in the compressed scale: fed
        the compressed value directly, a 45 dBZ storm is read as 0.5 dBZ and
        comes back as 0.004 mm per 6 minutes -- two orders of magnitude below
        the weakest scoring threshold, so every pixel is a miss and the
        baseline scores exactly zero.

        Args:
            x: the normalised radar field.

        Returns:
            The same field in dBZ.
        """
        return torch.expm1(x.clamp_min(0.0) * RADAR_LOG_SCALE).clamp(
            0.0, MAX_DBZ
        )

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_radar.shape
        # 选取组合反射率通道 (CR 通常位于 channel 6，按实际顺序读取)
        # 反归一化回 dBZ：光流与 Z-R 都在物理单位下定义。
        cr_last = self._to_dbz(x_radar[:, -1, 6:7])
        cr_prev = self._to_dbz(x_radar[:, -2, 6:7])

        # 有限差分近似位移场
        #
        # The velocity is estimated on the physical dBZ field, so it comes out
        # in pixels per frame: ``u`` is the x-component of the displacement
        # over one 6-minute step.
        dt = cr_last - cr_prev
        dy = (torch.roll(cr_last, -1, dims=2) - torch.roll(cr_last, 1, dims=2)) / 2.0
        dx = (torch.roll(cr_last, -1, dims=3) - torch.roll(cr_last, 1, dims=3)) / 2.0
        denom = dx**2 + dy**2 + GRADIENT_FLOOR_DBZ**2
        u = -(dt * dx) / denom
        v = -(dt * dy) / denom

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x_radar.device),
            torch.linspace(-1, 1, W, device=x_radar.device),
            indexing="ij"
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        # ``grid_sample`` reads the source at the coordinate it is given, so
        # sampling at ``base + d`` moves the content to ``x - d``.  A storm
        # travelling with velocity ``u`` therefore has to be sampled at
        # ``base - u * t``: with the sign the other way round the forecast
        # runs the storm backwards, which is what the previous version did.
        #
        # The displacement is in pixels per frame, and the grid spans
        # [-1, 1] over ``W - 1`` pixels, so one pixel is ``2 / (W - 1)`` in
        # grid units.  The old constant 0.05 was not that conversion and had
        # no relation to the grid size.
        pixels_to_grid = 2.0 / max(W - 1, 1)

        preds = []
        cur_frame = cr_last
        for t in range(1, self.out_len + 1):
            flow = torch.cat([u, v], dim=1).permute(0, 2, 3, 1) * (t * pixels_to_grid)
            sample_grid = (base_grid - flow).clamp(-1, 1)
            warped = F.grid_sample(cur_frame, sample_grid, mode="bilinear", align_corners=True)
            # Z-R 转换: Z = 200 * R^1.6 -> 换算为 6 分钟雨量 (mm)
            linear_z = torch.pow(10.0, warped.clamp_min(0.0) / 10.0)
            rain_rate = torch.pow(linear_z / 200.0, 1.0 / 1.6)
            rain_6min = rain_rate * 0.1
            preds.append(rain_6min)

        return torch.stack(preds, dim=1) + self.dummy * 0.0