#!/usr/bin/env python3
"""tools/_probe_swd_vs_fss.py: D4 损失机制探针

检验目的:
1. 空间位移敏感度: 当对流单体发生 1~6 像素 (1~6 km) 空间漂移时，各损失函数的饱和截断行为。
2. 梯度方向诊断 (平移拉力 vs 振幅阻尼):
   反向传播给预测场的梯度 grad = d(Loss)/d(Pred)，究竟是在向真实单体位置产生
   "定向生长拉力" (Target Growth)，还是仅仅在当前偏位处施加"削峰抹平" (Peak Damping)。
3. 零膨胀质量不守恒崩溃测试: 当预测雨量与真实雨量总量不相等时，SWD 的梯度数值稳定性。
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. 待检验的损失函数实现
# ==============================================================================

class SlicedWasserstein2D(nn.Module):
    """二维切片最优传输 (SWD) 损失."""
    def __init__(self, num_projections: int = 32, normalize_mass: bool = True):
        super().__init__()
        self.num_projections = num_projections
        self.normalize_mass = normalize_mass
        # 均匀采样 [0, pi) 上的投影方向
        theta = torch.linspace(0, np.pi, num_projections)
        self.register_buffer("directions", torch.stack([torch.cos(theta), torch.sin(theta)], dim=0))  # (2, K)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        device = pred.device

        # 归一化空间物理坐标到 [-1, 1]
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing="ij"
        )
        coords = torch.stack([gx.flatten(), gy.flatten()], dim=0)  # (2, HW)
        proj_coords = torch.matmul(self.directions.t(), coords)    # (K, HW)

        p_flat = pred.view(b * c, -1)
        t_flat = target.view(b * c, -1)

        # 质量守恒处理
        if self.normalize_mass:
            p_mass = p_flat.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            t_mass = t_flat.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            p_meas = p_flat / p_mass
            t_meas = t_flat / t_mass
        else:
            p_meas = p_flat
            t_meas = t_flat

        swd_loss = 0.0
        for k in range(self.num_projections):
            sort_idx = torch.argsort(proj_coords[k])
            # 排序后的一维分布累积质量差 (一维 Wasserstein 距离即累积分布的 L1 范数)
            p_cdf = torch.cumsum(p_meas[:, sort_idx], dim=-1)
            t_cdf = torch.cumsum(t_meas[:, sort_idx], dim=-1)
            swd_loss += F.l1_loss(p_cdf, t_cdf)

        return swd_loss / self.num_projections


class SoftFSSLoss(nn.Module):
    """空间邻域可微软 FSS 损失."""
    def __init__(self, threshold: float = 2.0, kernel_size: int = 5):
        super().__init__()
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 对称连续软化
        p_bin = torch.sigmoid((pred - self.threshold) * 4.0)
        t_bin = torch.sigmoid((target - self.threshold) * 4.0)

        # 边界 Replicate 填充，避免边缘零背景拉偏
        p_pad = F.pad(p_bin, [self.pad] * 4, mode="replicate")
        t_pad = F.pad(t_bin, [self.pad] * 4, mode="replicate")

        p_frac = F.avg_pool2d(p_pad, kernel_size=self.kernel_size, stride=1)
        t_frac = F.avg_pool2d(t_pad, kernel_size=self.kernel_size, stride=1)

        mse_frac = torch.mean((p_frac - t_frac) ** 2)
        ref_frac = torch.mean(p_frac ** 2 + t_frac ** 2) + 1e-5
        return mse_frac / ref_frac


class LogL1Loss(nn.Module):
    """基准对数空间 Smooth L1 回归损失."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_log = torch.log1p(pred.clamp_min(0.0))
        t_log = torch.log1p(target.clamp_min(0.0))
        return F.smooth_l1_loss(p_log, t_log)


# ==============================================================================
# 2. 合成单体发生器与梯度诊断引擎
# ==============================================================================

def make_synthetic_storm(
    h: int = 64,
    w: int = 64,
    center_y: float = 32.0,
    center_x: float = 32.0,
    peak_rain: float = 10.0,
    radius: float = 3.5,
    device: str = "cpu"
) -> torch.Tensor:
    """生成真实的极端对流单体 (高斯对流核，中心达到 10mm/6min 业务极值)."""
    gy, gx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij"
    )
    dist_sq = (gy - center_y) ** 2 + (gx - center_x) ** 2
    cell = peak_rain * torch.exp(-0.5 * dist_sq / (radius ** 2))
    cell[cell < 0.05] = 0.0  # 截断弱杂波
    return cell.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def diagnose_gradients(
    loss_fn: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
    shift_dx: float
) -> dict[str, float]:
    """诊断梯度场的物理向量属性:
    梯度下降更新方向为: Pred_new = Pred - lr * Grad.

    指标定义:
    1. damping_force: 在偏位的假阳性预测区 (Pred >= 2.0) 上的平均梯度。
       若 Grad > 0，说明损失函数正在命令该区域"削峰抹平" (减小雨量)。
    2. target_growth: 在真实单体所在的目标区 (Target >= 2.0) 上的平均负梯度 (-Grad)。
       若 -Grad > 0 (即 Grad < 0)，说明损失函数正在命令目标区域"原位生长出雨量"。
    3. transport_pull_ratio: 目标生长力 / (削峰抹平力 + 1e-5)。
       - 比值接近 0: 纯阻尼行为 (没有空间拉力，只懂得把对流核删掉)。
       - 比值 > 0.5: 存在显著的几何传输牵引力。
    """
    pred_var = pred.clone().detach().requires_grad_(True)
    loss = loss_fn(pred_var, target)
    loss.backward()

    grad = pred_var.grad.detach()

    pred_mask = (pred >= 2.0)
    target_mask = (target >= 2.0)

    damping = grad[pred_mask].mean().item() if pred_mask.any() else 0.0
    growth = (-grad[target_mask]).mean().item() if target_mask.any() else 0.0
    ratio = growth / (abs(damping) + 1e-5)

    return {
        "loss": loss.item(),
        "damping": damping,
        "growth": growth,
        "pull_ratio": ratio,
        "max_abs_grad": grad.abs().max().item(),
    }


# ==============================================================================
# 3. 主探针测试流程
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="D4 SWD vs SoftFSS 位移与梯度拉力探针")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 82)
    print("【D4 机制证明探针】位移敏感度与空间梯度拉力测试 (Synthetic Convective Core)")
    print(f"运行设备: {device} | 真实单体峰值: 10.0 mm/6min | 空间尺度: 64x64 格点")
    print("=" * 82)

    # 1. 初始化三种代表性损失函数
    losses = {
        "Log-L1 (基准)": LogL1Loss(),
        "SoftFSS (R5)": SoftFSSLoss(threshold=2.0, kernel_size=5),
        "SWD-Norm (D4)": SlicedWasserstein2D(num_projections=32, normalize_mass=True),
        "SWD-Unnorm": SlicedWasserstein2D(num_projections=32, normalize_mass=False),
    }
    # 工程修复（不改变任何数学/判据）：SlicedWasserstein2D 的 directions 是
    # register_buffer，不会自动跟随输入张量迁移设备；原脚本未做 .to(device)，
    # 在 cuda 下 forward 的 matmul 会因 cpu/cuda 混用直接抛 RuntimeError。
    for _fn in losses.values():
        _fn.to(device)

    shifts = [0, 1, 2, 3, 4, 6]  # 空间平移 0~6 像素 (对应 0~6 km)
    target = make_synthetic_storm(center_y=32.0, center_x=32.0, peak_rain=10.0, device=device)

    # --------------------------------------------------------------------------
    # 测试 A: 纯空间位移 (单体振幅 100% 正确，但位置偏离 0~6 km)
    # --------------------------------------------------------------------------
    print("\n[测试 A: 纯空间平移测试 (位移 dx 从 0 到 6 像素，峰值保持 10.0mm)]")
    print(f"{'Loss 方案':<15} | {'dx(px)':<6} | {'Loss':<10} | {'削峰阻尼(Damp)':<14} | {'目标牵引(Grow)':<14} | {'拉力/阻尼比':<10}")
    print("-" * 82)

    for name, loss_fn in losses.items():
        for dx in shifts:
            pred = make_synthetic_storm(center_y=32.0, center_x=32.0 + dx, peak_rain=10.0, device=device)
            diag = diagnose_gradients(loss_fn, pred, target, shift_dx=dx)
            print(f"{name:<15} | {dx:<6} | {diag['loss']:<10.4f} | {diag['damping']:<14.6f} | {diag['growth']:<14.6f} | {diag['pull_ratio']:<10.3f}")
        print("-" * 82)

    # --------------------------------------------------------------------------
    # 测试 B: 零膨胀质量不守恒扰动测试 (预测单体雨量被严重低估或高估)
    # --------------------------------------------------------------------------
    print("\n[测试 B: 质量不守恒扰动测试 (位移固定 dx=2px，预测单体峰值从 2mm 到 20mm 变动)]")
    print(f"{'Loss 方案':<15} | {'峰值(mm)':<8} | {'Loss':<10} | {'Max|Grad|':<12} | {'目标牵引(Grow)':<14} | {'状态判定':<12}")
    print("-" * 82)

    scales = [0.2, 0.5, 1.0, 1.5, 2.0]  # 对应预测峰值 2.0mm, 5.0mm, 10.0mm, 15.0mm, 20.0mm
    dx_fixed = 2.0

    for name, loss_fn in losses.items():
        for s in scales:
            peak = 10.0 * s
            pred = make_synthetic_storm(center_y=32.0, center_x=32.0 + dx_fixed, peak_rain=peak, device=device)
            diag = diagnose_gradients(loss_fn, pred, target, shift_dx=dx_fixed)

            # 判断梯度是否发生数值震荡或截断
            if np.isnan(diag['loss']) or np.isinf(diag['max_abs_grad']):
                status = "数值崩溃 (NaN/Inf)"
            elif diag['max_abs_grad'] > 100.0:
                status = "梯度爆炸"
            elif diag['growth'] <= 1e-6 and dx_fixed > 0:
                status = "拉力丧失(纯阻尼)"
            else:
                status = "有效传输梯度"

            print(f"{name:<15} | {peak:<8.1f} | {diag['loss']:<10.4f} | {diag['max_abs_grad']:<12.6f} | {diag['growth']:<14.6f} | {status:<12}")
        print("-" * 82)

    # --------------------------------------------------------------------------
    # 4. 自动给出 Gate 1 决策建议
    # --------------------------------------------------------------------------
    print("\n" + "=" * 82)
    print("【探针诊断判读与 Gate 1 决策标准】")
    print("1. 若 SWD 的 '拉力/阻尼比' 显著大于 0.3，且在 dx=2~4px 时目标牵引力 Growth 仍有连续负梯度:")
    print("   -> 证明 SWD 确实具备最优传输几何拉力，与 SoftFSS 的局部刚性截断具有本质机制差异。")
    print("2. 若 SWD-Norm 或 SWD-Unnorm 在质量不守恒 (测试 B) 下出现梯度爆炸或 Growth 归零:")
    print("   -> 证实降水场零膨胀特性导致最优传输测度未闭合，直接在 Gate 1 否决，禁止投入算力训练。")
    print("=" * 82)


if __name__ == "__main__":
    main()
