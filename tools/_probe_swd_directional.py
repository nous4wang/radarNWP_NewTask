#!/usr/bin/env python3
"""补充判据（D4 Gate 1）：梯度方向性 / 质心平移矩。

为什么要补：用户脚本 ``_probe_swd_vs_fss.py`` 的 ``pull_ratio`` 定义为
    growth / (|damping| + 1e-5)
其中 growth = 目标区平均负梯度、damping = 偏位预测区平均梯度。两个缺陷：
  1) 逐像素回归 Log-L1 天然在目标区产生生长梯度、在偏位区产生削峰梯度
     —— 这正是「双罚」的定义，该指标却把 Log-L1 也判成「有几何拉力」；
  2) SWD-Norm 的偏位区平均梯度恒为 0，分母塌到 1e-5，比值只是 growth 的缩放，
     没有「比」的物理含义；且平均把梯度场的空间方向性抹掉了。
真正的平移拉力应当表现为：预测单体靠近目标的一侧被命令生长（w>0）、
远离目标的一侧被命令削减（w<0），即梯度场存在空间一阶矩，把单体质心
朝目标方向推。本探针直接测这一阶矩（单位：像素）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import _probe_swd_vs_fss as P


def directional(loss_fn, pred, target, x_pred: float):
    """返回 (近侧平均增量, 远侧平均增量, 质心平移矩, loss).

    w = -grad 是梯度下降给预测场的增量方向：w>0 表示「生长」，w<0 表示「削减」。
    目标在 x=32，预测单体中心在 x_pred=32+dx，所以「近侧」= x < x_pred。
    质心平移矩 moment = Σ[w * (x - x_pred)] / Σ|grad|，负值 = 朝目标（左）推。
    """
    pv = pred.clone().detach().requires_grad_(True)
    loss = loss_fn(pv, target)
    loss.backward()
    g = pv.grad.detach()[0, 0]
    h, w = g.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    mask = pred[0, 0] >= 2.0
    w_inc = -g
    near = mask & (xx < x_pred)
    far = mask & (xx >= x_pred)
    g_near = w_inc[near].mean().item() if near.any() else float("nan")
    g_far = w_inc[far].mean().item() if far.any() else float("nan")
    denom = g[mask].abs().sum().item() + 1e-12
    moment = (w_inc[mask] * (xx[mask] - x_pred)).sum().item() / denom
    return g_near, g_far, moment, loss.item()


def main():
    device = torch.device("cpu")  # 纯合成场，CPU 确定性最好
    losses = {
        "Log-L1 (基准)": P.LogL1Loss(),
        "SoftFSS (R5)": P.SoftFSSLoss(threshold=2.0, kernel_size=5),
        "SWD-Norm (D4)": P.SlicedWasserstein2D(num_projections=32, normalize_mass=True),
        "SWD-Unnorm": P.SlicedWasserstein2D(num_projections=32, normalize_mass=False),
    }
    for fn in losses.values():
        fn.to(device)
    target = P.make_synthetic_storm(center_y=32.0, center_x=32.0, peak_rain=10.0, device=device)

    print("=" * 100)
    print("【补充判据】梯度方向性 / 质心平移矩（w=-grad；moment<0 表示朝目标方向推）")
    print("=" * 100)
    print(f"{'Loss':<15} | {'dx':<4} | {'近侧增量w(朝目标)':<18} | {'远侧增量w(背离)':<18} | {'质心平移矩(px)':<16} | 判读")
    print("-" * 100)
    for name, fn in losses.items():
        for dx in (1, 2, 3, 4, 6):
            x_pred = 32.0 + dx
            pred = P.make_synthetic_storm(center_y=32.0, center_x=x_pred, peak_rain=10.0, device=device)
            gn, gf, mom, l = directional(fn, pred, target, x_pred)
            if abs(mom) < 0.05:
                verdict = "无平移拉力(对称削峰/生长)"
            elif mom < 0:
                verdict = "朝目标平移拉力"
            else:
                verdict = "反向(远离目标)"
            print(f"{name:<15} | {dx:<4} | {gn:<18.6e} | {gf:<18.6e} | {mom:<16.4f} | {verdict}")
        print("-" * 100)

    print()
    print("判据：平移拉力要求同时满足 (a) 近侧 w>0 生长、远侧 w<0 削减，(b) moment<0。")
    print("      若近侧与远侧同号 → 只是整体削峰/整体生长，是双罚，不是最优传输。")


if __name__ == "__main__":
    main()
