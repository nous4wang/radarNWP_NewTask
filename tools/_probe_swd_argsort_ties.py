#!/usr/bin/env python3
"""D4 Gate 1 工程核查：固定网格投影的 argsort tie-breaking 与设备一致性。

背景
----
``code_artifact (2).md`` 的 ``SlicedWassersteinPrecipLoss`` 用
``torch.argsort(proj_coords[k])`` 排序**固定规则网格坐标**。规则网格沿任意
方向的投影存在大量并列值（ties），而 ``torch.argsort`` 在 CPU 与 CUDA 上的
tie-breaking 顺序并不保证一致。若排序置换随设备改变，则同一份 ``pred`` 在
CPU/GPU 上会得到不同的 CDF-L1 梯度 —— 直接违反 §35 的训练确定性要求。

本探针零算力、纯合成网格，量化两件事：
  1. 固定网格投影的并列值占比；
  2. 同一投影张量在 CPU 与 CUDA 上 argsort 是否逐元素一致。

产物写到远端 ``runs/probe_swd_argsort_ties.log``，供 §4.24 引用。
"""
from __future__ import annotations

import math

import torch


def main() -> None:
    h = w = 64
    k = 32

    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    coords = torch.stack([gx.flatten(), gy.flatten()], dim=0)  # (2, HW)

    theta = torch.linspace(0.0, math.pi, k)
    directions = torch.stack([torch.cos(theta), torch.sin(theta)], dim=0)  # (2, K)
    proj = torch.matmul(directions.t(), coords)  # (K, HW)

    tied = 0
    total = 0
    uniq_counts = []
    for row in range(k):
        _, counts = torch.unique(proj[row], return_counts=True)
        tied += int(counts[counts > 1].sum())
        total += int(proj[row].numel())
        uniq_counts.append(int(counts.numel()))

    print(f"projection grid : K={k} HW={h * w} elements={total}")
    print(f"tied elements   : {tied}/{total} = {tied / total:.4%}")
    print(f"unique per row  : min={min(uniq_counts)} max={max(uniq_counts)}")

    cpu_idx = torch.argsort(proj, dim=1)
    if not torch.cuda.is_available():
        print("cuda            : unavailable -> skip device comparison")
        return

    gpu_idx = torch.argsort(proj.cuda(), dim=1).cpu()
    diff = cpu_idx != gpu_idx
    print(f"cpu/cuda identical      : {torch.equal(cpu_idx, gpu_idx)}")
    print(f"rows with any difference: {int(diff.any(dim=1).sum())}/{k}")
    print(f"differing elements      : {int(diff.sum())}/{diff.numel()}")

    rows = torch.nonzero(diff.any(dim=1)).flatten().tolist()
    print(f"differing row indices   : {rows}")
    for r in rows:
        _, counts = torch.unique(proj[r], return_counts=True)
        print(
            f"  row {r:2d}: unique={int(counts.numel()):4d} "
            f"tied={int(counts[counts > 1].sum()):4d}/{proj[r].numel()} "
            f"n_diff={int(diff[r].sum())}"
        )


if __name__ == "__main__":
    main()
