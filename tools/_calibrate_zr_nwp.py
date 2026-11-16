#!/usr/bin/env python3
"""tools/_calibrate_zr_nwp.py: 雷达 Z-R 关系与 NWP 物理环境因子标定及持久性评测.

协议契约:
1. 在 train 划分拟合参数 (避免测试集泄漏).
2. 在 test 划分 (n=538) 上进行全场 301x301 汉宁缝合评测.
3. 产出 Arm 0~3 对比表, 回答 NWP 对地表降水反演的校准增益.

相比初稿的两处关键修正:
  * x_radar 的 CR 通道用 dataset.radar_vars.index("CR") 动态取 (原硬编码 6 取错变量).
  * NWP 的 RH1000 / PWAT 在数据集里是归一化值, 必须先按 transforms.NWP_NORM
    反归一化成物理量 (RH -> %, PWAT -> mm) 再进入物理公式; 否则 RH/PWAT 项会
    退化为常数, 根本捕捉不到低层湿度调制.
  * 拟合改用 numpy 对数线性最小二乘 (幂律模型在 log 空间本就线性), 不依赖 scipy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from radar_nwp_bench.data import NWPAlignment, RadarNWPDataset, build_dataloader
from radar_nwp_bench.data.transforms import NWP_NORM
from radar_nwp_bench.metrics import PatchStitcher, PrecipitationVerifier
from radar_nwp_bench.models.common import floor_at_zero

THRESHOLDS = (0.1, 0.5, 2.0, 4.0)
#: 持久性地板 (MASTER §4.15, 同 verifier/缝合).
FLOOR_CSI2 = 0.0924


def get_nwp_var_index(nwp_vars, name: str) -> int | None:
    """动态获取 NWP 变量在通道中的索引."""
    try:
        return list(nwp_vars).index(name)
    except ValueError:
        return None


def denorm_nwp(norm_value: np.ndarray, name: str) -> np.ndarray:
    """把数据集归一化后的 NWP 场还原成物理量."""
    mean, std = NWP_NORM[name]
    return norm_value * std + mean


def extract_paired_points(loader, cr_index: int, rh_idx, pwat_idx,
                          max_samples: int = 60000) -> dict[str, np.ndarray]:
    """从 DataLoader 采样对齐的 (Z_dBZ, RH%, PWAT_mm, Rain_True) 点对."""
    z_list, rh_list, pwat_list, y_list = [], [], [], []
    collected = 0
    for batch in loader:
        # x_radar: (B, 11, C, H, W); 起报时刻最后 1 帧 CR
        x_r = batch["x_radar"][:, -1, cr_index]
        cr_dbz = torch.expm1(x_r * 6.0).clamp(0.0, 75.0)
        # 真实降水真值: y_hist (B, 11, 1, H, W); 起报时刻为 -1 帧
        y_true = batch["y_hist"][:, -1, 0]
        # NWP 环境场: (B, 2, C, H, W); 取分析小时切片 0, 反归一化为物理量
        x_n = batch["x_nwp"][:, 0]
        rh_norm = x_n[:, rh_idx].cpu().numpy() if rh_idx is not None \
            else np.full_like(cr_dbz.cpu().numpy(), 80.0)
        pwat_norm = x_n[:, pwat_idx].cpu().numpy() if pwat_idx is not None \
            else np.full_like(cr_dbz.cpu().numpy(), 25.0)
        rh_phys = denorm_nwp(rh_norm, "RH1000")
        pwat_phys = denorm_nwp(pwat_norm, "PWAT")

        mask = (cr_dbz >= 15.0) | (y_true >= 0.05)
        if not mask.any():
            continue
        z_list.append(cr_dbz[mask].cpu().numpy())
        y_list.append(y_true[mask].cpu().numpy())
        rh_list.append(rh_phys[mask])
        pwat_list.append(pwat_phys[mask])
        collected += int(mask.sum().item())
        if collected >= max_samples:
            break

    return {
        "z": np.concatenate(z_list)[:max_samples],
        "y": np.concatenate(y_list)[:max_samples],
        "rh": np.concatenate(rh_list)[:max_samples],
        "pwat": np.concatenate(pwat_list)[:max_samples],
    }


def fit_models(data: dict[str, np.ndarray]) -> dict:
    """在训练点对上拟合各对比臂的物理参数 (numpy 对数线性最小二乘)."""
    z, y, rh, pwat = data["z"], data["y"], data["rh"], data["pwat"]
    z_linear = 10.0 ** (z / 10.0)
    y_pos = np.clip(y, 1e-3, None)

    # 0. 原始 Marshall-Palmer: Z=200*R^1.6 -> R_hourly=(Z/200)^(1/1.6); 6min:/10
    r_mp = ((z_linear / 200.0) ** (1.0 / 1.6)) / 10.0

    # Arm 1: 最优标量缩放因子 k * R_mp (最小二乘闭式解)
    k_opt = float(np.sum(r_mp * y) / (np.sum(r_mp ** 2) + 1e-12))

    # Arm 2: 本土化 Z-R 幂律 R = a * Z^b  (log 空间线性)
    log_z = np.log(np.clip(z_linear, 1e-3, None))
    log_y = np.log(y_pos)
    A2 = np.vstack([np.ones_like(log_z), log_z]).T
    coef2, *_ = np.linalg.lstsq(A2, log_y, rcond=None)
    a2, b2 = float(np.exp(coef2[0])), float(coef2[1])

    # Arm 3: 融入 NWP 湿度与水汽调节
    #   R = a * Z^b * (RH/100)^c * (PWAT/25)^d  (log 空间线性)
    rh_n = np.clip(rh / 100.0, 0.05, 1.2)
    pwat_n = np.clip(pwat / 25.0, 0.1, 2.0)
    A3 = np.vstack([
        np.ones_like(log_z),
        log_z,
        np.log(rh_n),
        np.log(pwat_n),
    ]).T
    coef3, *_ = np.linalg.lstsq(A3, log_y, rcond=None)
    a3, b3, c3, d3 = (float(np.exp(coef3[0])), float(coef3[1]),
                      float(coef3[2]), float(coef3[3]))

    return {
        "k_opt": k_opt,
        "popt_zr": (a2, b2),
        "popt_nwp": (a3, b3, c3, d3),
    }


def evaluate_arm(loader, dataset, arm_name: str, params: dict,
                 cr_index: int, rh_idx, pwat_idx) -> dict:
    """在独立测试集上评测指定反演公式的持久性外推技巧."""
    grid_h, grid_w = dataset.grid_shape
    stitcher = PatchStitcher(
        (grid_h, grid_w), n_frames=10, n_positions=dataset.n_patch_positions
    )
    verifier = PrecipitationVerifier(thresholds=THRESHOLDS)

    with torch.no_grad():
        for batch in loader:
            x_r = batch["x_radar"][:, -1, cr_index]
            z_dbz = torch.expm1(x_r * 6.0).clamp(0.0, 75.0)
            z_lin = 10.0 ** (z_dbz / 10.0)

            x_n = batch["x_nwp"][:, 0]
            if rh_idx is not None:
                rh_phys = torch.as_tensor(
                    denorm_nwp(x_n[:, rh_idx].cpu().numpy(), "RH1000")
                ).to(x_n.device)
            else:
                rh_phys = torch.ones_like(z_dbz) * 80.0
            if pwat_idx is not None:
                pwat_phys = torch.as_tensor(
                    denorm_nwp(x_n[:, pwat_idx].cpu().numpy(), "PWAT")
                ).to(x_n.device)
            else:
                pwat_phys = torch.ones_like(z_dbz) * 25.0

            if arm_name == "zr_mp_raw":
                r0 = ((z_lin / 200.0) ** (1.0 / 1.6)) / 10.0
            elif arm_name == "zr_global_scale":
                r0 = ((z_lin / 200.0) ** (1.0 / 1.6)) / 10.0 * params["k_opt"]
            elif arm_name == "zr_local_fit":
                a, b = params["popt_zr"]
                r0 = a * (z_lin ** b)
            elif arm_name == "zr_nwp_phys":
                a, b, c, d = params["popt_nwp"]
                rh_n = (rh_phys / 100.0).clamp(0.05, 1.2)
                pwat_n = (pwat_phys / 25.0).clamp(0.1, 2.0)
                r0 = a * (z_lin ** b) * (rh_n ** c) * (pwat_n ** d)
            else:
                raise ValueError(f"未知臂: {arm_name}")

            r0 = floor_at_zero(r0)  # (B, H, W)
            # 持久性外推: 复制未来 10 帧 → (B, 10, 1, H, W)
            pred_10 = r0.unsqueeze(1).unsqueeze(1).expand(-1, 10, -1, -1, -1)

            y_true = batch["y_rain"]
            pred_np = pred_10.cpu().numpy()
            true_np = y_true.cpu().numpy()
            sample_idx = batch["case_index"].cpu().numpy()
            pos_idx = batch["position_index"].cpu().numpy()
            origin = batch["patch_origin"].cpu().numpy()

            for b_i in range(pred_np.shape[0]):
                s_i, p_i = int(sample_idx[b_i]), int(pos_idx[b_i])
                orig = (int(origin[b_i][0]), int(origin[b_i][1]))
                stitcher.add(s_i, p_i, orig, pred_np[b_i, :, 0], true_np[b_i, :, 0])
                if stitcher.is_complete(s_i):
                    sp, st = stitcher.finalize(s_i)
                    verifier.update(sp[None], st[None])
                    stitcher.release(s_i)

    return verifier.report()["thresholds"]


def main():
    parser = argparse.ArgumentParser(description="雷达 Z-R 与 NWP 物理标定评测")
    parser.add_argument("--data-root", default="/home/wf/data/radarNWP/BJ_mmap")
    parser.add_argument("--out-json", default="runs/calibrate_zr_nwp.json")
    parser.add_argument("--max-samples", type=int, default=60000)
    args = parser.parse_args()

    # 1. 加载训练集并拟合参数
    print(">>> 正在从训练集提取配对数据并拟合物理参数...")
    train_ds = RadarNWPDataset(
        args.data_root, packed=True, in_len=11, out_len=10, patch_size=64,
        alignment=NWPAlignment.FORECAST, split="train", slide_stride=64,
        random_crop=False,
    )
    cr_index = list(train_ds.radar_vars).index("CR")
    rh_idx = get_nwp_var_index(train_ds.nwp_vars, "RH1000")
    if rh_idx is None:
        rh_idx = get_nwp_var_index(train_ds.nwp_vars, "RH")
    pwat_idx = get_nwp_var_index(train_ds.nwp_vars, "PWAT")
    print(f"    CR index={cr_index}, RH1000 index={rh_idx}, "
          f"PWAT index={pwat_idx}")
    train_loader = build_dataloader(
        train_ds, batch_size=32, shuffle=True, num_workers=4, seed=0
    )
    train_points = extract_paired_points(
        train_loader, cr_index, rh_idx, pwat_idx,
        max_samples=args.max_samples,
    )
    params = fit_models(train_points)

    print(">>> 物理参数拟合完成:")
    print(f"  Arm 1 (标量缩放倍率 k): {params['k_opt']:.4f} "
          f"(~1/1.778={1/1.778:.4f}, 消除 78% 偏置)")
    a2, b2 = params["popt_zr"]
    print(f"  Arm 2 (本土化 Z-R 幂律): a={a2:.6f}, b={b2:.4f}")
    a3, b3, c3, d3 = params["popt_nwp"]
    print(f"  Arm 3 (NWP 物理调节): a={a3:.6f}, b={b3:.4f}, "
          f"c_RH={c3:.4f}, d_PWAT={d3:.4f}")

    # 2. 在独立测试集 (n=538) 上全场打分
    print("\n>>> 正在独立洁净测试集 (n=538) 上评测各臂持久性技巧...")
    test_ds = RadarNWPDataset(
        args.data_root, packed=True, in_len=11, out_len=10, patch_size=64,
        alignment=NWPAlignment.FORECAST, split="test", slide_stride=64,
        random_crop=False,
    )
    tcr = list(test_ds.radar_vars).index("CR")
    trh = get_nwp_var_index(test_ds.nwp_vars, "RH1000") or \
        get_nwp_var_index(test_ds.nwp_vars, "RH")
    tpw = get_nwp_var_index(test_ds.nwp_vars, "PWAT")
    test_loader = build_dataloader(
        test_ds, batch_size=16, shuffle=False, num_workers=4, seed=0
    )

    results = {}
    arms = ["zr_mp_raw", "zr_global_scale", "zr_local_fit", "zr_nwp_phys"]
    for arm in arms:
        print(f"  正在评测 [{arm}]...")
        results[arm] = evaluate_arm(
            test_loader, test_ds, arm, params, tcr, trh, tpw
        )

    # 3. 打印综合对比表
    print("\n" + "=" * 84)
    print(f"{'对比臂 (Arm)':<20} | {'CSI@0.1':<8} | {'CSI@0.5':<8} | "
          f"{'CSI@2.0':<8} | {'CSI@4.0':<8} | {'BIAS@0.5':<8} | {'FAR@2.0':<8}")
    print("-" * 84)
    for arm in arms:
        rep = results[arm]
        c01 = rep.get("0.1", {}).get("csi", 0.0)
        c05 = rep.get("0.5", {}).get("csi", 0.0)
        c20 = rep.get("2.0", {}).get("csi", 0.0)
        c40 = rep.get("4.0", {}).get("csi", 0.0)
        bias05 = rep.get("0.5", {}).get("bias", 0.0)
        far20 = rep.get("2.0", {}).get("far", 0.0)
        tag = ""
        if arm == "zr_mp_raw":
            tag = f"  (floor={FLOOR_CSI2:.4f})"
        if arm in ("zr_global_scale", "zr_nwp_phys"):
            ok = ">=0.12" if c20 >= 0.12 else "below"
            tag = f"  -> CSI@2.0 {ok}"
        print(f"{arm:<20} | {c01:<8.4f} | {c05:<8.4f} | {c20:<8.4f} | "
              f"{c40:<8.4f} | {bias05:<8.4f} | {far20:<8.4f}{tag}")
    print("=" * 84)
    print(f"\n结论门槛: 持久性地板 CSI@2.0={FLOOR_CSI2:.4f}; "
          f"目标 >= 0.12; 天花板(完美QPE)=0.1653")

    # 4. 落盘
    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(
        json.dumps(
            {"floor_csi2": FLOOR_CSI2, "fitted_params": params,
             "results": results}, indent=2
        )
    )
    print(f">>> 评测报表已写入: {out_p}")


if __name__ == "__main__":
    main()
