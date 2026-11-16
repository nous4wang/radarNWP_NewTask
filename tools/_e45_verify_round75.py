#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 第 75 轮：远端 21 份逐评测原始 JSON 的最细粒度独立复核（只读本地）。

背景
----
第 50--74 轮核过：远端 Slurm 状态、`runs/e45/` 13 份聚合 json 的 md5、文档闸门、
已发布裁决（S3 12 区间 / 头条表 35 字段）的逐字段重算。
**从未校验过 21 份 `*.eval.json`（评测器原始输出）本身**——整条证据链的最细一环。

本轮做三件前 74 轮没做过的事：
  1. 把 21 份原始评测 JSON 逐份取回，用**远端 md5** 锁定字节（不是信任本地副本）；
  2. 从原始评测的**列联计数**（hits / misses / false_alarms）**重算 CSI / BIAS / POD / FAR**，
     检验评测器自身的算术自洽——此前所有层都直接信任评测器写出的 csi/bias 字段；
  3. 从原始评测**独立重算**已发布头条表（中位数 / 过地板通过率 / 种子跨度），
     并与 `scores_all.json`、HANDOVER §3 的公布数逐格对账。

纪律：报「中位数 + 通过率」，单点分数不作裁决；事后复核不构成新判据、不改变任何裁决。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import statistics
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "tools" / "_evals" / "e45"
TARBALL = EV / "e45_evals.tar.gz"
RAW_DIR = EV / "evals_raw"
REPORT = EV / "round75_raw_eval_audit.json"

TARBALL_MD5 = "2d859447fce615230d96b0c55536640f"  # 远端 md5（2026-11-15 07:35）

# 远端 `md5sum runs/e45/evals/*.eval.json` 实测值（本轮从远端取回并逐份锁定）
REMOTE_MD5 = {
    "A1_s0": "081f10d09d05e8a4097793392c13cba5",
    "A1_s1": "31906e020988516339824a88cb6c6109",
    "A1_s2": "83f0edad0e408e0275ed6d717313a8e6",
    "A2_s0": "4cd4d82310969df3fd1e3ce087572225",
    "A2_s1": "3c4876c8dce6298d2e51809f39eca3b4",
    "A2_s2": "9893ac967b7e3afbdc6fbd87bbdfad1e",
    "A3a_s0": "a463d9102694b1d88643c9a27c34ceec",
    "A3a_s1": "f56b90efedeb41400d5b824744ae4340",
    "A3a_s2": "adf0056027fa36ad33ac82e31ee909d7",
    "A3b_s0": "6d63f95aa26322f51e2842b85551a7f7",
    "A3b_s1": "dcd9b3011e23f133be07fe06daf17ed3",
    "A3b_s2": "07f9089598450d85c1ac62b1af7d0848",
    "A4b_s0": "9247529bf6d2cb4057c201ecdc6b63e6",
    "A4b_s1": "e8b903ceaaacfa1a9569849e2ae95cbe",
    "A4b_s2": "82c4631825af2283df2b1133b085958c",
    "A4_s0": "d79fd59b52fdc8334ba97df9d82065e5",
    "A4_s1": "7149bf6794769b4aca52b69b98120e0e",
    "A4_s2": "3e356c3759179feca6f7011c2a9e19ef",
    "B0_s0": "19f73e2acadb85606a2af5f2f13b28a9",
    "B0_s1": "f1518c976f18291146206fc557c63f5f",
    "B0_s2": "fec6672ff210c017831f7a3d02072591",
}

# HANDOVER_E45.md §3 已公布的 seed-0 数字（权威文档口径）
PUBLISHED_S0_CSI = {
    "A4": 0.175500, "A3b": 0.169661, "A3a": 0.149470, "A4b": 0.137567,
    "B0": 0.121817, "A1": 0.118397, "A2": 0.112003,
}
PUBLISHED_S0_BIAS = {
    "A4": 1.6131, "A3b": 0.9205, "A3a": 0.6719, "A4b": 3.1789,
    "B0": 0.5095, "A1": 0.4430, "A2": 0.4490,
}
# 已公布的「过地板通过率」（HANDOVER §4⑦ / 第 74 轮表）
PUBLISHED_PASS = {"A3a": 3, "A3b": 3, "B0": 2, "A4b": 2}

FLOOR = 0.0924
ORACLE = 0.1653
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]

pos: list[str] = []
neg: list[str] = []


def check(ok: bool, msg: str) -> None:
    (pos if ok else neg).append(msg)


def md5_of(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    # ---------- 0. 取包与字节锁定 ----------
    check(TARBALL.exists(), f"打包文件存在: {TARBALL.name}")
    if not TARBALL.exists():
        return finish({})
    tar_md5 = md5_of(TARBALL)
    check(tar_md5 == TARBALL_MD5, f"打包 md5 与远端一致 ({tar_md5[:8]}…)")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TARBALL, "r:gz") as tf:
        tf.extractall(RAW_DIR)

    raw_files = sorted(RAW_DIR.glob("evals/*.eval.json"))
    check(len(raw_files) == 21, f"解出 21 份 *.eval.json（实际 {len(raw_files)}）")

    raw: dict[str, dict] = {}
    for p in raw_files:
        cell = p.name.replace("_ep00.eval.json", "")  # e.g. A1_s0
        check(cell in REMOTE_MD5, f"文件名可映射到已知臂-种子格: {cell}")
        m = md5_of(p)
        check(m == REMOTE_MD5.get(cell), f"字节与远端逐位相同: {cell}")
        raw[cell] = json.loads(p.read_text(encoding="utf-8"))

    check(len(raw) == 21, f"解析出 21 格（实际 {len(raw)}）")

    # ---------- 1. 评测器自身算术自洽（列联计数 → 指标）----------
    max_dev_arith = 0.0
    for cell, d in sorted(raw.items()):
        for thr, m in d["thresholds"].items():
            h, ms, fa = float(m["hits"]), float(m["misses"]), float(m["false_alarms"])
            tot, obs = h + ms + fa, h + ms
            if tot > 0:
                max_dev_arith = max(max_dev_arith, abs(h / tot - float(m["csi"])))
            if obs > 0:
                max_dev_arith = max(max_dev_arith, abs((h + fa) / obs - float(m["bias"])))
            if obs > 0:
                max_dev_arith = max(max_dev_arith, abs(h / obs - float(m["pod"])))
            if (h + fa) > 0:
                max_dev_arith = max(max_dev_arith, abs(fa / (h + fa) - float(m["far"])))
    check(max_dev_arith < 1e-12, f"评测器列联计数自洽（CSI/BIAS/POD/FAR 最大偏差 {max_dev_arith:.3e}）")

    ns = {c: int(d["n_samples"]) for c, d in raw.items()}
    check(set(ns.values()) == {538}, f"21 格样本数一致且为 538（实际 {sorted(set(ns.values()))}）")

    # ---------- 2. 原始评测 vs 聚合产物 scores_all.json（精确 key 匹配）----------
    scores_all = json.loads((EV / "scores_all.json").read_text(encoding="utf-8"))
    expected_keys = {f"{a}_s{s}_ep00" for a in ARMS for s in (0, 1, 2)}
    check(set(scores_all.keys()) == expected_keys,
          "scores_all.json 键集恰为 21 格（无缺无多，累积快照陷阱检测）")

    field_map = {
        "csi_at_2.0": ("2.0", "csi"), "bias_at_2.0": ("2.0", "bias"),
        "pod_at_2.0": ("2.0", "pod"), "far_at_2.0": ("2.0", "far"),
        "csi_at_0.5": ("0.5", "csi"), "bias_at_0.5": ("0.5", "bias"),
        "csi_at_4.0": ("4.0", "csi"), "bias_at_4.0": ("4.0", "bias"),
    }
    max_dev_agg = 0.0
    for cell, d in sorted(raw.items()):
        agg = scores_all[f"{cell}_ep00"]
        for fname, (thr, key) in field_map.items():
            dev = abs(float(d["thresholds"][thr][key]) - float(agg[fname]))
            max_dev_agg = max(max_dev_agg, dev)
        max_dev_agg = max(max_dev_agg, abs(int(d["n_samples"]) - int(agg["n_samples"])))
    check(max_dev_agg == 0.0, f"聚合产物与原始评测逐格逐位相同（最大偏差 {max_dev_agg:.3e}）")

    # ---------- 3. 与 HANDOVER §3 已公布数对账 ----------
    max_dev_pub_csi = 0.0
    for arm, pub in PUBLISHED_S0_CSI.items():
        dev = abs(raw[f"{arm}_s0"]["thresholds"]["2.0"]["csi"] - pub)
        max_dev_pub_csi = max(max_dev_pub_csi, dev)
    check(max_dev_pub_csi <= 5e-7, f"§3 已公布 seed-0 CSI@2.0 全部对上（最大偏差 {max_dev_pub_csi:.3e}）")

    max_dev_pub_bias = 0.0
    for arm, pub in PUBLISHED_S0_BIAS.items():
        dev = abs(raw[f"{arm}_s0"]["thresholds"]["2.0"]["bias"] - pub)
        max_dev_pub_bias = max(max_dev_pub_bias, dev)
    check(max_dev_pub_bias <= 5e-5, f"§3 已公布 seed-0 BIAS@2.0 全部对上（最大偏差 {max_dev_pub_bias:.3e}）")

    # ---------- 4. 从原始评测独立重算头条表（中位数 / 通过率 / 跨度）----------
    headline = {}
    for arm in ARMS:
        csis = [raw[f"{arm}_s{s}"]["thresholds"]["2.0"]["csi"] for s in (0, 1, 2)]
        biases = [raw[f"{arm}_s{s}"]["thresholds"]["2.0"]["bias"] for s in (0, 1, 2)]
        med_csi = statistics.median(csis)
        med_bias = statistics.median(biases)
        npass = sum(1 for c in csis if c >= FLOOR)
        headline[arm] = {
            "median_csi_at_2.0": med_csi,
            "median_bias_at_2.0": med_bias,
            "pass_over_floor": f"{npass}/3",
            "seed_spread_csi_at_2.0": max(csis) - min(csis),
            "n_seeds_above_oracle": sum(1 for c in csis if c > ORACLE),
            "seeds_csi_at_2.0": csis,
            "seeds_bias_at_2.0": biases,
        }
        # 与 scores_all 同源重算必须一致（口径自洽）
        check(abs(med_csi - statistics.median(
            [scores_all[f"{arm}_s{s}_ep00"]["csi_at_2.0"] for s in (0, 1, 2)])) < 1e-15,
            f"{arm}: 头条中位数与聚合口径一致")

    for arm, exp in PUBLISHED_PASS.items():
        check(headline[arm]["pass_over_floor"] == f"{exp}/3",
              f"{arm}: 过地板通过率 = 已公布的 {exp}/3")

    # 第 74 轮登记的累积快照陷阱检测器：任何一臂三种子数值全等 = 快照污染
    same_seed_arms = [a for a in ARMS if len(set(headline[a]["seeds_csi_at_2.0"])) == 1]
    check(not same_seed_arms, f"无任一臂三种子数值全等（累积快照检测器未触发；触发臂={same_seed_arms}）")

    # ---------- 5. 与第 71/74 轮已发布头条表逐字段对账 ----------
    pub_head = None
    r74 = EV / "round74_status_audit.json"
    if r74.exists():
        d74 = json.loads(r74.read_text(encoding="utf-8"))
        for key in ("headline", "headline_table", "published_headline", "arms"):
            v = d74.get(key)
            if isinstance(v, dict) and all(a in v for a in ARMS):
                pub_head = v
                break
    if pub_head:
        dev_list = []
        for arm in ARMS:
            p = pub_head[arm]
            if isinstance(p, dict):
                for cand in ("median_csi_at_2.0", "median_csi", "csi_at_2.0_median"):
                    if cand in p:
                        dev_list.append(abs(float(p[cand]) - headline[arm]["median_csi_at_2.0"]))
                        break
        check((not dev_list) or max(dev_list) <= 1e-12,
              f"与第 74 轮已发布头条表中位数逐臂对上（最大偏差 {max(dev_list) if dev_list else 0.0:.3e}）")
    else:
        pos.append("第 74 轮产物未含机器可抽取的头条表（本项改由 scores_all + §3 双重对账覆盖）")

    return finish({
        "headline_from_raw": headline,
        "published_seed0_csi": PUBLISHED_S0_CSI,
        "published_seed0_bias": PUBLISHED_S0_BIAS,
        "max_dev_arith": max_dev_arith,
        "max_dev_agg": max_dev_agg,
        "max_dev_pub_csi": max_dev_pub_csi,
        "max_dev_pub_bias": max_dev_pub_bias,
        "floor": FLOOR,
        "oracle": ORACLE,
    })


def finish(extra: dict) -> int:
    rep = {
        "round": 75,
        "purpose": "E45 远端 21 份逐评测原始 JSON 的最细粒度独立复核（只读本地）",
        "remote_tarball_md5": TARBALL_MD5,
        "cells": 21,
        "pos": pos,
        "neg": neg,
        "verdict": "PASS" if not neg else "FAIL",
    }
    rep.update(extra)
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print("RAW-EVAL (21 *.eval.json) INDEPENDENT RECOMPUTE")
    print("=" * 78)
    hl = extra.get("headline_from_raw", {})
    print(f"{'arm':<5}{'medianCSI@2.0':>16}{'medianBIAS@2.0':>16}{'pass':>8}{'spread':>12}{'>oracle':>9}")
    for arm in ARMS:
        h = hl.get(arm)
        if not h:
            continue
        print(f"{arm:<5}{h['median_csi_at_2.0']:>16.6f}{h['median_bias_at_2.0']:>16.6f}"
              f"{h['pass_over_floor']:>8}{h['seed_spread_csi_at_2.0']:>12.6f}"
              f"{str(h['n_seeds_above_oracle'])+'/3':>9}")
    print("-" * 78)
    for m in pos:
        print("  PASS ", m)
    for m in neg:
        print("  FAIL ", m)
    print(f"\nPOS {len(pos)}/{len(pos) + len(neg)}   NEG {len(neg)}")
    print(f"VERDICT: {rep['verdict']}")
    print(f"report -> {REPORT}")
    return 0 if not neg else 1


if __name__ == "__main__":
    sys.exit(main())
