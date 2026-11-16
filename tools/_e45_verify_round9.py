"""E45 独立复核（第 9 轮）：从本地原始产物重算 S3 裁决与 3-seed 统计。

设计要点（来自前几轮踩坑）：
  - 不凭猜写键路径：先打印结构，再按「键名模式」递归捞叶子值。
  - 判读口径：中位数 + 过地板通过率；单点分数不单独出现。
"""
import json
import os
import statistics as st

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_evals", "e45")
FLOOR = 0.0924
ORACLE = 0.1653
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]


def load(name):
    with open(os.path.join(BASE, name), "r", encoding="utf-8") as f:
        return json.load(f)


def struct(obj, depth=0, maxdepth=3):
    """打印一层结构（键 + 类型），不打印标量值。"""
    pad = "  " * depth
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                print(f"{pad}{k}: dict({len(v)})")
                if depth < maxdepth:
                    struct(v, depth + 1, maxdepth)
            elif isinstance(v, list):
                print(f"{pad}{k}: list[{len(v)}]")
                if v and isinstance(v[0], dict) and depth < maxdepth:
                    struct(v[0], depth + 1, maxdepth)
            else:
                print(f"{pad}{k}: {type(v).__name__} = {v!r}"[:200])
    else:
        print(f"{pad}{type(obj).__name__}")


def walk(obj, path=""):
    """递归产出 (路径, 标量值)。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def harvest(obj, patterns):
    """按键名子串模式递归收集叶子值。"""
    out = []
    for p, v in walk(obj):
        tail = p.rsplit(".", 1)[-1].lower()
        if any(pat in tail for pat in patterns):
            out.append((p, v))
    return out


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


print("=" * 78)
print("PART 1 · s3_equalbias.json 结构")
print("=" * 78)
s3 = load("s3_equalbias.json")
struct(s3, maxdepth=2)

print()
print("=" * 78)
print("PART 2 · S3 裁决字段（按模式捞取）")
print("=" * 78)
verdict_hits = harvest(s3, ["verdict", "label", "conclusion"])
for p, v in verdict_hits:
    if isinstance(v, str):
        print(f"  {p} = {v}")

print()
print("-- 每个臂的增益/偏置相关标量 --")
gain_hits = harvest(s3, ["gain", "ci", "delta", "equal", "bias", "mean", "control", "n_"])
for p, v in gain_hits:
    n = num(v)
    if n is None:
        continue
    if "arm" in p.lower() or "arms" in p.lower():
        continue
    print(f"  {p} = {v}")

print()
print("=" * 78)
print("PART 3 · 3-seed 分数统计（中位数 + 过地板通过率）")
print("=" * 78)
seeds = {}
for s in (0, 1, 2):
    seeds[s] = load(f"scores_summary_s{s}.json")
print("scores_summary_s0.json 结构：")
struct(seeds[0], maxdepth=2)


def csi_of(d, arm):
    """从单 seed 汇总里取某臂的 CSI@2.0 / BIAS@2.0。"""
    res = []
    for p, v in walk(d):
        if arm not in p:
            continue
        tail = p.rsplit(".", 1)[-1].lower()
        if tail.startswith("csi") and "2" in tail:
            n = num(v)
            if n is not None:
                res.append((p, n))
    return res


# 先看一条臂的完整叶子路径，便于写出精确提取器
print()
print("-- A4 在 seed0 里的所有叶子（键路径 + 值）--")
for p, v in walk(seeds[0]):
    if "A4" in p:
        print(f"  {p} = {v!r}"[:180])
