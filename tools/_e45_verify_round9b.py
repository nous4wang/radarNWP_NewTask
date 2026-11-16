"""E45 独立复核（第 9 轮 · 第二段）：S3 增益 CI + 3-seed 中位数/通过率。

修掉第一段的过滤 bug（`if "arms" in path` 把所有臂内路径都跳过了）。
判读口径：中位数 + 过地板通过率；单点分数不单独出现。
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


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, obj


s3 = load("s3_equalbias.json")
print("meta:", {k: v for k, v in s3.items() if k != "arms"})
print()
print("== 单一臂的 raw / equal_bias 全部叶子（先看结构，不猜键） ==")
for p, v in walk(s3["arms"]["A1"]):
    print(f"  {p} = {v!r}"[:160])
print()

print("=" * 92)
print("S3 裁决表：六臂的原始增益 CI 与配平后增益 CI")
print("=" * 92)
print(f"{'臂':<5} {'verdict':<30} {'raw CI':<26} {'跨零?':<7} {'equal-bias CI':<26} {'跨零?':<7}")
verdicts = {}
for arm, blk in s3["arms"].items():
    raw = blk.get("raw", {})
    eb = blk.get("equal_bias", {})
    # S3 的 CI 键是 ci_lo/ci_hi（不是 ci_low/ci_high）；crosses_zero 是现成布尔字段
    raw_ci = (raw.get("ci_lo"), raw.get("ci_hi"))
    eb_ci = (eb.get("ci_lo"), eb.get("ci_hi"))
    raw_cross = raw.get("crosses_zero")
    eb_cross = eb.get("crosses_zero")
    if raw_cross is None:
        raw_cross = raw_ci[0] is not None and raw_ci[0] <= 0 <= raw_ci[1]
    if eb_cross is None:
        eb_cross = eb_ci[0] is not None and eb_ci[0] <= 0 <= eb_ci[1]
    verdicts[arm] = blk.get("s3_verdict")
    fmt = lambda c: f"[{c[0]:+.6f}, {c[1]:+.6f}]" if c[0] is not None else "n/a"
    print(f"{arm:<5} {blk.get('s3_verdict',''):<30} {fmt(raw_ci):<26} {str(raw_cross):<7} {fmt(eb_ci):<26} {str(eb_cross):<7}")

print()
print("== raw / equal_bias 里非 CI 的关键量 ==")
for arm, blk in s3["arms"].items():
    raw = blk.get("raw", {})
    eb = blk.get("equal_bias", {})
    print(f"  {arm:<5} raw: {json.dumps({k: v for k, v in raw.items() if not isinstance(v, (dict, list))}, ensure_ascii=False)}")
    print(f"  {arm:<5} eb : {json.dumps({k: v for k, v in eb.items() if not isinstance(v, (dict, list))}, ensure_ascii=False)}")

print()
print("=" * 92)
print("3-seed 统计（中位数 + 过地板通过率 + 超 Oracle 计数）")
print("=" * 92)
seeds = {s: load(f"scores_summary_s{s}.json") for s in (0, 1, 2)}
rows = []
for arm in ARMS:
    csi, bias = [], []
    for s in (0, 1, 2):
        key = f"{arm}_s{s}_ep00"
        d = seeds[s][key]
        csi.append(d["csi_at_2.0"])
        bias.append(d["bias_at_2.0"])
    med = st.median(csi)
    n_pass = sum(1 for c in csi if c >= FLOOR)
    pass_rate = n_pass / len(csi)
    over = sum(1 for c in csi if c > ORACLE)
    rows.append((arm, med, min(csi), max(csi), st.median(bias), pass_rate, over, n_pass))
    print(f"  {arm:<5} median CSI@2.0 = {med:.4f}  range {min(csi):.4f}–{max(csi):.4f}  "
          f"median BIAS = {st.median(bias):.4f}  过地板 {n_pass}/3  超Oracle {over}/3")

print()
# 注意：rows 的列序是 (臂, 中位数, min, max, 中位BIAS, 通过率, 超Oracle计数, 过地板次数)
# 对照基准必须取 r[1]（中位数），不是 r[2]（min）
base_row = next(r for r in rows if r[0] == "B0")
base_med = base_row[1]
base_pass = base_row[7]
print(f"对照 B0 中位数 = {base_med:.4f}（range {base_row[2]:.4f}–{base_row[3]:.4f}）；"
      f"高于对照的臂 = {[r[0] for r in rows if r[0] != 'B0' and r[1] > base_med]}")
print(f"对照 B0 自身过地板 = {base_pass}/3"
      + ("⇒ 地板口径本身不稳，不构成晋级依据" if base_pass < 3 else ""))
print()
print("六臂 S3 结论：全部判负 =" + str(all(
    v in ("GAIN_WAS_AREA_OR_AMPLITUDE", "NO_GAIN_TO_BEGIN_WITH") for v in verdicts.values())))
print("无一臂进十 seed（无臂 3/3 过地板且中位数高于对照）：" + str(not any(
    r[0] != "B0" and r[7] == 3 and r[1] > base_med for r in rows)))
