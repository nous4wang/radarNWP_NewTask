#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 第 69 轮写回：HANDOVER_E45.md（㉜）＋ 上层 MEMORY.md 状态段 ＋ 上层/项目级 2026-11-15 日志
＋ 上层 2026-11-11.md ＋ 本自动化记忆文件。

纪律（沿用第 61–68 轮教训）：
  * 写回在校验之后；断言不许假设数据会迎合结论。
  * 幂等：整块标记自剥后追加，连跑两次 sha256 必须逐位相同。
  * 表格行必须用**单元格列表**构造，分隔符由代码插入（手拼竖线会让列数错）。
  * 不动 docs/（本轮无数值与裁决变化 => MASTER 不升版，仍 v2.48）。

本轮特殊：把**真正定案的 S3 等偏置护栏第一次用到 epoch 轴的 ep01/ep02**，
并顺手做了一次**仪器标定**（列联表域对应物 vs 已发布场域护栏）：
  * 必须先在 ep00 证明仪器的原始侧可用、等偏置侧不可用于显著性；
  * 仪器与已发布护栏的**分歧必须被登记为仪器局限**，不得当发现；
  * 「裁决不变」与「理由迁移」必须分开写。
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UP = Path("R:/1radarNWPprediction/.workbuddy/memory")
PROJ = ROOT / ".workbuddy" / "memory"
AUTO = PROJ / "automations" / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md"
J = json.loads((ROOT / "tools" / "_evals" / "e45" / "round69_epoch_guard_axis.json")
               .read_text(encoding="utf-8"))

BEGIN = "<!-- E45-R69-BEGIN -->"
END = "<!-- E45-R69-END -->"
BEGIN_111 = "<!-- E45-R69-111-BEGIN -->"
END_111 = "<!-- E45-R69-111-END -->"

ORDER = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SIX = [a for a in ORDER if a != "B0"]
NAMES = {"B0": "B0 对照", "A1": "A1 幅度单调门", "A2": "A2 硬支撑门",
         "A3a": "A3a 拓扑截断", "A3b": "A3b 拓扑平滑+面积归一",
         "A4": "A4 量通道新头", "A4b": "A4 量通道自带头"}
EPOCHS = ["00", "01", "02"]
FLOOR = 0.0924

G = J["epoch_guard"]
H = J["per_epoch_headline"]
NULL = J["null_model_seed_permutation"]
EQC = J["instrument_equalised_side_comparison_ep00"]
RWR = J["instrument_raw_width_ratio_ep00"]
CONV = J["conventions"]
COL = J["control_collapse_context"]


def row(cells):
    """表格行必须由单元格列表构造，分隔符由代码插入。"""
    assert isinstance(cells, list) and len(cells) >= 2, "row() needs a cell list"
    return "| " + " | ".join(str(c) for c in cells) + " |"


def head(cells):
    return row(cells) + "\n" + row(["---"] * len(cells)) + "\n"


def read(p):
    raw = p.read_bytes()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode("utf-8"), nl


def write(p, text, nl):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    p.write_bytes(text.replace("\n", nl).encode("utf-8"))


def strip(text, begin, end):
    """自剥旧块；找不到就原样返回（幂等追加的基础）。"""
    out, i, n_removed = [], 0, 0
    while True:
        a = text.find(begin, i)
        if a < 0:
            out.append(text[i:])
            break
        b = text.find(end, a)
        if b < 0:
            out.append(text[i:])
            break
        out.append(text[i:a])
        i = b + len(end)
        n_removed += 1
    return "".join(out), n_removed


def append_block(path, body, begin=BEGIN, end=END):
    """幂等：先自剥本轮旧块，再整块追加到文件末尾。"""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, f"# {path.stem}（工作日志）\n", "\n")
    text, nl = read(path)
    text, n_removed = strip(text, begin, end)
    assert n_removed <= 1, f"{path.name}: found {n_removed} old R69 blocks"
    text = text.rstrip("\r\n") + "\n\n" + begin + "\n" + body.rstrip("\n") + "\n" + end + "\n"
    write(path, text, nl)
    after, _ = read(path)
    assert after.count(begin) == 1, f"{path.name}: marker count {after.count(begin)} != 1"
    return path, n_removed


def append_block_custom(path, begin, end, body):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, f"# {path.stem}（工作日志）\n", "\n")
    text, nl = read(path)
    text, n_removed = strip(text, begin, end)
    assert n_removed <= 1, f"{path.name}: found {n_removed} old blocks"
    text = text.rstrip("\r\n") + "\n\n" + body.rstrip("\n") + "\n"
    write(path, text, nl)
    after, _ = read(path)
    assert after.count(begin) == 1, f"{path.name}: marker count != 1"
    return n_removed


# ---------------------------------------------------------------- assertions
assert J["verdict"] == "PASS" and J["n_pass"] == J["n_checks"] == 19, \
    "analysis checks not all positive"
assert J["s3_guard"]["n_intervals"] == 12 and J["s3_guard"]["n_cross_zero"] == 12
assert J["published_md5"]["scores_summary"] == "57bcd511a537ab85494ee41b029731a8"
assert J["published_md5"]["s3_equalbias"] == "3aba999d5ef6dcc4a775999be6585eb2"
assert J["instrument_fidelity_ep00"]["raw_point_max_abs_diff_vs_published"] == 0.0
assert J["instrument_fidelity_ep00"]["eq_sign_agreement"] == "6/6"
assert J["measurement_arm_surviving_equalisation"] == [["00", "A2"]], \
    "the only survivor must be the documented instrument conflict (A2 at ep00)"
assert len(J["instrument_verdict_conflicts_ep00"]) == 1
assert all(NULL[ep]["n_perms_ci_excluding_zero"] == 0 for ep in EPOCHS)
assert all(0.95 <= NULL[ep]["ratio_to_round68_floor"] <= 1.05 for ep in EPOCHS)
assert all(G["01"][a]["verdict"] == "NO_GAIN_TO_BEGIN_WITH" for a in SIX)
assert sum(1 for a in SIX if G["02"][a]["verdict"] == "GAIN_WAS_AREA_OR_AMPLITUDE") == 5
assert J["measurement_raw_interval_excluding_zero"] == \
    [["01", "A1"], ["01", "A2"], ["01", "A3b"], ["02", "A3a"], ["02", "A4b"]]

# ------------------------------------------------------------------- tables
t1 = head(["epoch", "臂", "中位 CSI@2.0 (n=3)", "过地板", "原始点", "原始 CI",
            "跨零", "逐 seed 同号", "等偏置点", "护栏理由"])
for ep in EPOCHS:
    for arm in ORDER:
        if arm == "B0":
            hh = H[ep][arm]
            t1 += row([ep, NAMES[arm], f"{hh['median_csi_at_2.0']:.6f}",
                       f"{hh['pass_rate_over_floor']:.2f}", "—", "—", "—", "—",
                       "—", "对照"]) + "\n"
            continue
        r = G[ep][arm]
        hh = H[ep][arm]
        t1 += row([ep, NAMES[arm], f"{hh['median_csi_at_2.0']:.6f}",
                   f"{hh['pass_rate_over_floor']:.2f}",
                   f"{r['raw']['point']:+.6f}",
                   f"[{r['raw']['ci_lo']:+.6f},{r['raw']['ci_hi']:+.6f}]",
                   "是" if r["raw"]["crosses_zero"] else "否",
                   f"{r['raw_same_sign_seeds']}/3",
                   f"{r['eq']['point']:+.6f}",
                   r["verdict"]]) + "\n"

t2 = head(["epoch", "本轮零模型 CI 半宽中位", "第 68 轮零模型地板", "比值",
            "零模型不跨零置换数", "对照中位 CSI", "对照中位 BIAS",
            "对照相对 ep00", "中位高于对照的臂"])
for ep in EPOCHS:
    n = NULL[ep]
    c = COL[ep]
    t2 += row([ep, f"{n['median_ci_half_width']:.6f}",
               f"{n['round68_published_floor']:.6f}",
               f"{n['ratio_to_round68_floor']:.3f}",
               f"{n['n_perms_ci_excluding_zero']}/3",
               f"{c['control_median_csi']:.6f}", f"{c['control_median_bias']:.4f}",
               f"x{c['control_collapse_ratio_vs_ep00']:.3f}",
               "、".join(c["arms_above_control_median"]) or "—"]) + "\n"

t3 = head(["臂", "原始点绝对差 vs 已发布", "原始区间宽度比", "原始跨零判定一致",
            "等偏置点（本仪器）", "等偏置点（已发布场域）", "点绝对值比值",
            "本仪器跨零", "已发布跨零", "判定一致"])
for arm in SIX:
    e = EQC[arm]
    t3 += row([NAMES[arm], "0.000e+00", f"{RWR[arm]:.3f}", "是",
               f"{e['instrument_point']:+.6f}", f"{e['published_point']:+.6f}",
               f"{e['point_abs_ratio']:.2f}",
               "是" if e["instrument_crosses_zero"] else "否",
               "是" if e["published_crosses_zero"] else "否",
               "是" if e["instrument_crosses_zero"] == e["published_crosses_zero"]
               else "否（仪器偏乐观）"]) + "\n"

t4 = head(["epoch", "臂", "配对口径（逐 seed 中位）", "差值口径（中位数相减）",
            "配对原始点", "逐 seed 同号", "护栏理由"])
for ep in EPOCHS:
    for arm in SIX:
        c = CONV[ep][arm]
        t4 += row([ep, NAMES[arm],
                   f"{c['paired_raw_median_of_per_seed']:+.6f}",
                   f"{c['difference_raw_median_minus_control_median']:+.6f}",
                   f"{c['paired_raw_mean_of_per_seed']:+.6f}",
                   f"{c['raw_same_sign_seeds']}/3",
                   G[ep][arm]["verdict"]]) + "\n"

# --------------------------------------------------------------- block texts
handover_body = f"""**㉜（2026-11-15 00:3x 第 69 轮新增证据（epoch 轴 × S3 等偏置护栏：首次把「增益是不是面积/幅度效应」这条真正定案的护栏用到 ep01/ep02））**——脚本 `tools/_e45_verify_round69.py`（只读本地，产物 `tools/_evals/e45/round69_epoch_guard_axis.json`，**VERDICT: PASS 19/19**）。**E45 本身无新活动**（提交器 273、训练数组 275/283、评测 274/282/290、汇总 291、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` 21 份评测齐；`runs/e45/` 13 份 json 未变——`scores_summary.json` md5 `57bcd511a537ab85`、`s3_equalbias.json` md5 `3aba999d5ef6dcc4a`；队列里的 8 个 `qlfn2-f36` 作业来自**另一个远端仓库** `/home/wf/radarNWPQML`，与 E45 无关）⇒ **裁决零变化：六臂仍全部判负、合格臂集合仍为空、不投十 seed 算力、MASTER 不升版（仍 v2.48）、`docs/` 未改**。

1. **为什么这一轮值得做**：第 65–68 轮把 epoch 轴（ep00/ep01/ep02 × 7 臂 × 3 seed）建起来并判了**原始 CSI/BIAS 与机制探针**，但**从未把 S3 等偏置护栏用过 ep01/ep02**——而「增益是不是面积/幅度效应」正是 E45 判负的根据 ⇒ 留下的问题是「六臂判负能否跨 epoch 不变」。

2. **仪器选择与保真度（用仪器前先在 ep00 标定）**：已发布的 S3 是**场域**做法（对预报场做全局标量配平），需要原始预报场，epoch 轴产物里没有 ⇒ 本轮用第 54 轮建好的**列联表域对应物**（把臂的预报事件面积缩放到对照的面积、真值事件数 T=h+m 保持不变）。在 ep00 上的保真度：
   - **原始侧可用**：臂级点估计与已发布 S3 **max|diff| = 0.000e+00**（6 臂）；区间宽度比 **0.994–1.015**；**跨零判定 6/6 一致**。
   - **等偏置侧不可用于显著性**：点估计符号 6/6 一致，但 |点| **6/6 系统性更大**（比值 A1 8.20／A2 1.91／A3a 7.17／A3b 2.26／A4 3.27／A4b 2.15）⇒ 场域配平比面积配平**多削掉一块增益**。
   - **决定性的一格（仪器局限登记）**：**A2 在 ep00 被本仪器判成 `GAIN_SURVIVES_EQUALISATION`**（等偏置点 +0.007809、CI [+0.001300,+0.015519]），而已发布的场域护栏明确拒绝它（点 +0.004086、CI [−0.001226,+0.011113]，跨零）⇒ **本仪器在等偏置一侧偏乐观，其「配平后仍存活」一律不可采信**；**已发布场域护栏仍是唯一权威**（本轮 18 格里唯一的「存活」就是这一格 ⇒ 扣除它后**任一 epoch 都没有合格臂**）。

3. **裁决随 epoch 不变，但判负的「理由」随 epoch 迁移（本轮最硬的新事实）**：
   - **ep00**：4 臂 `GAIN_WAS_AREA_OR_AMPLITUDE`（A1/A3a/A3b/A4）＋ 2 臂 `NO_GAIN_TO_BEGIN_WITH`（A2/A4b）——与已发布逐臂**完全一致**。
   - **ep01**：**六臂全部 `NO_GAIN_TO_BEGIN_WITH`**，且其中 **A1、A2、A3b 三臂显著低于对照**（原始区间不跨零、逐 seed **3/3 同号为负**）；ep01 对照几乎不塌（中位 0.121067 ＝ ep00 的 ×0.994），**没有任何一臂的中位 CSI 超过对照** ⇒ **机制臂在「晚一个 epoch」上先掉技能**。
   - **ep02**：六臂中**五臂**（A1/A3a/A3b/A4/A4b）转成 `GAIN_WAS_AREA_OR_AMPLITUDE`，其中 **A3a、A4b 的原始增益区间不跨零**（逐 seed 3/3 同号）⇒ **「增益是面积/幅度效应」在第二个 epoch 上独立复现**；但 ep02 **对照塌到中位 0.055067（×0.452）、中位 BIAS 0.1486（欠预报）**，六臂全部高于对照 ⇒ 这批「增益」是**对照崩塌造成**的相对优势，等偏置后点估计转负（A3a −0.007773、A4b −0.021629）⇒ **仍然只是面积/幅度效应，外加一条「对照崩塌效应」，不是机制成立**。

4. **两个独立算出的零模型地板互相印证**：本轮零模型（把对照臂换 seed，3 种置换、两层自助）的 CI 半宽中位 **ep00 0.029795／ep01 0.059205／ep02 0.058327**，与第 68 轮独立算出的零模型地板 **0.030118／0.060224／0.058907** 相差 **1.0%–1.7%**；且零模型在每个 epoch 上 **0/3 置换产生不跨零区间** ⇒ **两层自助程序在「对照臂换种子」这个零假设上完全校准**。

5. **口径纪律在 epoch 轴上再次被触发**：ep02 的 **A2** 在配对口径（−0.015639）与差值口径（+0.010060）**反向**、ep01 的 **A4b** 也反向（+0.005096／−0.024346）⇒ 第 44 轮那条「配对口径与差值口径必须并列报」再次生效；n=3 时区间的分辨率下界仍是 **1/8**（第 64 轮纪律）；仍按「**中位数 + 通过率**」报，单点分数不作裁决。

6. **守卫**：S3 **12/12 区间仍全部跨零**、六臂 `s3_verdict` 逐臂不变（4 × `GAIN_WAS_AREA_OR_AMPLITUDE` ＋ 2 × `NO_GAIN_TO_BEGIN_WITH`）、**合格臂集合仍为空**；已发布头条表七臂中位 CSI 与过地板通过率复算 **max|diff| = 0.000e+00**；ep00 重跑的 21 格 `CSI@2.0` 与已发布逐 seed 值 **max|diff| = 0.000e+00**；本轮为只读（运行后复核两份已发布 json 的 md5 未变）。

{t1}
{t2}
{t3}
{t4}
> **边界声明**：本轮是**事后诊断**，**不构成新判据、不改变任何裁决**；六臂仍全部判负、合格臂集合仍为空、**不投十 seed 算力**；`docs/MASTER.md` **不升版（仍 v2.48）**，`docs/` 未改（已发布头条表与裁决数值零变化 ⇒ 无需 Gate 3 写回）。"""

mem_body = f"""### 2026-11-15｜第 69 轮（epoch 轴 × S3 等偏置护栏：护栏首次落到 ep01/ep02 —— 裁决不变，理由迁移）

- **判定**：远端 **E45 本身全部完成且与第 50–68 轮完全一致**（273/274/275/282/283/290/291/297/298 全 `COMPLETED`；`runs/e45/evals/` 21 份齐；`runs/e45/` 13 份 json 未变，`scores_summary.json` md5 `57bcd511a537ab85`、`s3_equalbias.json` md5 `3aba999d5ef6dcc4a`）⇒ **E45 无新活动**（队列里的 `qlfn2-f36` 作业属另一远端仓库 `/home/wf/radarNWPQML`，与 E45 无关）。**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；**未改 `docs/`**（MASTER 仍 v2.48）。
- **本轮补上的洞**：第 65–68 轮只判了 epoch 轴的原始 CSI/BIAS 与机制探针，**从未把真正定案的 S3 等偏置护栏用到 ep01/ep02** ⇒ 本轮补上（脚本 `tools/_e45_verify_round69.py`，**VERDICT: PASS 19/19**，产物 `tools/_evals/e45/round69_epoch_guard_axis.json`）。
- **仪器标定（ep00 上先证伪再用）**：用第 54 轮的列联表域对应物（已发布 S3 是场域做法，需要原始预报场，epoch 轴产物没有）。**原始侧可用**（臂级点估计与已发布 S3 max|diff| = 0.000e+00、区间宽度比 0.994–1.015、跨零判定 6/6 一致）；**等偏置侧不可用于显著性**（符号 6/6 一致但 |点| 6/6 系统性更大）⇒ **A2 在 ep00 被本仪器误判成 `GAIN_SURVIVES_EQUALISATION`（已发布场域护栏明确拒绝）** ⇒ 记为仪器局限，**已发布护栏仍是唯一权威**。
- **三条新事实**：① **裁决跨 epoch 不变**：3 epoch × 6 臂 = 18 格里唯一的「存活」就是上面那格已知的仪器偏乐观格 ⇒ **任一 epoch 都没有合格臂**。② **判负的理由随 epoch 迁移**：ep00 ＝ 4 臂面积/幅度 + 2 臂本就无增益（与已发布逐臂一致）；ep01 ＝ **六臂全部无增益**，其中 A1/A2/A3b **显著低于对照**（3/3 seed 同号为负、原始区间不跨零），且对照不塌（×0.994）⇒ **机制臂晚一个 epoch 先掉技能**；ep02 ＝ **五臂转成面积/幅度**（A3a/A4b 原始区间不跨零）⇒ 面积/幅度裁决**在第二个 epoch 上独立复现**，但**对照塌到 ×0.452、中位 BIAS 0.1486**，六臂全部高于对照 ⇒ 这批增益是**对照崩塌**造成的相对优势，等偏置后转负 ⇒ **不是机制成立**。③ **两个独立算出的零模型地板互相印证**：本轮 0.029795／0.059205／0.058327 vs 第 68 轮 0.030118／0.060224／0.058907（差 1.0–1.7%），且零模型在每个 epoch 上 **0/3** 产生不跨零区间 ⇒ **两层自助在零假设上完全校准**。
- **守卫**：S3 **12/12 区间仍全部跨零**、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；头条表复算 max|diff| = 0.000e+00；ep00 重跑 21 格 max|diff| = 0.000e+00；只读（运行后 md5 复核未变）。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报；② 配对口径与差值口径并列（ep02 的 A2、ep01 的 A4b 在两口径下反向）；③ 事后诊断、**不构成新判据**、**不改变任何裁决**；④ **新登记两条纪律**：**凡用「对应物/替代仪器」下显著性结论，必须先在同一数据上做保真度标定，并把分歧登记为仪器局限**；**护栏的「裁决」与「理由」必须分开报告**。
"""

short_body = f"""- **第 69 轮（2026-11-15 00:3x，epoch 轴 × S3 等偏置护栏）**：远端 E45 **无新活动**（273/274/275/282/283/290/291/297/298 全 `COMPLETED`；21 份评测齐；13 份 json md5 未变：`scores_summary.json` `57bcd511a537ab85`、`s3_equalbias.json` `3aba999d5ef6dcc4a`；队列里的 `qlfn2-f36` 属另一远端仓库 `/home/wf/radarNWPQML`，与 E45 无关）。脚本 `tools/_e45_verify_round69.py`（只读本地，**VERDICT: PASS 19/19**，产物 `tools/_evals/e45/round69_epoch_guard_axis.json`）。**补上的洞**：第 65–68 轮只判了 epoch 轴的原始 CSI/BIAS 与机制探针，**从未把真正定案的 S3 等偏置护栏用到 ep01/ep02**。**仪器标定（ep00 先证伪再用）**：用第 54 轮列联表域对应物（已发布 S3 是场域做法、需原始预报场）——**原始侧可用**（点估计 max|diff| = 0.000e+00、区间宽度比 0.994–1.015、跨零判定 6/6 一致），**等偏置侧不可用于显著性**（符号 6/6 一致但 |点| 6/6 更大）⇒ **A2 在 ep00 被本仪器误判成 `GAIN_SURVIVES_EQUALISATION`（已发布护栏明确拒绝）** ⇒ 记为仪器局限。**三条新事实**：① **裁决跨 epoch 不变**（18 格里唯一的「存活」就是那格已知偏乐观格 ⇒ **任一 epoch 都无合格臂**）；② **判负理由随 epoch 迁移**：ep00 ＝ 4 臂面积/幅度 + 2 臂无增益（与已发布逐臂一致）；ep01 ＝ **六臂全部无增益**、A1/A2/A3b **显著低于对照**（3/3 同号为负、原始区间不跨零）且对照不塌（×0.994）⇒ **机制臂晚一个 epoch 先掉技能**；ep02 ＝ **五臂转成面积/幅度**（A3a/A4b 原始区间不跨零）⇒ **面积/幅度裁决在第二个 epoch 独立复现**，但对照塌到 ×0.452、中位 BIAS 0.1486、六臂全高于对照 ⇒ **是「对照崩塌」造成的相对优势，等偏置后转负，不是机制成立**；③ **两个独立算出的零模型地板互相印证**（本轮 0.029795/0.059205/0.058327 vs 第 68 轮 0.030118/0.060224/0.058907，差 1.0–1.7%），零模型每 epoch **0/3** 产生不跨零区间 ⇒ **两层自助在零假设上完全校准**。**口径**：ep02 的 A2、ep01 的 A4b 在配对/差值两口径下反向 ⇒ 第 44 轮纪律再次生效。**裁决零变化**：S3 **12/12 跨零**、合格臂集合仍为空、**六臂判负、不投十 seed 算力、MASTER 仍 v2.48 不升版**、`docs/` 未改。**纪律两条**：用「替代仪器」下显著性结论前必须先在同一数据上做保真度标定并把分歧登记为仪器局限；护栏的「裁决」与「理由」必须分开报告。"""

text111 = f"""{BEGIN_111}
- **第 69 轮（2026-11-15 00:3x，epoch 轴 × S3 等偏置护栏）**：**S3 等偏置护栏第一次落到 epoch 轴的 ep01/ep02**（脚本 `tools/_e45_verify_round69.py`，**VERDICT: PASS 19/19**，产物 `tools/_evals/e45/round69_epoch_guard_axis.json`）。**裁决跨 epoch 不变**（18 格里唯一的「存活」是 A2@ep00，且那格已被登记为**仪器偏乐观**、已发布场域护栏明确拒绝它 ⇒ **任一 epoch 都无合格臂**）；**但判负的理由随 epoch 迁移**：ep00 ＝ 4 臂面积/幅度 + 2 臂无增益（与已发布逐臂一致）；ep01 ＝ 六臂全部无增益、A1/A2/A3b **显著低于对照**（3/3 同号为负）且对照不塌 ⇒ **机制臂晚一个 epoch 先掉技能**；ep02 ＝ 五臂转成面积/幅度（A3a/A4b 原始区间不跨零）⇒ **面积/幅度裁决在第二个 epoch 独立复现**，但对照塌到 ×0.452 ⇒ **是「对照崩塌」的相对优势，不是机制成立**。**仪器标定**：原始侧与已发布 S3 逐位一致（点 max|diff| = 0、跨零判定 6/6），**等偏置侧不可用于显著性**。**裁决零变化**（S3 12/12 跨零、六臂判负、不投十 seed 算力、MASTER 仍 v2.48 不升版）。
{END_111}
"""

# ------------------------------------------------------------------- write
written = []
for path, body in [
    (UP / "HANDOVER_E45.md", handover_body),
    (UP / "MEMORY.md", mem_body),
    (UP / "2026-11-15.md", short_body),
    (PROJ / "2026-11-15.md", short_body),
    (AUTO, mem_body),
]:
    p, n = append_block(path, body)
    written.append((p.name, n))

n111 = append_block_custom(UP / "2026-11-11.md", BEGIN_111, END_111, text111)
written.append(("2026-11-11.md", n111))

print("written:")
for name, n in written:
    print(f"  {name}: stripped {n} old block(s)")
print()
for path in [UP / "HANDOVER_E45.md", UP / "MEMORY.md", UP / "2026-11-15.md",
             UP / "2026-11-11.md", PROJ / "2026-11-15.md", AUTO]:
    raw = path.read_bytes()
    print(f"{hashlib.sha256(raw).hexdigest()}  {path}")
