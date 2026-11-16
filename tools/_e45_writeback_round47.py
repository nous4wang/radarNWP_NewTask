"""E45 round-47 write-back (idempotent, marker-guarded).

Round-47 = new storm-event (per-case) axis + arithmetic re-derivation of the
published headline table.  No verdict change, no number change => MASTER is NOT
re-versioned; evidence goes to HANDOVER §4⑫, upper MEMORY.md status section and
the three daily logs (project-level + upper-level), plus the automation memory.

Idempotency rules learned in rounds 41/43/45:
  * the guard marker MUST be an exact substring of the inserted text;
  * each target needs its own guard (a shared marker only guards one file);
  * after writing, assert the marker occurs exactly once in the target;
  * a second run must report every target as "unchanged".
"""

from pathlib import Path

MARK = "第 47 轮（风暴场次级轴首次判读）"

HANDOVER = Path("R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
MEMORY = Path("R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
LOG_1111 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
LOG_1113 = Path("R:/1radarNWPprediction/.workbuddy/memory/2026-11-13.md")
PROJ_LOG_1113 = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-13.md")
AUTOMATION_MEM = Path("R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/automations/"
                      "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md")

HANDOVER_ANCHOR = "### 下面是结案前的执行记录（存档用，不必重跑）"

HANDOVER_BLOCK = f"""
**⑫（2026-11-13 22:0x {MARK}）风暴场次级（storm-event）轴的首次判读 ＋ 已发布头条表的算术复算**——脚本 `tools/_e45_verify_round47.py`（只读本地，**POS 108/108 VERDICT: PASS**）：前 46 轮判过聚合层／机制探针层／S1 结构量层／窗级共四层，**从未把 538 窗按所属风暴场次（8 个）聚合成事件级指标**。本轮补上这一层（复制单元＝风暴场次，是「某臂是否让一次风暴更准」的天然单位），并首次对已发布的头条表做**算术复算**。

1. **头条表的算术被独立复算通过**：由 `scores_summary_s{{0,1,2}}.json` 逐 seed 重算七臂的 `median_csi_at_2.0`、`median_bias_at_2.0`、过地板通过率、`seed_spread`（max/min）、`n_seeds` ⇒ 与已发布 `scores_summary.json`（MASTER §4.47 卡片的数据源）**逐项一致**（35/35 项，容差 1e−9）⇒ **头条表的算术无误**（此前只核过「文档数字 ↔ JSON」，从未核过「JSON ↔ 逐 seed 原始产物」这一层）。
2. **事件级零模型地板 = 0.010772**：用**对照臂 B0 的两个不同 seed 互比**（纯 seed 噪声、不含任何机制），三对控制的场次级配对中位差为 **−0.005333／−0.010772／−0.002902**，且三对的**场次级 bootstrap CI 全部跨零**、**每对都是 3/8 场次为正**。
3. **六臂的场次级效应全部落在零模型地板之内** ⇒ **六臂判负在风暴事件级独立复现**：场次级配对中位差（三 seed 的中位数）A1 **+0.003400**、A2 **−0.008177**、A3a **+0.002913**、A3b **+0.009664**、A4 **+0.009857**、A4b **+0.007783**，**绝对值均 ≤ 0.010772**（地板）；差值口径 A3a +0.005367／A3b +0.007040／A4 +0.005075 为正、其余为负，量级同样在地板上下。
4. **BIAS 爆表在事件级复现**：事件级 BIAS 中位数相对对照的比值＝**A4b 4.99×、A4 3.33×、A3a 1.73×、A3b 1.62×**（过预报），A1 0.78×／A2 0.83×（欠预报）⇒ **连在事件级看起来「为正」的 A3a／A3b 也是过预报的**（它们在窗级差值口径为负、在事件级为正 ⇒ 两层口径不一致，进一步说明其「增益」不稳）。
5. **E45 首个非伪重复的置信区间**：场次级配对 bootstrap（10000 次、rng 20261113、**复制单元＝风暴场次 n=8**，不是窗）⇒ 18 个（臂×seed）格里**只有 2 格 CI 不跨零：A3a_s0 与 A3b_s0**（其余 16 格跨零，含 A4b_s2 的强负 −0.0446）⇒ **两条不跨零的恰好又是 seed 0**（与第 40 轮「A4 增益 99.2% 来自 seed 0」同向）⇒ **无任何一臂在事件级拿到跨种子的证据**。
   ⚠️ **口径边界**：场次级 CI 只覆盖「同一次采样下跨风暴的内部一致性」，**跨种子不确定性仍是 n=3**，本轮无法消除 ⇒ **不构成新判据、不改变任何裁决**。

**结论**：**裁决不变——六臂全部判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版**（无数值与裁决变化）；待办仍是三件实现缺陷修复 ＋ 重提机制族。
"""

MEMORY_BLOCK = f"""
- **（2026-11-13 22:0x {MARK}：裁决不变，新证据＝风暴场次级（storm-event）轴的首次判读 ＋ 头条表算术复算）** 远端 E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283、评测 274/282/290、汇总 **291**、只读诊断 297/298）、`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全、队列空、提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ 与前四十六轮完全一致；本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。十一套既有闸门（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`／`_e45_verify_round40/43/44/45/46`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。**新脚本 `tools/_e45_verify_round47.py`（只读本地，POS 108/108 VERDICT: PASS）**：**前 46 轮从未把 538 窗按 8 个风暴场次聚合成事件级指标** ⇒ 本轮补上这一层。① **头条表算术复算通过**：由逐 seed 产物重算七臂中位数／过地板通过率／seed 极差／n_seeds ⇒ 与 `scores_summary.json` 逐项一致（35/35）⇒ **MASTER §4.47 卡片的数据源算术无误**（此前只核过「文档↔JSON」，未核过「JSON↔逐 seed 原始产物」）。② **事件级零模型地板 0.010772**（对照臂 B0 换 seed 互比：三对 −0.005333／−0.010772／−0.002902，CI 全跨零，每对 3/8 场次为正）。③ **六臂场次级效应全部落在地板之内**（配对中位差 A1 +0.003400、A2 −0.008177、A3a +0.002913、A3b +0.009664、A4 +0.009857、A4b +0.007783）⇒ **六臂判负在事件级独立复现**。④ **BIAS 爆表在事件级复现**：A4b 4.99×、A4 3.33×、A3a 1.73×、A3b 1.62×（过预报）、A1 0.78×／A2 0.83×（欠预报）⇒ 连「事件级为正」的 A3a／A3b 也是过预报的。⑤ **E45 首个非伪重复的 CI**（复制单元＝风暴场次 n=8）：18 个（臂×seed）格中仅 **A3a_s0 与 A3b_s0** 两格不跨零，**恰好又是 seed 0** ⇒ 无任何一臂拿到跨种子证据。**边界**：事后诊断、**不构成新判据**（S3 护栏固定 thr=2.0 mm、跨种子单元固定为 seed）⇒ **六臂仍全部判负、无一臂进十 seed、不投十 seed 算力**。
"""

LOG_BLOCK = f"""
**补记，{MARK}（2026-11-13 22:0x）**

- **远端**：E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283、评测 274/282/290、汇总 **291**、只读诊断 297/298）；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；队列空；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **与前四十六轮完全一致，无新活动**。本轮**未下载、未 sbatch、未改远端状态、未改 `docs/`**。
- **闸门**：十一套（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`／`_e45_verify_round40/43/44/45/46`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**（无数值与裁决变化）。
- **新证据（风暴场次级轴首次判读 ＋ 头条表算术复算）**——`tools/_e45_verify_round47.py`（只读本地，**POS 108/108 VERDICT: PASS**）：
  1. **头条表算术复算通过**：由 `scores_summary_s{{0,1,2}}.json` 逐 seed 重算七臂的中位数／过地板通过率／seed 极差（max/min）／n_seeds ⇒ 与已发布 `scores_summary.json` **35/35 逐项一致**（容差 1e−9）⇒ MASTER §4.47 卡片的数据源算术无误（此前只核过「文档数字↔JSON」）。
  2. **事件级零模型地板 = 0.010772**（对照臂 B0 换 seed 互比，三对配对中位差 −0.005333／−0.010772／−0.002902；CI 全跨零；每对 3/8 场次为正）。
  3. **六臂场次级效应全部落在地板之内**（配对中位差：A1 +0.003400、A2 −0.008177、A3a +0.002913、A3b +0.009664、A4 +0.009857、A4b +0.007783）⇒ **六臂判负在风暴事件级独立复现**；差值口径只有 A3a/A3b/A4 为正且同样在地板量级。
  4. **BIAS 爆表在事件级复现**：A4b 4.99×、A4 3.33×、A3a 1.73×、A3b 1.62×（过预报），A1 0.78×／A2 0.83×（欠预报）⇒ 连事件级「为正」的 A3a／A3b 也是过预报的。
  5. **首个非伪重复的 CI**（复制单元＝风暴场次 n=8 的配对 bootstrap，10000 次、rng 20261113）：18 个（臂×seed）格仅 **A3a_s0、A3b_s0** 两格不跨零，**恰好又是 seed 0** ⇒ 无一臂拿到跨种子证据。

**口径**：仍按中位数＋通过率报；配对与差值口径并列；场次级轴只作诊断（跨种子不确定性仍是 n=3）。

**边界**：事后诊断，**不构成新判据**（S3 护栏固定 thr=2.0 mm、跨种子单元固定为 seed）⇒ **裁决不变：六臂判负、无一臂进十 seed、不投十 seed 算力**；**MASTER 不升版** ⇒ 新证据只进 `HANDOVER_E45.md` §4⑫、上层 MEMORY.md 状态段与三份逐日日志。
"""

AUTOMATION_BLOCK = f"""
### 2026-11-13 22:0x｜第 47 轮（风暴场次级轴首次判读）：状态复核（无远端改动，裁决不变）＋ 新证据＝风暴场次级（storm-event）轴首次判读 ＋ 头条表算术复算

**判定：远端 E45 全部完成且与前四十六轮完全一致；裁决不变（六臂判负、无一臂进十 seed、不投十 seed 算力）；本轮无远端改动（未 sbatch、未下载、未改远端状态、未动 `docs/`）。**

**远端**：提交器 **273 COMPLETED**（05:04:04）、训练数组 275/283 各 7/7、评测 274/282/290、汇总 **291** 14:23、只读诊断 297/298 全部 `COMPLETED`；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；队列空；日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。

**闸门**：十一套（`_check_tables`／`_check_v245/246/247/248_row`／`_e45_docs_numbers`／`_e45_verify_round40/43/44/45/46`）全部 `VERDICT: PASS` ⇒ MASTER 仍 **v2.48**、**不升版**。

**新证据**——`tools/_e45_verify_round47.py`（只读本地，**POS 108/108 VERDICT: PASS**）：① **头条表算术复算通过**（由逐 seed 产物重算七臂中位数／过地板通过率／seed 极差／n_seeds ⇒ 与 `scores_summary.json` 35/35 一致）；② **事件级零模型地板 0.010772**（对照臂 B0 换 seed 互比，三对配对中位差 −0.005333／−0.010772／−0.002902，CI 全跨零）；③ **六臂场次级效应全部落在地板之内**（配对中位差 A1 +0.003400、A2 −0.008177、A3a +0.002913、A3b +0.009664、A4 +0.009857、A4b +0.007783）⇒ **六臂判负在事件级独立复现**；④ **BIAS 爆表在事件级复现**（A4b 4.99×、A4 3.33×、A3a 1.73×、A3b 1.62× 过预报）；⑤ **首个非伪重复 CI**（复制单元＝风暴场次 n=8）：18 个（臂×seed）格仅 A3a_s0、A3b_s0 两格不跨零，**恰好又是 seed 0**。

**写回**：`HANDOVER_E45.md` §4⑫ ＋ 上层 MEMORY.md 状态段 ＋ 三份逐日日志（脚本 `tools/_e45_writeback_round47.py`，幂等，标记＝插入文本原文子串，连跑两次第二遍全部 `unchanged`，各文件标记计数 == 1）。

**本轮踩坑（可复用规则）**：**正则贪婪会把场次名与补丁序号一起吃掉**——`^(CP_\\d+_\\d+_\\d+)_` 把 538 窗压成 **1 个**场次（脚本报 FAIL 才暴露）⇒ 正则必须锚到分隔符：`^(CP_\\d+_\\d+)_(\\d+):`；另 `pass_rate_over_floor` 在产物里是字符串 `"2/3"` ⇒ 数值比较前必须做 `as_num` 归一；`boot_ci` 的重抽次数与样本数两个参数曾同名 ⇒ 一次跑崩。
"""


def append_block(path, block):
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        return "unchanged"
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + block, encoding="utf-8")
    return "written"


def insert_before(path, anchor, block):
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        return "unchanged"
    if anchor not in text:
        raise SystemExit(f"ANCHOR NOT FOUND in {path}: {anchor!r} (refusing to silently skip)")
    idx = text.index(anchor)
    path.write_text(text[:idx] + block.lstrip("\n") + "\n" + text[idx:], encoding="utf-8")
    return "written"


def main():
    targets = []
    targets.append(("HANDOVER_E45.md", insert_before, HANDOVER, HANDOVER_ANCHOR, HANDOVER_BLOCK))
    targets.append(("MEMORY.md", append_block, MEMORY, None, MEMORY_BLOCK))
    targets.append(("memory/2026-11-11.md", append_block, LOG_1111, None, LOG_BLOCK))
    targets.append(("memory/2026-11-13.md", append_block, LOG_1113, None, LOG_BLOCK))
    targets.append(("project 2026-11-13.md", append_block, PROJ_LOG_1113, None, LOG_BLOCK))
    targets.append(("automation memory", append_block, AUTOMATION_MEM, None, AUTOMATION_BLOCK))

    for name, fn, path, anchor, block in targets:
        if not path.exists():
            raise SystemExit(f"MISSING TARGET: {path}")
        res = fn(path, anchor, block) if anchor else fn(path, block)
        print(f"  {name:>22}: {res}")
        # invariant: the guard marker must appear exactly once in the target
        n = path.read_text(encoding="utf-8").count(MARK)
        if n != 1:
            raise SystemExit(f"MARKER COUNT != 1 in {path}: {n}")
    print("all targets written/unchanged; marker count == 1 in every target")


if __name__ == "__main__":
    main()
