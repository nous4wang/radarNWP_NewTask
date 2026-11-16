"""E45 第 16 次独立复核的记录写入 + MEMORY.md 轮次标注自检。

一次完成：
  1) 自动化记忆文件追加「第 16 次独立复核」分节（标题带时间戳，作为轮次事实来源）；
  2) 上层与项目级当日日志 2026-11-12.md 各追加同一结论段；
  3) 上层 MEMORY.md 状态段追加第 16 次复核注记（**带轮次标注自检**：把自动化记忆文件
     的「时间戳 → 轮次」映射与 MEMORY.md 现有标注逐条比对，错标即报错）；
  4) 幂等：已存在标记则跳过，不重复写。

锚点用「前十四轮复核结论完全一致，无新增偏离。）**」这一段末尾文本（唯一），
不使用「第 N 次」这类会被后续轮次复用的序号做幂等锚点（第 15 轮踩过的坑）。
"""
import os
import re
from datetime import datetime

AUTO = r"R:\1radarNWPprediction\radar-nwp-bench\.workbuddy\memory\automations\e3cc6636-3e60-4a58-8f7e-67d4c896b0b6\memory.md"
MEM = r"R:\1radarNWPprediction\.workbuddy\memory\MEMORY.md"
UPPER_LOG = r"R:\1radarNWPprediction\.workbuddy\memory\2026-11-12.md"
WS_LOG = r"R:\1radarNWPprediction\radar-nwp-bench\.workbuddy\memory\2026-11-12.md"

TS = datetime.now().strftime("%Y-%m-%d %H:%M")
ROUND = 16
MARK = f"{TS} 第 {ROUND} 次独立复核"

AUTO_SECTION = f"""
---

## {TS}（本次运行 · 第 {ROUND} 次独立复核）

**判定：远端 E45 全部完成且与前十五轮完全一致；裁决与写回均无需改动；本轮无远端改动（未下载、未 sbatch、未改 docs）。**

- 远端：提交器 **273** `COMPLETED`（05:04:04）、训练数组 275/283、评测 274/282/290 全 `COMPLETED`、汇总 **291** `COMPLETED`（00:14:23）、**21 份评测齐**、**队列空**；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- **md5 13/13 全同**：远端 `runs/e45/` 13 个 json 与本地 `tools/_evals/e45/` 13 个 json 逐一相同 ⇒ 无需重新下载。
- **S3 独立重算**（新脚本 `tools/_e45_verify_round16.py`，不 import 旧脚本）：阈值 2.0 mm、epoch 00、control=B0、n_boot 10000、rng 20261109；六臂**原始增益 CI 与配平后 CI 全部跨零**，12 组 `crosses_zero` 标志与重算逐项一致；A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH` ⇒ **六臂判负、不投十 seed 算力**成立。
- **3-seed 中位数 / 过地板通过率**（地板 0.0924、Oracle 0.1653）：B0 0.1218/2-3、A1 0.1184/2-3、A2 0.1120/2-3、A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3 ⇒ 中位数高于对照的只有 A4/A4b，**BIAS 同时爆表**（0.8362 / 2.8215，对照 0.5095）；**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳，不构成晋级依据。
- **机制读数逐臂逐 seed 重算**（键在 `arms/<臂>/readouts/`）：门 A1 `corr(mask,amp)` +0.6103/+0.6145/+0.6881、A2 +0.4035/+0.3113/+0.3879（全 ≥0），`mask_at_peak` 0.7224～1.0（全 >0.5）⇒ **两臂 3/3 PASS，门指向结构上已解决**；A3a `grad_top1pct_share` 0.5526/0.5666/0.4782、A3b 0.9712/0.9915/0.9775，`field_max_mm` 三 seed **全 402.4288 > 30** ⇒ **两臂 0/3 PASS，拓扑支结掉**；A4 `scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384、`scale_is_constant` 全 False ⇒ **3/3**；**A4b 只 2/3**（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）；`mean_out_equals_T` A4 [True,False,True]、A4b 全 False ⇒ 仍属实现缺陷；**`maxratio_as_trained` 三 seed 全 `nan`（读数缺失）**。
- **写回复核**：`_sync_v245.py` 连跑 **0 改动**（幂等，五个目标文件均 `unchanged`）、`_check_tables.py` **VERDICT: PASS**（`introduced=[]`）、`_check_v245_row.py` **VERDICT: PASS**（v2.44/v2.45 行均 4 cells，与 §12 表头一致）；MASTER 头部 **v2.45**、§4.47、§9 E45 行、§12 v2.44/v2.45 两行、INDEX E45 行全部在位。

### 本轮新增可复用经验（本轮脚本自身踩的坑）

- **按 seed 分文件的键格式是 `<臂>_s<seed>_ep00`**（如 `A1_s0_ep00`），**不是** `<臂>` 直挂字典 ⇒ 第 16 轮脚本首版按 `tok[1] == "s"` 切分，解析出 **0 条**记录（真实是 `tok[1] == "s0"`）。**根因与既往同源：凭猜写键路径而非先转储结构。**
- **"互校一致"在空集合上会空洞通过**：首版按 seed 分文件解析失败后，逐记录互校对 **0 条**记录比较 ⇒ 报 PASS（假通过）。**正确做法：互校断言必须同时校验「实际比对了 N 条」，N=0 时直接失败**；本轮已加 `sa_cmp == 21` 与 `len(per_seed) == 21` 两条计数断言。
- **累计型产物文件要认准「累积」语义**：`scores_summary_s1.json` 含 s0+s1、`_s2.json` 含 s0+s1+s2（**`_s2.json` 与合并版 `scores_all.json` 的 md5 完全相同**）⇒ 合并时按 (臂, seed) 取值并对重复处做一致性核对，而不是按文件名里的 seed 直接覆盖。
"""

LOG_SECTION = f"""
### {TS} 第 {ROUND} 次独立复核（E45 远端状态监控）

- **远端**：提交器 **273** `COMPLETED`（05:04:04）、训练数组 275/283、评测 274/282/290 全 `COMPLETED`、汇总 **291** `COMPLETED`（00:14:23）、**21 份评测齐**、队列空；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- **md5 13/13 全同**：远端 `runs/e45/` 13 个 json 与本地 `tools/_evals/e45/` 13 个 json 逐一相同 ⇒ 无需重新下载，**未下载、未 sbatch、未改 docs**。
- **S3 独立重算**（新脚本 `tools/_e45_verify_round16.py`，不 import 旧脚本）：六臂**原始增益 CI 与配平后 CI 全部跨零**，12 组 `crosses_zero` 标志与重算逐项一致；A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH` ⇒ **六臂判负、不投十 seed 算力**成立。
- **3-seed 中位数 / 过地板通过率**（地板 0.0924、Oracle 0.1653）：B0 0.1218/2-3、A1 0.1184/2-3、A2 0.1120/2-3、A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3 ⇒ 中位数高于对照的只有 A4/A4b，**BIAS 同时爆表**（0.8362 / 2.8215，对照 0.5095）；**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳。
- **机制读数逐臂逐 seed 重算**（键在 `arms/<臂>/readouts/`）：门 A1 `corr(mask,amp)` +0.6103/+0.6145/+0.6881、A2 +0.4035/+0.3113/+0.3879（全 ≥0），`mask_at_peak` 0.7224～1.0（全 >0.5）⇒ 两臂 3/3 PASS；A3a `grad_top1pct_share` 0.5526/0.5666/0.4782、A3b 0.9712/0.9915/0.9775，`field_max_mm` 三 seed **全 402.4288 > 30** ⇒ 两臂 0/3 PASS，拓扑支结掉；A4 `scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384、`is_constant` 全 False ⇒ 3/3；**A4b 只 2/3**（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）；`maxratio_as_trained` 三 seed 全 `nan`；`mean_out_equals_T` A4 [True,False,True]、A4b 全 False ⇒ 仍属实现缺陷。
- **写回复核**：`_sync_v245.py` 连跑 **0 改动**（幂等）、`_check_tables.py` **VERDICT: PASS**、`_check_v245_row.py` **VERDICT: PASS**；MASTER 头部 **v2.45**、§4.47、§9、§12 v2.44/v2.45、INDEX E45 行全部在位。
- **本轮新增经验**：① 按 seed 分文件的键格式是 `<臂>_s<seed>_ep00`，且 s1/s2 是**累积**文件（`_s2.json` 与 `scores_all.json` md5 相同）；② **互校断言必须带「实际比对了 N 条」计数**，否则空集合会空洞通过（本轮首版即栽在这：解析 0 条却报 PASS）；③ 结论与待办（不变）——六臂判负、不投十 seed 算力；门指向结构上已解决但不是瓶颈；拓扑支按预注册结掉；量通道不是瓶颈（A4 三 seed 成立、A4b 只 2/3 成立）；重开须重新提机制族。待办四：三件实现缺陷（幅度截断未作用于输出场／`mean(out)=T`／`maxratio_as_trained` 空值）＋ 判据工具区分「臂不稳」与「地板口径不稳」。
"""

MEM_NOTE = (
    f" **（{TS} 第 {ROUND} 次独立复核：远端 273/291 全 `COMPLETED`、队列空、**21 份评测齐**、"
    "提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、"
    "远端 `runs/e45/` 13 个 json 与本地 md5 **13/13 全同**；S3 独立重算（新脚本 "
    "`tools/_e45_verify_round16.py`，不 import 旧脚本）：六臂**原始增益 CI 与配平后 CI 全部跨零**，"
    "12 组 `crosses_zero` 标志与重算逐项一致，A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`、"
    "A2/A4b = `NO_GAIN_TO_BEGIN_WITH`；3-seed 中位数/过地板通过率（地板 0.0924、Oracle 0.1653，"
    "**由 `scores_summary_s{0,1,2}.json` 三份按 seed 分文件重算**并与 `scores_all.json` 逐记录互校、"
    "与 `scores_summary.json` 中位数互校一致）：B0 0.1218/2-3、A1 0.1184/2-3、A2 0.1120/2-3、"
    "A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3 ⇒ 中位数高于对照者仅 A4/A4b 且 "
    "BIAS 同时爆表（0.8362 / 2.8215）、**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳；"
    "机制读数逐臂逐 seed 重算与 MASTER v2.45 逐项一致（门 A1 +0.6103/+0.6145/+0.6881、"
    "A2 +0.4035/+0.3113/+0.3879 全 ≥0，`mask_at_peak` 0.72～1.0 全 >0.5 ⇒ 3/3；A3a 0.5526/0.5666/0.4782、"
    "A3b 0.9712/0.9915/0.9775，`field_max_mm` 三 seed 全 402.4288 > 30 ⇒ 0/3；A4 `scale_cv` 2.0040/1.2358/1.7046、"
    "`scale_truth_corr` 0.7429/0.5521/0.6384、`is_constant` 全 False ⇒ 3/3，**A4b 只 2/3**"
    "（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）；`maxratio_as_trained` 三 seed 全 `nan`；"
    "`mean_out_equals_T` A4 [True,False,True]、A4b 全 False ⇒ 仍属实现缺陷）；写回脚本连跑 **0 改动**、"
    "两个校验器 `VERDICT: PASS` ⇒ 结论无误，**未改 docs、未下载、未 sbatch**。"
    "本轮新脚本自身修了两个口径 bug：① 按 seed 分文件的键格式是 `<臂>_s<seed>_ep00`（首版按 "
    "`tok[1]==\"s\"` 切分解析出 0 条）；② **互校断言缺「实际比对了 N 条」计数，空集合会空洞通过**"
    "（首版 0 条比较却报 PASS）⇒ 已加 `len(per_seed)==21` 与 `sa_cmp==21` 两条计数断言。"
    "前十五轮复核结论完全一致，无新增偏离。）**"
)

ANCHOR = "前十四轮复核结论完全一致，无新增偏离。）**"


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def append(p, text, marker):
    if os.path.exists(p):
        s = read(p)
        if marker in s:
            print(f"  skip (already present): {p}")
            return False
    else:
        s = ""
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s.rstrip("\n") + "\n" + text)
    print(f"  appended: {p}")
    return True


print("=== 1) 自动化记忆文件 ===")
append(AUTO, AUTO_SECTION, f"## {TS}（本次运行 · 第 {ROUND} 次独立复核）")

print("=== 2) 当日日志（上层 + 项目级） ===")
append(UPPER_LOG, LOG_SECTION, f"### {TS} 第 {ROUND} 次独立复核（E45 远端状态监控）")
append(WS_LOG, LOG_SECTION, f"### {TS} 第 {ROUND} 次独立复核（E45 远端状态监控）")

print("=== 3) 上层 MEMORY.md 状态段追加注记 ===")
mem = read(MEM)
if MARK in mem:
    print("  skip (already present)")
else:
    if mem.count(ANCHOR) != 1:
        raise SystemExit(f"锚点不唯一（出现 {mem.count(ANCHOR)} 次），拒绝写入：{ANCHOR!r}")
    mem = mem.replace(ANCHOR, ANCHOR + MEM_NOTE)
    with open(MEM, "w", encoding="utf-8", newline="\n") as f:
        f.write(mem)
    print("  appended: MEMORY.md 状态段")

print("=== 4) 轮次标注自检（事实来源＝自动化记忆文件标题） ===")
auto = read(AUTO)
mapping = {ts: int(n) for ts, n in re.findall(
    r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2})（本次运行 · 第 (\d+) 次独立复核", auto, re.M)}
print("  自动化记忆文件轮次映射:", ", ".join(f"{t}=第{n}次" for t, n in mapping.items()))
mem_pairs = re.findall(r"（(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) 第 (\d+) 次独立复核", read(MEM))
bad = []
for ts, n in mem_pairs:
    truth = mapping.get(ts, "?")
    if str(truth) != n:
        bad.append((ts, n, truth))
    print(f"   MEMORY.md {ts} -> 第 {n} 次  {'OK' if str(truth) == n else f'MISLABEL(应为第 {truth} 次)'}")
if bad:
    raise SystemExit(f"轮次标注错位，停止：{bad}")
print(f"  自检通过：{len(mem_pairs)} 条标注全部对位（最新 {TS} = 第 {ROUND} 次）")
