"""E45 第 15 次复核：本轮结论追加到三处记忆文件 + 上层 MEMORY.md 复核注记更新。

规则：幂等（若已含「第 15 次」则跳过追加）；找不到标记时报错不静默跳过。
"""
import os
import re

ROUNDS = 15
STAMP = "2026-11-12 05:59"

SECTION = """
## {stamp}（本次运行 · 第 {r} 次独立复核）

**判定：远端 E45 全部完成且与前十四轮完全一致；裁决与写回均无需改动；本轮无远端改动（未下载、未 sbatch、未改 docs）。**

- 远端：提交器 **273** `COMPLETED`（05:04:04）、训练数组 275/283、评测 274/282/290 全 `COMPLETED`、汇总 **291** `COMPLETED`（00:14:23）、**21 份评测齐**、**队列空**；提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`。
- **md5 13/13 全同**：远端 `runs/e45/` 13 个 json 与本地 `tools/_evals/e45/` 13 个 json 逐一相同（本地目录 14 项 = 13 json + `_s2_scores_raw.txt`）⇒ 无需重新下载。
- **S3 独立重算**（新脚本 `tools/_e45_verify_round15.py`，不 import 旧脚本、直接从原始产物重取）：六臂**原始增益 CI 与配平后 CI 全部跨零**，12 组 `crosses_zero` 标志与重算逐项一致；A1/A3a/A3b/A4 = `GAIN_WAS_AREA_OR_AMPLITUDE`，A2/A4b = `NO_GAIN_TO_BEGIN_WITH` ⇒ **六臂判负、不投十 seed 算力**成立。
- **3-seed 中位数 / 过地板通过率**（地板 0.0924、Oracle 0.1653；由 `scores_all.json` 重算并与 `scores_summary.json` **互校一致**）：B0 0.1218/2-3、A1 0.1184/2-3、A2 0.1120/2-3、A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3 ⇒ 中位数高于对照的只有 A4/A4b，**BIAS 同时爆表**（0.8362 / 2.8215，对照 0.5095）；**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳，不构成晋级依据。
- **机制读数逐臂逐 seed 重算**（键在 `arms/<臂>/readouts/`）：门 A1 `corr(mask,amp)` +0.6103/+0.6145/+0.6881、A2 +0.4035/+0.3113/+0.3879（全 ≥0），`mask_at_peak` 0.7224～1.0（全 >0.5）⇒ **两臂 3/3 PASS，门指向结构上已解决**；A3a `grad_top1pct_share` 0.5526/0.5666/0.4782、A3b 0.9712/0.9915/0.9775，`field_max_mm` 三 seed **全 402.4288 > 30** ⇒ **两臂 0/3 PASS，拓扑支结掉**；A4 `scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384、`scale_is_constant` 全 False ⇒ **3/3**；**A4b 只 2/3**（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）；`mean_out_equals_T` A4 [True,False,True]、A4b 全 False ⇒ 仍属实现缺陷；**`maxratio_as_trained` 三 seed 全 `nan`（读数缺失）**。
- **写回复核**：`_sync_v245.py` 连跑 **0 改动**（幂等，五个目标文件均 `unchanged`）、`_check_tables.py` **VERDICT: PASS**（`introduced=[]`）、`_check_v245_row.py` **VERDICT: PASS**（v2.44/v2.45 行均 4 cells，与 §12 表头一致）；MASTER 头部 **v2.45**、§4.47、§9 E45 行、§12 v2.44/v2.45 两行、INDEX E45 行、`MEMORY.md` 指针 v2.45 全部在位。

### 本轮新增可复用经验

- **断言"读数缺失"必须区分 None / NaN / 有效值三态**：本轮首跑 `maxratio_as_trained` 写成 `not isinstance(x,(int,float))`，而实际值是 **NaN（float）** ⇒ 被判成"有值"从而把预注册负结果刷红（VERDICT: FAIL）。正确写法 `x is None or (isinstance(x,(int,float)) and x != x)`。**负结果断言也要自己会失败**：判据脚本把"该失败的没失败"和"该通过的没通过"混在一个红绿通道里时，一次误报就会让整份复核的告警意义失效。

### 结论与待办（不变）

六臂判负、不投十 seed 算力；门指向结构上已解决但不是瓶颈；拓扑支按预注册结掉；量通道不是瓶颈（A4 三 seed 成立、A4b 只 2/3 成立）；重开须重新提机制族。待办四：三件实现缺陷（幅度截断未作用于输出场／`mean(out)=T`／`maxratio_as_trained` 空值）＋ 判据工具区分「臂不稳」与「地板口径不稳」。
""".format(stamp=STAMP, r=ROUNDS)

APPEND_TARGETS = [
    r"R:\1radarNWPprediction\radar-nwp-bench\.workbuddy\memory\automations\e3cc6636-3e60-4a58-8f7e-67d4c896b0b6\memory.md",
    r"R:\1radarNWPprediction\radar-nwp-bench\.workbuddy\memory\2026-11-12.md",
    r"R:\1radarNWPprediction\.workbuddy\memory\2026-11-12.md",
]

MARK = "第 15 次独立复核"
MEMORY_MD = r"R:\1radarNWPprediction\.workbuddy\memory\MEMORY.md"


def append_once(path, text, mark):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cur = f.read()
        if mark in cur:
            print(f"  SKIP (already has marker): {path}")
            return
        with open(path, "a", encoding="utf-8") as f:
            if not cur.endswith("\n"):
                f.write("\n")
            f.write(text)
        print(f"  APPENDED: {path}")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.lstrip("\n"))
        print(f"  CREATED: {path}")


print("=== 追加本轮结论到记忆文件 ===")
for p in APPEND_TARGETS:
    append_once(p, SECTION, MARK)

print("=== 更新上层 MEMORY.md 的复核注记 ===")
with open(MEMORY_MD, encoding="utf-8") as f:
    mem = f.read()
hits = [ln for ln in mem.splitlines() if "第 14 次" in ln and ("复核" in ln or "独立" in ln)]
if not hits:
    print("  [WARN] 未找到含「第 14 次」的复核注记行，未修改 MEMORY.md")
    for ln in mem.splitlines():
        if "复核" in ln:
            print("    候选行:", ln[:160])
else:
    for ln in hits:
        print("  命中行:", ln[:200])
    new_mem = re.sub(r"第 14 次(独立)?复核", "第 %d 次独立复核" % ROUNDS, mem)
    if "第 15 次独立复核" in new_mem:
        with open(MEMORY_MD, "w", encoding="utf-8") as f:
            f.write(new_mem)
        print("  UPDATED: MEMORY.md 复核注记 -> 第 15 次")
    else:
        print("  [WARN] 替换后未出现标记，未写入")
print("=== DONE ===")
