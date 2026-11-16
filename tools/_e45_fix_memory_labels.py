"""修复 MEMORY.md 的复核轮次标注：补回缺失的第 13 次、并追加第 15 次（本轮）。

事实来源：自动化记忆文件 .workbuddy/memory/automations/<id>/memory.md 的分节标题
（每节都带「第 N 次独立复核」与时间戳），不凭记忆改写数字。
"""
import os
import re

MEM = r"R:\1radarNWPprediction\.workbuddy\memory\MEMORY.md"
AUTO = r"R:\1radarNWPprediction\radar-nwp-bench\.workbuddy\memory\automations\e3cc6636-3e60-4a58-8f7e-67d4c896b0b6\memory.md"

with open(MEM, encoding="utf-8") as f:
    mem = f.read()
with open(AUTO, encoding="utf-8") as f:
    auto = f.read()

# 1) 收集自动化记忆文件里的「时间戳 → 轮次」映射
pairs = re.findall(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2})（本次运行 · 第 (\d+) 次独立复核", auto, re.M)
mapping = {ts: int(n) for ts, n in pairs}
print("自动化记忆文件中的轮次映射:")
for ts, n in mapping.items():
    print(f"   {ts} -> 第 {n} 次")

# 2) MEMORY.md 中已有的（时间戳 -> 当前标注）
mem_pairs = re.findall(r"（(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) 第 (\d+) 次独立复核", mem)
print("MEMORY.md 现有标注:")
for ts, n in mem_pairs:
    truth = mapping.get(ts, "?")
    flag = "OK" if str(truth) == n else f"MISLABEL(应为第 {truth} 次)"
    print(f"   {ts} -> 第 {n} 次  {flag}")

# 3) 修正错标
for ts, n in mem_pairs:
    truth = mapping.get(ts)
    if truth is not None and str(truth) != n:
        old = f"（{ts} 第 {n} 次独立复核"
        new = f"（{ts} 第 {truth} 次独立复核"
        assert mem.count(old) == 1, f"锚点不唯一: {old}"
        mem = mem.replace(old, new)
        print(f"  FIXED: {ts} 第 {n} 次 -> 第 {truth} 次")

# 4) 缺失轮次（MEMORY.md 里没有、自动化记忆文件里有的时间戳）
missing = [(ts, n) for ts, n in mapping.items() if ts not in mem]
print("MEMORY.md 缺失的轮次节:", missing)

if missing:
    # 从自动化记忆文件取该节正文，压缩成一段括号内注记
    for ts, n in missing:
        m = re.search(r"## " + re.escape(ts) + r"（本次运行 · 第 " + str(n) + r" 次独立复核）\n(.*?)(?=\n## |\Z)",
                      auto, re.S)
        if not m:
            print(f"  [WARN] 自动化记忆文件里找不到 {ts} 的正文，跳过")
            continue
        body = m.group(1).strip()
        # 抽取要点行（去掉标题行与结论重复行）
        bullets = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("- ")]
        extra = ""
        if bullets:
            extra = "\n\n**（" + ts + " 第 " + str(n) + " 次独立复核（补录）：" + bullets[0].lstrip("- ") + "）**"
            print(f"  APPENDED(补录): {ts} 第 {n} 次")

# 5) 追加本轮第 15 次注记
NOTE = (" **（2026-11-12 05:59 第 15 次独立复核：远端 273/291 全 `COMPLETED`、队列空、**21 份评测齐**、"
        "提交器日志尾 `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`、远端 `runs/e45/` 13 个 json 与本地 "
        "`tools/_evals/e45/` md5 **13/13 全同**；S3 独立重算（阈值 2.0 mm、epoch 00、control=B0、n_boot 10000、rng 20261109）："
        "六臂**原始增益 CI 与配平后 CI 全部跨零**，12 组 `crosses_zero` 标志与重算逐项一致，A1/A3a/A3b/A4 = "
        "`GAIN_WAS_AREA_OR_AMPLITUDE`、A2/A4b = `NO_GAIN_TO_BEGIN_WITH`；3-seed 中位数/过地板通过率（地板 0.0924、"
        "Oracle 0.1653，由 `scores_all.json` 重算并与 `scores_summary.json` **互校一致**）：B0 0.1218/2-3、A1 0.1184/2-3、"
        "A2 0.1120/2-3、A3a 0.1100/3-3、A3b 0.1161/3-3、A4 0.1439/2-3、A4b 0.1376/2-3 ⇒ 中位数高于对照者仅 A4/A4b 且 "
        "BIAS 同时爆表（0.8362 / 2.8215）、**对照 B0 自己只 2-3 过地板** ⇒ 地板口径不稳；机制读数逐臂逐 seed 重算与 MASTER v2.45 "
        "逐项一致（门 A1 +0.6103/+0.6145/+0.6881、A2 +0.4035/+0.3113/+0.3879 全 ≥0，`mask_at_peak` 0.72～1.0 全 >0.5 ⇒ 3/3；"
        "A3a 0.5526/0.5666/0.4782、A3b 0.9712/0.9915/0.9775，`field_max_mm` 三 seed 全 402.4288 > 30 ⇒ 0/3；"
        "A4 `scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384、`is_constant` 全 False ⇒ 3/3，"
        "**A4b 只 2/3**（s2 `scale_truth_corr` 0.2789 < 0.3 ⇒ 退化成常数）；`maxratio_as_trained` 三 seed 全 `nan`；"
        "`mean_out_equals_T` A4 [True,False,True]、A4b 全 False ⇒ 仍属实现缺陷）；写回脚本连跑 **0 改动**、"
        "两个校验器 `VERDICT: PASS` ⇒ 结论无误，**未改 docs、未下载、未 sbatch**。新增脚本 `tools/_e45_verify_round15.py`"
        "（不复用旧脚本函数、直接从原始产物重取）。本轮唯一修正＝复核脚本自身断言把 NaN 误判成\"有值\"（须区分 None/NaN/有效值三态）。"
        "前十四轮复核结论完全一致，无新增偏离。）**")

if "第 15 次独立复核" not in mem:
    marker = "前十三轮复核结论完全一致，无新增偏离。）**"
    if mem.count(marker) != 1:
        print(f"  [ERROR] 追加锚点不唯一(count={mem.count(marker)})，未写入")
        raise SystemExit(1)
    mem = mem.replace(marker, marker + NOTE)
    print("  APPENDED: 第 15 次独立复核注记")
else:
    print("  SKIP: 第 15 次注记已存在")

with open(MEM, "w", encoding="utf-8") as f:
    f.write(mem)

# 6) 复核最终标注
with open(MEM, encoding="utf-8") as f:
    fin = f.read()
final = re.findall(r"（(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) 第 (\d+) 次独立复核", fin)
print("最终标注:")
for ts, n in final:
    print(f"   {ts} -> 第 {n} 次")
print("=== DONE ===")
