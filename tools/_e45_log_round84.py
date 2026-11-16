"""Append the E45 round-84 record to the memory files (idempotent).

READ-ONLY with respect to docs/ and the remote: this script only appends a
marked block to the handover / daily logs / MEMORY.md status section, and
refuses to write if the marker already exists (no silent double-append).
"""

from __future__ import annotations

import os
import sys

ROOT = r"R:/1radarNWPprediction"
BLOCK = """<!-- E45-R84-BEGIN -->
### 2026-11-15｜第 84 轮（E45 远端运行状态复核：权威链终态 ＋ **裁决从原始产物独立重导** ＋ 发布表与重算表逐项对账）

- **远端终态（远端动作＝0）**：权威链 **273（提交器）／274·282·290（stage2）／275·283（训练数组）／291（summary）／297·298（诊断）／314（gpuprobe）／315·316·317（epoch 轴）全部 `COMPLETED`**；提交器 273 日志尾部仍是 `summary job id : 291` ＋ `the summary (job 291) finished: COMPLETED` ＋ `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **S3 等偏置对照当年即由汇总作业自动跑完 ⇒ 本轮无需（也不重复）手动 sbatch**；`runs/e45/evals/` **21 份 `.eval.json`**、`runs/e45/` **13 份** run 级 json 齐；`squeue` 里在跑的是 333, 334, 332, 330, 331, 329, 328, 327 个 **`qlfn2-f37`**，属同一 `wf` 账号下的**另一条项目线**（与 E45 无关，不干预）⇒ **E45 无新活动**。
- **本轮新增审计面（真正的新的东西）：裁决从原始产物独立重导**——新脚本 `tools/_e45_verify_round84.py`（只读本地，**POS 13/13 VERDICT: PASS**），**不复用任何既有脚本的结论**，而是从 `scores_summary_s{0,1,2}.json`／`diag_probe_s{0,1,2}.json`／`s3_equalbias.json` **重新推导**六臂裁决：① **13 份 run 级 json 远端↔本地 md5 13/13 逐位相同**（本轮实测远端 md5 清单，与本地逐份重算比对）；② **发布表与重算表对账 7/7 命中**（`scores_summary.json` 的 `median_csi_at_2.0` vs 由逐 seed 快照重算的中位数，max|diff| < 1e−9）⇒ **文档里印的数与原始产物仍逐位一致**；③ **对照臂自己的过地板通过率＝2/3**（地板 0.0924 是单一种子口径）⇒ 「过地板通过率不能单独用于晋级」在数据上再次坐实；④ **S3 护栏 12/12 CI 格全部跨零、合格臂集合为空**（阈值 2.0 mm、control=B0、n_boot=10000 与产物一致 ⇒ 护栏本身未被改写）；⑤ **预注册机制判据逐臂逐 seed 重判**：A1/A2 **3/3 过**（`corr(mask,amp)` +0.6103/+0.6145/+0.6881 与 +0.4035/+0.3113/+0.3879；`mask_at_peak` 0.8217/0.7599/0.7224 与 1.0×3）；A3a/A3b **0/3 过**（`grad_top1pct_share` 0.5526/0.5666/0.4782 与 0.9712/0.9915/0.9775 全 ≥ 0.5；场最大值三 seed 全 402.43 mm > 30）；A4 **3/3 过**（`scale_cv` 2.0040/1.2358/1.7046、`scale_truth_corr` 0.7429/0.5521/0.6384）；A4b **2/3 过**（seed2 触发的是**相关性分支** corr 0.2789 < 0.3，而该格 `scale_cv` 3.0188 是六格最高、**cv 分支从未触发** ⇒ 实为**量级塌缩＋与真值脱钩**，不是「退化成常数」）；⑥ `mean(out)=T` 不达格＝**A4s1/A4bs0/A4bs1/A4bs2** ⇒ 仍按 §11.8 记**实现缺陷**，不作机制结论。
- **裁决零变化**：S3 12/12 跨零 ⇒ **六臂全部判负、无一臂进十 seed、不投十 seed 算力**；门指向（结构上已解决但不是瓶颈）／拓扑支（按预注册结掉）／量通道（不是瓶颈）三条机制线结论不变；`maxratio_as_trained` 的预注册判据 `> 0.9` 在探针修好后已判读为 **A1 0/3、A2 3/3**（第 41 轮）⇒ 不改变任何裁决。
- **文档闸门**：`_e45_verify_round81.py` **POS 27/27 VERDICT: PASS**（CONTENT 类失败＝0；S3 12 区间全跨零；发布中位数 maxdev 4.53e−07；过地板通过率逐臂一致；超 Oracle 0.1653 的只有 A3b、A4 各 1/3＝seed 0）。

| 臂 | 中位 CSI@2.0 | 过地板通过率 | 中位 BIAS@2.0 |
| --- | --- | --- | --- |
| B0（对照） | 0.121817 | 2/3 | 0.5095 |
| A1 | 0.118397 | 2/3 | 0.4430 |
| A2 | 0.112003 | 2/3 | 0.4490 |
| A3a | 0.110015 | 3/3 | 0.6394 |
| A3b | 0.116055 | 3/3 | 0.6188 |
| A4 | 0.143896 | 2/3 | 0.8362 |
| A4b | 0.137567 | 2/3 | 2.8215 |

- **写回**：**未改 `docs/`**（本轮无新数字、无裁决变化 ⇒ **不升版**，MASTER 仍 **v2.50**）；按惯例只把本轮登记进 HANDOVER／逐日日志／MEMORY 状态段。
- **本轮产物**：`tools/_e45_verify_round84.py`、`tools/_evals/e45/round84_status_audit.json`。
- **纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 事后复核、**不构成新判据、不改变任何裁决**；③ **本轮远端动作＝0**（未下载、未 sbatch、未改远端状态）；④ 写回用**带标记的幂等追加**，标记已存在即报错退出，绝不静默重复追加（第 82 轮事故的教训）。
<!-- E45-R84-END -->
"""

MEM_BULLET = """- **E45 状态复核（第 84 轮，2026-11-15）**：远端权威链全 `COMPLETED`（273／274·282·290／275·283／291／297·298／314／315·316·317）、`runs/e45/evals/` 21 份 `.eval.json`、`runs/e45/` 13 份 run 级 json；**S3 由 summary 291 当年自动跑完 ⇒ 本轮无需手动 sbatch**；队列里 8 个 `qlfn2-f37` 属另一条项目线、不干预 ⇒ **E45 无新活动**。**本轮新增审计面：裁决从原始产物独立重导**（新脚本 `tools/_e45_verify_round84.py`，POS 13/13 VERDICT: PASS，不复用既有脚本结论）：① 13 份 run 级 json 远端↔本地 md5 **13/13 逐位相同**；② **发布表与重算表对账 7/7 命中**（`scores_summary.json` 的中位 CSI@2.0 vs 逐 seed 快照重算，max|diff| < 1e−9）⇒ 文档里印的数与原始产物仍逐位一致；③ **对照臂自己只有 2/3 过地板**（地板 0.0924 是单一种子口径）⇒ 过地板通过率不能单独用于晋级；④ **S3 护栏 12/12 CI 格全跨零、合格臂集合为空**（thr 2.0 mm、control=B0、n_boot=10000 未被改写）；⑤ **预注册机制判据逐臂逐 seed 重判**：A1/A2 3/3 过、A3a/A3b 0/3 过（share 全 ≥ 0.5、场最大全 402.43 mm）、A4 3/3 过、A4b 2/3 过（seed2 触发**相关性分支** corr 0.2789 < 0.3，cv 分支从未触发）；⑥ `mean(out)=T` 不达格＝A4s1/A4bs0/A4bs1/A4bs2 ⇒ 仍记实现缺陷、不作机制结论。**裁决不变**（六臂判负、不投十 seed 算力）。**未改 `docs/`**（无新数字、无裁决变化 ⇒ 不升版，MASTER 仍 v2.50）。
"""

TARGETS = [
    os.path.join(ROOT, ".workbuddy", "memory", "2026-11-11.md"),
    os.path.join(ROOT, ".workbuddy", "memory", "HANDOVER_E45.md"),
    os.path.join(ROOT, "radar-nwp-bench", ".workbuddy", "memory", "2026-11-15.md"),
]

ok = True
for p in TARGETS:
    if not os.path.exists(p):
        print(f"[FAIL] missing target: {p}")
        ok = False
        continue
    with open(p, encoding="utf-8") as fh:
        text = fh.read()
    n_begin = text.count("<!-- E45-R84-BEGIN -->")
    n_end = text.count("<!-- E45-R84-END -->")
    if n_begin or n_end:
        print(f"[SKIP] marker already present in {p} (begin={n_begin}, end={n_end}) -> no double-append")
        continue
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("\n" + BLOCK)
    print(f"[OK] appended round-84 block -> {p}")

# MEMORY.md status section: insert the round-84 bullet right after the round-83 bullet
mem = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")
with open(mem, encoding="utf-8") as fh:
    lines = fh.readlines()
if any("第 84 轮" in ln for ln in lines):
    print("[SKIP] round-84 bullet already present in MEMORY.md")
else:
    idx = None
    for i, ln in enumerate(lines):
        if "第 83 轮" in ln and ln.lstrip().startswith("- **E45 状态复核"):
            idx = i
    if idx is None:
        print("[FAIL] could not locate the round-83 status bullet in MEMORY.md")
        ok = False
    else:
        lines.insert(idx + 1, MEM_BULLET)
        with open(mem, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        print(f"[OK] inserted round-84 status bullet into MEMORY.md at line {idx + 2}")

sys.exit(0 if ok else 1)
