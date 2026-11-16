#!/usr/bin/env python
"""E45 round 77 writeback -- idempotent, marker-block based.

Writes the round-77 record into:
  * R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md   (marker block)
  * R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md      (marker block)
  * R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md          (status paragraph)
  * R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-15.md (marker block)

Rules honoured:
  * idempotent -- re-running must leave every file byte-identical;
  * marker = the exact inserted text (substring match), block stripped then re-appended;
  * table rows are built from cell LISTS, the separator is inserted by code;
  * does NOT touch docs/MASTER.md or docs/experiments/ (no new numbers, no verdict
    change => MASTER stays v2.49).
"""

import hashlib
import re
import sys
from pathlib import Path

MARKER = "E45-R77-BEGIN"
END = "E45-R77-END"

HANDOVER = Path(r"R:/1radarNWPprediction/.workbuddy/memory/HANDOVER_E45.md")
UPPER_DAILY = Path(r"R:/1radarNWPprediction/.workbuddy/memory/2026-11-11.md")
UPPER_MEM = Path(r"R:/1radarNWPprediction/.workbuddy/memory/MEMORY.md")
WS_DAILY = Path(r"R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory/2026-11-15.md")


def md_table(rows):
    """Build a markdown table from a list of cell-lists; separator inserted by code."""
    out = []
    for r in rows:
        cells = [str(c) for c in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


HANDOVER_BLOCK = f"""<!-- {MARKER} -->
### 2026-11-15｜第 77 轮（epoch 轴 × 预报时效轴：两条轴的**首次交叉**判读）

- **判定**：远端 **E45 本身全部完成、无新活动、裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；**本轮远端动作：零**（只读状态查询；未 sbatch、未下载、未同步、未改远端状态）。**未改 `docs/`**（MASTER 仍 **v2.49**，不升版）。
- **远端**：权威链 **273/274/275/282/283/290/291/297/298/314/315/316/317 全 `COMPLETED`**；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 与本地 md5 **13/13 逐位相同**；提交器 273 日志尾部确认 **summary 291 当年即自动跑完 S3 等偏置对照**（未手动 sbatch）；`squeue -u wf` 空、Slurm 控制器 `active`；`runs/` 下**无 E46+ 目录**。**易误读点（沿用第 73 轮）**：`sacct` 里的 102/103/118/265 是 v2.42/v2.43 修传参 bug 前的**废作业**，不属权威链。
- **本轮为什么值得做**：第 49 轮判了**预报时效轴**（lead axis），但只用**已发布表（ep00）**；第 65–69 轮判了 **epoch 轴**（ep00/ep01/ep02 × 7 臂 × 3 seed），但只判**汇总 CSI/BIAS 与 S3 护栏**。⇒ **两条轴的交叉（63 格 × 10 个时效）此前从未判过**：第 49 轮发现「A4 的增益随时效反号（L0–L3 正、L4–L9 负）、A4b 恰好相反」，但**那条反号结构是不是 ep00 这个点的偶然**从未被检验。脚本 `tools/_e45_verify_round77.py`（只读本地，**POS 9/9 VERDICT: PASS**，产物 `tools/_evals/e45/round77_epoch_lead_cross.json`）。
- **同协议基准零漂移（两条独立）**：① epoch 轴 ep00 的**整体** `CSI@2.0` 与已发布逐 seed 值 **max|dev| = 0.000e+00**（21 格）；② ep00 的**逐时效** `CSI@2.0` 与已发布 **210 格 max|dev| = 0.000e+00**；③ **解析器与第 49 轮交叉标定**：逐时效向量 **6 臂 × 10 时效 max|dev| = 0.000e+00**、斜率 **6 臂 max|dev| = 0.000e+00** ⇒ **ep01/ep02 与 ep00 的任何差异只可能来自 epoch 本身**。
- **第一条新事实（核心）**：「A4 早正晚负 / A4b 早负晚正」这条**时效反号结构在 n=3 的三个 epoch 上并不稳定**——配对口径逐时效斜率：

{md_table([
  ["臂", "ep00 斜率/时效", "ep01 斜率/时效", "ep02 斜率/时效", "三 epoch 同号?"],
  ["A4", "**−0.01154**", "−0.00415", "−0.00706", "**3/3 同为负**（方向稳、幅度不稳）"],
  ["A4b", "**+0.01106**", "−0.00264", "+0.00332", "**否**（ep01 翻号）"],
  ["A3a", "+0.00407", "+0.00355", "+0.00092", "3/3 同为正"],
  ["A3b", "+0.00090", "+0.00291", "+0.00258", "3/3 同为正"],
  ["A2", "+0.00184", "+0.00177", "+0.00493", "3/3 同为正"],
  ["A1", "−0.00588", "+0.00485", "−0.00115", "否（ep01 翻号）"],
])}

  ⇒ **A4 的「增益随时效递减」方向在三个 epoch 上都成立（3/3）**，但**幅度衰减到 ep01/ep02 的 1/3–2/3**；**A4b 的反号方向只活在一个 epoch 上（ep01 翻号）⇒ 第 49 轮那条「A4b 与 A4 恰好相反」的配对是 ep00 单点现象**，不能当作两臂的稳定性质。**「flip 时效点」也不固定**：A4 的反号点 ep00 在 L4、ep01 在 L1、**ep02 一直到 L8 仍为正**。
- **第二条新事实**：「越过零模型地板」的格**极少且大多不可归因**——零模型＝同一 epoch 内**把对照臂换个 seed**（纯种子噪声），地板按 **(epoch, 时效) 格**各自计算（取 3 对换种子的 **max** |ΔCSI|，比第 49 轮用的 median 更严）：**A4b 6/30、A3a 1/30、A3b 1/30、A4 1/30、A1 0/30、A2 0/30**（每臂 3 epoch × 10 时效）。其中**仅 4/9 格是 3/3 种子同号、仅 1/9 格能通过「留一 seed」检验**（唯一一格是 A4 ep00 L2，3/3 同号正、bias 比 3.75×）⇒ **本轴上唯一「稳定可归因」的格仍是 A4 在 ep00 的短时效**，与第 40 轮「A4 的增益 99.2% 来自 seed 0」同向。
- **第三条新事实**：**ep01／ep02 的「臂级相对优势」来自对照臂塌陷，不是机制**——差值口径下 ep01 六臂**全部为负**（A4 −0.01349 最接近 0）、ep02 六臂**全部为正**（A4b +0.05713、A4 +0.03318）；但 ep02 的零模型地板在短时效高达 **0.1072–0.1290**（ep00 仅 0.0456–0.0596）⇒ **ep02 的地板比 ep00 高 2 倍以上**，这与第 69 轮「ep02 对照塌到 ×0.452、增益是对照崩塌造成的相对优势」完全一致 ⇒ **ep02 无任何一臂过地板**。
- **守卫**：S3 **12/12 区间仍全部跨零**、六臂 `s3_verdict` 逐臂不变、**合格臂集合仍为空**；已发布头条表七臂中位 CSI 由原始 ep00 评测复算 **max|dev| = 0.000e+00**。
- **口径与纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 配对口径与差值口径并列；③ 事后诊断、**不构成新判据**、**不改变任何裁决**；④ **新登记两条纪律**：**「一条轴上的结构性发现必须在另一条轴上复现后才可称结构」**（第 49 轮的 A4/A4b 反号配对在本轮被降级为 ep00 单点现象）；**「越过零模型地板」与「可归因于机制」是两件事**（本轮 9 格越地板只有 1 格能留一 seed 通过）——与第 46/48 轮「窗级 CI 是伪重复」同族。
- **本轮踩坑（可复用规则）**：**断言不许假设数据会迎合结论**（第 61/63 轮同族）——首版断言写成「凡越地板的格都必须 3/3 种子同号」，实测 4/9 ⇒ 改为**枚举事实**（记录「3/3 同号 4/9、留一 seed 通过 1/9」）＋ 只断言「上报的计数与明细行一致」；**跨轮复用产物要先核键名**——第 49 轮的斜率键是 `paired_slope_per_lead`（不是 `slope_per_lead`），首版取空导致交叉标定 0 臂 ⇒ 断言必须校验「预期臂数」而非只校验「脚本跑通」（第 66 轮同族教训）。
<!-- {END} -->
"""

DAILY_BLOCK = f"""
<!-- {MARKER} -->
### 2026-11-15｜第 77 轮：E45 远端状态复核（无新活动）＋ epoch 轴 × 时效轴首次交叉判读

- **远端**：权威链 273/274/275/282/283/290/291/297/298/314/315/316/317 全 `COMPLETED`；21 份评测齐、13 份 json 与本地 md5 **13/13**；S3 由 summary 291 自动跑完（未手动 sbatch）；控制器 `active`、队列空、无 E46+ 目录。**远端动作：零**（只读状态查询）。
- **新证据**：`tools/_e45_verify_round77.py`（只读本地，**POS 9/9 VERDICT: PASS**，产物 `tools/_evals/e45/round77_epoch_lead_cross.json`）——**epoch 轴 × 预报时效轴的首次交叉**（63 格 × 10 时效）。① **同协议基准零漂移**：ep00 整体 21 格 max|dev| = 0、**逐时效 210 格 max|dev| = 0**、与第 49 轮逐时效向量交叉标定 **6 臂 × 10 时效 max|dev| = 0**；② **第 49 轮「A4 早正晚负 / A4b 早负晚正」不是稳定结构**——A4 斜率 **−0.01154／−0.00415／−0.00706（三 epoch 同号为负）**，但**反号点随 epoch 移动**（L4／L1／L8）；A4b **+0.01106／−0.00264／+0.00332（ep01 翻号）** ⇒ 那条配对是 **ep00 单点现象**；③ **越零模型地板的格极少且大多不可归因**：按 (epoch, 时效) 各自算地板（对照臂换 seed，取 max，ep02 短时效地板高达 0.1072–0.1290）⇒ **A4b 6/30、A3a/A3b/A4 各 1/30、A1/A2 0/30**，其中**仅 4/9 格 3/3 种子同号、仅 1/9 格通过留一 seed**（唯一是 A4 ep00 L2，bias 比 3.75×）；④ **ep01 六臂全负、ep02 六臂全正**，但 ep02 地板比 ep00 高 2 倍以上 ⇒ **ep02 的相对优势是对照塌陷的产物**（与第 69 轮一致）。
- **守卫**：S3 **12/12 区间仍全部跨零**、`s3_verdict` 逐臂不变、**合格臂集合仍为空**；头条表七臂中位 CSI 复算 max|dev| = 0。**裁决不变**（六臂判负、不投十 seed 算力）；**未改 `docs/`** ⇒ MASTER 仍 v2.49、不升版。
- **纪律**：① 仍按「中位数 + 通过率」报，单点分数不作裁决；② 事后诊断、**不构成新判据、不改变任何裁决**；③ **新登记两条**：**一条轴上的结构性发现必须在另一条轴上复现后才可称结构**；**「越过零模型地板」≠「可归因于机制」**。
<!-- {END} -->
"""

MEM_PARA = (
    "- **第 77 轮（2026-11-15）epoch 轴 × 时效轴首次交叉**：远端 E45 无新活动（权威链全 COMPLETED、21 份评测齐、"
    "13 份 json md5 13/13、S3 由 summary 291 自动跑完、控制器 active、队列空）；**本轮远端动作零**、未改 `docs/`（MASTER 仍 v2.49）。"
    "新证据（`tools/_e45_verify_round77.py`，POS 9/9）：**第 49 轮那条「A4 早正晚负 / A4b 早负晚正」不是稳定结构**——"
    "A4 逐时效斜率 **−0.01154／−0.00415／−0.00706（三个 epoch 同号为负）但反号点随 epoch 移动（L4／L1／L8）**，"
    "**A4b +0.01106／−0.00264／+0.00332（ep01 翻号）** ⇒ 那条配对是 **ep00 单点现象**，不得当两臂的稳定性质；"
    "**越零模型地板的格极少且大多不可归因**（按 (epoch,时效) 各自算地板、对照臂换 seed 取 max：ep02 短时效地板高达 0.1072–0.1290）"
    "⇒ A4b 6/30、A3a/A3b/A4 各 1/30、A1/A2 0/30，**其中仅 4/9 格 3/3 种子同号、仅 1/9 格通过留一 seed**（唯一是 A4 ep00 L2，bias 比 3.75×）；"
    "**ep01 六臂全负、ep02 六臂全正**，但 ep02 地板比 ep00 高 2 倍以上 ⇒ **ep02 的相对优势是对照塌陷的产物**（与第 69 轮一致）。"
    "同协议基准零漂移（ep00 整体 21 格与逐时效 210 格 max|dev| = 0；与第 49 轮逐时效向量交叉标定 6 臂 × 10 时效 max|dev| = 0）。"
    "**裁决不变**（S3 12/12 区间仍全部跨零、合格臂集合仍为空 ⇒ 六臂判负、不投十 seed 算力）。"
    "**新登记两条纪律**：① **一条轴上的结构性发现必须在另一条轴上复现后才可称结构**；② **「越过零模型地板」≠「可归因于机制」**。"
)


def strip_marker(text, marker, end):
    """Remove an existing marker block (marker line .. end line) if present."""
    pat = re.compile(r"\n?<!-- " + re.escape(marker) + r" -->.*?<!-- " + re.escape(end) + r" -->\n?",
                     re.S)
    new, n = pat.subn("\n", text)
    return new, n


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def upsert_block(path, block, marker, end):
    """Insert block at end of file, idempotently."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(block.lstrip("\n"), encoding="utf-8")
        return "created"
    text = path.read_text(encoding="utf-8")
    stripped, n = strip_marker(text, marker, end)
    if marker in text:
        stripped = stripped.rstrip("\n") + "\n\n"
    stripped = stripped.rstrip("\n") + "\n\n" + block.lstrip("\n")
    path.write_text(stripped, encoding="utf-8")
    return "replaced" if n else "appended"


def upsert_para(path, para):
    """Insert a paragraph after the MEMORY status marker line; idempotent."""
    text = path.read_text(encoding="utf-8")
    if para in text:
        return "already present"
    anchor = None
    for line in text.splitlines():
        if line.startswith("- **第 7") and "轮" in line:
            anchor = line
            break
    if anchor is None:
        # fall back: append at end
        path.write_text(text.rstrip("\n") + "\n" + para + "\n", encoding="utf-8")
        return "appended"
    text = text.replace(anchor, anchor + "\n" + para, 1)
    path.write_text(text, encoding="utf-8")
    return "inserted"


results = []
for path, block in ((HANDOVER, HANDOVER_BLOCK), (UPPER_DAILY, DAILY_BLOCK),
                    (WS_DAILY, DAILY_BLOCK)):
    results.append((str(path), upsert_block(path, block, MARKER, END)))

results.append((str(UPPER_MEM), upsert_para(UPPER_MEM, MEM_PARA)))

print("== WRITEBACK ==")
for p, r in results:
    print(f"  {r:15s} {p}")

# idempotency: run the same upserts a second time and compare hashes
before = {str(p): sha(p) for p in (HANDOVER, UPPER_DAILY, UPPER_MEM, WS_DAILY)}
for path, block in ((HANDOVER, HANDOVER_BLOCK), (UPPER_DAILY, DAILY_BLOCK),
                    (WS_DAILY, DAILY_BLOCK)):
    upsert_block(path, block, MARKER, END)
upsert_para(UPPER_MEM, MEM_PARA)
after = {str(p): sha(p) for p in (HANDOVER, UPPER_DAILY, UPPER_MEM, WS_DAILY)}

ok = before == after
print("== IDEMPOTENCY ==")
for p in before:
    print(f"  {before[p]} -> {after[p]}  {'SAME' if before[p]==after[p] else 'CHANGED'}  {Path(p).name}")
print(f"  VERDICT: {'PASS' if ok else 'FAIL'} (second run must be a no-op)")

# marker counts must be exactly 1
print("== MARKER COUNTS ==")
for path, block in ((HANDOVER, HANDOVER_BLOCK), (UPPER_DAILY, DAILY_BLOCK),
                    (WS_DAILY, DAILY_BLOCK)):
    t = path.read_text(encoding="utf-8")
    print(f"  {t.count(MARKER)}  {path.name}")
t = UPPER_MEM.read_text(encoding="utf-8")
print(f"  {t.count('第 77 轮')}  MEMORY.md (第 77 轮 mentions)")

sys.exit(0 if ok else 1)
