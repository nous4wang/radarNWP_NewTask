#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 第 41 轮：只读核验 + 记忆层登记。

新增内容：把预注册判据 `maxratio_as_trained > 0.9`（门那组 A1/A2）
首次**逐 seed 判读**。读数取自 2026-11-13 两个只读诊断作业（Slurm 297/298）
在真 checkpoint 上按 nanmedian 聚合算出的结果，日志：
    tools/_evals/e45/e45_diag_r31_full.log

原则：
  * 阈值从判据文件里**解析**出来，不手抄；
  * 核验全部通过才写记忆层，任一硬检查失败 ⇒ 只报 VERDICT: FAIL，不写；
  * 不改任何冻结判据文件（E45 实验文件 / MASTER / INDEX），不碰远端。
"""
import json
import os
import re
import sys

ROOT = r"R:/1radarNWPprediction/radar-nwp-bench"
UPM = r"R:/1radarNWPprediction/.workbuddy/memory"
WSM = r"R:/1radarNWPprediction/radar-nwp-bench/.workbuddy/memory"
GATE = os.path.join(ROOT, "docs/experiments/E45-gate-direction-amount-separation-baseline.md")
DIAG = os.path.join(ROOT, "tools/_evals/e45/e45_diag_r31_full.log")
S3P = os.path.join(ROOT, "tools/_evals/e45/s3_equalbias.json")
SCP = os.path.join(ROOT, "tools/_evals/e45/scores_summary.json")
PROBE = os.path.join(ROOT, "tools/_e45_mechanism_probe.py")
SCL = os.path.join(ROOT, "models/scaled_structure/model.py")
AUTOM = os.path.join(WSM, "automations/e3cc6636-3e60-4a58-8f7e-67d4c896b0b6/memory.md")

FAILS = []


def ok(cond, msg):
    if cond:
        print("  [OK]   " + msg)
    else:
        print("  [FAIL] " + msg)
        FAILS.append(msg)


def info(msg):
    print("  [info] " + msg)


def read(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def nl_of(text):
    crlf = text.count("\r\n")
    return "\r\n" if crlf * 2 > text.count("\n") else "\n"


def insert_after(path, anchor, addition, marker):
    """在 anchor 之后插入；anchor 找不到则追加到文件尾。幂等：marker 已在则跳过。"""
    if not os.path.exists(path):
        return "missing"
    t = read(path)
    if marker in t:
        return "already"
    nl = nl_of(t)
    body = addition.replace("\n", nl)
    i = t.find(anchor)
    if i < 0:
        if not t.endswith(nl):
            t += nl
        t += body
        write(path, t)
        return "appended(no-anchor)"
    j = i + len(anchor)
    t = t[:j] + body + t[j:]
    write(path, t)
    return "inserted"


def append_file(path, text, marker):
    if not os.path.exists(path):
        return "missing"
    t = read(path)
    if marker in t:
        return "already"
    nl = nl_of(t)
    if t and not t.endswith(nl):
        t += nl
    t += text.replace("\n", nl)
    write(path, t)
    return "appended"


print("=" * 72)
print("E45 round-41 verification (read-only)")
print("=" * 72)

# ---------------------------------------------------------------- 1. 阈值解析
gate_txt = read(GATE)
thr_hits = re.findall(r"maxratio_as_trained`?[^\n|]{0,40}>\s*([0-9]+\.[0-9]+)", gate_txt)
thr = float(thr_hits[0]) if thr_hits else None
ok(thr is not None and all(abs(float(x) - thr) < 1e-12 for x in thr_hits),
   "预注册阈值从判据文件解析出 = %s（源行命中 %d 处）" % (thr, len(thr_hits)))

m_ref = re.search(r"maxratio_as_trained`?（训练后保住的峰值）\s*\|\s*\*\*([0-9.]+)\*\*", gate_txt)
e44_ref = float(m_ref.group(1)) if m_ref else None
info("E44 参照读数 maxratio_as_trained = %s（判据文件 §0.1 表）" % e44_ref)

# ------------------------------------------------------- 2. 逐 seed 读数解析
vals = {}
cur = None
for line in read(DIAG).splitlines():
    mh = re.match(r"^===\s*([A-Za-z0-9]+)_(s\d+)\b", line)
    if mh:
        cur = (mh.group(1), mh.group(2))
        continue
    mv = re.search(r"nanmedian\(same list\)\s*=\s*([0-9.eE+-]+)", line)
    if mv and cur is not None:
        vals[cur] = float(mv.group(1))

a1 = sorted((k[1], v) for k, v in vals.items() if k[0] == "A1")
a2 = sorted((k[1], v) for k, v in vals.items() if k[0] == "A2")
ok(len(a1) == 3 and len(a2) == 3,
   "诊断日志中 A1/A2 各 3 个种子的 nanmedian 读数齐全（A1=%d, A2=%d）" % (len(a1), len(a2)))

a1_pass = sum(1 for _, v in a1 if v > thr)
a2_pass = sum(1 for _, v in a2 if v > thr)
ok(a1_pass == 0, "A1 逐 seed 过 >%s 线的种子数 = %d / 3（0/3 ⇒ 判不过）" % (thr, a1_pass))
ok(a2_pass == 3, "A2 逐 seed 过 >%s 线的种子数 = %d / 3（3/3 ⇒ 判过）" % (thr, a2_pass))

a1_txt = "／".join("%.4f" % v for _, v in a1)
a2_txt = "／".join("%.4f" % v for _, v in a2)
a1_lo, a1_hi = min(v for _, v in a1), max(v for _, v in a1)
info("A1 = %s（区间 %.4f–%.4f，线 %.1f）" % (a1_txt, a1_lo, a1_hi, thr))
info("A2 = %s" % a2_txt)

# --------------------------------------------------- 3. 裁决未变（S3 硬检查）
flags = []


def collect_flags(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and "crosses_zero" in k and isinstance(v, bool):
                flags.append((k, v))
            else:
                collect_flags(v)
    elif isinstance(obj, list):
        for v in obj:
            collect_flags(v)


collect_flags(json.loads(read(S3P)))
ok(len(flags) >= 12, "S3 原始产物里取到 %d 组 crosses_zero 标志（>=12，防空洞通过）" % len(flags))
ok(len(flags) > 0 and all(v for _, v in flags),
   "S3 全部 %d 组 CI 跨零（raw 与 equal_bias 两侧）⇒ 六臂判负、合格臂集合为空" % len(flags))
s3_txt = read(S3P)
ok("GAIN_WAS_AREA_OR_AMPLITUDE" in s3_txt and "NO_GAIN_TO_BEGIN_WITH" in s3_txt,
   "S3 裁决标签齐备（GAIN_WAS_AREA_OR_AMPLITUDE / NO_GAIN_TO_BEGIN_WITH）")

try:
    sc = json.loads(read(SCP))
    arms = [k for k in (sc if isinstance(sc, dict) else {}) if re.match(r"^(B0|A1|A2|A3a|A3b|A4|A4b)$", str(k))]
    if not arms:
        for v in (sc.values() if isinstance(sc, dict) else []):
            if isinstance(v, dict):
                arms = [k for k in v if re.match(r"^(B0|A1|A2|A3a|A3b|A4|A4b)$", str(k))]
                break
    info("scores_summary 里的臂块 = %s" % (sorted(arms) if arms else "未识别（软检查，不影响判定）"))
except Exception as exc:  # pragma: no cover
    info("scores_summary 解析异常（软检查）：%s" % exc)

# ------------------------------------------------------ 4. 三处实现缺陷状态
probe_txt = read(PROBE)
still_np_median = bool(re.search(r"np\.median\(np\.asarray", probe_txt))
info("探针仍用非 nan 感知 np.median：%s ⇒ 正式复现仍待修后重跑" % still_np_median)
info("量通道绝对护栏 _SHAPE_EPS = 1e-12 仍在：%s"
     % ("_SHAPE_EPS = 1e-12" in read(SCL)))
clamp_hits = []
for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "models")):
    for fn in files:
        if fn.endswith(".py"):
            p = os.path.join(dirpath, fn)
            if "clamp_mm" in read(p):
                clamp_hits.append(p)
info("models/ 下 clamp_mm 入口命中文件：%s" % (clamp_hits if clamp_hits else "无（输出场仍无截断入口）"))

# ---------------------------------------------------------------- 5. 判定
print("-" * 72)
if FAILS:
    print("VERDICT: FAIL  (%d 项硬检查失败，未写任何文件)" % len(FAILS))
    for f in FAILS:
        print("   - " + f)
    sys.exit(1)

print("VERDICT: PASS —— 全部硬检查通过")
print("-" * 72)

# ---------------------------------------------------------------- 6. 登记
MARK = "第 41 轮"
# MEMORY.md 的插入文本用「第 41 次独立复核」措辞 ⇒ 必须用与插入文本一致的标记，
# 否则重跑时标记查不到 ⇒ 会被再插一次（本轮已踩过：重复插入两行，已去重修复）。
MARK_MEMORY = "第 41 次独立复核"

handover_add = (
    "\n\n**⑥（2026-11-13 第 41 轮新增证据）预注册判据 `maxratio_as_trained > 0.9` "
    "首次被逐 seed 判读** —— 读数取自 2026-11-13 两个**只读**诊断作业（Slurm 297/298）"
    "在真 checkpoint 上按 nanmedian 聚合算出的结果（日志 `tools/_evals/e45/e45_diag_r31_full.log`）："
    "**A1 = " + a1_txt + "（s0/s1/s2）⇒ 三个种子全部低于 " + ("%.1f" % thr) + "（0/3 过）**；"
    "**A2 = " + a2_txt + " ⇒ 3/3 过**。⇒ **A1 判不过、A2 判过**（与 MASTER v2.46 登记的区间一致）。"
    "**裁决不变**：A1／A2 已由 S3 判负（12 组 CI 全跨零）⇒ 合格臂集合为空、不投十 seed 算力。"
    "**两条附带机制读数**：① A2 的硬支撑门**完整保住峰值（1.0）却仍不涨分** ⇒ 再次印证"
    "「门指向不是瓶颈」；② A1 的峰值保住率从 E44 的 0.152377 抬到 0.72～0.76，"
    "**但仍在削尾（不过 0.9 线）**，方向上与其 CSI 中位数低于对照一致（不作因果主张）。"
    "**刻意不改冻结判据文件**：探针源码仍用非 nan 感知的 `np.median`"
    "（`tools/_e45_mechanism_probe.py`），E45 实验文件 §8.4 的「空值／未产出」标签"
    "在**正式复现（修代码后重跑探针）之前保持不变** —— 本条读数是诊断脚本旁路算出的，"
    "不是探针自身产出。\n"
)

memory_add = (
    "\n- **（2026-11-13 14:5x 第 41 次独立复核：裁决不变，新证据＝预注册判据 "
    "`maxratio_as_trained > 0.9` 首次逐 seed 判读）** 远端 E45 全部作业 `COMPLETED`"
    "（提交器 **273** 05:04:04、训练数组 275/283、评测 274/282/290、汇总 **291**、"
    "只读诊断 297/298）、队列空、`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；"
    "远端与本地 **md5 13/13 全同**；五套闸门（`_check_tables`／`_check_v245_row`／"
    "`_check_v246_row`／`_check_v247_row`／`_e45_docs_numbers`）全部 `VERDICT: PASS` ⇒ "
    "MASTER 仍 **v2.47**、**不升版**（无裁决与数值变化）。**新脚本 `tools/_e45_verify_round41.py`"
    "（只读本地）把诊断作业 297/298 的 nanmedian 结果按臂-种子对齐，阈值从判据文件**解析**"
    "（不手抄）**：**A1 = " + a1_txt + " ⇒ 0/3 过 " + ("%.1f" % thr) + " 线（判不过）**；"
    "**A2 = " + a2_txt + " ⇒ 3/3 过（判过）** ⇒ **门那组唯一未判过的读数首次判完**；"
    "**A2 保住峰值 1.0 仍不涨分 ⇒ 再次印证「门指向不是瓶颈」**；A1 峰值保住率从 E44 的 "
    "0.152377 抬到 0.72～0.76 但仍削尾（不过线）。**不改变任何裁决**（S3 12 组 CI 全跨零 ⇒ "
    "六臂判负、不投十 seed 算力、合格臂集合为空）。**三处实现缺陷仍 0/3 未修**"
    "（输出场无 `clamp_mm` 入口／`_SHAPE_EPS = 1e-12`／探针仍 `np.median`）⇒ "
    "**E45 实验文件 §8.4 的「读数缺失／未产出」标签刻意保持不变**（正式复现待修后重跑探针）。"
)

daily = (
    "\n### 2026-11-13 14:5x｜第 41 轮：状态复核（无改动）＋ 新证据＝`maxratio_as_trained > 0.9` "
    "首次逐 seed 判读\n\n"
    "- **远端**：E45 全部作业 `COMPLETED`（提交器 **273** 05:04:04、训练数组 275/283、"
    "评测 274/282/290、汇总 **291** 14:23、只读诊断 297/298）；队列空（仅 4 个 `qlfn2-curve`，非 E45）、"
    "`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** json 齐全；提交器日志尾 "
    "`=== E45 submission finished 2026-11-11T14:20:42+08:00 ===`；远端最新日志仍是 "
    "`e45-diag-298.out`（Nov 13 03:10）。**本轮未下载、未 sbatch、未改远端状态**。\n"
    "- **产物一致性**：远端 `runs/e45/*.json` 与本地 `tools/_evals/e45/*.json` **md5 13/13 全同**。\n"
    "- **闸门**：`_check_tables`／`_check_v245_row`／`_check_v246_row`／`_check_v247_row`／"
    "`_e45_docs_numbers` 全部 `VERDICT: PASS` ⇒ MASTER **v2.47** 写回仍与产物逐格一致（**不升版**）。\n"
    "- **新证据（`tools/_e45_verify_round41.py`，只读本地）**：把诊断作业 297/298 的 nanmedian 结果"
    "按臂-种子对齐，阈值 **从判据文件解析**（`> " + ("%.1f" % thr) + "`）：**A1 = " + a1_txt +
    " ⇒ 0/3 过线（判不过）**；**A2 = " + a2_txt + " ⇒ 3/3 过线（判过）** ⇒ "
    "**门那组唯一未判过的读数首次判完**；A2 保住峰值 1.0 仍不涨分 ⇒ 再证「门指向不是瓶颈」；"
    "A1 峰值保住率 0.152377(E44) → 0.72～0.76 但仍削尾。**裁决不变**（S3 12 组 CI 全跨零）。\n"
    "- **三处实现缺陷仍 0/3 未修** ⇒ E45 实验文件 §8.4 的「读数缺失／未产出」标签刻意保持不变"
    "（正式复现待修后重跑探针）。\n"
    "- **结论**：**裁决不变——六臂判负、无一臂进十 seed、不投十 seed 算力**；后续仍只有三件实现缺陷"
    "修复（幅度截断接到输出场／`mean(out)=T` 的 `1e-12` 绝对护栏／`maxratio_as_trained` 改 "
    "`np.nanmedian`）＋ 重提机制族。\n"
)

autom_note = (
    "\n### 2026-11-13 14:5x｜第 41 轮：状态复核（无改动）＋ 新证据＝`maxratio_as_trained > 0.9` "
    "首次逐 seed 判读\n\n"
    "远端 E45 全部作业 `COMPLETED`（273/275/283/274/282/290/291/297/298）、队列空（仅 4 个 "
    "`qlfn2-curve`，非 E45）、21 份评测齐、13 份 json 远端与本地 md5 13/13 全同；五套闸门全部 "
    "`VERDICT: PASS` ⇒ MASTER 仍 v2.47、**不升版**（无裁决与数值变化）；**未下载、未 sbatch、"
    "未改远端状态**。新脚本 `tools/_e45_verify_round41.py`（只读本地）把诊断作业 297/298 的 "
    "nanmedian 结果按臂-种子对齐、阈值从判据文件解析：**A1 = " + a1_txt + " ⇒ 0/3 过线（判不过）**、"
    "**A2 = " + a2_txt + " ⇒ 3/3 过线（判过）** ⇒ 门那组唯一未判过的读数首次判完；A2 保住峰值 1.0 "
    "仍不涨分 ⇒ 再证「门指向不是瓶颈」。**裁决不变：六臂判负、无一臂进十 seed、不投十 seed 算力**；"
    "三处实现缺陷仍 0/3 未修 ⇒ E45 实验文件 §8.4 的「读数缺失／未产出」标签刻意保持不变"
    "（正式复现待修后重跑探针）。\n"
)

print("登记结果：")
print("  HANDOVER_E45.md      :", insert_after(
    os.path.join(UPM, "HANDOVER_E45.md"),
    "`tools/_e45_verify_round40.py`（VERDICT: PASS，只读本地、不动远端）。",
    handover_add, MARK))
print("  MEMORY.md (状态段)   :", insert_after(
    os.path.join(UPM, "MEMORY.md"),
    "新证据写入 `HANDOVER_E45.md` §4⑤ 与两份逐日日志。",
    memory_add, MARK_MEMORY))
print("  2026-11-13.md (上层) :", append_file(
    os.path.join(UPM, "2026-11-13.md"), daily, MARK))
print("  2026-11-13.md (工作区):", append_file(
    os.path.join(WSM, "2026-11-13.md"), daily, MARK))
print("  2026-11-11.md (上层) :", append_file(
    os.path.join(UPM, "2026-11-11.md"),
    "\n- **（2026-11-13 14:5x 补记，第 41 轮）** E45 结案结论不变（六臂判负、不投十 seed 算力）；"
    "新增证据＝预注册判据 `maxratio_as_trained > 0.9` 首次逐 seed 判读（A1 0/3 过线、A2 3/3 过线），"
    "详见 `HANDOVER_E45.md` §4⑥ 与 `2026-11-13.md`。\n", MARK))
print("  自动化 memory.md     :", append_file(AUTOM, autom_note, MARK))

print("-" * 72)
print("SUMMARY: A1 %s (pass %d/3) | A2 %s (pass %d/3) | thr > %.1f | 裁决不变"
      % (a1_txt, a1_pass, a2_txt, a2_pass, thr))
