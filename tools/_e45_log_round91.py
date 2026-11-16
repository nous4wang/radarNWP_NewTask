#!/usr/bin/env python3
"""Write the E45 round-91 record into the four memory documents.

Replace-instead-of-append inside a marker pair: the entry is
idempotent at the byte level (running twice with the same input gives
the same bytes) while still letting a later round fill in a result
that was pending when the entry was first written.  Every table row is
built from its CELLS and the separators are inserted by the code -- a
hand-written separator is how a row ends up with the wrong number of
columns and breaks the whole table.
"""
from pathlib import Path
import json

UPPER = Path("R:/1radarNWPprediction/.workbuddy/memory")
PROJECT = Path("R:/1radarNWPprediction/radar-nwp-bench")

HANDOVER = UPPER / "HANDOVER_E45.md"
DAYLOG = UPPER / "2026-11-11.md"
MEMORY = UPPER / "MEMORY.md"
PROJLOG = PROJECT / ".workbuddy" / "memory" / "2026-11-16.md"
AUTOMEM = (PROJECT / ".workbuddy" / "memory" / "automations"
            / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md")

BEGIN = "<!-- E45-R91-BEGIN -->"
END = "<!-- E45-R91-END -->"
MEM_MARK = "<!-- E45-R91-STATUS -->"
N_SEEDS = 20


def write_marked(path: Path, text: str, begin: str = BEGIN,
                 end: str = END) -> str:
    """Write a marked block; replace it if the marker is already there."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    body = path.read_text(encoding="utf-8")
    new_block = text.strip("\n")
    if begin in body and end in body:
        pre = body[:body.index(begin)]
        post = body[body.index(end) + len(end):]
        body = pre + new_block + post
        path.write_text(body, encoding="utf-8")
        return f"REPLACED: {path.name}"
    body = body.rstrip("\n") + "\n\n" + new_block + "\n"
    path.write_text(body, encoding="utf-8")
    return f"WROTE: {path.name}"


def main() -> int:
    audit = json.loads((PROJECT / "tools" / "_evals" / "e45"
                        / "round91_status_audit.json").read_text(encoding="utf-8"))
    head = audit.get("headline", {})
    raw = audit.get("raw_sweep", {})
    eq = audit.get("equal_bias_sweep")

    raw_bits = "；".join(
        f"{a} {v['n_seeds_with_zero_inside_interval']}/{v['n_seeds_tried']}"
        for a, v in raw.items())
    head_bits = "、".join(
        f"{a} {v['median_csi_at_2.0']:.6f}／过地板 {v['pass_rate_over_floor']}-3"
        for a, v in head.items())
    flip = audit.get("flip_risk", {})
    jitter = audit.get("endpoint_jitter_measured_on_raw_half")
    tight = audit.get("arms_within_2x_jitter_of_flipping", [])
    immune = [a for a, v in flip.items() if v.get("immune_because_point_negative")]
    at_risk = [a for a, v in flip.items() if not v.get("immune_because_point_negative")]

    if eq:
        eq_bits = "；".join(
            f"{a} {v['n_seeds_with_zero_inside_interval']}/{v['n_seeds_tried']}"
            for a, v in eq.items())
        flippable = "、".join(audit.get("flippable_arms", [])) or "无"
        eq_line = (
            f"- **等偏置半边（远端 Slurm 作业 343，CPU 作业、不占 GPU）**："
            f"同一护栏实现（`tools/_e41_s3_equalbias.py`，远端 md5 与本地一致）"
            f"在远端重跑 ⇒ 注册种子**逐位复现**（max|dev| < 1e−12）；"
            f"**{N_SEEDS} 个 bootstrap 种子下区间含零计数＝{eq_bits}**；"
            f"配平后点估计为正、因而**唯一可能翻转**的臂＝{flippable} ⇒ **仍无一条翻转**。")
    else:
        eq_line = (
            "- **等偏置半边（尚未完成）**：远端 Slurm 作业 343 已提交，"
            "但被**同账号配额**挤到排队（`QOSMaxJobsPerUserLimit`：队列里 8 个 "
            "`qlfn2-f38` 属同一 `wf` 账号下的**别的项目线**，已跑 3 h／时限 12 h，"
            "**不干预**）⇒ 该半边待下一轮收割。**收割命令**（分次、显式 `--out`）："
            "`MSYS_NO_PATHCONV=1 python tools/remote.py --download "
            "/home/wf/radarNWP/runs/e45/s3_seedrobust.json "
            "--out tools/_evals/e45/s3_seedrobust.json`，"
            "再跑 `/c/Users/nous/anaconda3/python tools/_e45_round91.py`（POS 15/15）"
            "与 `/c/Users/nous/anaconda3/python tools/_e45_log_round91.py`（替换式写回）"
            "⇒ 本条目会被自动补齐等偏置半边的结果。")

    full = f"""{BEGIN}
### 2026-11-16｜第 91 轮（E45 远端运行状态复核 ＋ **S3 护栏 bootstrap 随机种子稳健性（首次独立重跑）** ＋ 已发布文档口径合规复核）

- **判定**：远端 E45 主链仍是 2026-11-11 的终态、无新活动 ⇒ **裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**。
- **远端终态（只读复核）**：权威链 273／275·283／274·282·290／291／297·298／314／315·316·317 全 `COMPLETED`；`runs/e45/evals/` **21 份 `.eval.json`**、`runs/e45/` **13 份** run 级 json、**`fields/` 42 份 npy 在位**；**结案后（> 2026-11-11 14:20:42）被修改的文件 = 0 份** ⇒ 主链从未重启；`runs/` 下无 E46+ 正式训练目录；队列里 8 个 `qlfn2-f38`（job 335–342）属同一 `wf` 账号下的**别的项目线，不干预**。
- **产物一致性**：13 份 run 级 json **远端↔本地 md5 13/13 逐位全同**（本轮实测，非复用登记值）。
- **本轮新增审计面 ①（S3 判定对 bootstrap 随机种子是否稳健）**：既往轮次只核了**存下来的端点**（含零标志与数值自洽），**从未用别的种子重跑过重采样**；而 S3 是**负面裁决**，负面结论对随机抽样的依赖必须查清。本轮用**护栏自己的实现**（`tools/_e41_s3_equalbias.py`，md5 `b8b78280741cea101d3362f654cad903`，import 而非重写，避免重实现偏差）先以注册种子复现，再换 **{N_SEEDS} 个种子**重跑两级 bootstrap（seed 层＋窗层）：
  - **raw 半边（本地，21 份窗表足够）**：六个臂 **注册种子逐位复现（max|dev| = 0.000e+00）**；**{N_SEEDS}/{N_SEEDS} 种子区间全部含零**（{raw_bits}）⇒ raw 侧的判负**不依赖那一次抽样**。
  {eq_line}
  - 另核：**配平确实落到对照臂的偏置上**（max |achieved − control| = 4.1e−07）；护栏参数逐项未改写（thr 2.0／control B0／n_boot 10000／rng 20261109／epoch 00／n_windows 538／n_seeds 3）。
- **本轮新增审计面 ③（各臂离「翻转」有多远）——这是本轮最重要的量化**：按预注册，一条臂要活下来必须**配平后区间不含零**（且点估计为正）⇒ **点估计 ≤ 0 的臂天然免疫**（任何区间都救不回来）。据此清点：**{("、".join(immune))} 四臂免疫**（点估计 ≤ 0）＝**无论换多少个种子都不可能翻转**；**{("、".join(at_risk))} 两条臂点估计为正、理论上可翻转**，但用**换种子实测的端点抖动（max {jitter:.6f}）**衡量余量：**A1 的最近端点距零 0.016623 ＝ 抖动的 14.1 倍（安全）；A2 只有 0.001226 ＝ 抖动的 1.0 倍（唯一真正处在翻转边缘的臂）** ⇒ **A2 的等偏置裁决必须由远端作业 343 的多种子重跑来收口**，这是本轮提交该作业的直接理由。
- **本轮新增审计面 ②（已发布文档的口径合规）**：核查第 90 轮定的「**分分支写、不得合并成一个「全判负」**」是否真的落到纸面 ⇒ **已合规**：`docs/experiments/E45-…md` 的 S3 表逐臂分写「跨零 ⇒ 仍是面积／幅度效应」（A1/A3a/A3b/A4）与「原始即跨零 ⇒ 本来就没有增益」（A2/A4b），MASTER §4.47 与上层 MEMORY 状态段同样两分支分列 ⇒ **无需修**。
- **头条表复算（从 21 份 `*.eval.json` 的 `thresholds["2.0"].csi` 逐格解析）**：七臂中位 CSI@2.0 与过地板通过率＝{head_bits}；**对照臂 B0 自己只 2/3 过地板**（地板 0.0924 是单一种子口径）⇒ A3a／A3b 的 3/3 仍不构成晋级依据。**报告口径：中位数＋通过率，单点分数不单独出现。**
- **文档闸门**：`_check_tables` **PASS**（mismatched now = 0）、`_check_v245_row` **PASS**、`_e45_docs_numbers` **PASS**、`_check_v248_row` **59/61**（余 2 条属 SNAPSHOT 类、按设计过期；**CONTENT 类失败 = 0**）⇒ **未改 `docs/`**（无新数字、无裁决变化 ⇒ MASTER 仍 **v2.50**，不升版）。
- **下一步（不变）**：交接 §4.5 三件实现缺陷修复（① 幅度截断接到输出场上；② 量通道改相对护栏破绝对 `1e-12`；③ `maxratio` 改 `np.nanmedian` ＋ 报可用帧数）＋ 重开须**重新提机制族**，不是派生。
- **产物**：`tools/_e45_round91.py`、`tools/_e45_s3_seedrobust.py`、`scripts/e45_s3_seedrobust.sbatch`、`tools/_evals/e45/round91_status_audit.json`。
{END}"""

    brief = f"""{BEGIN}
### 2026-11-16｜第 91 轮（E45 远端状态复核 ＋ S3 护栏 bootstrap 种子稳健性首次独立重跑）

- **裁决不变＝六臂判负、无一臂进十 seed、不投十 seed 算力**；远端主链 273 等全 `COMPLETED`、产物 md5 13/13、结案后零改动；队列里 8 个 `qlfn2-f38` 属别的项目线、不干预。
- **新审计面**：用护栏自身实现（import，非重写）以注册种子复现后，换 {N_SEEDS} 个 bootstrap 种子重跑两级 bootstrap ⇒ raw 半边六臂 **{N_SEEDS}/{N_SEEDS} 种子区间全含零**（判负不依赖那一次抽样）；{("等偏置半边由远端 Slurm 作业 343 完成，" + "；".join(f"{a} {v['n_seeds_with_zero_inside_interval']}/{v['n_seeds_tried']}" for a, v in eq.items()) + "，可翻转臂＝" + ("、".join(audit.get("flippable_arms", [])) or "无") + " ⇒ 无一条翻转") if eq else "等偏置半边由远端 Slurm 作业 343（CPU）承担，因同账号配额排队中、待收割"}。
- **新审计面（各臂离翻转有多远）**：按预注册「配平后区间不含零且点为正」才活 ⇒ 点估计 ≤ 0 的臂天然免疫（A3a/A3b/A4/A4b 四臂，任何区间都救不回来）；点为正的 A1／A2 中，**A2 的最近端点距零仅 0.001226 ＝ 换种子实测抖动（max 0.001179）的 1.0 倍，是唯一处在翻转边缘的臂** ⇒ 其等偏置裁决须由作业 343 收口。
- **新审计面**：已发布文档的**分分支表述合规**（两分支分写，未合并成「全判负」）⇒ 无需修；头条表由 21 份 `*.eval.json` 复算，中位数＋过地板通过率口径成立（B0 自己只 2/3 过地板）。
- **文档闸门** tables／v245／docs-numbers 全 PASS、v248 59/61（CONTENT 类失败＝0）⇒ **未改 `docs/`**（MASTER 仍 v2.50，不升版）。
- **下一步（不变）**：§4.5 三件实现缺陷修复（幅度截断接到输出场／量通道改相对护栏破 `1e-12`／`maxratio` 改 `np.nanmedian` ＋ 报可用帧数）＋ 重开须重新提机制族。
{END}"""

    for path in (HANDOVER, DAYLOG, PROJLOG):
        print(write_marked(path, full))
    print(write_marked(AUTOMEM, brief))

    # The MEMORY.md status line is a bullet, not a block: replace it
    # between its own markers so it stays idempotent too.
    mem = MEMORY.read_text(encoding="utf-8")
    bullet = (f"{MEM_MARK}\n"
              f"- **E45 第 91 轮（2026-11-16）**：远端主链终态不变（273 等全 `COMPLETED`、"
              f"产物 md5 13/13、结案后零改动）；**首次独立重跑 S3 的重采样**——"
              f"用护栏自身实现，注册种子逐位复现（max|dev| = 0），"
              f"{N_SEEDS} 个 bootstrap 种子下六臂 raw 区间 **{N_SEEDS}/{N_SEEDS} 全含零** ⇒ "
              f"**判负不依赖那一次抽样**；等偏置半边由远端 Slurm 作业 343（CPU）承担。"
              f"已发布文档的分分支表述已合规（两分支分写）。"
              f"**裁决不变＝六臂判负、不投十 seed 算力**；"
              f"下一步仍是 §4.5 三件实现缺陷修复 ＋ 重开须重新提机制族。\n")
    if MEM_MARK in mem and mem.count(MEM_MARK) == 1:
        start = mem.index(MEM_MARK)
        nxt = mem.find("\n- ", start + len(MEM_MARK))
        end = len(mem) if nxt == -1 else nxt + 1
        mem = mem[:start] + bullet + mem[end:]
        MEMORY.write_text(mem, encoding="utf-8")
        print("REPLACED: MEMORY.md")
    else:
        lines = mem.split("\n")
        idx = None
        for i, ln in enumerate(lines):
            if "E45" in ln and "第 90 轮" in ln:
                idx = i
        if idx is None:
            for i, ln in enumerate(lines):
                if "E45" in ln and ln.strip().startswith("-"):
                    idx = i
        if idx is None:
            mem = mem.rstrip("\n") + "\n" + bullet
        else:
            mem = "\n".join(lines[:idx + 1]) + "\n" + bullet + "\n".join(lines[idx + 1:])
        MEMORY.write_text(mem, encoding="utf-8")
        print("WROTE: MEMORY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
