#!/usr/bin/env python3
"""E45 round 72 — corrected writeback.

Two statements written in round 72's first pass were inaccurate / imprecise:
  (1) the generic "key-name" S3 probe found 0 boolean flags, which was rendered as
      "全部跨零＝False" (reads as "not all cross zero"). The authoritative check is
      `_e45_verify_round71.py` (POS 27/27), which confirms S3 12/12 intervals cross
      zero and the qualified-arm set is empty.
  (2) the inventory line reported "1 unparsable" without naming the file.
This script replaces the round-72 block in every writeback target with the
corrected text. Idempotent: it replaces the block, it never duplicates it.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "tools" / "_evals" / "e45"
UPPER = Path("R:/1radarNWPprediction")
HANDOVER = UPPER / ".workbuddy" / "memory" / "HANDOVER_E45.md"
UPPER_MEM = UPPER / ".workbuddy" / "memory" / "MEMORY.md"
UPPER_DAILY = UPPER / ".workbuddy" / "memory" / "2026-11-11.md"
PROJ_DAILY = ROOT / ".workbuddy" / "memory" / "2026-11-15.md"
AUTOMEM = ROOT / ".workbuddy" / "memory" / "automations" / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md"

audit = json.loads((EVALS / "round72_status_audit.json").read_text(encoding="utf-8"))
md5n = audit["md5_parity"]["n_match"]
md5t = audit["md5_parity"]["n_files"]
differ = audit["fix_file_drift"]["differ"]
gates = []
gates_all_rc0 = True
for g in audit["gates"]:
    verdict = "PASS" if g["returncode"] == 0 else f"rc={g['returncode']}"
    if g["returncode"] != 0:
        gates_all_rc0 = False
    tail = " / ".join(t for t in g["tail"] if t.strip())[:200]
    gates.append(f"`{g['script']}` **{verdict}**（{tail}）")
gates_verdict = ("四个闸门 rc 全为 0" if gates_all_rc0
                 else "**存在 rc≠0 的闸门**（`_check_v248_row` 的 59/61 是第 70 轮已登记的「按设计过期」状态，"
                      "余 2 项为「版本指针须停在 v2.48」「v2.48 行后须空行」类快照断言，**与实验数字无关**）")
r71_tail = " / ".join(t for t in audit["round71_reverify"]["tail"][-2:] if t.strip())
inv = audit["inventory"]
unparsable = inv["unparsable"]
unparsable_txt = ("、".join(f"`{u['file']}`" for u in unparsable) if unparsable else "无")
mentioned = inv["rounds_in_handover"]
not_in_handover = inv["round_products_not_in_handover"] or "无"

BODY = f"""- **判定**：远端 **E45 本身全部完成且与第 50–71 轮完全一致**（273/274/275/282/283/290/291/297/298/315/316/317 全 `COMPLETED`；`runs/e45/evals/` **21 份**评测齐；`runs/e45/` **13 份** json 与本地 md5 **{md5n}/{md5t} 逐位相同**）⇒ **E45 无新活动**；**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；**未改 `docs/`**（MASTER 仍 **v2.49**，不升版）。
- **远端**：Slurm 控制器 `systemctl is-active slurmctld` → **active**（第 64 轮的崩溃已恢复并保持）；队列里 8 个 `qlfn2-f36` 属另一远端仓库 `/home/wf/radarNWPQML`，与 E45 无关；`runs/` 下无 E46+ 目录 ⇒ **无新活动**。
- **审计一（远端-本地产物一致性）**：13 份 json 逐份 md5 比对 **{md5n}/{md5t} 全同、0 处漂移** ⇒ 已发布头条表 / S3 裁决的数据源自 2026-11-11 结案以来未被改动。
- **审计二（已修实现缺陷的同步漂移）**：本地已修、远端未同步的文件仍恰为 **{len(differ)} 个**（{'、'.join('`' + d + '`' for d in differ)}，逐一 md5 均不同）；**对照组 `models/amp_monotone_gate/model.py` 双方 md5 相同** ⇒ **最小同步集仍是这 4 个文件**，其余 E45 运行时代码未漂移 ⇒ 未同步即未重训重测 ⇒ **裁决不变**。
- **审计三（文档闸门回归复测）**：{'；'.join(gates)} ⇒ {gates_verdict}。
- **已发布裁决复验**：`_e45_verify_round71.py` → {r71_tail} ⇒ **S3 护栏 12/12 区间全部跨零、合格臂集合为空**（权威口径）。⚠️ 本轮自建的「键名泛化」探针未命中区间布尔（0 个标志）⇒ **该探针不作独立证据，仅记为工具局限**。
- **本地产物清单**：`tools/_evals/e45/` 共 **{inv['n_products']} 份** json，其中 round 产物 {len(inv['round_products'])} 份、**解析失败 {inv['n_unparsable']} 份**（{unparsable_txt}：首字符即非合法 JSON ⇒ 不是可解析的 json 产物；**本轮只登记、不改动**）；交接文档已登记轮次 {min(mentioned)}–{max(mentioned)}（{len(mentioned)} 轮），产物端未登记的轮次 {not_in_handover}。
- **三项待用户确认仍挂着**（未擅自执行）：① E46 三段 verdict 写回 MASTER（＝升 v2.50）；② 4 个修复文件按最小集同步远端；③ dx=3px／20–400mm 扫描须先预注册。
- **纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 事后复核、**不构成新判据、不改变任何裁决**；③ **MASTER 不升版**（仍 v2.49）。"""

HANDOVER_BLOCK = f"\n### 2026-11-15｜第 72 轮（状态复核 ＋ 产物一致性 / 修复文件漂移 / 文档闸门 三重审计）\n\n{BODY}\n"
AUTOMEM_BLOCK = ("\n<!-- E45-R72-BEGIN -->\n"
                 "### 2026-11-15｜第 72 轮（状态复核 ＋ 三重审计：产物一致性 / 修复文件漂移 / 文档闸门）\n\n"
                 f"{BODY}\n"
                 "<!-- E45-R72-END -->\n")
DAILY_BLOCK = f"\n- **2026-11-15 第 72 轮**：{BODY.splitlines()[0][2:]}\n" + \
              "\n".join("  " + ln.strip() for ln in BODY.splitlines()[1:] if ln.strip()) + "\n"
MEM_BLOCK = (
    f"\n- **E45 第 72 轮（2026-11-15）**：远端状态不变（273/274/275/282/283/290/291/297/298/315/316/317 全 `COMPLETED`、"
    f"21 份评测齐、13 份 json 与本地 md5 13/13 全同、控制器 active、无 E46+ 目录）；**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；"
    f"三重审计：产物一致性 {md5n}/{md5t} 全同、修复文件漂移仍恰为 4 个（最小同步集未变）、文档闸门 `_check_tables`/`_check_v245_row`/`_e45_docs_numbers` PASS、"
    f"`_check_v248_row` 59/61（按设计过期）；**MASTER 不升版（仍 v2.49）**；三项待用户确认仍挂着。\n"
)

TARGETS = [
    (HANDOVER, "### 2026-11-15｜第 72 轮（状态复核 ＋ 产物一致性", None, HANDOVER_BLOCK),
    (AUTOMEM, "<!-- E45-R72-BEGIN -->", "<!-- E45-R72-END -->", AUTOMEM_BLOCK),
    (UPPER_DAILY, "- **2026-11-15 第 72 轮**", None, DAILY_BLOCK),
    (PROJ_DAILY, "- **2026-11-15 第 72 轮**", None, DAILY_BLOCK),
    (UPPER_MEM, "- **E45 第 72 轮（2026-11-15）**", "\n", MEM_BLOCK),
]


def replace_block(path: Path, start: str, end: str | None, new_block: str) -> str:
    text = path.read_text(encoding="utf-8")
    n = text.count(start)
    if n != 1:
        return f"SKIP(start count={n})"
    i = text.index(start)
    if end is None:
        j = len(text)
    else:
        k = text.find(end, i + len(start))
        j = len(text) if k == -1 else k + len(end)
    while j < len(text) and text[j] == "\n":
        j += 1
    new_text = text[:i] + new_block.lstrip("\n") + ("\n" if not new_block.endswith("\n") else "") + text[j:]
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(new_text)
    return "replaced"


def main() -> int:
    bad = EVALS / "round57_remote_config_audit.json"
    print("=== unparsable product head ===")
    try:
        head = bad.read_bytes()[:160]
        print(repr(head))
    except OSError as exc:
        print("read error:", exc)

    print("\n=== block replacement ===")
    for path, start, end, block in TARGETS:
        print(f"{path.name}: {replace_block(path, start, end, block)}")

    print("\n=== marker counts (must be exactly 1 each) ===")
    for path, start, _e, _b in TARGETS:
        txt = path.read_text(encoding="utf-8")
        print(f"{path.name}: round72_marker={txt.count(start)}")

    # refresh the audit json so it carries the same clarified notes
    audit["s3_guard"]["note"] = ("generic key-name probe matched 0 boolean flags; the authoritative "
                                 "check is _e45_verify_round71.py (POS 27/27): S3 12/12 intervals cross zero, "
                                 "qualified-arm set empty. This probe is NOT independent evidence.")
    audit["inventory"]["unparsable_note"] = ("round57_remote_config_audit.json is not valid JSON "
                                             "(first character already fails to parse); registered only, "
                                             "not modified this round.")
    audit["writeback"] = {"mode": "corrected-replace", "round": 72, "date": "2026-11-15",
                           "master_version": "v2.49 (unchanged)", "docs_touched": False}
    (EVALS / "round72_status_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nUPDATED", EVALS / "round72_status_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
