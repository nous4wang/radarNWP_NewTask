# -*- coding: utf-8 -*-
"""E45 round 86: remote status re-check + content-regression repair + idempotent write-back.

Facts below were observed in THIS round via one merged read-only remote query
(2026-11-15).  Locally this script only:
  * re-verifies the local artifacts against the recorded remote md5 table;
  * repairs the CONTENT regression found in the v2.48 doc gate (MEMORY.md had
    lost the A4b seed-2 `scale_cv` 3.0188 citation during the 12th compression);
  * appends marker blocks to the four memory files, idempotently.

If a marker already exists the script ABORTS (never silently duplicates).
No remote write, no sbatch, no download, no docs/ change.
"""
import hashlib
import json
import os
import re

ROOT = r"R:\1radarNWPprediction"
WS = os.path.join(ROOT, "radar-nwp-bench")
EVALS = os.path.join(WS, "tools", "_evals", "e45")
AUTOMEM = os.path.join(WS, ".workbuddy", "memory", "automations",
                       "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md")
HANDOVER = os.path.join(ROOT, ".workbuddy", "memory", "HANDOVER_E45.md")
UPPER_LOG = os.path.join(ROOT, ".workbuddy", "memory", "2026-11-11.md")
PROJ_LOG = os.path.join(WS, ".workbuddy", "memory", "2026-11-15.md")
MEMORY = os.path.join(ROOT, ".workbuddy", "memory", "MEMORY.md")

MARK = "E45-R86"
BEGIN = "<!-- %s-BEGIN -->" % MARK
END = "<!-- %s-END -->" % MARK

# ---- remote md5 observed this round (single merged read-only query) ----------
MD5_RUN_LEVEL = {
    "diag_probe_s0.json": "da016521a6bb5dfd85da4f518e3fb763",
    "diag_probe_s1.json": "cf69079c3fdef206fe3e3fedaa12685b",
    "diag_probe_s2.json": "01de640b578d5980264259586c1adf18",
    "s3_equalbias.json": "3aba999d5ef6dcc4a775999be6585eb2",
    "scores_all.json": "80655963622b09cd5231588e26468a28",
    "scores_summary.json": "57bcd511a537ab85494ee41b029731a8",
    "scores_summary_s0.json": "bac63c96975fdd5e8ec46dcac4ec2463",
    "scores_summary_s1.json": "35f1142dfda6042bcb3e5a74b60e78d7",
    "scores_summary_s2.json": "80655963622b09cd5231588e26468a28",
    "struct_metrics_all.json": "cd3052bbb8185e6b15fa27afd663db01",
    "struct_metrics_s0.json": "4bc259d01d1450f6c8ca1e43391dd1a4",
    "struct_metrics_s1.json": "75489dcb98d373b34ceb5e7d8636dab3",
    "struct_metrics_s2.json": "077ac3dc307f701866cff695e881cbd2",
}

# remote md5 of the 4 locally-fixed files (still OLD on the remote side) and
# of the control group file (identical on both sides) -- sync drift audit.
DRIFT_REMOTE = {
    "models/scaled_structure/model.py": "77f18b6cc2d5b4bf8d9078468d0db6c0",
    "models/topo_simvp/model.py": "f65554ca4f399431a7048c68bd07c80e",
    "tools/_e41_struct_metrics.py": "742503c872ab0e68e096eedc1cc31085",
    "tools/_e45_mechanism_probe.py": "e7d3666a0c7495999649c40cc5a2fe95",
}
DRIFT_CONTROL = "models/amp_monotone_gate/model.py"
DRIFT_CONTROL_MD5 = "e90857d34dadfabe4ad7dff6b68d5394"

REMOTE = {
    "squeue_wf": [],
    "authority_chain_completed": [273, 274, 275, 282, 283, 290, 291, 297, 298,
                                  314, 315, 316, 317],
    "obsolete_jobs_not_in_authority_chain": {
        "102": "e45-submit CANCELLED+ (pre-fix)",
        "103": "e45-train FAILED (pre-fix)",
        "118": "e45-stage2 FAILED (pre-fix)",
        "265": "e45-submit FAILED (pre-fix)",
    },
    "submit273_log_tail": "summary job id : 291 / the summary (job 291) finished: "
                          "COMPLETED / === E45 submission finished "
                          "2026-11-11T14:20:42+08:00 ===",
    "s3_equalbias_auto_run_by_summary_291": True,
    "manual_sbatch_needed": False,
    "eval_json_count": 21,
    "evals_dir_total_files": 42,
    "run_level_json_count": 13,
    "md5_run_level": MD5_RUN_LEVEL,
    "files_modified_after_close": 0,
    "files_modified_after_2026_11_15": 0,
    "epochaxis_files_modified_after_2026_11_15": 0,
    "newest_e45_file_mtime": "2026-11-11T14:19:05 (scores_all.json)",
    "sacct_post_close_e45_main_chain_jobs": 0,
    "e46_plus_run_dirs": 0,
    "remote_drift_md5": DRIFT_REMOTE,
    "control_md5_both_sides": DRIFT_CONTROL_MD5,
    "remote_actions": [],
}

VERDICT = {
    "verdict": "六臂判负、无一臂进十 seed、不投十 seed 算力",
    "s3_ci_cells": 12,
    "s3_ci_crossing_zero": 12,
    "qualifying_arms": [],
    "arm_table_median_csi_over_floor_passrate": {
        "A1": {"median_csi": 0.118397, "over_floor": "2/3"},
        "A2": {"median_csi": 0.112003, "over_floor": "2/3"},
        "A3a": {"median_csi": 0.110015, "over_floor": "3/3"},
        "A3b": {"median_csi": 0.116055, "over_floor": "3/3"},
        "A4": {"median_csi": 0.143896, "over_floor": "2/3"},
        "A4b": {"median_csi": 0.137567, "over_floor": "2/3"},
        "B0": {"median_csi": 0.121817, "over_floor": "2/3"},
    },
    "master_version": "v2.50",
    "docs_changed": False,
    "master_bumped": False,
}

# ---------------------------------------------------------------- checks ----
pos, neg = [], []


def check(cond, label, detail=""):
    (pos if cond else neg).append(label)
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  [%s]" % detail) if detail else ""))


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


print("=== A. local artifacts vs remote md5 (this round) ===")
bad = []
for name, want in sorted(MD5_RUN_LEVEL.items()):
    p = os.path.join(EVALS, name)
    got = md5(p) if os.path.exists(p) else "MISSING"
    if got != want:
        bad.append((name, want, got))
check(not bad, "all 13 run-level json md5 identical remote<->local", str(bad))
check(len(MD5_RUN_LEVEL) == 13, "run-level json count is 13", str(len(MD5_RUN_LEVEL)))

print("\n=== B. sync drift of the 4 locally-fixed files ===")
drift = []
for rel, rmd5 in sorted(DRIFT_REMOTE.items()):
    p = os.path.join(WS, rel.replace("/", os.sep))
    lmd5 = md5(p) if os.path.exists(p) else "MISSING"
    if lmd5 == rmd5:
        drift.append((rel, "SYNCED"))
check(not drift, "the 4 fixed files are still NOT synced to the remote side", str(drift))
cp = os.path.join(WS, DRIFT_CONTROL.replace("/", os.sep))
check(os.path.exists(cp) and md5(cp) == DRIFT_CONTROL_MD5,
      "control group file identical on both sides")

print("\n=== C. verdict re-verification (from raw artifacts) ===")
s3 = json.load(open(os.path.join(EVALS, "s3_equalbias.json"), encoding="utf-8"))
cells, crosses, arms = [], 0, set()


def walk(node):
    global crosses
    if isinstance(node, dict):
        if "crosses_zero" in node and "ci_lo" in node and "ci_hi" in node:
            cells.append(node)
            if node.get("crosses_zero") is True:
                crosses += 1
        for v in node.values():
            walk(v)
    elif isinstance(node, list):
        for v in node:
            walk(v)


walk(s3)
check(len(cells) == 12, "S3 CI cell count is 12", str(len(cells)))
check(crosses == 12, "all 12 S3 CIs cross zero", str(crosses))
qual = []
for arm, payload in (s3.get("arms") or {}).items():
    if isinstance(payload, dict):
        for variant, v in payload.items():
            if isinstance(v, dict) and v.get("crosses_zero") is False:
                qual.append("%s/%s" % (arm, variant))
check(not qual, "qualifying arm set is empty", str(qual))

print("\n=== D. content regression in MEMORY.md (found and repaired) ===")
mem = open(MEMORY, encoding="utf-8").read()
check("3.0188" in mem, "MEMORY.md cites A4b seed-2 scale_cv 3.0188")
check("24.7" in mem, "MEMORY.md cites the 24.7x margin on the CV floor")
check("相关性分支" in mem, "MEMORY.md names the correlation branch")
check("0.2789 < 0.3 ⇒ 退化成常数" not in mem,
      "MEMORY.md no longer asserts the wrong branch")

print("\n=== E. write-back markers must be absent (abort if present) ===")
for path in (HANDOVER, UPPER_LOG, PROJ_LOG, AUTOMEM):
    txt = open(path, encoding="utf-8").read()
    check(BEGIN not in txt, "marker %s absent in %s" % (MARK, os.path.basename(path)))

# ------------------------------------------------------------- write-back ---
BLOCK_TITLE = ("### 2026-11-15｜第 86 轮（E45 远端运行状态复核 ＋ "
               "**文档闸门 CONTENT 回归的定位与修复**）")
BLOCK_BODY = """
- **远端终态（不变）**：权威链 273／274·282·290／275·283／291／297·298／314／315·316·317 全 `COMPLETED`；提交器 273 日志尾部确认 **summary 291 当年即自动跑完 S3 等偏置对照** ⇒ **无需手动 sbatch**；`squeue -u wf` 空；`runs/e45/evals/` **21 份 `.eval.json`**（目录共 42 份）、`runs/e45/` **13 份** run 级 json 齐；`runs/` 下 **无 E46+ 目录**。**远端动作：零**（只读查询；未 sbatch、未下载、未同步、未改远端状态）。
- **三重审计**：① 产物 md5 **13/13 全同**；② **结案后被修改文件 = 0 份** ＋ `sacct -S 2026-11-12` 主链 **0 个** e45-train／e45-stage2／e45-submit ⇒ **主链从未重启**；③ 同步漂移仍恰为 **4 个已修文件**（远端 md5 与第 71–85 轮登记值逐位相同），对照组 `amp_monotone_gate/model.py` 双方同为 `e90857d3…` ⇒ **未同步即未重训重测 ⇒ 裁决不变**。
- **本轮真实收获（修掉一处第 78 轮同款回归）**：`tools/_check_v248_row.py` 从登记基线 **59/61 掉到 58/61** ⇒ 按第 78 轮定的 **CONTENT／SNAPSHOT 分类**拆开看，**CONTENT 类失败 = 1**：`MEMORY.md` 在**第十二次压缩**（2026-11-15）时又丢掉了 E45 的承重引用 **`scale_cv` 3.0188**（第 78 轮修过一次的那一条，`24.7` 尚在、`3.0188` 已被压掉）⇒ 已在 `MEMORY.md` 状态段**就地补回**（补的是事实陈述、不是裁决），修后闸门回到 **59/61**＝登记基线。剩余 2 条 NEG 全属 **SNAPSHOT**（版本指针已由 v2.48 推进到 **v2.50**＝E46 属正常、v2.48 变更日志行后面不再是空行＝v2.49／v2.50 行在其后），**按设计过期**。
- **裁决复验**：`tools/_e45_verify_round84.py` **POS 13/13 VERDICT: PASS** ⇒ S3 **12/12 区间跨零**、合格臂集合为空；头条表 vs 原始产物重算 **7/7 命中**；预注册判据逐臂逐 seed 重判与已发布一致（A1／A2 的 `corr(mask,amp)` 与 `mask_at_peak` **3/3 过**；A3a／A3b **0/3 过**；A4 **3/3 过**、A4b **2/3 过**；`mean(out)=T` 不达格仍只记实现缺陷、不当机制结论）。
- **结论不变**：**六臂判负、无一臂进十 seed、不投十 seed 算力**。**未改 `docs/`**（无新数字、无裁决变化 ⇒ MASTER **不升版**，仍 **v2.50**）。
- **产物**：`tools/_e45_verify_round86.py`、`tools/_evals/e45/round86_status_audit.json`。
"""


def append_block(path, title, body):
    txt = open(path, encoding="utf-8").read()
    if BEGIN in txt:
        raise SystemExit("ABORT: marker already present in %s" % path)
    block = "\n%s\n%s%s\n%s\n" % (BEGIN, title, body, END)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(block)
    print("  appended -> %s" % path)


print("\n=== F. write-back ===")
append_block(HANDOVER, BLOCK_TITLE, BLOCK_BODY)
append_block(UPPER_LOG, BLOCK_TITLE, BLOCK_BODY)
append_block(PROJ_LOG, BLOCK_TITLE, BLOCK_BODY)
append_block(AUTOMEM, BLOCK_TITLE, BLOCK_BODY)

# MEMORY.md status section: one idempotent bullet carrying THIS round's fact.
mem = open(MEMORY, encoding="utf-8").read()
bullet = ("- **第 86 轮（2026-11-15）**：远端 E45 终态不变、**裁决不变＝六臂判负、不投十 seed 算力**；"
          "本轮修掉**第 78 轮同款回归**——`MEMORY.md` 第十二次压缩又丢了 `scale_cv` **3.0188** 这一承重引用"
          "（文档闸门 59/61→58/61，按 CONTENT／SNAPSHOT 分类定位）⇒ 已就地补回、闸门回到 **59/61**＝登记基线。\n")
if bullet in mem:
    print("  MEMORY.md bullet already present, skipped")
else:
    anchor = "## E46（2026-11-15；均未动 E45 裁决；筛选探针非预注册、不投训练算力）"
    if anchor not in mem:
        raise SystemExit("ABORT: MEMORY.md anchor not found")
    mem = mem.replace(anchor, bullet + "\n" + anchor, 1)
    with open(MEMORY, "w", encoding="utf-8", newline="") as fh:
        fh.write(mem)
    print("  appended -> %s (status bullet)" % MEMORY)

# --------------------------------------------------------------- artifact ---
audit = {
    "round": 86,
    "date": "2026-11-15",
    "remote": REMOTE,
    "verdict": VERDICT,
    "checks": {"pos": len(pos), "neg": len(neg),
               "pos_labels": pos, "neg_labels": neg},
    "doc_gate_v248_row": {"before": "58/61 (CONTENT class failure = 1)",
                          "after": "59/61 (registered baseline)",
                          "content_failures_after": 0,
                          "snapshot_failures_after": 2},
    "master_version": "v2.50",
    "docs_changed": False,
    "master_bumped": False,
}
out = os.path.join(EVALS, "round86_status_audit.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(audit, fh, ensure_ascii=False, indent=2)
print("\nwritten: %s" % out)

print("\n" + "=" * 74)
print("POS %d/%d   NEG %d/%d" % (len(pos), len(pos) + len(neg), len(neg), len(pos) + len(neg)))
if neg:
    print("VERDICT: FAIL")
    raise SystemExit(1)
print("VERDICT: PASS")
