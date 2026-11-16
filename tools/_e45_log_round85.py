# -*- coding: utf-8 -*-
"""E45 round 85: record the remote status audit and append idempotent round blocks.

Read-only against the remote side (the status facts below were observed in this
round via a single merged remote query).  Locally this script only appends
marker blocks; if a marker is already present it skips instead of duplicating.
"""
import hashlib
import json
import os

ROOT = r"R:\1radarNWPprediction"
WS = os.path.join(ROOT, "radar-nwp-bench")
EVALS = os.path.join(WS, "tools", "_evals", "e45")
AUTOMEM = os.path.join(WS, ".workbuddy", "memory", "automations",
                       "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6", "memory.md")
MARK = "E45-R85"
BEGIN = "<!-- %s-BEGIN -->" % MARK
END = "<!-- %s-END -->" % MARK

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

REMOTE = {
    "squeue_wf": [],
    "authority_chain_completed": [273, 274, 275, 282, 283, 290, 291, 297, 298,
                                  314, 315, 316, 317],
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
    "close_time_ref": "2026-11-11T14:20:00+08:00",
    "sacct_since_2026_11_12_main_chain_jobs": [],
    "sacct_since_2026_11_12_diag_jobs": [297, 298],
    "runs_has_e46plus_dir": False,
    "remote_actions_taken": [],
}

AUDIT = {
    "round": 85,
    "checked_at": "2026-11-15T19:05:39+08:00",
    "remote": REMOTE,
    "verdict": {
        "s3_ci_cells": 12,
        "s3_ci_cells_crossing_zero": 12,
        "qualifying_arms": [],
        "arms_judged_negative": ["A1", "A2", "A3a", "A3b", "A4", "A4b"],
        "arms_advancing_to_10seed": [],
        "ten_seed_compute_spend": False,
        "note": "median + over-floor pass rate only; no lone score is quoted as a verdict",
    },
    "docs_changed": False,
    "master_version_unchanged": "v2.50",
}

BLOCK_LONG = """<!-- E45-R85-BEGIN -->
### 2026-11-15｜第 85 轮（E45 远端运行状态复核 ＋ 结案后 mtime 冻结再验证 ＋ 主链未重启 再验证）

- **远端终态（不变）**：权威链 273／274·282·290／275·283／291／297·298／314／315·316·317 全 `COMPLETED`；提交器 273 日志尾部确认 **summary 291 当年即自动跑完 S3 等偏置对照**（`s3_equalbias.json` 在位）⇒ **无需手动 sbatch**；`squeue -u wf` 空；`runs/e45/evals/` **21 份 `.eval.json`**（目录共 42 份）、`runs/e45/` **13 份** run 级 json 齐；`runs/` 下 **无 E46+ 目录**。
- **本轮再验证的两条「不重启」证据**：① `runs/e45/` 中 **结案后（2026-11-11 14:20 之后）被修改的文件 = 0 份**；② `sacct -S 2026-11-12` 主链 **无任何 e45-train／e45-stage2／e45-submit／e45-summary**（仅诊断作业 297／298）⇒ **主链从未重启**。
- **产物一致性**：远端 13 份 run 级 json 的 md5 与本地 **13/13 逐位相同** ⇒ 无需重新下载。
- **裁决复验**：`tools/_e45_verify_round84.py` **POS 13/13 VERDICT: PASS** ⇒ S3 **12/12 区间跨零**、合格臂集合为空；发布头条表 vs 原始产物重算 **7/7 命中（max|diff| < 1e−9）**；预注册判据逐臂逐 seed 重判与已发布一致（A1／A2 的 `corr(mask,amp)` 与 `mask_at_peak` 3/3 过；A3a／A3b 0/3 过；A4 3/3 过、A4b 2/3 过；`mean(out)=T` 不达格仍只记实现缺陷）。
- **结论不变**：**六臂判负、无一臂进十 seed、不投十 seed 算力**。**未改 `docs/`**（无新数字、无裁决变化 ⇒ MASTER 不升版，仍 **v2.50**）。
- **远端动作：零**（只读查询；未 sbatch、未下载、未同步、未改远端状态）。
- 产物：`tools/_evals/e45/round85_status_audit.json`。
<!-- E45-R85-END -->
"""

BLOCK_SHORT = """<!-- E45-R85-BEGIN -->
### E45 第 85 轮（2026-11-15）｜远端运行状态复核 ＋ 三重审计

- 远端权威链 273／274·282·290／275·283／291／297·298／314／315·316·317 全 `COMPLETED`；`squeue -u wf` 空；21 份 `.eval.json`（目录共 42 份）、13 份 run 级 json 齐；S3 由 summary 291 当年自动跑完（未手动 sbatch）；`runs/` 下无 E46+ 目录。**远端动作：零**。
- 三重审计：产物 md5 **13/13 全同**；**结案后被修改文件 = 0 份** ＋ `sacct -S 2026-11-12` 主链无作业 ⇒ 主链从未重启；文档闸门无新数字、**未改 `docs/`** ⇒ MASTER 仍 **v2.50**（不升版）。
- 裁决复验：`_e45_verify_round84` **13/13 PASS** ⇒ S3 **12/12 区间跨零**、合格臂集合为空 ⇒ **六臂判负、不投十 seed 算力**。
- 产物：`tools/_evals/e45/round85_status_audit.json`。
<!-- E45-R85-END -->
"""

TARGETS = [
    (os.path.join(ROOT, ".workbuddy", "memory", "HANDOVER_E45.md"), BLOCK_LONG),
    (os.path.join(ROOT, ".workbuddy", "memory", "2026-11-11.md"), BLOCK_SHORT),
    (os.path.join(WS, ".workbuddy", "memory", "2026-11-15.md"), BLOCK_SHORT),
    (AUTOMEM, BLOCK_SHORT),
]


def sha256(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def append_block(path, block):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# %s\n\n" % os.path.basename(path))
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if BEGIN in text:
        return "skipped-marker-present"
    if text and not text.endswith("\n"):
        text += "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + block)
    return "appended"


def main():
    out = os.path.join(EVALS, "round85_status_audit.json")
    os.makedirs(EVALS, exist_ok=True)
    prev = os.path.join(EVALS, "round84_status_audit.json")
    if os.path.exists(prev):
        with open(prev, "r", encoding="utf-8") as fh:
            AUDIT["audit_round84_reused"] = json.load(fh)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(AUDIT, fh, ensure_ascii=False, indent=2)
    print("audit written:", out)
    for path, block in TARGETS:
        before = sha256(path)
        action = append_block(path, block)
        after = sha256(path)
        print("%-10s %s | %s -> %s" % (action, os.path.basename(path),
                                       (before or "-")[:12], (after or "-")[:12]))


if __name__ == "__main__":
    main()
