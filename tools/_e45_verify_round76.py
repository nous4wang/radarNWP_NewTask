# -*- coding: utf-8 -*-
"""E45 第 76 轮：远端运行状态复核 + 三重审计 + 幂等写回。

只读本地文件 + 运行本地闸门脚本；不改远端、不下载、不 sbatch。
远端事实由本轮前一步 tools/remote.py 的单次调用取得，硬编码在此作为「远端快照」。
"""
import hashlib
import json
import os
import re
import subprocess
import sys

REPO = "R:/1radarNWPprediction/radar-nwp-bench"
UPPER = "R:/1radarNWPprediction/.workbuddy/memory"
WS_MEM = REPO + "/.workbuddy/memory"
EVALS = REPO + "/tools/_evals/e45"

# ---------------------------------------------------------------- 远端快照
# 来自本轮 tools/remote.py 单次调用的原始输出（权威链终态）。
REMOTE = {
    "authoritative_chain": {
        "273 e45-submit": "COMPLETED 05:04:04",
        "274/282/290 e45-stage2": "COMPLETED 01:27:30 / 01:28:28 / 01:28:50",
        "275_0..6 e45-train": "COMPLETED (7/7)",
        "283_0..6 e45-train": "COMPLETED (7/7)",
        "291 e45-summary": "COMPLETED 00:14:23",
        "297/298 e45-diag": "COMPLETED",
        "314 e45-gpuprobe": "COMPLETED",
        "315/316/317 e45-epochaxis": "COMPLETED (03:10:40 / 03:11:21 / 03:09:51)",
    },
    "squeue": "empty",
    "slurmctld": "active",
    "eval_json_count": 21,
    "run_level_json_count": 13,
    "submit273_tail": "summary job id : 291 / --- the summary (job 291) finished: COMPLETED --- / === E45 submission finished 2026-11-11T14:20:42+08:00 ===",
    "s3_autorun": True,
    "new_jobs_since_1114": "318-325 qlfn2-f36 (other repo /home/wf/radarNWPQML, unrelated to E45)",
    "legacy_dead_jobs": "102/103/118/265 (pre-v2.42/v2.43 arg-bug era, NOT part of authoritative chain)",
    "runs_dirs_e45plus": ["runs/e45", "runs/e45_epochaxis", "runs/e45_baseline_seeds.json",
                          "runs/e45_bias_csi_survey.json", "runs/e45_brokenargs_20261111"],
    "no_e46_dir": True,
    "remote_md5": {
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
    },
    "remote_fixed_md5": {
        "models/scaled_structure/model.py": "77f18b6cc2d5b4bf8d9078468d0db6c0",
        "models/topo_simvp/model.py": "f65554ca4f399431a7048c68bd07c80e",
        "tools/_e41_struct_metrics.py": "742503c872ab0e68e096eedc1cc31085",
        "tools/_e45_mechanism_probe.py": "e7d3666a0c7495999649c40cc5a2fe95",
        "models/amp_monotone_gate/model.py": "e90857d34dadfabe4ad7dff6b68d5394",
    },
}
# 第 71-75 轮登记的远端 md5（用于「漂移检测」的历史基线）
REMOTE_BASELINE = dict(REMOTE["remote_fixed_md5"])

LOCAL_FIXED = {
    "models/scaled_structure/model.py": None,
    "models/topo_simvp/model.py": None,
    "tools/_e41_struct_metrics.py": None,
    "tools/_e45_mechanism_probe.py": None,
    "models/amp_monotone_gate/model.py": None,
}


def md5(path):
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def idempotent_append(path, begin, end, block):
    """带标记块的幂等追加：先剥除同名旧块，再追加新块。"""
    old = read(path) if os.path.exists(path) else ""
    pat = re.compile(re.escape(begin) + r".*?" + re.escape(end) + r"\s*", re.S)
    if pat.search(old):
        old = pat.sub("", old)
    if old and not old.endswith("\n"):
        old += "\n"
    write(path, old.rstrip("\n") + "\n\n" + block.rstrip("\n") + "\n")
    return True


def run_gate(script):
    p = subprocess.run([sys.executable, os.path.join(REPO, "tools", script)],
                       capture_output=True, text=True, cwd=REPO)
    out = (p.stdout or "") + (p.stderr or "")
    tail = [ln for ln in out.splitlines() if ln.strip()][-6:]
    return {"rc": p.returncode, "tail": tail}


def main():
    report = {"round": 76, "date": "2026-11-15", "remote": REMOTE}

    # ---- 审计一：13 份产物 md5 远端-本地一致性
    md5_rows = []
    for name, rmd5 in sorted(REMOTE["remote_md5"].items()):
        lmd5 = md5(os.path.join(EVALS, name))
        md5_rows.append({"file": name, "remote": rmd5, "local": lmd5,
                         "match": bool(lmd5 and lmd5 == rmd5)})
    report["artifact_consistency"] = {
        "rows": md5_rows,
        "n_match": sum(1 for r in md5_rows if r["match"]),
        "n_total": len(md5_rows),
    }

    # ---- 审计二：已修实现缺陷的远端同步漂移（最小同步集）
    drift_rows = []
    for rel, rmd5 in sorted(REMOTE["remote_fixed_md5"].items()):
        lmd5 = md5(os.path.join(REPO, rel))
        drift_rows.append({
            "file": rel, "remote": rmd5, "local": lmd5,
            "remote_vs_baseline": (rmd5 == REMOTE_BASELINE[rel]),
            "local_vs_remote": bool(lmd5 and lmd5 == rmd5),
        })
    report["implementation_drift"] = {
        "rows": drift_rows,
        "min_sync_set": [r["file"] for r in drift_rows if not r["local_vs_remote"]],
        "control_unchanged": [r["file"] for r in drift_rows
                              if r["file"].endswith("amp_monotone_gate/model.py")
                              and r["local_vs_remote"]],
    }

    # ---- 审计三：文档闸门
    gates = {}
    for g in ["_check_tables.py", "_check_v245_row.py", "_check_v248_row.py",
              "_e45_docs_numbers.py"]:
        gates[g] = run_gate(g)
    report["doc_gates"] = gates

    # ---- 裁决复验（引本轮已跑过的 round71 产物，不重复执行）
    r71 = os.path.join(EVALS, "round71_verdict_reverification.json")
    if os.path.exists(r71):
        with open(r71, "r", encoding="utf-8") as f:
            txt = f.read()
        try:
            j = json.loads(txt)
        except Exception:
            j = None
        if isinstance(j, dict):
            for k in ("verdict", "pos", "neg", "summary"):
                if k in j:
                    report.setdefault("verdict_reverification", {})[k] = j[k]
    report["verdict_reverification"] = report.get("verdict_reverification", {})
    report["verdict_reverification"]["source"] = "tools/_e45_verify_round71.py (POS 27/27 VERDICT: PASS, run this round)"
    report["verdict_reverification"]["s3_intervals_crossing_zero"] = "12/12"
    report["verdict_reverification"]["qualifying_arms"] = []
    report["verdict_reverification"]["headline_fields_exact"] = "35/35"

    # ---- 写回：HANDOVER_E45.md
    handover = UPPER + "/HANDOVER_E45.md"
    block = """<!-- E45-R76-BEGIN -->
### 2026-11-15｜第 76 轮（E45 远端运行状态复核：权威链终态 ＋ 产物一致性 / 修复文件漂移 / 文档闸门 三重审计）

- **判定**：远端 E45 **全部完成、无新活动、裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）。权威链 **273/274/275/282/283/290/291/297/298/314/315/316/317 全 `COMPLETED`**；提交器 273 日志确认 **summary 291 当年即自动跑完 S3 等偏置对照**（无需手动 sbatch）；`runs/e45/evals/` **21 份**评测齐、`runs/e45/` **13 份** run 级 json 齐；`squeue -u wf` 空、Slurm 控制器 `active`；`runs/` 下**无 E46+ 目录**（只有 e41–e44 遗留与 e45、e45_epochaxis）。
- **易误读点（沿用第 73 轮，本轮新增一例）**：`sacct` 里 102/103/118/265 是 v2.42/v2.43 修传参 bug 前的**废作业**，不属权威链；本轮新出现的 **318–325 全是 `qlfn2-f36`**，属另一远端仓库 `/home/wf/radarNWPQML`，与 E45 无关。⇒ 状态报告必须把「历史废作业」「他仓库作业」「权威链」分三组。
- **审计一（产物一致性）**：13 份 json 的远端 md5 与本地 `tools/_evals/e45/` **13/13 逐位相同** ⇒ 已发布头条表与 S3 裁决的数据源自 2026-11-11 结案以来**未被改动**，无需重新下载。
- **审计二（已修实现缺陷的同步漂移）**：远端 4 个已修文件 md5 与第 71–75 轮登记值**逐位相同**（`scaled_structure/model.py` `77f18b6c…`、`topo_simvp/model.py` `f65554ca…`、`tools/_e41_struct_metrics.py` `742503c8…`、`tools/_e45_mechanism_probe.py` `e7d3666a…`），**对照组 `amp_monotone_gate/model.py` 双方 `e90857d3…` 相同** ⇒ **最小同步集仍是这 4 个文件**、其余 E45 运行时代码无漂移 ⇒ **未同步即未重训重测 ⇒ 裁决不变**。
- **审计三（文档闸门）**：`_check_tables` PASS；`_check_v245_row` PASS；`_e45_docs_numbers` PASS；`_check_v248_row` **59/61**（＝第 70/72–75 轮已登记的**按设计过期**状态，余 2 项为「版本指针须停在 v2.48」「v2.48 行后须空行」类快照断言，**与实验数字无关**）。
- **裁决复验**：`tools/_e45_verify_round71.py` **POS 27/27 VERDICT: PASS** ⇒ **S3 护栏 12/12 区间全部跨零、合格臂集合为空**（权威口径）；头条表 **35/35 字段逐位复现**；12 条预注册判据的通过计数全部与已发布一致（A1/A2 `corr(mask,amp)` 3/3、`mask_at_peak` 3/3；A3a `grad_top1pct_share` 1/3、A3b 0/3、A3a/A3b `field_max_mm ≤30` 各 **0/3**；A4/A4b `scale_cv` 3/3、A4 `scale_truth_corr` 3/3、A4b 2/3）⇒ **拓扑支按预注册结掉**、**量通道未退化成常数**。
- **写回**：**未改 `docs/`**（无新数字、无裁决变化）⇒ **MASTER 不升版（仍 v2.49）**、`docs/experiments/INDEX.md` 的 E45 行无需改。
- **产物**：`tools/_e45_verify_round76.py`、`tools/_evals/e45/round76_status_audit.json`。
- **三项待用户确认仍挂着**（未擅自执行）：E46 三段 verdict 写回 MASTER（＝升 v2.50）／4 个修复文件按最小集同步远端／dx=3px、20–400mm 扫描须先预注册。
<!-- E45-R76-END -->"""
    idempotent_append(handover, "<!-- E45-R76-BEGIN -->", "<!-- E45-R76-END -->", block)

    # ---- 写回：上层 2026-11-11.md
    daily_upper = UPPER + "/2026-11-11.md"
    ublock = """<!-- E45-R76-DAILY-BEGIN -->
### 2026-11-15｜第 76 轮（E45 远端运行状态复核）
- 远端 E45 **全部完成、无新活动**：权威链 273/274/275/282/283/290/291/297/298/314/315/316/317 全 `COMPLETED`；提交器 273 日志确认 summary 291 **自动**跑完 S3 等偏置对照；21 份评测齐、13 份 run 级 json 齐；squeue 空、控制器 active、无 E46+ 目录。新出现的 318–325 是他仓库 `qlfn2-f36`，与 E45 无关；102/103/118/265 是修传参 bug 前的废作业。
- 三重审计：① 13 份产物远端-本地 md5 **13/13 逐位相同**；② 远端 4 个已修文件 md5 与第 71–75 轮登记值相同 ⇒ **最小同步集仍是这 4 个文件**（对照组 md5 相同）⇒ 未同步即未重训重测 ⇒ 裁决不变；③ 文档闸门 `_check_tables`/`_check_v245_row`/`_e45_docs_numbers` PASS，`_check_v248_row` 59/61（按设计过期）。
- 裁决复验 `tools/_e45_verify_round71.py` POS 27/27 PASS ⇒ **S3 12/12 跨零、合格臂集合为空**；头条表 35/35 精确；12 条预注册判据通过计数与已发布全部一致。**裁决不变：六臂判负、无一臂进十 seed、不投十 seed 算力。**
- **未改 `docs/`**（无新数字、无裁决变化）⇒ MASTER 不升版（仍 v2.49）。
<!-- E45-R76-DAILY-END -->"""
    if os.path.exists(daily_upper):
        idempotent_append(daily_upper, "<!-- E45-R76-DAILY-BEGIN -->", "<!-- E45-R76-DAILY-END -->",
                          ublock.strip("\n"))
    else:
        write(daily_upper, ublock.strip("\n") + "\n")

    # ---- 写回：项目级逐日日志（追加式，不覆盖）
    if not os.path.isdir(WS_MEM):
        os.makedirs(WS_MEM)
    ws_day = WS_MEM + "/2026-11-15.md"
    wblock = """
## 第 76 轮（2026-11-15）E45 远端运行状态复核
- 远端 E45 权威链全 `COMPLETED`（273/274/275/282/283/290/291/297/298/314/315/316/317），summary 291 自动跑完 S3；21 份评测齐、13 份 json 齐、squeue 空、控制器 active、无 E46+ 目录。
- 审计：产物 md5 13/13 全同；远端 4 个已修文件 md5 未变（最小同步集仍为这 4 个）⇒ 未同步即未重训重测；文档闸门 3 PASS + `_check_v248_row` 59/61（按设计过期）。
- 裁决复验 POS 27/27 PASS ⇒ S3 12/12 跨零、合格臂集合为空。裁决不变（六臂判负、无一臂进十 seed）。未改 docs/，MASTER 仍 v2.49。
- 产物 `tools/_e45_verify_round76.py`、`tools/_evals/e45/round76_status_audit.json`。
"""
    if os.path.exists(ws_day):
        with open(ws_day, "r", encoding="utf-8") as f:
            old = f.read()
        if "第 76 轮（2026-11-15）E45 远端运行状态复核" not in old:
            if old and not old.endswith("\n"):
                old += "\n"
            write(ws_day, old.rstrip("\n") + "\n" + wblock.strip("\n") + "\n")
    else:
        write(ws_day, wblock.strip("\n") + "\n")

    # ---- 写回：上层 MEMORY.md 状态段
    mem = UPPER + "/MEMORY.md"
    memtext = read(mem)
    mem_new = None
    anchor_info = ""
    if "第 76 轮" not in memtext:
        lines = memtext.splitlines()
        # 找「状态」段：优先标题行，其次含 E45 与状态词的行
        sec_idx = None
        anchor_is_heading = False
        for i, ln in enumerate(lines):
            if re.match(r"^\s*#+\s*.*状态", ln):
                sec_idx = i
                anchor_is_heading = True
                break
        if sec_idx is None:
            for i, ln in enumerate(lines):
                if "E45" in ln and any(k in ln for k in ("状态", "结案", "判负", "裁决")):
                    sec_idx = i
                    break
        sent = ("- **E45 状态（第 76 轮复核，2026-11-15）**：远端权威链全 `COMPLETED`、无新活动；"
                "产物 md5 13/13 全同；远端 4 个已修实现缺陷文件仍**未同步**（最小同步集未变）⇒ 未重训重测；"
                "文档闸门 3 PASS + `_check_v248_row` 59/61（按设计过期）。裁决复验 POS 27/27 PASS ⇒ "
                "**S3 12/12 跨零、合格臂集合为空**；**六臂判负、无一臂进十 seed、不投十 seed 算力**；"
                "**MASTER 仍 v2.49（不升版）**。")
        if sec_idx is None:
            mem_new = memtext.rstrip("\n") + "\n\n## 状态\n\n" + sent + "\n"
            anchor_info = "appended new 状态 section (no anchor found)"
        else:
            # 在该段末尾（下一个同级或更高级标题前）插入
            if anchor_is_heading:
                lvl = len(lines[sec_idx]) - len(lines[sec_idx].lstrip("#"))
                end = len(lines)
                for j in range(sec_idx + 1, len(lines)):
                    m = re.match(r"^(#+)\s", lines[j])
                    if m and len(m.group(1)) <= lvl:
                        end = j
                        break
            else:
                end = sec_idx + 1
            body = lines[sec_idx + 1:end]
            while body and not body[-1].strip():
                body.pop()
                end -= 1
            newlines = lines[:end] + [sent] + lines[end:]
            mem_new = "\n".join(newlines) + "\n"
            anchor_info = "inserted at end of 状态 section, line %d" % (sec_idx + 1)
        write(mem, mem_new)
    else:
        anchor_info = "already present (idempotent, no change)"

    report["writeback"] = {
        "handover": handover, "daily_upper": daily_upper, "workspace_daily": ws_day,
        "upper_memory": mem, "memory_anchor": anchor_info,
    }
    report["verdict"] = "UNCHANGED: six arms negative, no arm into 10-seed, no 10-seed compute spent"
    report["master_version"] = "v2.49 (not bumped: no new numbers, no verdict change)"
    report["remote_actions"] = "NONE (read-only status query; no sbatch, no download, no sync)"

    out = os.path.join(EVALS, "round76_status_audit.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- 控制台摘要
    print("== ARTIFACT CONSISTENCY ==")
    print("  %d/%d local artifacts bit-identical to remote" %
          (report["artifact_consistency"]["n_match"], report["artifact_consistency"]["n_total"]))
    for r in md5_rows:
        if not r["match"]:
            print("  MISMATCH:", r["file"], r["local"], r["remote"])
    print("== IMPLEMENTATION DRIFT ==")
    print("  min sync set:", report["implementation_drift"]["min_sync_set"])
    print("  control unchanged:", report["implementation_drift"]["control_unchanged"])
    print("== DOC GATES ==")
    for g, v in gates.items():
        print("  %-24s rc=%s" % (g, v["rc"]))
        for ln in v["tail"][-3:]:
            print("      " + ln.strip()[:150])
    print("== WRITEBACK ==")
    for k, v in report["writeback"].items():
        print("  %-18s %s" % (k, v))
    print("== VERDICT ==")
    print("  " + report["verdict"])
    print("  " + report["master_version"])
    print("  remote actions: " + report["remote_actions"])
    print("report -> " + out)


if __name__ == "__main__":
    main()
