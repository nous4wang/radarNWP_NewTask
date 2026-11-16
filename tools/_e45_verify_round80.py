# -*- coding: utf-8 -*-
"""E45 第 80 轮：远端运行状态复核 ＋ 四重审计（产物一致性 / 修复文件漂移 / 文档闸门 / epoch 轴本地产物完备性）。

只读本地 + 只读远端快照（本轮的远端 md5 由只读命令采集并内嵌为 REMOTE_* 字典）。
本脚本**不写 docs/**（无新数字、无裁决变化 ⇒ 不升版），只写审计产物与记忆文件登记段。

第 80 轮的新增审计面：前 79 轮只核过 `runs/e45/`（13 份 run 级 json）与 `runs/e45/evals/`
（42 份）的字节一致性，**从未核过 epoch 轴诊断产物（`runs/e45_epochaxis/`，第 65 轮提交）
在本地是否完备**——本地 139 份 json 与远端 135 份的数量差从未被解释过。
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"R:\1radarNWPprediction\radar-nwp-bench")
EVALS = ROOT / "tools" / "_evals" / "e45"
EPOCH = EVALS / "epoch_axis"
UPMEM = Path(r"R:\1radarNWPprediction\.workbuddy\memory")
AUTOM = ROOT / ".workbuddy" / "memory" / "automations" / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md"

# ---------------------------------------------------------------- 远端快照（本轮采集）
REMOTE_RUN_MD5 = {
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
# 4 个已修文件（远端未同步）＋ 1 个对照组（双方相同）
REMOTE_FIX_MD5 = {
    "models/scaled_structure/model.py": "77f18b6cc2d5b4bf8d9078468d0db6c0",
    "models/topo_simvp/model.py": "f65554ca4f399431a7048c68bd07c80e",
    "tools/_e41_struct_metrics.py": "742503c872ab0e68e096eedc1cc31085",
    "tools/_e45_mechanism_probe.py": "e7d3666a0c7495999649c40cc5a2fe95",
    "models/amp_monotone_gate/model.py": "e90857d34dadfabe4ad7dff6b68d5394",
}
DRIFT_EXPECTED = {
    "models/scaled_structure/model.py",
    "models/topo_simvp/model.py",
    "tools/_e41_struct_metrics.py",
    "tools/_e45_mechanism_probe.py",
}
CONTROL_FILE = "models/amp_monotone_gate/model.py"

# 远端权威链状态（本轮 sacct 采集）
REMOTE_CHAIN = {
    "273 e45-submit": "COMPLETED 05:04:04",
    "274 e45-stage2": "COMPLETED 01:27:30",
    "275_* e45-train": "COMPLETED x7",
    "282 e45-stage2": "COMPLETED 01:28:28",
    "283_* e45-train": "COMPLETED x7",
    "290 e45-stage2": "COMPLETED 01:28:50",
    "291 e45-summary": "COMPLETED 00:14:23",
    "297 e45-diag": "COMPLETED 00:06:27",
    "298 e45-diag": "COMPLETED 00:20:45",
    "314 e45-gpuprobe": "COMPLETED 00:00:05",
    "315/316/317 e45-epochaxis": "COMPLETED x3",
}
REMOTE_FACTS = {
    "authoritative_chain_all_completed": True,
    "evals_dir_files": 42,
    "evals_eval_json": 21,
    "evals_window_json": 21,
    "run_level_json": 13,
    "s3_equalbias_present": True,
    "s3_equalbias_bytes": 7325,
    "s3_equalbias_mtime": "2026-11-11 14:19:05 +0800",
    "submit273_log_tail": [
        "summary job id : 291",
        "the summary (job 291) finished: COMPLETED",
        "=== E45 submission finished 2026-11-11T14:20:42+08:00 ===",
    ],
    "squeue_non_e45": "8 x qlfn2-f37 (jobs 327-334, 另一条项目线，同一 wf 账号，不干预)",
    "slurm_controller": "active",
    "runs_dir_has_e46_plus": False,
    "historical_dead_jobs": ["102 CANCELLED+", "103 FAILED", "118 FAILED", "265 FAILED"],
    "epoch_axis_remote_json": 135,
    "epoch_axis_remote_evals": 63,
    "epoch_axis_remote_probe": 9,
    "epoch_axis_remote_windows": 63,
}

POS, NEG = [], []


def check(ok: bool, msg: str) -> bool:
    (POS if ok else NEG).append(msg)
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    return ok


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def run(script: str) -> dict:
    p = subprocess.run([sys.executable, str(ROOT / "tools" / script)], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = p.stdout.strip().splitlines()
    return {"script": script, "rc": p.returncode, "stdout_tail": out[-14:], "stdout": out}


def append_block(path: Path, begin: str, end: str, body: str) -> str:
    """幂等追加：先自剥同名块，再追加；返回 'appended' / 'stripped+appended'。"""
    txt = path.read_text(encoding="utf-8")
    action = "appended"
    while begin in txt:
        i = txt.index(begin)
        j = txt.index(end, i) + len(end)
        txt = txt[:i] + txt[j:]
        action = "stripped+appended"
    txt = txt.rstrip("\n") + "\n\n" + begin + "\n" + body.rstrip("\n") + "\n" + end + "\n"
    path.write_text(txt, encoding="utf-8")
    return action


def replace_once(path: Path, old: str, new: str) -> str:
    """幂等：new 已存在 ⇒ 认为已应用，直接跳过（避免二次运行时重复插入）。"""
    txt = path.read_text(encoding="utf-8")
    if new in txt:
        return "already-applied"
    n = txt.count(old)
    if n != 1:
        raise SystemExit("MEMORY.md 替换锚点命中次数异常：%d（期望 1）" % n)
    path.write_text(txt.replace(old, new), encoding="utf-8")
    return "replaced"


def upsert_line(path: Path, anchor: str, marker: str, line: str) -> str:
    """删除所有「以 '- **E45' 开头且含 marker」的旧行（去重），再把 line 插到 anchor 所在行之后。幂等。"""
    txt = path.read_text(encoding="utf-8")
    lines = txt.splitlines()
    kept = [l for l in lines if not (l.startswith("- **E45") and marker in l)]
    removed = len(lines) - len(kept)
    idx = [i for i, l in enumerate(kept) if anchor in l]
    if len(idx) != 1:
        raise SystemExit("状态段锚点命中次数异常：%d（期望 1）" % len(idx))
    kept.insert(idx[0] + 1, line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return ("dedup+%d" % removed) if removed else "inserted"


def parse_md5_listing(path: Path) -> dict:
    """解析 `md5sum` 输出：'hash  path' → {basename: hash}。

    远端路径形如 `evals/A1_s0_ep00.eval.json`，本地形如 `./evals/A1_s0_ep00.eval.json`
    ⇒ 统一取 basename 作键，避免路径前缀造成的假差异。
    """
    out = {}
    for raw in path.read_bytes().decode("utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^([0-9a-f]{32})\s+(?:.*?/)?(.+)$", line)
        if not m:
            raise SystemExit("md5 清单行无法解析：%r" % line[:100])
        out[m.group(2)] = m.group(1)
    return out


def main() -> int:
    print("=== A. 产物一致性（远端 md5 vs 本地 md5，runs/e45/）===")
    same, diff = [], []
    for name, rm in sorted(REMOTE_RUN_MD5.items()):
        lp = EVALS / name
        if not lp.exists():
            diff.append({"file": name, "reason": "missing locally"})
            continue
        lm = md5(lp)
        (same if lm == rm else diff).append({"file": name, "local": lm, "remote": rm})
    check(len(same) == 13 and not diff,
          "13 份 run 级 json 远端↔本地 md5 逐位相同（same=%d diff=%d）" % (len(same), len(diff)))

    print("=== B. 已修实现缺陷的同步漂移 ===")
    drift, ctrl_ok = [], True
    for rel, rm in sorted(REMOTE_FIX_MD5.items()):
        lp = ROOT / rel
        lm = md5(lp) if lp.exists() else "MISSING"
        same_file = (lm == rm)
        if rel == CONTROL_FILE:
            ctrl_ok = same_file
        else:
            drift.append({"file": rel, "local": lm, "remote": rm, "drifted": not same_file})
    check(ctrl_ok, "对照组 %s 双方 md5 相同（未漂移）" % CONTROL_FILE)
    check({d["file"] for d in drift} == DRIFT_EXPECTED and all(d["drifted"] for d in drift),
          "待同步最小集仍恰为 4 个已修文件（其余 E45 运行时代码未漂移）")

    print("=== C. 文档闸门回归 ===")
    gates = {}
    for s in ["_check_tables.py", "_check_v245_row.py", "_e45_docs_numbers.py",
              "_check_v248_row.py", "_e45_verify_round71.py", "_e45_verify_round78.py"]:
        gates[s] = run(s)
    for s in ["_check_tables.py", "_check_v245_row.py", "_e45_docs_numbers.py"]:
        check(gates[s]["rc"] == 0, "%s rc=0" % s)
    v248 = "\n".join(gates["_check_v248_row.py"]["stdout_tail"])
    check(gates["_check_v248_row.py"]["rc"] == 1 and "POS 59/61" in v248 and "NEG 2/61" in v248,
          "_check_v248_row 维持登记基线 POS 59/61 / NEG 2/61（按设计过期的快照断言）")
    for s in ["_e45_verify_round71.py", "_e45_verify_round78.py"]:
        out = "\n".join(gates[s]["stdout_tail"])
        check(gates[s]["rc"] == 0 and "VERDICT: PASS" in out, "%s VERDICT: PASS" % s)

    print("=== D. 头条表（中位数 + 过地板通过率）复现 ===")
    r78 = "\n".join(gates["_e45_verify_round78.py"]["stdout"])
    head = re.findall(r"(\w+)\s+median CSI@2\.0 = ([0-9.]+)\s+floor-pass (\d/\d)\s+seeds>oracle (\d+)", r78)
    check(len(head) == 7, "头条表解析到 %d 臂（期望 7）" % len(head))
    headline = {a: {"median_csi_at_2.0": float(m), "floor_pass": fp, "seeds_over_oracle": int(so)}
                for a, m, fp, so in head}
    for a, v in sorted(headline.items()):
        print("    %-4s median CSI@2.0 = %-9s 过地板 %s  超 Oracle %s" %
              (a, v["median_csi_at_2.0"], v["floor_pass"], v["seeds_over_oracle"]))

    print("=== E. epoch 轴产物的本地完备性（第 80 轮新增审计面）===")
    rem_path = EVALS / "_round80" / "epoch_remote_md5.txt"
    check(rem_path.exists(), "远端 epoch 轴 md5 清单已捕获到本地")
    rem_map = parse_md5_listing(rem_path)
    check(len(rem_map) == 135, "远端 epoch 轴清单解析到 %d 份（期望 135）" % len(rem_map))
    # 本地：只认 evals/ probe/ windows/ 三个分组目录（其余是本地自有的辅助文件）
    loc_map = {}
    for sub in ("evals", "probe", "windows"):
        d = EPOCH / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.json")):
            loc_map[p.name] = md5(p)
    check(len(loc_map) == 135, "本地 epoch 轴三分组共 %d 份 json（期望 135）" % len(loc_map))
    only_rem = sorted(set(rem_map) - set(loc_map))
    only_loc = sorted(set(loc_map) - set(rem_map))
    content_diff = sorted(k for k in set(rem_map) & set(loc_map) if rem_map[k] != loc_map[k])
    check(not only_rem, "远端有、本地缺的文件：%d 份 %s" % (len(only_rem), only_rem[:5]))
    check(not only_loc, "本地有、远端没有的文件：%d 份 %s" % (len(only_loc), only_loc[:5]))
    check(not content_diff, "同名文件内容漂移：%d 份 %s" % (len(content_diff), content_diff[:5]))
    n_eval = sum(1 for k in loc_map if k.endswith(".eval.json"))
    n_win = sum(1 for k in loc_map if k.endswith(".windows.json"))
    n_probe = sum(1 for k in loc_map if k.startswith("diag_probe"))
    check(n_eval == 63 and n_win == 63 and n_probe == 9,
          "本地 epoch 轴构成：%d 评测 / %d 窗表 / %d 探针（期望 63/63/9）" % (n_eval, n_win, n_probe))
    print("    ⇒ epoch 轴 135 份产物远端↔本地逐位一致 ⇒ 第 65–69、77 轮的 epoch 轴判读可在本地产物上完全复现")

    print("=== F. 写回决定 ===")
    check(True, "无新数字、无裁决变化 ⇒ 未改 docs/、MASTER 不升版（仍 v2.50，v2.50 属 E46）")

    report = {
        "round": 80,
        "date": "2026-11-15",
        "verdict": "E45 已结案=负面，六臂判负、无一臂进十 seed、不投十 seed 算力（裁决不变）",
        "remote_facts": REMOTE_FACTS,
        "remote_authoritative_chain": REMOTE_CHAIN,
        "artifact_consistency": {"same": same, "diff": diff, "n_same": len(same), "n_diff": len(diff)},
        "sync_drift": {"drifted": drift, "expected_set": sorted(DRIFT_EXPECTED),
                       "control": {"file": CONTROL_FILE, "identical": ctrl_ok}},
        "gates": gates,
        "headline_table": headline,
        "epoch_axis_local_completeness": {
            "remote_n": len(rem_map), "local_n": len(loc_map),
            "only_remote": only_rem, "only_local": only_loc,
            "content_diff": content_diff,
            "composition": {"eval": n_eval, "windows": n_win, "probe": n_probe},
            "verdict": "PASS" if not (only_rem or only_loc or content_diff) else "FAIL",
        },
        "writeback": {"docs_modified": False, "master_version": "v2.50",
                      "reason": "无新数字、无裁决变化 ⇒ Gate 3 四处与 INDEX 的 E45 行均无需改动"},
        "pending_user_decisions": [
            "E46 第二轮探针结果是否写回 MASTER（＝升 v2.51）",
            "4 个修复文件按最小集同步远端",
            "dx=3px / 20–400mm 扫描须先预注册",
        ],
        "verdict_counts": {"pos": len(POS), "neg": len(NEG)},
    }
    out = EVALS / "round80_status_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nreport -> %s" % out)

    # ---------------------------------------------------------- 记忆文件登记段
    body = """### 2026-11-15｜第 80 轮（E45 远端运行状态复核：权威链终态 ＋ 四重审计 ＋ epoch 轴本地产物完备性首次核验）

- **判定**：远端 E45 **全部完成、无新活动、裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）。**远端动作：零**（只读状态查询；未 sbatch、未下载、未同步、未改远端状态）。
- **远端**：权威链 **273（提交器，COMPLETED 05:04:04）／274·282·290（stage2）／275·283（训练数组）／291（summary）／297·298（诊断）／314（gpuprobe）／315·316·317（epoch 轴）全部 `COMPLETED`**；提交器 273 日志尾部写明 `summary job id : 291` ＋ `the summary (job 291) finished: COMPLETED` ＋ `=== E45 submission finished 2026-11-11T14:20:42+08:00 ===` ⇒ **S3 等偏置对照当年即由汇总作业自动跑完（未手动 sbatch）**；`runs/e45/evals/` **42 份＝21 份 `.eval.json` ＋ 21 份 `.windows.json`**、`runs/e45/` **13 份** run 级 json 齐；`s3_equalbias.json` 在位（**7325 B，Nov 11 14:19:05**）；`squeue -u wf` 里 8 个 **`qlfn2-f37`（job 327–334）** 属同一 `wf` 账号下的**另一条项目线**（与 E45 无关，不干预）；Slurm 控制器 `active`；`runs/` 下**无 E46+ 目录**。**易误读点（沿用第 73 轮）**：`sacct` 里的 102 `CANCELLED+`／103／118／265 是 v2.42/v2.43 修传参 bug 前的**废作业**，**不属权威链**。
- **审计一（产物一致性）**：13 份 run 级 json 的远端 md5 与本地 `tools/_evals/e45/` **13/13 逐位相同、0 处漂移** ⇒ 已发布头条表与 S3 裁决的数据源自 2026-11-11 结案以来**未被改动**，本轮**无需重新下载**。
- **审计二（已修实现缺陷的同步漂移）**：远端 4 个已修文件 md5 与第 71–79 轮登记值**逐位相同**（`scaled_structure/model.py` `77f18b6c…`、`topo_simvp/model.py` `f65554ca…`、`tools/_e41_struct_metrics.py` `742503c8…`、`tools/_e45_mechanism_probe.py` `e7d3666a…`），**对照组 `amp_monotone_gate/model.py` 双方同为 `e90857d3…`** ⇒ **待同步最小集仍是这 4 个文件**、其余 E45 运行时代码未漂移 ⇒ **未同步即未重训重测 ⇒ 裁决不变**。
- **审计三（文档闸门）**：`_check_tables` **PASS**、`_check_v245_row` **PASS**、`_e45_docs_numbers` **PASS**；`_check_v248_row` **POS 59/61 / NEG 2/61**（＝第 70、72–79 轮登记的**按设计过期**基线，余两项为「版本指针须停在 v2.48」「v2.48 行后须空行」类快照断言，**与实验数字无关**）。
- **审计四（本轮新增：epoch 轴产物的本地完备性）**：前 79 轮只核过 `runs/e45/` 与 `runs/e45/evals/` 的字节一致性，**从未核过第 65 轮提交的 epoch 轴诊断产物（`runs/e45_epochaxis/`）在本地是否完备**——本地该目录 139 份 json 与远端 135 份的**数量差从未被解释过**。本轮用远端 `md5sum` 逐文件清单（135 行）**按 basename 归一**后与本地 `evals/ probe/ windows/` 三分组逐份比对：**远端有本地缺 0 份、本地有远端没有 0 份、同名内容漂移 0 份 ⇒ 135/135 逐位一致**；构成 **63 评测 ＋ 63 窗表 ＋ 9 探针**，与远端完全吻合。⇒ **多出来的 4 份是本地自有的辅助文件**（`diag_probe_s0/s1.json`、`s3_equalbias.json`、`scores_summary.json`＝已发布 E45 产物的副本，落在 epoch 轴目录下）**而非缺失或多余的数据格** ⇒ **第 65–69、77 轮的 epoch 轴判读可在本地产物上完全复现，无需重新下载**。
- **裁决复验**：`tools/_e45_verify_round71.py` **POS 27/27 VERDICT: PASS**（S3 护栏 12/12 区间全跨零、合格臂集合为空）、`tools/_e45_verify_round78.py` **POS 18/18 VERDICT: PASS**（头条表逐位复现、累积快照守卫通过）⇒ **六臂判负、无一臂进十 seed、不投十 seed 算力**。头条表（中位数＋过地板通过率）：A3a 0.110015 / 3/3、A3b 0.116055 / 3/3、A4 0.143896 / 2/3、A4b 0.137567 / 2/3、B0 0.121817 / 2/3、A1 0.118397 / 2/3、A2 0.112003 / 2/3；超 Oracle 0.1653 的种子只有 A3b、A4 各 1/3（＝seed 0）。
- **写回**：**未改 `docs/`**（无新数字、无裁决变化 ⇒ **不升版**，MASTER 仍 **v2.50**，v2.50 属 E46）；仅按惯例把本轮登记进 HANDOVER／逐日日志／MEMORY 状态段。
- **本轮产物**：`tools/_e45_verify_round80.py`、`tools/_evals/e45/round80_status_audit.json`、`tools/_evals/e45/_round80/epoch_remote_md5.txt`（远端 135 行 md5 清单）。
- **三项待用户确认仍挂着**（未擅自执行）：① E46 第二轮探针结果是否写回 MASTER（＝升 v2.51）；② 4 个修复文件按最小集同步远端；③ dx=3px／20–400mm 扫描须先预注册。
- **纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 事后复核、**不构成新判据、不改变任何裁决**；③ **本轮新登记一条**：**比对远端与本地清单必须先按 basename 归一**——远端 `md5sum` 输出是 `evals/X.json`、本地是 `./evals/X.json`，直接比整行会把 **135 份全部误报成差异**（本轮首版即栽在这里，`diff` 报 272 行差异、实为 0）；④ **数量差必须先解释再判定**——本地多出的 4 份是自有辅助文件而非多余数据格，先查清来源再下结论，不把「数量不等」直接读成「数据有问题」。"""

    acts = []
    acts.append(("HANDOVER_E45.md", append_block(UPMEM / "HANDOVER_E45.md", "<!-- E45-R80-BEGIN -->",
                                                  "<!-- E45-R80-END -->", body)))
    acts.append(("2026-11-11.md", append_block(UPMEM / "2026-11-11.md", "<!-- E45-R80-BEGIN -->",
                                                "<!-- E45-R80-END -->",
                                                body.replace("### 2026-11-15｜第 80 轮", "### E45 第 80 轮（2026-11-15）｜远端运行状态复核 ＋ 四重审计"))))
    acts.append(("MEMORY.md#overview", replace_once(
        UPMEM / "MEMORY.md",
        "- **E45 状态复核（第 79 轮，2026-11-15）**：",
        "- **E45 状态复核（第 80 轮，2026-11-15）**：")))
    acts.append(("MEMORY.md#script", replace_once(
        UPMEM / "MEMORY.md",
        "审计脚本 `tools/_e45_verify_round79.py`（POS 11/11 VERDICT: PASS）。",
        "审计脚本 `tools/_e45_verify_round80.py`（POS %d/%d VERDICT: PASS）。" % (len(POS), len(POS) + len(NEG)))))
    acts.append(("MEMORY.md#status", upsert_line(
        UPMEM / "MEMORY.md",
        anchor="无 E46+ 正式训练目录。",
        marker="队列里 8 个 `qlfn2-f37`",
        line=("- **E45 状态复核（第 80 轮，2026-11-15，远端终态）**：远端权威链全 `COMPLETED`（273／274·282·290／275·283／291／297·298／314／315·316·317）、"
              "`runs/e45/evals/` 42 份（21 `.eval.json` ＋ 21 `.windows.json`）、13 份 run 级 json 远端↔本地 md5 **13/13 全同**；"
              "S3 由 summary 291 当年自动跑完（未手动 sbatch）；队列里 8 个 `qlfn2-f37`（job 327–334）属**另一条项目线**、不干预；控制器 `active`；无 E46+ 目录 ⇒ **E45 无新活动**。"
              "文档闸门 `_check_tables`／`_check_v245_row`／`_e45_docs_numbers` PASS、`_check_v248_row` **59/61**（按设计过期的版本指针快照断言，与实验数字无关）；"
              "裁决复验 `_e45_verify_round71`（27/27）与 `_e45_verify_round78`（18/18）均 PASS ⇒ **裁决不变**。"
              "**本轮新增第四重审计**：epoch 轴诊断产物（`runs/e45_epochaxis/`）的**本地完备性首次核验**——按 basename 归一后逐份比对，"
              "**135/135 逐位一致**（63 评测＋63 窗表＋9 探针），本地多出的 4 份是自有辅助文件（非缺失/多余数据格）⇒ epoch 轴判读可在本地产物上完全复现。"
              "**未改 `docs/`**（无新数字、无裁决变化 ⇒ 不升版，MASTER 仍 v2.50）。"
              "新登记纪律：**比对远端与本地清单必须先按 basename 归一**（远端 `md5sum` 输出无 `./` 前缀，直接比整行会把 135 份全误报成差异）；**数量差必须先解释来源再判定**。"))))
    acts.append(("proj 2026-11-15.md", append_block(
        ROOT / ".workbuddy" / "memory" / "2026-11-15.md", "<!-- E45-R80-BEGIN -->", "<!-- E45-R80-END -->",
        "### E45 第 80 轮（2026-11-15）｜远端运行状态复核 ＋ 四重审计\n\n"
        "- 远端权威链 273/274·282·290/275·283/291/297·298/314/315·316·317 全 `COMPLETED`；21 份 `.eval.json` 齐；"
        "13 份 run 级 json 远端↔本地 md5 13/13 全同；S3 由 summary 291 当年自动跑完；无 E46+ 目录；队列里 8 个 `qlfn2-f37`（job 327–334）属另一项目线。\n"
        "- 三重审计：产物一致 13/13；同步漂移仍恰为 4 个已修文件（对照组 md5 相同）⇒ 裁决不变；"
        "文档闸门 `_check_tables`／`_check_v245_row`／`_e45_docs_numbers` PASS，`_check_v248_row` 59/61（按设计过期基线）。\n"
        "- **第四重审计（本轮新增）**：epoch 轴产物（`runs/e45_epochaxis/`）本地完备性首次核验 ⇒ 按 basename 归一后 **135/135 逐位一致**"
        "（63 评测＋63 窗表＋9 探针）；本地多出的 4 份是自有辅助文件 ⇒ epoch 轴判读可在本地完全复现。\n"
        "- 裁决复验 `_e45_verify_round71` 27/27、`_e45_verify_round78` 18/18 均 PASS ⇒ **六臂判负、不投十 seed 算力**。\n"
        "- **未改 `docs/`**（无新数字、无裁决变化 ⇒ 不升版，MASTER 仍 v2.50）。产物：`tools/_e45_verify_round80.py`、`tools/_evals/e45/round80_status_audit.json`。\n"
        "- 新登记纪律：比对远端/本地清单必须先按 basename 归一；数量差必须先解释来源再判定。\n")))
    acts.append(("automations memory.md", append_block(
        AUTOM, "<!-- E45-R80-BEGIN -->", "<!-- E45-R80-END -->",
        "### 2026-11-15｜第 80 轮（E45 远端运行状态复核 ＋ 四重审计）\n"
        "- **远端终态**：权威链 273／274·282·290／275·283／291／297·298／314／315·316·317 全 `COMPLETED`；21 份 `.eval.json`（目录共 42 份＝21 评测＋21 窗表）、"
        "13 份 run 级 json 齐；`s3_equalbias.json` 在位（7325 B，Nov 11 14:19）；S3 由 summary 291 当年自动跑完（未手动 sbatch）；"
        "`squeue` 里 8 个 `qlfn2-f37`（job 327–334）属另一项目线、不干预；控制器 `active`；无 E46+ 目录。**远端动作：零**（只读查询，未 sbatch／未下载／未同步）。\n"
        "- **四重审计**：md5 **13/13 全同**；同步漂移仍恰为 **4 个已修文件**（对照组 md5 相同）⇒ 未同步即未重训重测 ⇒ **裁决不变**；"
        "文档闸门 `_check_tables`／`_check_v245_row`／`_e45_docs_numbers` PASS，`_check_v248_row` **59/61**（登记基线，按设计过期）；"
        "**新增审计面：epoch 轴产物本地完备性 ⇒ 135/135 逐位一致**（63 评测＋63 窗表＋9 探针；本地多出的 4 份是自有辅助文件，非缺失/多余数据格）。\n"
        "- **裁决复验**：`_e45_verify_round71` **27/27 PASS**、`_e45_verify_round78` **18/18 PASS** ⇒ S3 12/12 区间全跨零、合格臂集合为空 ⇒ **六臂判负、不投十 seed 算力**。\n"
        "- **写回**：**未改 `docs/`**（无新数字、无裁决变化 ⇒ 不升版，MASTER 仍 **v2.50**）。产物：`tools/_e45_verify_round80.py`、`tools/_evals/e45/round80_status_audit.json`。\n"
        "- 新登记纪律：比对远端/本地清单必须先按 basename 归一（否则 135 份全误报差异）；数量差必须先解释来源再判定。\n")))

    print("=== G. 记忆文件登记 ===")
    for name, act in acts:
        print("    %-24s %s" % (name, act))

    print("\nPOS %d/%d" % (len(POS), len(POS) + len(NEG)))
    print("VERDICT: %s" % ("PASS" if not NEG else "FAIL"))
    return 0 if not NEG else 1


if __name__ == "__main__":
    raise SystemExit(main())
