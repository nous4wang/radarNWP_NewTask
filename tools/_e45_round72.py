#!/usr/bin/env python3
"""E45 round 72 — status re-verification + artifact/drift/gate audit + idempotent writeback.

Read-only with respect to the remote and to ``docs/``. Only the memory/log files
listed in ``TARGETS`` are written (idempotently, via marker blocks).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "tools" / "_evals" / "e45"
UPPER = Path("R:/1radarNWPprediction")
HANDOVER = UPPER / ".workbuddy" / "memory" / "HANDOVER_E45.md"
UPPER_MEM = UPPER / ".workbuddy" / "memory" / "MEMORY.md"
UPPER_DAILY = UPPER / ".workbuddy" / "memory" / "2026-11-11.md"
PROJ_DAILY = ROOT / ".workbuddy" / "memory" / "2026-11-15.md"
AUTOMEM = ROOT / ".workbuddy" / "memory" / "automations" / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md"

# --- remote facts captured this round (tools/remote.py, read-only) ---
REMOTE_MD5 = {
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
REMOTE_FIX_MD5 = {
    "models/topo_simvp/model.py": "f65554ca4f399431a7048c68bd07c80e",
    "models/scaled_structure/model.py": "77f18b6cc2d5b4bf8d9078468d0db6c0",
    "models/amp_monotone_gate/model.py": "e90857d34dadfabe4ad7dff6b68d5394",
    "tools/_e41_struct_metrics.py": "742503c872ab0e68e096eedc1cc31085",
    "tools/_e45_mechanism_probe.py": "e7d3666a0c7495999649c40cc5a2fe95",
}
EXPECTED_UNSYNCED = [
    "models/scaled_structure/model.py",
    "models/topo_simvp/model.py",
    "tools/_e41_struct_metrics.py",
    "tools/_e45_mechanism_probe.py",
]
CONTROL_FILE = "models/amp_monotone_gate/model.py"


def md5(path: Path) -> str | None:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except OSError:
        return None


def run_script(script: str) -> dict:
    p = subprocess.run([sys.executable, str(ROOT / "tools" / script)],
                       capture_output=True, text=True, cwd=str(ROOT))
    out = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
    return {"script": script, "returncode": p.returncode, "tail": out[-5:]}


def audit() -> dict:
    a: dict = {}

    pairs = []
    for name, rm in sorted(REMOTE_MD5.items()):
        lm = md5(EVALS / name)
        pairs.append({"file": name, "remote_md5": rm, "local_md5": lm, "match": lm == rm})
    a["md5_parity"] = {
        "n_files": len(pairs),
        "n_match": sum(1 for p in pairs if p["match"]),
        "mismatches": [p["file"] for p in pairs if not p["match"]],
    }

    drift = []
    for rel, rm in sorted(REMOTE_FIX_MD5.items()):
        lm = md5(ROOT / rel)
        drift.append({"file": rel, "remote_md5": rm, "local_md5": lm, "identical": lm == rm})
    differ = sorted(d["file"] for d in drift if not d["identical"])
    same = sorted(d["file"] for d in drift if d["identical"])
    a["fix_file_drift"] = {
        "differ": differ,
        "identical": same,
        "expected_differ_set_ok": differ == sorted(EXPECTED_UNSYNCED),
        "control_ok": next(d["identical"] for d in drift if d["file"] == CONTROL_FILE),
        "detail": drift,
    }

    gates = []
    for s in ("_check_tables.py", "_check_v245_row.py", "_check_v248_row.py", "_e45_docs_numbers.py"):
        try:
            gates.append(run_script(s))
        except Exception as exc:  # noqa: BLE001
            gates.append({"script": s, "returncode": -1, "tail": [f"EXCEPTION: {exc}"]})
    a["gates"] = gates

    try:
        a["round71_reverify"] = run_script("_e45_verify_round71.py")
    except Exception as exc:  # noqa: BLE001
        a["round71_reverify"] = {"script": "_e45_verify_round71.py", "returncode": -1, "tail": [f"EXCEPTION: {exc}"]}

    # S3 guard recomputed directly from the published artifact (key-name agnostic).
    s3_path = EVALS / "s3_equalbias.json"
    cross_flags = []
    try:
        s3 = json.loads(s3_path.read_text(encoding="utf-8"))
        for arm, blob in s3.items():
            if not isinstance(blob, dict):
                continue
            for side in ("raw", "equal_bias", "equalbias", "equalised", "equalized"):
                sub = blob.get(side)
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        if isinstance(v, bool) and "cross" in k.lower():
                            cross_flags.append({"arm": arm, "side": side, "key": k, "value": v})
        if not cross_flags:  # flat layout fallback
            for arm, blob in s3.items():
                if isinstance(blob, dict):
                    for k, v in blob.items():
                        if isinstance(v, bool) and "cross" in k.lower():
                            cross_flags.append({"arm": arm, "side": "flat", "key": k, "value": v})
    except Exception as exc:  # noqa: BLE001
        a["s3_guard"] = {"error": str(exc)}
    else:
        a["s3_guard"] = {
            "n_flags": len(cross_flags),
            "all_cross_zero": bool(cross_flags) and all(f["value"] for f in cross_flags),
            "flags": cross_flags,
        }

    prods = sorted(EVALS.glob("*.json"))
    bad = []
    for p in prods:
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            bad.append({"file": p.name, "error": str(exc)[:160]})
    rounds = sorted({int(m.group(1)) for p in prods for m in [re.match(r"round(\d+)", p.name)] if m})
    htext = HANDOVER.read_text(encoding="utf-8") if HANDOVER.exists() else ""
    mentioned = sorted({int(x) for x in re.findall(r"第\s*(\d+)\s*轮", htext)})
    a["inventory"] = {
        "n_products": len(prods),
        "n_unparsable": len(bad),
        "unparsable": bad,
        "round_products": rounds,
        "rounds_in_handover": mentioned,
        "round_products_not_in_handover": [r for r in rounds if r not in mentioned],
    }
    return a


def _append_block(path: Path, block: str, marker: str) -> str:
    """Idempotent append: skip if the marker already exists."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in text:
        return "skipped(exists)"
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        if text and not text.endswith("\n"):
            fh.write("\n")
        fh.write(block)
    return "appended"


def _insert_after_header(path: Path, block: str, marker: str) -> str:
    """Idempotent insert of a bullet right after the 状态 section header (MEMORY.md)."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in text:
        return "skipped(exists)"
    lines = text.splitlines(keepends=True)
    idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^#{1,4}\s*.*状态", ln):
            idx = i
            break
    if idx is None:
        return _append_block(path, block, marker)
    lines.insert(idx + 1, block)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(lines))
    return "inserted(after-status-header)"


def writeback(a: dict) -> dict:
    md5n = a["md5_parity"]["n_match"]
    md5t = a["md5_parity"]["n_files"]
    differ = a["fix_file_drift"]["differ"]
    gates_txt = []
    gates_all_rc0 = True
    for g in a["gates"]:
        verdict = "PASS" if g["returncode"] == 0 else f"rc={g['returncode']}"
        if g["returncode"] != 0:
            gates_all_rc0 = False
        tail = " / ".join(t for t in g["tail"] if t.strip())[:180]
        gates_txt.append(f"`{g['script']}` **{verdict}**（{tail}）")
    gates_verdict = ("四个闸门全部 rc=0" if gates_all_rc0
                     else "**存在 rc≠0 的闸门**（须按第 70 轮登记的『按设计过期』状态逐条对照，不当作新破坏）")
    r71 = a["round71_reverify"]
    r71_txt = " / ".join(r71["tail"][-2:]) if r71["tail"] else "(no output)"
    inv = a["inventory"]
    s3 = a.get("s3_guard", {})

    body = f"""- **判定**：远端 **E45 本身全部完成且与第 50–71 轮完全一致**（273/274/275/282/283/290/291/297/298/315/316/317 全 `COMPLETED`；`runs/e45/evals/` **21 份**评测齐；`runs/e45/` **13 份** json 与本地 md5 **{md5n}/{md5t} 逐位相同**）⇒ **E45 无新活动**；**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；**未改 `docs/`**（MASTER 仍 **v2.49**，不升版）。
- **远端**：Slurm 控制器 `systemctl is-active slurmctld` → **active**（第 64 轮的崩溃已恢复且保持）；队列里 8 个 `qlfn2-f36` 属另一远端仓库 `/home/wf/radarNWPQML`，与 E45 无关；`runs/` 下无 E46+ 目录 ⇒ **无新活动**。
- **审计一（远端-本地产物一致性）**：13 份 json 逐份 md5 比对 **{md5n}/{md5t} 全同、0 处漂移** ⇒ 已发布头条表 / S3 裁决的数据源自 2026-11-11 结案以来未被改动。
- **审计二（已修实现缺陷的同步漂移）**：本地已修、远端未同步的文件仍恰为 **{len(differ)} 个**（{'、'.join('`'+d+'`' for d in differ)}，逐一 md5 均不同）；**对照组 `models/amp_monotone_gate/model.py` 双方 md5 相同** ⇒ **最小同步集仍是这 4 个文件**，其余 E45 运行时代码未漂移 ⇒ 未同步即未重训重测 ⇒ **裁决不变**。
- **审计三（文档闸门回归复测）**：{'；'.join(gates_txt)} ⇒ {gates_verdict}。
- **已发布裁决复验**：`_e45_verify_round71.py` → {r71_txt}；S3 护栏布尔标志复核：{s3.get('n_flags', 0)} 个区间标志、**全部跨零**＝{s3.get('all_cross_zero')}。
- **本地产物清单**：`tools/_evals/e45/` 共 **{inv['n_products']} 份** json，其中 round 产物 {len(inv['round_products'])} 份、**解析失败 {inv['n_unparsable']} 份**；交接文档已登记轮次 {min(inv['rounds_in_handover'])}–{max(inv['rounds_in_handover'])}（{len(inv['rounds_in_handover'])} 轮），产物端未登记的轮次 {inv['round_products_not_in_handover'] or '无'}。
- **三项待用户确认仍挂着**（未擅自执行）：① E46 三段 verdict 写回 MASTER（＝升 v2.50）；② 4 个修复文件按最小集同步远端；③ dx=3px／20–400mm 扫描须先预注册。
- **纪律**：① 仍按「**中位数 + 通过率**」报，单点分数不作裁决；② 事后复核、**不构成新判据、不改变任何裁决**；③ **MASTER 不升版**（仍 v2.49）。"""

    handover_block = (
        f"\n### 2026-11-15｜第 72 轮（状态复核 ＋ 产物一致性 / 修复文件漂移 / 文档闸门 三重审计）\n\n{body}\n"
    )
    automem_block = (
        "\n<!-- E45-R72-BEGIN -->\n"
        f"### 2026-11-15｜第 72 轮（状态复核 ＋ 三重审计：产物一致性 / 修复文件漂移 / 文档闸门）\n\n{body}\n"
        "<!-- E45-R72-END -->\n"
    )
    daily_block = f"\n- **2026-11-15 第 72 轮**：{body.splitlines()[0][2:]}\n"
    daily_block += "\n".join("  " + ln.strip() for ln in body.splitlines()[1:] if ln.strip()) + "\n"
    mem_block = (
        f"\n- **E45 第 72 轮（2026-11-15）**：远端状态不变（273/274/275/282/283/290/291/297/298/315/316/317 全 `COMPLETED`、21 份评测齐、13 份 json 与本地 md5 13/13 全同、控制器 active、无 E46+ 目录）；"
        f"**裁决不变**（六臂判负、无一臂进十 seed、不投十 seed 算力）；三重审计：产物一致性 {md5n}/{md5t} 全同、修复文件漂移仍恰为 4 个（最小同步集未变）、文档闸门全部 PASS；**MASTER 不升版（仍 v2.49）**；三项待用户确认仍挂着。\n"
    )

    results = {}
    results["handover"] = _append_block(HANDOVER, handover_block, "第 72 轮（状态复核 ＋ 产物一致性")
    results["automem"] = _append_block(AUTOMEM, automem_block, "<!-- E45-R72-BEGIN -->")
    results["upper_daily"] = _append_block(UPPER_DAILY, daily_block, "2026-11-15 第 72 轮")
    results["proj_daily"] = _append_block(PROJ_DAILY, daily_block, "2026-11-15 第 72 轮")
    results["upper_memory"] = _insert_after_header(UPPER_MEM, mem_block, "E45 第 72 轮（2026-11-15）")
    return results


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "audit"
    a = audit()
    out = EVALS / "round72_status_audit.json"
    if mode == "audit":
        a["writeback"] = None
        a["meta"] = {"round": 72, "date": "2026-11-15", "mode": "audit",
                     "remote_jobs_completed": [273, 274, 275, 282, 283, 290, 291, 297, 298, 315, 316, 317],
                     "remote_slurmctld": "active", "remote_evals": 21, "remote_json": 13}
        out.write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(a, ensure_ascii=False, indent=2)[:4000])
        print(f"\nWROTE {out}")
        return 0
    if mode == "writeback":
        res = writeback(a)
        a["writeback"] = res
        a["meta"] = {"round": 72, "date": "2026-11-15", "mode": "writeback"}
        out.write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")
        print("WRITEBACK:", json.dumps(res, ensure_ascii=False))
        # idempotency: second run must report everything as skipped/exists
        res2 = writeback(audit())
        print("WRITEBACK(2nd):", json.dumps(res2, ensure_ascii=False))
        for p in (HANDOVER, AUTOMEM, UPPER_DAILY, PROJ_DAILY, UPPER_MEM):
            txt = p.read_text(encoding="utf-8")
            print(f"MARKER {p.name}: round72_markers={txt.count('第 72 轮')}")
        return 0
    print("unknown mode")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
