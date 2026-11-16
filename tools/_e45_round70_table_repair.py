"""E45 round 70 (part 2): generic, line-accurate table-row repair.

Rows are repaired by LINE NUMBER (not by guessing prose strings), and the
repair rule is mechanical:

  * inside an inline code span, a raw ``|`` splits the table row, so it is
    escaped as ``\\|`` (GFM renders ``\\|`` inside a code span as a literal
    ``|``, so the rendered text does not change);
  * already-escaped ``\\|`` sequences are left alone;
  * a row that still has more cells than its header after the code-span
    repair has its surplus prose pipes escaped too (row 11 of MASTER.md).

Every step asserts: the script errors out instead of silently skipping.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(r"R:\1radarNWPprediction\radar-nwp-bench")
MASTER = ROOT / "docs" / "MASTER.md"
E45DOC = ROOT / "docs" / "experiments" / "E45-gate-direction-amount-separation-baseline.md"
ARTEFACT = ROOT / "tools" / "_evals" / "e45" / "round70_status_gate_repair.json"

ESC = re.compile(r"\\\|")
CODE_SPAN = re.compile(r"`[^`]*`")
SENTINEL = "\u0000"


def cells(line: str) -> list[str]:
    s = ESC.sub(SENTINEL, line)
    parts = s.split("|")
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return parts


def escape_code_spans(line: str) -> str:
    """Escape raw pipes inside inline code spans; leave existing ``\\|`` alone."""
    def fix(m: re.Match) -> str:
        body = m.group(0)
        body = ESC.sub(SENTINEL, body)          # protect existing escapes
        body = body.replace("|", "\\|")          # escape the raw ones
        return body.replace(SENTINEL, "\\|")
    return CODE_SPAN.sub(fix, line)


def header_cells(lines: list[str], idx0: int) -> int:
    """Cell count of the header governing row ``idx0`` (0-based)."""
    for j in range(idx0 - 1, max(idx0 - 400, -1), -1):
        if re.fullmatch(r"\|[\s:|-]+\|", lines[j].strip()):
            return len(cells(lines[j]))
    raise AssertionError(f"no table separator found above row {idx0 + 1}")


def repair(path: Path, line_no: int) -> dict:
    lines = path.read_text(encoding="utf-8").split("\n")
    idx = line_no - 1
    row = lines[idx]
    want = header_cells(lines, idx)
    before = len(cells(row))
    new = escape_code_spans(row)
    if len(cells(new)) != want:
        # surplus prose pipes: escape them all but the structural ones
        keep = want
        out, seen, buf = [], 0, []
        parts = new.split("|")
        if parts and not parts[0].strip():
            parts = parts[1:]
        if parts and not parts[-1].strip():
            parts = parts[:-1]
        merged: list[str] = []
        for p in parts:
            if seen < keep:
                merged.append(p)
                seen += 1
            else:
                merged[-1] = merged[-1] + " \\| " + p
        new = "| " + " | ".join(merged) + " |"
    got = len(cells(new))
    assert got == want, f"{path.name}:{line_no} still {got} cells (want {want})"
    lines[idx] = new
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"file": path.name, "line": line_no, "cells_before": before,
            "cells_after": got, "header_cells": want}


RESULTS = [
    repair(MASTER, 11),
    repair(MASTER, 2026),
    repair(MASTER, 2032),
    repair(E45DOC, 355),
]

# ---- verify: no row in either file may exceed its header's cell count ----
def audit(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").split("\n")
    bad, header = [], None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("|"):
            header = None
            continue
        n = len(cells(line))
        if header is None:
            header = n
        elif n != header:
            bad.append({"line": i + 1, "cells": n, "header": header})
    return bad

AUDIT = {p.name: audit(p) for p in (MASTER, E45DOC)}
for name, bad in AUDIT.items():
    assert not bad, f"{name} still has malformed rows: {bad}"

ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
if ARTEFACT.exists():
    art = json.loads(ARTEFACT.read_text(encoding="utf-8"))
else:
    art = {}
art["table_repairs"] = RESULTS
art["table_audit_after"] = AUDIT
ARTEFACT.write_text(json.dumps(art, ensure_ascii=False, indent=2), encoding="utf-8")

for r in RESULTS:
    print(f"  {r['file']}:{r['line']}  {r['cells_before']} -> {r['cells_after']} "
          f"(header {r['header_cells']})")
print("audit after repair: no malformed rows in MASTER.md / E45 doc")
print("artefact:", ARTEFACT)
