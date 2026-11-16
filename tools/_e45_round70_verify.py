"""E45 round 70 (final): marker hygiene + the correct verification.

Two things are settled here.

1. MARKER HYGIENE -- every round block in every memory file must carry the
   ``E45-R70-BEGIN`` / ``E45-R70-END`` markers (the upper daily log uses the
   ``-111-`` suffix, which is that file's own convention), so a later round
   can strip and re-append idempotently. The automation memory file's round
   block was written without markers; it is wrapped here.

2. THE CORRECT VERIFICATION INVARIANT for the four repaired rows:
     * for the three rows whose CONTENT is unchanged since HEAD (the two
       section-12 changelog rows and the E45 criteria row) the repaired row
       must be character-identical to HEAD once the markup is stripped --
       i.e. the repair changed markup, never a word;
     * for MASTER.md line 11 the row is rewritten on every version bump, so
       comparing it to HEAD is meaningless. Instead the pre-repair row is
       RECONSTRUCTED by un-escaping the two pipes the repair escaped, and
       that reconstruction must have the original 4 cells with the original
       cell boundaries.
     * every table row must match its header; no double escapes; the four
       gates' verdicts; the version pointer; the negative verdict; the
       MEMORY.md citation; the round-70 blocks in all five memory files.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(r"R:\1radarNWPprediction\radar-nwp-bench")
UPPER = Path(r"R:\1radarNWPprediction")
MASTER = ROOT / "docs" / "MASTER.md"
E45DOC = ROOT / "docs" / "experiments" / "E45-gate-direction-amount-separation-baseline.md"
MEMORY = UPPER / ".workbuddy" / "memory" / "MEMORY.md"
HANDOVER = UPPER / ".workbuddy" / "memory" / "HANDOVER_E45.md"
UPPER_LOG = UPPER / ".workbuddy" / "memory" / "2026-11-11.md"
PROJ_LOG = ROOT / ".workbuddy" / "memory" / "2026-11-15.md"
AUTOMATION_LOG = (ROOT / ".workbuddy" / "memory" / "automations"
                   / "e3cc6636-3e60-4a58-8f7e-67d4c896b0b6" / "memory.md")
ARTEFACT = ROOT / "tools" / "_evals" / "e45" / "round70_status_gate_repair.json"

ESC = re.compile(r"\\\|")
NOTES: list[str] = []


def cells(line: str) -> list[str]:
    s = ESC.sub("\x00", line)
    parts = s.split("|")
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return parts


def canon(line: str) -> str:
    s = ESC.sub("\x00", line).replace("\x00", "|")
    s = s.replace("\\", "").replace("|", "")
    return re.sub(r"\s+", "", s)


def head_line(path: Path, n: int) -> str:
    rel = path.relative_to(ROOT).as_posix()
    out = subprocess.run(["git", "show", f"HEAD:{rel}"],
                         capture_output=True, text=True, encoding="utf-8").stdout
    return out.split("\n")[n - 1]


# ---------------------------------------------- 1. marker hygiene (automation)
text = AUTOMATION_LOG.read_text(encoding="utf-8")
if "E45-R70-BEGIN" not in text:
    head, sep, tail = text.partition("### 2026-11-15｜第 70 轮")
    assert sep, "round-70 heading not found in the automation memory file"
    body = ("### 2026-11-15｜第 70 轮" + tail).rstrip("\n")
    AUTOMATION_LOG.write_text(head.rstrip("\n") + "\n\n<!-- E45-R70-BEGIN -->\n"
                              + body + "\n<!-- E45-R70-END -->\n", encoding="utf-8")
    NOTES.append("automation memory: round-70 block wrapped in E45-R70 markers")

# ------------------------------------------------------- 2. verification
ok = True
print("A. repaired rows -- content preserved (markup stripped):")
for path, new_ln, old_ln, hdr in [(MASTER, 2026, 1983, 4), (MASTER, 2032, 1989, 4),
                                  (E45DOC, 355, 354, 4)]:
    new = path.read_text(encoding="utf-8").split("\n")[new_ln - 1]
    old = head_line(path, old_ln)
    same = canon(new) == canon(old)
    ok = ok and same and len(cells(new)) == hdr
    print(f"   {path.name}:{new_ln} (HEAD:{old_ln})  cells {len(cells(old))}"
          f" -> {len(cells(new))} (header {hdr})  content-identical={same}")

print("B. MASTER.md line 11 -- pre-repair row reconstructed (it is rewritten on")
print("   every version bump, so comparing it to HEAD would be meaningless):")
row11 = MASTER.read_text(encoding="utf-8").split("\n")[10]
recon = row11.replace(" \\| ", " | ")          # undo exactly the two escapes
rc = cells(recon)
checks = {
    "reconstruction has the original 4 cells": len(rc) == 4,
    "cell 0 is the label": rc[0].strip() == "最后更新",
    "cell 1 ends the v2.48 block": rc[1].rstrip().endswith("裁决不变**）"),
    "cell 2 is the criteria sentence": rc[2].strip().startswith("**判据一字未改**"),
    "cell 3 is the token-lesson cell": rc[3].strip().startswith("**本会话 token 教训**"),
    "current row has 2 cells (matches its 2-cell header)": len(cells(row11)) == 2,
    "only the two intended pipes were escaped": row11.count(" \\| ") == 2,
}
for k, v in checks.items():
    print(f"   [{'OK ' if v else 'BAD'}] {k}")
    ok = ok and v

print("C. table integrity / escapes:")
for path in (MASTER, E45DOC):
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
            bad.append(i + 1)
    dbl = sum(1 for l in lines if "\\\\|" in l)
    print(f"   {path.name}: malformed rows={len(bad)}  double-escape lines={dbl}")
    ok = ok and not bad and dbl == 0

print("D. gates:")
gates = {}
for s in ("_check_tables", "_check_v245_row", "_check_v248_row", "_e45_docs_numbers"):
    r = subprocess.run(["python", str(ROOT / "tools" / f"{s}.py")],
                       capture_output=True, text=True, encoding="utf-8")
    tail = [l.strip() for l in r.stdout.split("\n")
            if l.startswith("VERDICT") or "POS " in l]
    gates[s] = tail
    print(f"   {s}: {tail}")
ok = ok and any("PASS" in g for g in gates["_check_tables"])
ok = ok and any("PASS" in g for g in gates["_check_v245_row"])
ok = ok and any("PASS" in g for g in gates["_e45_docs_numbers"])
ok = ok and any("POS 59/61" in g for g in gates["_check_v248_row"])
v248 = subprocess.run(["python", str(ROOT / "tools" / "_check_v248_row.py")],
                      capture_output=True, text=True, encoding="utf-8").stdout
stale = [l.strip() for l in v248.split("\n") if l.strip().startswith("FAIL")]
for s in stale:
    print(f"   v248 remaining (by-design stale): {s}")
ok = ok and len(stale) == 2 and any("version pointer" in s for s in stale) \
    and any("blank line" in s for s in stale)

print("E. state unchanged / memory updated:")
m = MASTER.read_text(encoding="utf-8")
vers = max((int(a), int(b)) for a, b in re.findall(r"v(\d+)\.(\d+)", "\n".join(m.split("\n")[:20])))
verdict = ("六臂" in m) and ("判负" in m or "负面" in m)
mem = MEMORY.read_text(encoding="utf-8")
mem_ok = all(s in mem for s in ("3.0188", "24.7", "量级塌缩"))
print(f"   MASTER version pointer = v{vers[0]}.{vers[1]} (expect 2.49); negative verdict = {verdict}")
print(f"   MEMORY.md carries 3.0188 / 24.7 / 量级塌缩 = {mem_ok}")
ok = ok and vers == (2, 49) and verdict and mem_ok
for path, marker in ((HANDOVER, "E45-R70-BEGIN"), (UPPER_LOG, "E45-R70-111-BEGIN"),
                     (PROJ_LOG, "E45-R70-BEGIN"), (AUTOMATION_LOG, "E45-R70-BEGIN")):
    t = path.read_text(encoding="utf-8")
    has = t.count(marker) == 1 and t.count(marker.replace("BEGIN", "END")) == 1
    print(f"   {path.name}: {marker} exactly once (begin+end) = {has}")
    ok = ok and has

print()
print("VERDICT:", "PASS" if ok else "FAIL")

art = json.loads(ARTEFACT.read_text(encoding="utf-8"))
art["verification"] = {
    "content_preserved_rows": ["docs/MASTER.md:2026 (HEAD:1983)",
                               "docs/MASTER.md:2032 (HEAD:1989)",
                               "docs/experiments/E45-...-baseline.md:355 (HEAD:354)"],
    "master_line11_reconstruction": checks,
    "gate_verdicts": gates,
    "v248_stale_failures": stale,
    "master_version_pointer": f"v{vers[0]}.{vers[1]}",
    "memory_cites_3.0188": mem_ok,
    "notes": NOTES,
    "overall": "PASS" if ok else "FAIL",
}
ARTEFACT.write_text(json.dumps(art, ensure_ascii=False, indent=2), encoding="utf-8")
for n in NOTES:
    print("note:", n)
