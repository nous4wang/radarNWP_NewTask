r"""Check that every markdown table row has the same cell count as its header.

Rows whose cell text contains an escaped pipe (``\|``) inside a code span
are counted with that pipe as part of the text, not as a separator.
"""
import re
import subprocess
import sys
from pathlib import Path

ESC = re.compile(r"\\\|")


def cells(line):
    s = ESC.sub("ESC", line)
    parts = s.split("|")
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return parts


def scan(text):
    bad = []
    header = None
    for i, line in enumerate(text.split("\n")):
        s = line.strip()
        if not s.startswith("|"):
            header = None
            continue
        n = len(cells(line))
        if header is None:
            header = n
        elif n != header:
            bad.append((i + 1, n, header, s[:46]))
    return bad


def main():
    files = [
        "docs/MASTER.md",
        "docs/experiments/INDEX.md",
        "docs/experiments/E45-gate-direction-amount-separation-baseline.md",
    ]
    rc = 0
    for f in files:
        new = Path(f).read_text(encoding="utf-8")
        bn = scan(new)
        try:
            old = subprocess.run(
                ["git", "show", f"HEAD:{f}"],
                capture_output=True, text=True, encoding="utf-8",
            ).stdout
            bo = scan(old)
        except Exception:
            bo = []
        old_lines = {b[0] for b in bo}
        new_lines = {b[0] for b in bn}
        introduced = sorted(new_lines - old_lines)
        fixed = sorted(old_lines - new_lines)
        print(f"{f}: mismatched now={len(bn)} (HEAD={len(bo)}) "
              f"introduced={introduced} fixed={fixed}")
        for b in bn:
            print(f"    line {b[0]}: {b[1]} cells vs header {b[2]} :: {b[3]}")
        if introduced:
            rc = 1
    print("VERDICT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
