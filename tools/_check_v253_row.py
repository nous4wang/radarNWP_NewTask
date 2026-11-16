"""Standalone gate for the v2.53 write-back.

Checks, all read-only:

1.  every document that must carry the v2.53 marker carries it, and the
    five version pointers agree;
2.  the Gate-3 places in MASTER are all present (header version line,
    the section 4.47 card row, the section 4.48 status bullet, the
    section 9 row, a section 12 changelog row) and the new changelog
    row has the same cell count as the row it follows;
3.  the NEW finding is stated where it belongs -- that job 343 ran and
    the S3 guard's bootstrap-seed robustness is closed on BOTH halves,
    and that A2 (the arm at the flipping edge) does not flip;
4.  the negative verdict and the pre-registered promotion rule are
    still stated, and the three-over-the-floor arms are NOT promoted;
5.  the historical section 12 rows are all still present with one
    consistent cell count;
6.  no table row has a cell count that disagrees with its header (the
    pipe-escaping rule), and the version pointer is the newest one.

Note on the older gates: ``_check_v251_row.py`` and ``_check_v252_row.py``
assert that the version pointer IS v2.51 / v2.52, so they go stale by
design the moment the pointer advances -- exactly as
``_check_v248_row.py`` did when v2.49 arrived.  Those are SNAPSHOT-class
expiries, not content failures; this gate is the current one.

Exit code 0 iff every assertion holds.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

MASTER = ROOT / "docs" / "MASTER.md"
INDEX = ROOT / "docs" / "experiments" / "INDEX.md"
E45DOC = ROOT / "docs" / "experiments" / "E45-gate-direction-amount-separation-baseline.md"
MEMORY = Path(r"R:\1radarNWPprediction\.workbuddy\memory\MEMORY.md")
HANDOVER = Path(r"R:\1radarNWPprediction\.workbuddy\memory\HANDOVER_E45.md")

POS = 0
NEG = 0


def check(ok: bool, label: str) -> bool:
    global POS, NEG
    if ok:
        POS += 1
        print(f"  PASS  {label}")
    else:
        NEG += 1
        print(f"  FAIL  {label}")
    return ok


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"FILE MISSING: {path}")
    return path.read_text(encoding="utf-8")


def vtuple(text: str):
    """The HIGHEST version token in `text`.

    The first match is NOT the pointer: the INDEX E45 row and the E45
    experiment file carry the whole version history, so the pointer is
    the newest token, not the oldest.
    """
    toks = [(int(a), int(b)) for a, b in re.findall(r"v(\d+)\.(\d+)", text or "")]
    return max(toks) if toks else None


def cells(line: str) -> int:
    """The cell count of a markdown table row.

    A row is ``| a | b | c | d |``: four cells, five separators.  Empty
    trailing cells are not counted, and escaped pipes inside a cell do
    not separate it.
    """
    s = re.sub(r"\\\|", "ESC", line)
    parts = s.split("|")
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return len(parts)


def main() -> int:
    texts = {
        "MASTER": read(MASTER),
        "INDEX": read(INDEX),
        "E45DOC": read(E45DOC),
        "MEMORY": read(MEMORY),
        "HANDOVER": read(HANDOVER),
    }

    print("=" * 78)
    print("E45 round 94 -- gate for the v2.53 write-back")
    print("=" * 78)

    print("\n1. the version pointers")
    vers = {name: vtuple(t) for name, t in texts.items()}
    for name, v in vers.items():
        print(f"         {name:9s} = v{v[0]}.{v[1]}")
    check(len(set(vers.values())) == 1, "all five version pointers agree")
    check(vers["MASTER"] == (2, 53), "the version pointer is v2.53")

    print("\n2. the Gate-3 places in MASTER")
    m = texts["MASTER"]
    lines = m.split("\n")
    check(any(l.startswith("| 最后更新 |") and "v2.53" in l for l in lines),
          "MASTER header carries v2.53")
    check(any(l.startswith("| §4.47 | 修三处机制缺陷") and "v2.53" in l for l in lines),
          "MASTER section 4.47 card carries v2.53")
    check(any(l.startswith("- **状态**：4 个修复文件只在本地") and "v2.53" in l
              for l in lines),
          "MASTER section 4.48 status bullet carries v2.53")
    check(any(l.startswith("| **E45 / §4.47 修三处机制缺陷 + 摆正基准** |")
              and "v2.53" in l for l in lines),
          "MASTER section 9 row carries v2.53")
    idx = [i for i, l in enumerate(lines) if l.startswith("| 2026-11-16 | **v2.53** |")]
    check(len(idx) == 1, "MASTER has exactly one v2.53 changelog row")
    if idx:
        i = idx[0]
        prev = lines[i - 1]
        check(cells(prev) == cells(lines[i]),
              f"the v2.53 row has the same cell count as the row it follows "
              f"({cells(lines[i])} vs {cells(prev)})")
        check(prev.startswith("| 2026-11-16 | **v2.52** |"),
              "the v2.53 row follows the v2.52 row")

    print("\n3. the new finding is stated")
    for name in ("MASTER", "INDEX", "E45DOC", "MEMORY", "HANDOVER"):
        t = texts[name]
        check("343" in t, f"{name} names job 343")
        check("s3_seedrobust" in t, f"{name} names the seed-robustness artifact")
        check("20/20" in t or "20 个种子" in t or "20 个 bootstrap" in t,
              f"{name} states the 20-seed sweep")
    for name in ("MASTER", "E45DOC", "MEMORY", "HANDOVER"):
        t = texts[name]
        check("A2" in t, f"{name} names A2")
        check("不翻转" in t, f"{name} states that A2 does not flip")
        check("两侧" in t, f"{name} states that BOTH halves are closed")
    check("0.001075" in texts["MASTER"] or "0.001075" in texts["MEMORY"],
          "the A2 minimum margin is recorded")
    check("module_md5" in texts["MASTER"] or "module_md5" in texts["MEMORY"],
          "the guard-implementation identity is recorded")
    check("数值级重算" in texts["MASTER"],
          "MASTER states the numeric (not boolean) re-derivation")

    print("\n4. the verdict is still the negative one")
    for name in ("MASTER", "INDEX", "E45DOC", "MEMORY", "HANDOVER"):
        t = texts[name]
        check("六臂" in t and "判负" in t, f"{name} states the negative verdict")
    check("不投十 seed 算力" in texts["MASTER"],
          "MASTER states that no ten-seed compute is committed")
    check("3/3" in texts["MASTER"] and "低于同批对照 B0" in texts["MASTER"],
          "MASTER states that the two 3/3 arms sit below the control")
    check("不投十 seed 算力" in texts["HANDOVER"],
          "HANDOVER states that no ten-seed compute is committed")
    check("不投十 seed 算力" in texts["MEMORY"],
          "MEMORY states that no ten-seed compute is committed")

    print("\n5. the historical changelog rows are intact")
    hist = ["v2.44", "v2.45", "v2.46", "v2.47", "v2.48", "v2.49", "v2.50",
            "v2.51", "v2.52"]
    for v in hist:
        check(any(l.startswith("| 20") and f"**{v}**" in l for l in lines),
              f"the {v} changelog row is still present")
    counts = {cells(l) for l in lines if re.match(r"^\| 20\d\d-\d\d-\d\d \| \*\*v2\.", l)}
    check(len(counts) == 1, f"all changelog rows share one cell count {sorted(counts)}")

    print("\n6. table integrity (the pipe-escaping rule)")
    for name in ("MASTER", "INDEX", "E45DOC"):
        t = texts[name]
        header = None
        bad = []
        for i, line in enumerate(t.split("\n")):
            s = line.strip()
            if not s.startswith("|"):
                header = None
                continue
            n = cells(line)
            if header is None:
                header = n
            elif n != header:
                bad.append((i + 1, n, header))
        check(not bad, f"{name} has no row with a mismatched cell count ({bad[:3]})")

    print("\n7. the round-91 duplicate is gone")
    mem_lines = texts["MEMORY"].split("\n")
    n91 = sum(1 for l in mem_lines if l.startswith("- **E45 第 91 轮（2026-11-16）**"))
    check(n91 == 1, f"MEMORY carries exactly one round-91 bullet (found {n91})")
    n94 = sum(1 for l in mem_lines if l.startswith("- **E45 第 94 轮（2026-11-16）**"))
    check(n94 == 1, f"MEMORY carries exactly one round-94 bullet (found {n94})")

    print("\n" + "=" * 78)
    print(f"POS {POS}/{POS + NEG}   NEG {NEG}/{POS + NEG}")
    print("VERDICT: " + ("PASS" if NEG == 0 else "FAIL"))
    print("=" * 78)
    return 0 if NEG == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
