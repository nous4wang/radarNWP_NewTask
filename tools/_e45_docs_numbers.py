#!/usr/bin/env python3
"""E45 round 35 -- the numbers in the documents, checked against the data.

Why this file exists
--------------------
Earlier verification rounds of E45 checked the *verdict wording* of the
records (negative verdict, no SOTA claim, median + pass rate reported), and
``tools/_e45_recompute_raw.py`` checked that the stored summary agrees with
the 21 raw evaluation reports.  The last link was never closed: whether the
numbers actually printed in the authoritative documents are those numbers.

This script closes it by PARSING THE ACTUAL MARKDOWN TABLES rather than by
matching free prose:

1. E45 experiment file, the three-seed score table (section 8.4) -- for every
   arm: the median CSI@2.0, the three-seed range, the median BIAS@2.0, the
   pass rate and the over-Oracle count are compared against the recorded
   table, cell by cell.
2. E45 experiment file, the seed-0 score table (section 8.2) -- compared
   against the recorded seed-0 summary.
3. E45 experiment file, the S3 equal-bias table -- the printed point estimate
   and both interval bounds are compared against the recorded S3 output, and
   every row must carry the "crosses zero" verdict.
4. MASTER.md and INDEX.md -- a narrow contradiction check: a number that is
   quoted AS A MEDIAN next to an E45 arm and that sits within 0.02 of that
   arm's own median but is not equal to it (a corrupted median) is a defect.
   Numbers that are merely near a median are not flagged, because a line may
   legitimately quote a per-seed value, a historical ten-seed median or a
   mechanism reading.

Nothing is written and no document is touched.

Usage:
    python tools/_e45_docs_numbers.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(ROOT, "tools", "_evals", "e45")

SUMMARY = os.path.join(EVALS, "scores_summary.json")
SUMMARY_S0 = os.path.join(EVALS, "scores_summary_s0.json")
S3 = os.path.join(EVALS, "s3_equalbias.json")
E45_DOC = os.path.join(
    ROOT, "docs", "experiments", "E45-gate-direction-amount-separation-baseline.md")
MASTER = os.path.join(ROOT, "docs", "MASTER.md")
INDEX = os.path.join(ROOT, "docs", "experiments", "INDEX.md")

FLOOR = 0.0924
ORACLE = 0.1653
ARMS = ("B0", "A1", "A2", "A3a", "A3b", "A4", "A4b")

#: a printed value may move by half a unit in its last printed place
TOL = 0.5e-4 + 1e-9

#: how close a quoted number must be to a median to be treated as a claim
#: about that median
NEAR = 0.02


def _clean(cell: str) -> str:
    return cell.replace("**", "").replace("`", "").replace("*", "").strip()


def _num(cell: str) -> float:
    return float(_clean(cell).replace("−", "-"))


def _load(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _rows(path: str) -> List[Tuple[int, List[str]]]:
    """Every markdown table row of a file as (line number, cells)."""
    out: List[Tuple[int, List[str]]] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            cells = stripped.strip("|").split("|")
            if all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells if c.strip()):
                continue
            out.append((number, cells))
    return out


def _arm_of(cells: List[str]) -> str:
    first = _clean(cells[0]) if cells else ""
    return first if first in ARMS else ""


def _find_header(rows, *needles: str) -> int:
    for index, (number, cells) in enumerate(rows):
        joined = "|".join(_clean(c) for c in cells)
        if all(needle in joined for needle in needles):
            return index
    return -1


def check_three_seed_table(problems: List[str]) -> None:
    """Section 8.4 score table: median, range, bias, pass rate, oracle count."""
    reference = _load(SUMMARY)
    rows = _rows(E45_DOC)
    header_index = _find_header(rows, "中位数", "过地板通过率")
    if header_index < 0:
        problems.append("E45 experiment file: the three-seed score table was not found")
        return

    seen: List[str] = []
    for number, cells in rows[header_index + 1:]:
        arm = _arm_of(cells)
        if not arm:
            if seen:
                break
            continue
        seen.append(arm)
        row = reference.get(arm)
        if row is None:
            problems.append(f"E45 L{number}: arm {arm} is not in the recorded table")
            continue
        if len(cells) < 6:
            problems.append(f"E45 L{number}: the {arm} row has only {len(cells)} cells")
            continue

        med_csi_doc = _num(cells[1])
        med_csi_ref = float(row["median_csi_at_2.0"])
        if abs(med_csi_doc - med_csi_ref) > TOL:
            problems.append(f"E45 L{number} {arm}: median CSI@2.0 printed {med_csi_doc}, "
                            f"recorded {med_csi_ref:.6f}")

        rng = _clean(cells[2]).split("–")
        if len(rng) != 2:
            problems.append(f"E45 L{number} {arm}: the range cell {cells[2]!r} is not a range")
        else:
            seed_values = [float(v) for v in row["csi_at_2.0"].values()]
            lo_ref, hi_ref = min(seed_values), max(seed_values)
            if abs(_num(rng[0]) - lo_ref) > TOL:
                problems.append(f"E45 L{number} {arm}: range low printed {_num(rng[0])}, "
                                f"recorded {lo_ref:.6f}")
            if abs(_num(rng[1]) - hi_ref) > TOL:
                problems.append(f"E45 L{number} {arm}: range high printed {_num(rng[1])}, "
                                f"recorded {hi_ref:.6f}")

        med_bias_doc = _num(cells[3])
        med_bias_ref = float(row["median_bias_at_2.0"])
        if abs(med_bias_doc - med_bias_ref) > TOL:
            problems.append(f"E45 L{number} {arm}: median BIAS@2.0 printed {med_bias_doc}, "
                            f"recorded {med_bias_ref:.6f}")

        rate_doc = _clean(cells[4])
        rate_ref = str(row["pass_rate_over_floor"])
        if rate_doc != rate_ref:
            problems.append(f"E45 L{number} {arm}: pass rate printed {rate_doc!r}, "
                            f"recorded {rate_ref!r}")

        oracle_doc = _clean(cells[5])
        seed_values = [float(v) for v in row["csi_at_2.0"].values()]
        oracle_ref = sum(1 for v in seed_values if v > ORACLE)
        if oracle_doc != f"{oracle_ref}/3":
            problems.append(f"E45 L{number} {arm}: over-Oracle count printed {oracle_doc!r}, "
                            f"recorded {oracle_ref}/3")

        print(f"  {arm:>4s}  median CSI {med_csi_doc:.4f} (rec {med_csi_ref:.6f})  "
              f"BIAS {med_bias_doc:.4f} (rec {med_bias_ref:.6f})  "
              f"pass {rate_doc} (rec {rate_ref})  oracle {oracle_doc}")

    missing = [arm for arm in ARMS if arm not in seen]
    if missing:
        problems.append(f"E45 experiment file: the score table has no row for {missing}")
    else:
        print(f"  all {len(ARMS)} arms present in the three-seed score table")


def check_seed0_table(problems: List[str]) -> None:
    """Section 8.2 seed-0 table against the recorded seed-0 summary."""
    reference = _load(SUMMARY_S0)
    rows = _rows(E45_DOC)
    header_index = _find_header(rows, "臂", "CSI@2.0", "BIAS@2.0")
    if header_index < 0:
        problems.append("E45 experiment file: the seed-0 score table was not found")
        return
    checked = 0
    for number, cells in rows[header_index + 1:]:
        arm = _arm_of(cells)
        if not arm:
            break
        row = reference.get(f"{arm}_s0_ep00")
        if row is None:
            problems.append(f"E45 L{number}: arm {arm} is not in the seed-0 summary")
            continue
        for index, (field, label) in enumerate(
                (("csi_at_2.0", "CSI@2.0"), ("bias_at_2.0", "BIAS@2.0"),
                 ("csi_at_4.0", "CSI@4.0")), start=1):
            if len(cells) <= index:
                break
            doc_value = _num(cells[index])
            ref_value = float(row[field])
            decimals = len(_clean(cells[index]).split(".")[-1])
            tol = 0.5 * 10 ** (-decimals) + 1e-9
            if abs(doc_value - ref_value) > tol:
                problems.append(f"E45 L{number} {arm}: seed-0 {label} printed {doc_value}, "
                                f"recorded {ref_value:.6f}")
        checked += 1
    print(f"  seed-0 table: {checked} arm rows checked against the recorded seed-0 summary")


def check_s3_table(problems: List[str]) -> None:
    """S3 equal-bias table against the recorded S3 output, cell by cell."""
    recorded = _load(S3)["arms"]
    rows = _rows(E45_DOC)
    header_index = _find_header(rows, "原始增益", "配平后增益")
    if header_index < 0:
        problems.append("E45 experiment file: the S3 table was not found")
        return

    checked = 0
    for number, cells in rows[header_index + 1:]:
        arm = _arm_of(cells)
        if not arm:
            break
        entry = recorded.get(arm)
        if entry is None:
            problems.append(f"E45 L{number}: arm {arm} is not in the recorded S3 output")
            continue
        if len(cells) < 4:
            problems.append(f"E45 L{number}: the S3 row for {arm} has too few cells")
            continue
        for column, key in ((1, "raw"), (2, "equal_bias")):
            # the table writes a negative bound with the U+2212 MINUS SIGN, so
            # the sign has to be part of the token, not just the digits
            text = _clean(cells[column]).replace("−", "-")
            numbers = [float(t) for t in re.findall(r"-?\d+\.\d+", text)]
            if len(numbers) < 3:
                problems.append(f"E45 L{number} {arm}: S3 column {column} does not print a "
                                f"point estimate and two bounds")
                continue
            point, lo, hi = numbers[0], numbers[1], numbers[2]
            for label, printed, key_name in (("point", point, "point"),
                                             ("ci_lo", lo, "ci_lo"),
                                             ("ci_hi", hi, "ci_hi")):
                ref = float(entry[key][key_name])
                # the table prints four or five decimals of a bootstrap estimate
                tol = 1.1e-4
                if abs(printed - ref) > tol:
                    problems.append(f"E45 L{number} {arm}: S3 {key}.{label} printed "
                                    f"{printed}, recorded {ref:.6f}")
        verdict = _clean(cells[3])
        if "跨零" not in verdict:
            problems.append(f"E45 L{number} {arm}: the S3 verdict does not say the interval "
                            f"crosses zero ({verdict!r})")
        for key in ("raw", "equal_bias"):
            if not entry[key]["crosses_zero"]:
                problems.append(f"E45 L{number} {arm}: the recorded {key} interval does not "
                                f"cross zero, but the table prints a verdict for it")
        checked += 1
    print(f"  S3 table: {checked} arm rows checked against the recorded S3 output")


def check_prose_contradictions(problems: List[str]) -> None:
    """A number quoted AS A MEDIAN next to an E45 arm must not be a corrupted median.

    The scan is clause-scoped: a line is split at Chinese punctuation, and a
    number is only treated as a median claim when the SAME clause names an E45
    arm and carries the median keyword.  Without that scoping a line that
    mentions several models would drag in the medians of models that are not
    E45 arms at all.
    """
    reference = _load(SUMMARY)
    keyword = re.compile(r"(?:中位|median)")
    number = re.compile(r"-?\d+\.\d+")
    for name, path in (("MASTER.md", MASTER), ("INDEX.md", INDEX)):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for number_, line in enumerate(handle, 1):
                if "E45" not in line and "e45" not in line:
                    continue
                for clause in re.split(r"[。；，、]", line):
                    if not keyword.search(clause):
                        continue
                    arms_on_line = [arm for arm in ARMS
                                    if re.search(rf"(?<![A-Za-z0-9]){re.escape(arm)}(?![A-Za-z0-9])", clause)]
                    if not arms_on_line:
                        continue
                    allowed = {FLOOR, ORACLE}
                    for arm in arms_on_line:
                        row = reference.get(arm)
                        if row is None:
                            continue
                        allowed.add(float(row["median_csi_at_2.0"]))
                        allowed.add(float(row["median_bias_at_2.0"]))
                    for token in number.findall(clause.replace("−", "-")):
                        value = float(token)
                        if any(abs(value - a) <= 0.0006 for a in allowed):
                            continue
                        if any(abs(value - a) <= NEAR for a in allowed):
                            problems.append(
                                f"{name} L{number_}: the line quotes {token} as a median "
                                f"near {'/'.join(arms_on_line)}, but the recorded medians are "
                                f"{sorted(round(a, 4) for a in allowed - {FLOOR, ORACLE})}")
    print("  prose contradiction scan over MASTER.md and INDEX.md done")


def main() -> int:
    for path in (SUMMARY, SUMMARY_S0, S3, E45_DOC):
        if not os.path.exists(path):
            print(f"missing input: {path}")
            print("VERDICT: FAIL")
            return 1

    print("=" * 78)
    print("E45 round 35 -- the numbers printed in the documents vs the recorded data")
    print("=" * 78)

    problems: List[str] = []

    print("\n1) the three-seed score table (E45 experiment file, section 8.4)")
    check_three_seed_table(problems)
    print("\n2) the seed-0 score table (E45 experiment file, section 8.2)")
    check_seed0_table(problems)
    print("\n3) the S3 equal-bias table (E45 experiment file, section 8.4)")
    check_s3_table(problems)
    print("\n4) prose contradictions in MASTER.md and INDEX.md")
    check_prose_contradictions(problems)

    print()
    print("-" * 78)
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        print()
        print("VERDICT: FAIL")
        return 1
    print("every number in the three-seed, seed-0 and S3 tables agrees with the recorded")
    print("data, and no authoritative document contradicts the recorded table")
    print()
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
