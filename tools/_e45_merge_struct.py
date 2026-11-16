#!/usr/bin/env python3
"""E45 -- merge the per-seed structure metrics into one file.

The structure quantities (S1) are measured once per seed, because the
measurement reads the dumped forecast fields of that seed and the
fields do not fit in memory all at once.  The S1 VERDICT, on the other
hand, compares an arm against the control over all three seeds: it
takes the median of each quantity per arm, and a per-seed file would
give it a single seed's worth.

So this tool merges the per-seed files into one, keyed by tag.  The
merged file has the same shape as a single file's, so the verdict tool
reads it unchanged.

The merge is a union of the per-tag entries, and a tag that appears in
two inputs is a bug in the schedule rather than a duplicate worth
keeping: the same arm at the same seed cannot be measured twice.  Such
a tag is reported and the tool exits non-zero rather than silently
keeping the first or the last, because either choice would make the
verdict read a number that no run produced.

Nothing here trains, evaluates, or reads the fields: it only joins
JSON that the structure-metrics tool already wrote.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="the per-seed structure-metrics JSON files")
    ap.add_argument("--out", required=True,
                    help="where to write the merged JSON")
    args = ap.parse_args()

    merged: Dict[str, Any] = {
        "n_windows": None,
        "out_len": None,
        "grid": None,
        "aggregation": "median over (window, lead) frames",
        "probe_implementation": "tools/_struct_probe_one_case.py::metrics",
        "tags": {},
    }
    seen: Dict[str, str] = {}
    problems: List[str] = []

    for path_text in args.inputs:
        path = Path(path_text)
        if not path.is_file():
            problems.append(f"missing input: {path}")
            continue
        with path.open(encoding="utf-8") as handle:
            blob = json.load(handle)
        tags = blob.get("tags")
        if not isinstance(tags, dict):
            problems.append(f"{path} has no 'tags' object")
            continue
        for key in ("n_windows", "out_len", "grid"):
            if merged[key] is None:
                merged[key] = blob.get(key)
        for tag, entry in tags.items():
            if tag in seen:
                problems.append(
                    f"tag '{tag}' appears in both {seen[tag]} and {path}"
                )
                continue
            seen[tag] = str(path)
            merged["tags"][tag] = entry

    if problems:
        for line in problems:
            print(f"ERROR: {line}", file=sys.stderr)
        print("the merged file would be wrong; nothing was written",
              file=sys.stderr)
        return 1

    if not merged["tags"]:
        print("ERROR: the inputs held no tags at all", file=sys.stderr)
        return 1

    Path(args.out).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"merged {len(merged['tags'])} tag(s) into {args.out}")
    for tag in sorted(merged["tags"]):
        print(f"  {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
