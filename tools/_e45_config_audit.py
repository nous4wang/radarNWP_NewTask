#!/usr/bin/env python3
"""E45 round-57 -- remote-side provenance digest of the 21 training configs.

READ-ONLY.  For every runs/e45/<ARM>_s<SEED>/ it parses the training
config.json and emits the values of the contract-relevant keys, so the
caller can check VALUE-LEVEL provenance (not just "the key exists" --
every config carries all flags, with defaults, so only the value
discriminates).

Prints a compact JSON digest on stdout (small enough to read directly)
and also writes the full blob to /tmp for the record.
"""
import json
import os
import re
import sys

ROOT = "/home/wf/radarNWP/runs/e45"
CELL_RE = re.compile(r"^[A-Za-z0-9]+_s\d+$")

# the keys that define the evaluation/training contract
WANT = [
    "model", "loss", "alpha_csi", "topo_clamp_mm", "topo_truth_smooth_sigma",
    "topo_area_normalize", "topo_weight", "topo_threshold", "no_nwp", "zero_nwp",
    "data_root", "split", "epochs", "batch_size", "lr", "rain_threshold",
    "filter_dry", "dry_keep_prob", "patch_size", "in_len", "out_len",
    "nwp_pipeline_version", "config",
]


def main():
    digest = {}
    for name in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, name)
        if not os.path.isdir(d) or not CELL_RE.match(name):
            continue
        found = {}
        for dirpath, _dirnames, filenames in os.walk(d):
            if "config.json" not in filenames:
                continue
            path = os.path.join(dirpath, "config.json")
            rel = os.path.relpath(path, ROOT)
            try:
                with open(path) as fh:
                    obj = json.load(fh)
            except Exception as exc:
                found[rel] = {"_error": repr(exc)}
                continue
            if not isinstance(obj, dict):
                continue
            rec = {k: obj.get(k) for k in WANT if k in obj}
            rec["_all_keys"] = len(obj)
            found[rel] = rec
        digest[name] = found

    out = json.dumps(digest, ensure_ascii=False, sort_keys=True, indent=1)
    with open("/tmp/e45_config_digest_round57.json", "w") as fh:
        fh.write(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
