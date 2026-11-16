#!/usr/bin/env python3
"""E45 round 33 -- the remote/local SOURCE-CODE identity check.

The thirty-two previous rounds verified the ARTIFACTS (the thirteen
result JSONs), the DOCUMENTS (MASTER / INDEX / E45 file / MEMORY) and
the JOB LEDGER -- but never the CODE that produced them.  That is the
check that matters before a re-run: the three implementation defects
registered in MASTER v2.46 are still unfixed in the source tree, so
the next action on this experiment will be a re-measurement, and a
re-measurement is only meaningful if the remote tree and the local
tree are the same code.

Line endings are normalised before hashing.  The local checkout is
CRLF (Windows) while the copies on the remote are LF, so a raw md5
comparison reports EVERY file as different and drowns the signal.  On
the remote ``sed 's/\\r$//'`` strips the carriage returns, locally
``bytes.replace(b"\\r\\n", b"\\n")`` does the same, and the two sides
are then on the same footing.  A file whose normalised hashes agree
but whose raw hashes differ is reported as EOL-ONLY, not as a drift.

Read-only: it runs one command over the single sanctioned channel
(``tools/remote.py``) and hashes local files.  It writes nothing to
the remote and changes nothing locally.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The files that decide what E45 measured.  The model files define the
#: arms, the two sbatch files and the arm table define how they were
#: run, and the four tool files define how the numbers were read out.
CRITICAL = [
    "models/topo_simvp/model.py",
    "models/scaled_structure/model.py",
    "models/amp_monotone_gate/model.py",
    "models/amp_hard_support/model.py",
    "radar_nwp_bench/metrics/precipitation.py",
    "radar_nwp_bench/metrics/topology_losses.py",
    "scripts/_e45_arms.sh",
    "scripts/e45_train.sbatch",
    "scripts/e45_stage2.sbatch",
    "scripts/e45_summary.sbatch",
    "scripts/e45_diag.sbatch",
    "tools/_e45_mechanism_probe.py",
    "tools/_e41_s3_equalbias.py",
    "tools/_e45_collect_scores.py",
    "tools/_e45_merge_struct.py",
    "tools/_e44_diag_probe.py",
]

#: The three defects MASTER v2.46 located, and the file each one
#: lives in.  Checked here because "the fix has been located" is not
#: "the fix has been written": round 32 found that none of the three
#: had touched the tree.
DEFECT_WHY = [
    ("models/topo_simvp/model.py",
     "topology clamp never reaches the output field"),
    ("models/scaled_structure/model.py",
     "shape normalisation still uses an absolute guard"),
    ("tools/_e45_mechanism_probe.py",
     "probe aggregation is still not NaN-aware"),
]


def norm_md5(path: Path) -> str:
    """The md5 of the file with CRLF folded to LF."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.md5(data).hexdigest()


def raw_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def remote_hashes() -> dict[str, tuple[str, str]]:
    """One remote round trip: normalised and raw md5 of every file.

    The remote command prints ``<path> <normalised> <raw>`` per line.
    """
    inner = "; ".join(
        "printf '%s ' \"{f}\"; "
        "sed 's/\\r$//' \"{f}\" | md5sum | cut -d' ' -f1 | tr -d '\\n'; "
        "printf ' '; md5sum \"{f}\" | cut -d' ' -f1".format(f=f)
        for f in CRITICAL
    )
    cmd = "cd /home/wf/radarNWP && " + inner
    proc = subprocess.run(
        [sys.executable, "tools/remote.py", "--", cmd],
        cwd=REPO, capture_output=True, text=True,
    )
    out: dict[str, tuple[str, str]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and all(len(p) == 32 for p in parts[1:]):
            out[parts[0]] = (parts[1], parts[2])
    if not out:
        print("REMOTE HASH FETCH FAILED")
        print(proc.stdout[-1500:])
        print(proc.stderr[-800:])
    return out


def main() -> int:
    remote = remote_hashes()
    if not remote:
        print("VERDICT: FAIL (no remote hashes)")
        return 1

    same = eol_only = drift = missing = 0
    print(f"{'file':<46} {'verdict':<10} detail")
    print("-" * 92)
    for f in CRITICAL:
        p = REPO / f
        if f not in remote:
            print(f"{f:<46} {'MISSING':<10} absent from the remote tree")
            missing += 1
            continue
        rn, rr = remote[f]
        ln, lr = norm_md5(p), raw_md5(p)
        if rn == ln and rr == lr:
            print(f"{f:<46} {'SAME':<10} normalised+raw identical")
            same += 1
        elif rn == ln:
            print(f"{f:<46} {'EOL-ONLY':<10} content identical; CRLF/LF only")
            eol_only += 1
        else:
            print(f"{f:<46} {'DRIFT':<10} local={ln[:10]} remote={rn[:10]}")
            drift += 1

    print("-" * 92)
    print(f"SAME={same}  EOL-ONLY={eol_only}  DRIFT={drift}  MISSING={missing}")

    print("\n--- the three registered defects: is the code changed yet? ---")
    fixed = 0
    for f, why in DEFECT_WHY:
        text = (REPO / f).read_text(encoding="utf-8", errors="replace")
        if f.endswith("topo_simvp/model.py"):
            # Unfixed: the output field is hard-clamped and there is no
            # ``clamp_mm`` entry point.  Fixed: the entry point exists.
            defect = ("expm1(log_mm.clamp(-10.0, 6.0))" in text
                      and "clamp_mm" not in text)
        elif f.endswith("scaled_structure/model.py"):
            # Unfixed: the guard is the bare absolute constant.  Fixed:
            # it is relative (the constant is scaled by a magnitude).
            m = re.search(r"^_SHAPE_EPS\s*=.*$", text, re.M)
            line = m.group(0).strip() if m else ""
            defect = bool(line) and bool(re.search(r"1e-12\s*$", line))
        else:
            # Unfixed: a non-NaN-aware median.  Fixed: np.nanmedian.
            defect = "np.median(" in text and "np.nanmedian(" not in text
        if defect:
            state = "STILL-PRESENT"
        else:
            state = "FIXED"
            fixed += 1
        print(f"  {f:<44} {state:<14} ({why})")
    print(f"  defects already fixed in the tree: {fixed}/3")

    ok = drift == 0 and missing == 0 and same + eol_only == len(CRITICAL)
    print("\nVERDICT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
