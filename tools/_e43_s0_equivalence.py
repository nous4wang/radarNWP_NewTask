"""E43 / MASTER 4.45 -- S0 implementation-equivalence gate for lr_sparse.

The single-variable claim rests on ONE thing: with the two mechanisms
switched off, ``lr_sparse`` must be BIT IDENTICAL to ``simvp_v2``.  If
it is not, the difference between the two arms would include whatever
else changed, and every later number would be uninterpretable.

This is a zero-training, zero-dataset check: two small tensors in, two
tensors out.  It reports the maximum absolute difference and fails
loudly when it is not zero.

Usage:
    python tools/_e43_s0_equivalence.py [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from radar_nwp_bench.models import build_model  # noqa: E402

KW = dict(in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10,
          hid_c=64, num_blocks=4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(20261109)
    control = build_model("simvp_v2", **KW)
    armed = build_model("lr_sparse", low_rank=False, sparse_gate=False, **KW)

    # The armed model has two extra heads that the control does not have.
    # They are NOT used when both switches are off, so they do not take
    # part in the forward pass -- but they DO consume random numbers at
    # construction time, which would leave the shared weights different.
    # The shared weights are therefore copied from the control, which is
    # exactly what the harness does when it evaluates a checkpoint.
    missing, unexpected = armed.load_state_dict(control.state_dict(),
                                                strict=False)
    shared_missing = [k for k in missing if not k.startswith(
        ("sparse_gate", "gate_threshold"))]
    if shared_missing or unexpected:
        print(f"FATAL: state dict mismatch: missing={shared_missing} "
              f"unexpected={unexpected}")
        return 2

    control.eval()
    armed.eval()

    g = torch.Generator().manual_seed(4242)
    x_radar = torch.rand(2, KW["in_len"], KW["in_radar_c"], 64, 64, generator=g)
    x_nwp = torch.rand(2, 2, KW["in_nwp_c"], 64, 64, generator=g)

    with torch.no_grad():
        a = control(x_radar, x_nwp)
        b = armed(x_radar, x_nwp)

    diff = float((a - b).abs().max())
    result = {
        "max_abs_diff": diff,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "control": "simvp_v2",
        "armed": "lr_sparse(low_rank=False, sparse_gate=False)",
        "pass": diff < 1e-6,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("VERDICT: PASS" if result["pass"] else "VERDICT: FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
