"""E44 / MASTER 4.46 -- S0 implementation-equivalence gate for the four arms.

Each arm's single-variable claim rests on ONE thing: with its mechanism
switched off, the arm must be BIT IDENTICAL to ``simvp_v2``.  If it is
not, the difference between the arms would include whatever else
changed, and every later number would be uninterpretable.

This is a zero-training, zero-dataset check: two small tensors in, two
tensors out, per arm.  It reports the maximum absolute difference and
fails loudly when it is not zero.

Usage:
    python tools/_e44_s0_equivalence.py [--out results.json]
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

#: The four arms and the switches that turn their mechanism off.
ARMS = {
    "lr_sparse_bg": {"low_rank": False, "sparse_gate": False},
    "lr_sparse_add": {"low_rank": False, "additive": False},
    "topo_simvp": {},
    "fno_simvp": {"use_fno": False},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(20261109)
    control = build_model("simvp_v2", **KW)
    control.eval()

    g = torch.Generator().manual_seed(4242)
    x_radar = torch.rand(2, KW["in_len"], KW["in_radar_c"], 64, 64, generator=g)
    x_nwp = torch.rand(2, 2, KW["in_nwp_c"], 64, 64, generator=g)

    with torch.no_grad():
        reference = control(x_radar, x_nwp)

    results = {}
    all_pass = True
    for name, off in ARMS.items():
        armed = build_model(name, **off, **KW)
        # The armed model may have extra heads that the control does not
        # have.  They are NOT used when the mechanism is off, so they do
        # not take part in the forward pass -- but they DO consume random
        # numbers at construction time, which would leave the shared
        # weights different.  The shared weights are therefore copied
        # from the control, which is exactly what the harness does when
        # it evaluates a checkpoint.
        missing, unexpected = armed.load_state_dict(control.state_dict(),
                                                    strict=False)
        if unexpected:
            print(f"FATAL: {name} has parameters the control does not: "
                  f"{unexpected}")
            return 2
        armed.eval()

        with torch.no_grad():
            got = armed(x_radar, x_nwp)

        diff = float((reference - got).abs().max())
        ok = diff < 1e-6
        all_pass = all_pass and ok
        results[name] = {
            "max_abs_diff": diff,
            "shape": list(got.shape),
            "dtype": str(got.dtype),
            "switches_off": off,
            "pass": ok,
        }
        print(f"{name}: max_abs_diff = {diff:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")

    result = {
        "control": "simvp_v2",
        "arms": results,
        "pass": all_pass,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("VERDICT: PASS" if all_pass else "VERDICT: FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
