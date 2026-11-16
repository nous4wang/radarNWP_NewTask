#!/usr/bin/env python3
"""E41 S0 gate -- prove the 'rain' head is bit-identical to simvp_v2.

Runs locally or on the remote host; needs no dataset, no checkpoint, no GPU.
If this fails, the whole single-variable ablation matrix is void and must be
fixed BEFORE any training compute (E41 section 5, S0).

Usage:
    python tools/_s0_equivalence.py
    python tools/_s0_equivalence.py --hid-c 64 --num-blocks 4 --size 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from radar_nwp_bench.models import build_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hid-c", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=4)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--in-len", type=int, default=11)
    ap.add_argument("--out-len", type=int, default=10)
    ap.add_argument("--out", default="tools/_evals/s0_equivalence.json")
    args = ap.parse_args()

    kw = dict(in_radar_c=10, in_nwp_c=27, in_len=args.in_len,
              out_len=args.out_len, hid_c=args.hid_c,
              num_blocks=args.num_blocks)
    base = build_model("simvp_v2", **kw)
    rain = build_model("struct_hurdle", head_mode="rain", **kw)
    rain.load_state_dict(base.state_dict())
    base.eval()
    rain.eval()

    g = torch.Generator().manual_seed(20261109)
    x = torch.rand(1, args.in_len, 10, args.size, args.size, generator=g)
    n = torch.rand(1, 2, 27, args.size, args.size, generator=g)
    with torch.no_grad():
        a = base(x, n)
        b = rain(x, n)

    max_abs = float(torch.max((a - b).abs()))
    result = {
        "gate": "E41 S0",
        "head_mode": "rain",
        "kwargs": kw,
        "input_size": args.size,
        "max_abs_diff": max_abs,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "pass": bool(max_abs < 1e-6 and a.dtype == torch.float32),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    print("VERDICT: S0 PASS" if result["pass"] else "VERDICT: S0 FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
