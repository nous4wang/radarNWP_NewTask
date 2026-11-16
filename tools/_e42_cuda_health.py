"""E42 follow-up -- check whether CUDA is still usable on the remote host.

READ-ONLY.  It only asks torch to allocate a tiny tensor on each visible
card and reports what happens.  It writes nothing and trains nothing.

Why this exists: at 15:15 the remote ``nvidia-smi`` started failing with
"Failed to initialize NVML: Unknown Error" even though the device nodes
still exist.  NVML is the monitoring library, not the driver itself, so a
NVML failure does not necessarily mean the cards are unusable -- but the
next training round depends on the answer, so it is checked directly
through torch rather than through the monitoring tool.
"""
from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"torch {torch.__version__}")
    print(f"cuda available : {torch.cuda.is_available()}")
    try:
        n = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001
        print(f"device_count FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"device count   : {n}")
    if n == 0:
        print("VERDICT: NO CUDA DEVICE VISIBLE")
        return 1

    ok = 0
    for i in range(n):
        try:
            x = torch.ones(1024, device=f"cuda:{i}")
            s = float(x.sum())
            # A matmul exercises the actual kernels, not just the allocator.
            y = x @ x
            print(f"  cuda:{i} alloc OK (sum={s:.0f}, mm={float(y.sum()):.0f})")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  cuda:{i} FAILED: {type(exc).__name__}: {exc}")
    print(f"VERDICT: {ok}/{n} cards usable")
    return 0 if ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
