#!/usr/bin/env python3
"""E41 remote readiness smoke: registry, both heads and both loss branches.

Pure CPU, no dataset and no checkpoint. Run on the benchmark server with
``python tools/remote.py --script tools/_e41_smoke.py``. A failure here means
the E41 ablation must not start.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/home/wf/radarNWP")

import torch
from radar_nwp_bench.models import build_model, list_models
from radar_nwp_bench.metrics.struct_losses import (
    HurdleCriterion,
    HurdleNLL,
    UOTCriterion,
    UnbalancedSinkhornDivergence,
)

g = torch.Generator().manual_seed(20261109)
x = torch.rand(1, 11, 10, 32, 32, generator=g)
n = torch.rand(1, 2, 27, 32, 32, generator=g)
y = (torch.rand(1, 10, 1, 32, 32, generator=g) * 4).clamp_min(0.0)

m_h = build_model("struct_hurdle", head_mode="hurdle").eval()
m_r = build_model("struct_hurdle", head_mode="rain").eval()
with torch.no_grad():
    p_h = m_h(x, n)
    p_r = m_r(x, n)

loss_h, _ = HurdleCriterion()(p_h, y, model=m_h)
loss_u, _ = UOTCriterion()(p_r, y, model=m_r)

result = {
    "torch": torch.__version__,
    "cuda": bool(torch.cuda.is_available()),
    "registered": "struct_hurdle" in list_models(),
    "hurdle_head": type(m_h).__name__,
    "rain_head": type(m_r).__name__,
    "pred_shape": list(p_h.shape),
    "hurdle_loss": float(loss_h),
    "uot_loss": float(loss_u),
    "finite": bool(torch.isfinite(loss_h) and torch.isfinite(loss_u)),
}
# The training criterion stays on CPU while pred lives on cuda:0.  This
# reproduces that exact mix (the buffer is NOT moved with .cuda()).
cuda_ok = True
if torch.cuda.is_available():
    div = UnbalancedSinkhornDivergence(grid=8, n_iters=5)
    p = torch.rand(1, 2, 1, 32, 32, device="cuda")
    t = torch.rand(1, 2, 1, 32, 32, device="cuda")
    v, _ = div(p, t)
    cuda_ok = bool(torch.isfinite(v)) and v.device.type == "cuda"
    result["cuda_uot_loss"] = float(v)
    result["cuda_buffer_device"] = "cpu (criterion) / cuda (pred)"
else:
    result["cuda_uot_loss"] = None
result["cuda_buffer_ok"] = cuda_ok

print(json.dumps(result, indent=2))
ok = (
    result["registered"]
    and result["finite"]
    and result["cuda_buffer_ok"]
    and result["pred_shape"] == [1, 10, 1, 32, 32]
)
print("VERDICT: SMOKE PASS" if ok else "VERDICT: SMOKE FAIL")
raise SystemExit(0 if ok else 1)
