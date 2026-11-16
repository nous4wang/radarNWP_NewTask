#!/usr/bin/env python3
"""D3 zero experiment: v2(use_gate=False) must be bit-identical to v1.

Runs on the remote with --script; it builds small models only, reads no
dataset and trains nothing.  If this fails, the D3 paired ablation is not a
one-variable comparison and D3-A cannot be interpreted.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/wf/radarNWP")

import torch
from radar_nwp_bench.models import build_model, list_models

print("=== D3 zero experiment (no data, no training) ===")
torch.manual_seed(0)
a = build_model("physical_qpe")
torch.manual_seed(0)
b = build_model("physical_qpe_v2", use_gate=False)
torch.manual_seed(0)
c = build_model("physical_qpe_v2", use_gate=True)

ka, kb, kc = a.state_dict(), b.state_dict(), c.state_dict()
print("v2 registered            :", "physical_qpe_v2" in list_models())
print("v1 n_params              :", sum(p.numel() for p in a.parameters()))
print("v2(use_gate=False) params:", sum(p.numel() for p in b.parameters()))
print("v2(use_gate=True)  params:", sum(p.numel() for p in c.parameters()))
print("v1 vs v2(False) same keys:", set(ka) == set(kb))
print("v1 vs v2(False) max|diff|:",
      max((ka[k] - kb[k]).abs().max().item() for k in ka))
print("v2(True) extra keys      :", sorted(set(kc) - set(ka)))

x = torch.randn(2, 11, 10, 32, 32)
nwp = torch.randn(2, 2, 27, 32, 32)
with torch.no_grad():
    ya, yb = a(x, nwp), b(x, nwp)
print("forward v1 vs v2(False) max|diff|:", (ya - yb).abs().max().item())
print("forward shape:", tuple(ya.shape))
print("=== ZERO EXPERIMENT DONE ===")
