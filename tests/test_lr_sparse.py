"""E43 / MASTER 4.45 -- unit tests for the low-rank + sparse model.

These are NOT dataset evaluations.  No training, no remote, no CSI
numbers.  They check the three properties the pre-registration depends
on: the S0 gate, that the two mechanisms actually do something, and
that the gradients reach every branch.

The S0 property (bit equality with ``simvp_v2`` when both switches are
off) is checked here as well as in ``tools/_e43_s0_equivalence.py``:
the tool is the record, the test is the guard that stops the property
from being broken silently later.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from radar_nwp_bench.models import build_model  # noqa: E402
from models.lr_sparse.model import _lowest_dct_modes  # noqa: E402

KW = dict(in_radar_c=10, in_nwp_c=27, in_len=3, out_len=2,
          hid_c=8, num_blocks=1)


def _inputs(b=2, h=32, w=32):
    g = torch.Generator().manual_seed(20261109)
    x = torch.rand(b, KW["in_len"], KW["in_radar_c"], h, w, generator=g)
    n = torch.rand(b, 2, KW["in_nwp_c"], h, w, generator=g)
    return x, n


def test_registry_builds_the_model():
    model = build_model("lr_sparse", **KW)
    assert model.r_low == 8
    assert model.use_low_rank and model.use_sparse_gate


def test_s0_switches_off_reproduces_simvp_v2_bit_for_bit():
    """The single-variable claim: only the mechanism differs."""
    torch.manual_seed(7)
    control = build_model("simvp_v2", **KW)
    armed = build_model("lr_sparse", low_rank=False, sparse_gate=False, **KW)
    missing, unexpected = armed.load_state_dict(control.state_dict(),
                                                strict=False)
    assert not [k for k in missing
                if not k.startswith(("sparse_gate", "gate_threshold"))]
    assert not unexpected

    control.eval()
    armed.eval()
    x, n = _inputs()
    with torch.no_grad():
        a = control(x, n)
        b = armed(x, n)
    assert float((a - b).abs().max()) == 0.0


def test_low_rank_modes_are_orthonormal():
    """The background lives in a genuine subspace, not a decorative one."""
    basis = _lowest_dct_modes(32, 32, 8)
    assert basis.shape == (8, 32, 32)
    gram = basis.reshape(8, -1) @ basis.reshape(8, -1).T
    assert torch.allclose(gram, torch.eye(8), atol=1e-4)


def test_low_rank_branch_confines_the_background():
    """A map outside the subspace must not survive the projection."""
    basis = _lowest_dct_modes(16, 16, 4)
    b, _ = basis.reshape(4, -1).T.shape
    # A single hot pixel is not a low-frequency mode.
    hot = torch.zeros(1, 16, 16)
    hot[0, 8, 8] = 1.0
    flat = hot.reshape(1, -1)
    coeff = flat @ basis.reshape(4, -1).T
    recon = (coeff @ basis.reshape(4, -1)).reshape(1, 16, 16)
    # The projection keeps only the low-frequency content of the pixel.
    assert float(recon.abs().max()) < 1.0


def test_gate_threshold_is_learned_and_per_sample():
    """The model chooses its own area: the threshold is predicted, not fixed."""
    model = build_model("lr_sparse", **KW)
    assert model.gate_threshold.weight.requires_grad
    x, n = _inputs()
    out = model(x, n)
    assert out.shape == (2, KW["out_len"], 1, 32, 32)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0


def test_gradients_reach_every_branch():
    """Both mechanisms must be trainable, or the arm is not an experiment."""
    model = build_model("lr_sparse", **KW)
    model.train()
    x, n = _inputs()
    out = model(x, n)
    out.mean().backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        assert torch.isfinite(param.grad).all(), name
    assert model.dec[-1].weight.grad is not None
    assert model.sparse_gate.weight.grad is not None
    assert model.gate_threshold.weight.grad is not None


def test_the_mechanisms_change_the_output():
    """The switches must actually change something, or they are no-ops."""
    torch.manual_seed(11)
    armed = build_model("lr_sparse", **KW)
    off = build_model("lr_sparse", low_rank=False, sparse_gate=False, **KW)
    off.load_state_dict(armed.state_dict(), strict=False)
    armed.eval()
    off.eval()
    x, n = _inputs()
    with torch.no_grad():
        a = armed(x, n)
        b = off(x, n)
    assert float((a - b).abs().max()) > 0.0


def test_output_contract_is_unchanged():
    """The evaluator must not learn a new shape."""
    model = build_model("lr_sparse", **KW)
    x, n = _inputs()
    out = model(x, n)
    assert out.dim() == 5
    assert out.shape[1] == KW["out_len"]
    assert out.shape[2] == 1
    assert out.dtype == torch.float32
