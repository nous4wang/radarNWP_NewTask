#!/usr/bin/env python3
"""E45 / MASTER 4.47 -- S0 implementation-equivalence gate for E45's arms.

Each arm's single-variable claim rests on ONE thing: with its mechanism
switched off, the arm must be BIT IDENTICAL to ``simvp_v2``.  If it is
not, the difference between the two arms would include whatever else
changed, and every later number would be uninterpretable.

The arms and the switches that turn their mechanism off:

* ``amp_monotone_gate``  -- ``low_rank=False, monotone_gate=False``
* ``amp_hard_support``   -- ``low_rank=False, hard_gate=False``
* ``scaled_structure``   -- ``use_scaled=False``
* ``scaled_structure_own`` -- ``use_scaled=False``
* ``topo_simvp``         -- no switch at all: the arm IS ``simvp_v2`` and
  the mechanism lives in the loss

This is a zero-training, zero-dataset check: two small tensors in, two
tensors out, per arm.  It reports the maximum absolute difference and
fails loudly when it is not zero.

The shared weights are COPIED from the control rather than compared, and
the key sets are AUDITED.  The new heads are not read when the mechanism
is off, so they do not take part in the forward pass -- but they DO consume
random numbers at construction time, which would leave the shared weights
different.  The audit is what makes the copy honest:

* every key the two models share must have the SAME SHAPE, or the two
  models are not the same architecture;
* the keys the armed model has and the control does not are its NEW
  HEADS, and only those may be left missing after the copy;
* no key the control has may be UNEXPECTED in the armed model, or the
  armed model is missing a piece of the shared trunk.

Usage:
    python tools/_e45_s0_equivalence.py [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from radar_nwp_bench.config import load_config  # noqa: E402
from radar_nwp_bench.models import build_model  # noqa: E402
from radar_nwp_bench.models.registry import model_class  # noqa: E402

#: The arms and the switches that turn their mechanism off.
ARMS: Dict[str, Dict[str, Any]] = {
    "amp_monotone_gate": {"off": {"low_rank": False, "monotone_gate": False}},
    "amp_hard_support": {"off": {"low_rank": False, "hard_gate": False}},
    "scaled_structure": {"off": {"use_scaled": False}},
    "scaled_structure_own": {"off": {"use_scaled": False}},
    "topo_simvp": {"off": {}},
}


def _trunk_kwargs(model_name: str) -> Dict[str, Any]:
    """The constructor kwargs the control accepts, taken from the arm's own file.

    The arm's configuration file records the hyperparameters it was trained
    with, and the trunk arguments are the ones both models share.  They
    are read from the file rather than hard-coded so that a change to the
    arm's configuration cannot silently leave the control built with a
    different width than the arm.
    """
    import inspect

    cfg = load_config(model_name)
    kwargs = dict(cfg.get("model") or {})
    accepted = set(
        inspect.signature(model_class("simvp_v2").__init__).parameters
    )
    trunk = {k: v for k, v in kwargs.items() if k in accepted}
    # The shape arguments are what the two models must agree on; the rest
    # of the control's signature has defaults that match the arm's file.
    for key in ("in_radar_c", "in_nwp_c", "in_len", "out_len",
                "hid_c", "num_blocks"):
        if key in kwargs:
            trunk[key] = kwargs[key]
    return trunk


def _arm_kwargs(model_name: str, off: Dict[str, Any]) -> Dict[str, Any]:
    """The armed model's kwargs: the arm's own file, with the switches off."""
    cfg = load_config(model_name)
    kwargs = dict(cfg.get("model") or {})
    kwargs.update(off)
    return kwargs


def _root_module(key: str) -> str:
    """The top-level module a state-dict key belongs to."""
    return key.split(".", 1)[0]


def _audit_keys(
    armed: torch.nn.Module,
    control: torch.nn.Module,
) -> Dict[str, Any]:
    """Audit the two state dicts and copy the shared weights across.

    The audit asks three questions, and they are asked about the MODULES
    rather than about the key names, so that renaming a head does not
    change what the gate checks:

    1. Do the two models share every key the CONTROL has?  A key the
       control has and the armed model does not would mean the arm is
       missing a piece of the shared trunk -- the comparison would then
       not be a single-variable one.
    2. Do the shared keys have the SAME SHAPE?  A shape difference means
       the two models are not the same architecture, and the copy below
       could not be complete.
    3. Are the keys left MISSING after the copy confined to modules the
       control does not have at all?  Those are the arm's NEW HEADS: they
       are not read when the mechanism is off, so they do not take part
       in the forward pass, but they do consume random numbers at
       construction time, which is exactly why the shared weights are
       copied rather than compared.  A missing key inside a module the
       control ALSO has would be a piece of the shared trunk that the
       copy failed to deliver, and that is a failure.

    Args:
        armed: the arm with its mechanism switched off.
        control: the ``simvp_v2`` control.

    Returns:
        A mapping with the audit's findings.
    """
    a_sd = armed.state_dict()
    c_sd = control.state_dict()
    shared = sorted(set(a_sd) & set(c_sd))
    extra = sorted(set(a_sd) - set(c_sd))

    shape_mismatch = [
        k for k in shared if tuple(a_sd[k].shape) != tuple(c_sd[k].shape)
    ]

    if shape_mismatch:
        # The two models are not the same architecture, so the copy
        # cannot be complete and the bit-equality question is not even
        # well posed.  The copy is not attempted: ``load_state_dict``
        # would raise on the mismatched shapes, and a stack trace is a
        # worse report than the audit's own finding.
        missing: List[str] = []
        unexpected: List[str] = []
    else:
        missing, unexpected = armed.load_state_dict(c_sd, strict=False)
        missing = sorted(missing)
        unexpected = sorted(unexpected)

    control_roots = {_root_module(k) for k in c_sd}
    # A missing key must belong to a module the control does not have:
    # that is the arm's own new head, never a piece of the shared trunk.
    stray_missing = [
        k for k in missing if _root_module(k) in control_roots
    ]
    # A key the control has must not be unexpected in the armed model.
    unexpected_shared = [k for k in unexpected if k in c_sd]

    return {
        "n_shared_keys": len(shared),
        "n_new_head_keys": len(extra),
        "new_head_keys": extra,
        "new_head_modules": sorted({_root_module(k) for k in extra}),
        "shape_mismatch": shape_mismatch,
        "missing_after_copy": missing,
        "unexpected_after_copy": unexpected,
        "stray_missing": stray_missing,
        "unexpected_shared": unexpected_shared,
        "audit_ok": (
            not shape_mismatch
            and not stray_missing
            and not unexpected_shared
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = torch.Generator().manual_seed(4242)
    x_radar = torch.rand(2, 11, 10, 64, 64, generator=g)
    x_nwp = torch.rand(2, 2, 27, 64, 64, generator=g)

    results: Dict[str, Any] = {}
    all_pass = True

    for name, spec in ARMS.items():
        off = dict(spec["off"])
        control = build_model("simvp_v2", **_trunk_kwargs(name))
        armed = build_model(name, **_arm_kwargs(name, off))

        audit = _audit_keys(armed, control)
        if not audit["audit_ok"]:
            print(f"FATAL: {name}: the state dicts do not line up:")
            print(f"  shape_mismatch={audit['shape_mismatch']}")
            print(f"  stray_missing={audit['stray_missing']}")
            print(f"  unexpected_shared={audit['unexpected_shared']}")
            return 2

        control.eval()
        armed.eval()

        with torch.no_grad():
            reference = control(x_radar, x_nwp)
            got = armed(x_radar, x_nwp)

        diff = float((reference - got).abs().max())
        bitwise = bool(torch.equal(reference, got))
        # The criterion is BIT equality, not "close enough": a difference
        # of one ULP would still mean the two forward passes are not the
        # same computation.
        ok = diff == 0.0 and bitwise
        all_pass = all_pass and ok

        results[name] = {
            "max_abs_diff": diff,
            "bitwise_equal": bitwise,
            "shape": list(got.shape),
            "dtype": str(got.dtype),
            "switches_off": off,
            "n_shared_keys": audit["n_shared_keys"],
            "n_new_head_keys": audit["n_new_head_keys"],
            "new_head_modules": audit["new_head_modules"],
            "missing_after_copy": audit["missing_after_copy"],
            "unexpected_after_copy": audit["unexpected_after_copy"],
            "audit_ok": audit["audit_ok"],
            "pass": ok,
        }
        print(f"{name}: max_abs_diff = {diff:.3e}  "
              f"bitwise={bitwise}  {'PASS' if ok else 'FAIL'}")

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
