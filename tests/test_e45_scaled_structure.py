"""Unit tests for the E45 A4 arm (MASTER 4.47).

The arm's claim is a single one: the AMOUNT of the forecast and the
STRUCTURE of the forecast are two different objects, so improving the
structure cannot move the amount.  E44 measured the opposite on its own
arms -- the additive decomposition (A2) and the FNO decoder (A4) both
brought ``BIAS@2.0`` down while their shape metrics improved -- which
is what happens when one tensor has to carry both roles.

The mechanism is::

    body  = dec[:-1](future_feats)
    amp   = dec[-1](body)                   # (B*T, 1, H, W), log-mm
    low   = P_low(amp)                      # E44-A2's low-rank term
    sparse = sparse_head(body)              # E44-A2's sparse term
    F_log = low + sparse                    # E44-A2's decomposition
    F_mm  = floor_at_zero(expm1(F_log.clamp(-10.0, 6.0)))
    S     = F_mm / (mean(F_mm) + eps)       # a unit-mean shape field
    T     = <one scalar per frame>
    out   = T.view(-1, 1, 1, 1) * S         # mean(out) == T

The structure operator is E44-A2's decomposition reused verbatim -- the
arm has to PAIR a structure improvement that was MEASURED (A2's
``tv_z`` -27 %, ``lap_z`` -36 %) with an amount that does not move, and
a head invented here would measure nothing of the sort.

Two modes carry ``T``: ``"head"`` gives it to a new scalar head and
``"own"`` takes it from the model's own ordinary decode head.  The two
are DIFFERENT EXPERIMENTS and are therefore registered under DIFFERENT
MODEL NAMES -- ``scaled_structure`` and ``scaled_structure_own`` --
because a runtime switch does not change the state dict and an
evaluation run that forgot to pass it would silently evaluate the
other mechanism.

**S0 gate: with the mechanism switched off the arm IS simvp_v2, bit for
bit** (``max_abs_diff == 0.0``).  The check is on the OUTPUT with the
SAME weights: two independently initialised models obviously hold
different numbers, and comparing them would say nothing.  A difference
of more than the mechanism would void the comparison, so this is
checked here rather than discovered on the cluster.

The rest of the tests are on the properties the mechanism rests on:
the output's spatial mean IS the amount, the shape operator cannot move
that mean, the structure operator really is E44-A2's additive
decomposition, the two modes differ in what they carry, and the output
contract still holds.  No training, no dataset, no CSI numbers.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from radar_nwp_bench.models.registry import build_model, list_models

#: The two registered names of the arm, and the amount each one carries.
_ARMS = ("scaled_structure", "scaled_structure_own")

#: The seed every model is built under, so the arms' initial weights
#: are the same on every run of the suite.
_SEED = 4242


def _config(name: str) -> dict:
    from radar_nwp_bench.config import load_config

    cfg = load_config(name)
    return dict(cfg.get("model") or {})


def _control_kwargs(kwargs: dict) -> dict:
    """The subset of the arm's kwargs that ``simvp_v2`` accepts.

    The arms take three arguments the control does not (``r_low``,
    ``scale_mode`` and ``use_scaled``), so the control is built from the
    trunk arguments only.
    """
    from radar_nwp_bench.models.registry import model_class

    accepted = set(inspect.signature(model_class("simvp_v2").__init__).parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}


def _inputs(seed: int = 4242, batch: int = 2, size: int = 64):
    """A batch in the shape the models actually receive.

    The radar is ``(B, T_in=11, C=10, H, W)`` and the NWP is
    ``(B, T_nwp=2, C=27, H, W)`` -- the NWP's time axis has two frames
    and its channel count is 27, which is the convention the dataset
    writes and ``resolve_nwp`` expects.
    """
    g = torch.Generator().manual_seed(seed)
    x_radar = torch.rand(batch, 11, 10, size, size, generator=g)
    x_nwp = torch.rand(batch, 2, 27, size, size, generator=g)
    return x_radar, x_nwp


def _decoder_predicts_rain(model) -> None:
    """Make the ordinary decode head predict rain.

    The structure operator is E44-A2's decomposition, so its
    millimetre field is ``floor_at_zero(expm1(low + sparse))``: where
    the log-millimetre field is negative the field is zero, and where
    it is positive the field is millimetres.  An UNTRAINED decoder
    predicts a negative log-millimetre field almost everywhere, so the
    decomposition's millimetre field is identically zero, the shape
    field is zero, and the output is the zero field.

    That is a true statement about an untrained model, but it makes
    every assertion about the AMOUNT or the SHAPE of the output
    vacuous -- there is no shape to speak of.  The fixture therefore
    gives the decoder a positive bias first, so the millimetre field
    exists.  The bias moves the amount and the shape together, and it
    moves neither of them in a way that would flatter the properties
    under test: the amount is still the decoder's own, the shape is
    still the decomposition's.
    """
    with torch.no_grad():
        model.dec[-1].bias.add_(0.5)


def _armed(name: str):
    """The arm, with a millimetre field that is not identically zero.

    The model is built under a fixed seed so its initial weights are
    the same on every run of the suite, and the decoder is given the
    rain fixture of :func:`_decoder_predicts_rain`.
    """
    torch.manual_seed(_SEED)
    model = build_model(name, **_config(name))
    model.eval()
    _decoder_predicts_rain(model)
    return model


def _spatial_mean(out: torch.Tensor) -> torch.Tensor:
    """The per-frame spatial mean of a ``(B, T, 1, H, W)`` field."""
    return out.mean(dim=(3, 4)).reshape(-1)


def _spatial_std(out: torch.Tensor) -> torch.Tensor:
    """The per-frame spatial standard deviation of the same field."""
    return out.std(dim=(3, 4)).reshape(-1)


def _decoder_outputs(model, x_radar, x_nwp) -> dict:
    """The decoder's body and amplitude, read off the model's own ``dec``.

    The arm splits the decoder into ``dec[:-1]`` (the body) and
    ``dec[-1]`` (the amplitude), so a hook on the last module of the
    body and on the amplitude module captures exactly the two tensors
    the mechanism is built from -- no second copy of the trunk is
    needed to check the decomposition.
    """
    captured: dict = {}

    def _take_body(_module, _inputs, output):
        captured["body"] = output.detach()

    def _take_amp(_module, _inputs, output):
        captured["amp"] = output.detach()

    modules = list(model.dec)
    body_hook = modules[-2].register_forward_hook(_take_body)
    amp_hook = modules[-1].register_forward_hook(_take_amp)
    try:
        with torch.no_grad():
            model(x_radar, x_nwp)
    finally:
        body_hook.remove()
        amp_hook.remove()
    return captured


# --------------------------------------------------------------------- S0


@pytest.mark.parametrize("name", _ARMS)
@pytest.mark.parametrize("scale_mode", ["head", "own"])
def test_s0_bit_equivalence_with_simvp_v2(name, scale_mode):
    """With the mechanism off, the arm IS simvp_v2.

    The control is given the arm's weights, so any difference in the
    output comes from the computation graph alone.  The arm owns extra
    heads -- the sparse term of the decomposition, and in the ``head``
    mode the amount head as well -- which the control does not have;
    they are not read on the S0 path, but they do consume random
    numbers at construction time, so the shared weights are copied
    across rather than compared.
    """
    kwargs = _config(name)
    kwargs.update({"scale_mode": scale_mode, "use_scaled": False})

    torch.manual_seed(_SEED)
    control = build_model("simvp_v2", **_control_kwargs(kwargs))
    torch.manual_seed(_SEED)
    armed = build_model(name, **kwargs)

    # The two must have the same parameters under the same names, or the
    # comparison is not a single-variable one.
    c_sd = control.state_dict()
    a_sd = armed.state_dict()
    shared = set(c_sd) & set(a_sd)
    assert shared, f"no shared parameters between simvp_v2 and {name}"
    for key in shared:
        assert c_sd[key].shape == a_sd[key].shape, (
            f"{name}: {key} has shape {tuple(a_sd[key].shape)}, "
            f"simvp_v2 has {tuple(c_sd[key].shape)}"
        )

    missing, unexpected = armed.load_state_dict(c_sd, strict=False)
    assert not unexpected, f"{name} has parameters simvp_v2 does not: {unexpected}"
    assert all(key.startswith("sparse_head.") or key.startswith("scale_head.")
               for key in missing), (
        f"{name}: unexpected missing parameters {sorted(missing)}"
    )

    control.eval()
    armed.eval()

    x_radar, x_nwp = _inputs()
    with torch.no_grad():
        expected = control(x_radar, x_nwp)
        got = armed(x_radar, x_nwp)
    assert got.shape == expected.shape
    max_abs_diff = (got - expected).abs().max().item()
    assert max_abs_diff == 0.0, (
        f"{name} with the mechanism off is not bit-equal to simvp_v2: "
        f"max_abs_diff={max_abs_diff}"
    )


@pytest.mark.parametrize("name", _ARMS)
@pytest.mark.parametrize("scale_mode", ["head", "own"])
def test_s0_path_reads_neither_new_head(name, scale_mode):
    """On the S0 path the new heads must not be run at all.

    The bit equality above only says the OUTPUT is unchanged; a head
    that is computed and then discarded would still consume random
    numbers and still sit in the checkpoint.  A forward hook is the
    witness: it fires if and only if the head is actually called.
    """
    kwargs = _config(name)
    kwargs.update({"scale_mode": scale_mode, "use_scaled": False})
    torch.manual_seed(_SEED)
    model = build_model(name, **kwargs)
    model.eval()

    called: list = []

    def _spy(tag):
        def _hook(_module, _inputs, _output):
            called.append(tag)
        return _hook

    handles = [model.sparse_head.register_forward_hook(_spy("sparse_head"))]
    if model.scale_head is not None:
        handles.append(model.scale_head.register_forward_hook(_spy("scale_head")))
    try:
        x_radar, x_nwp = _inputs(seed=31)
        with torch.no_grad():
            model(x_radar, x_nwp)
    finally:
        for handle in handles:
            handle.remove()

    assert not called, f"{name}: the S0 path ran {called}"


# ------------------------------------------------- the amount and the shape


@pytest.mark.parametrize("name", _ARMS)
def test_the_output_mean_is_the_amount(name):
    """The amount IS the output's spatial mean.

    This is the whole mechanism: ``S`` is divided by its own spatial
    mean, so ``mean(S) = 1`` and ``mean(out) = T * mean(S) = T``.  The
    total is therefore decided by the amount channel alone.
    """
    model = _armed(name)

    x_radar, x_nwp = _inputs(seed=7)
    with torch.no_grad():
        out = model(x_radar, x_nwp)

    T = model.last_debug["T"]
    assert T.shape == (2 * 10,), f"T has shape {tuple(T.shape)}"
    assert float(T.min()) > 0.0, (
        f"{name}: the amount is zero, so the assertion below is vacuous"
    )

    got = _spatial_mean(out)
    rel = ((got - T).abs() / T.abs().clamp_min(1e-12)).max().item()
    assert rel < 1e-5, (
        f"{name}: the output's mean is not the amount: max_rel_diff={rel}"
    )


@pytest.mark.parametrize("name", _ARMS)
def test_the_shape_operator_cannot_move_the_amount(name):
    """Changing the structure operator must not change the amount.

    The amount is what the mechanism isolates, so a structure operator
    that could move the total would defeat it.  The check is made on the
    OUTPUT's spatial mean -- the quantity the verifier sees -- and the
    shape is shown to have changed at the same time, so a mean that
    stayed put because nothing happened would not pass.
    """
    model = _armed(name)

    x_radar, x_nwp = _inputs(seed=11)
    with torch.no_grad():
        before = model(x_radar, x_nwp)
    mean_before = _spatial_mean(before)
    std_before = _spatial_std(before)
    assert float(std_before.min()) > 0.0, (
        f"{name}: the output has no spatial structure, so the assertion "
        "below is vacuous"
    )

    # Perturb the structure operator itself.
    torch.manual_seed(_SEED + 1)
    with torch.no_grad():
        model.sparse_head.weight.add_(
            torch.randn_like(model.sparse_head.weight) * 0.1
        )
    with torch.no_grad():
        after = model(x_radar, x_nwp)
    mean_after = _spatial_mean(after)
    std_after = _spatial_std(after)

    rel = ((mean_after - mean_before).abs()
           / mean_before.abs().clamp_min(1e-12)).max().item()
    assert rel < 1e-5, (
        f"{name}: the structure operator moved the amount: max_rel_diff={rel}"
    )
    assert float((std_after - std_before).abs().max()) > 1e-3, (
        f"{name}: perturbing the structure operator did not change the shape"
    )


@pytest.mark.parametrize("name", _ARMS)
def test_the_structure_operator_is_the_additive_decomposition(name):
    """The structure operator must be E44-A2's ``low + sparse``.

    The arm's whole point is to pair a structure improvement that was
    MEASURED (A2's ``tv_z`` -27 %, ``lap_z`` -36 %) with an amount that
    does not move.  A structure operator that is anything else would
    measure something else, so the decomposition is recomputed here
    from the model's own decoder and the shared DCT basis and compared
    with the published ``F_log``.
    """
    from models.lr_sparse.model import _lowest_dct_modes

    model = _armed(name)

    x_radar, x_nwp = _inputs(seed=37)
    captured = _decoder_outputs(model, x_radar, x_nwp)
    body, amp = captured["body"], captured["amp"]

    H, W = amp.shape[-2:]
    basis = _lowest_dct_modes(H, W, model.r_low).to(
        device=amp.device, dtype=amp.dtype
    )
    flat = amp.reshape(amp.shape[0], -1)
    coeff = flat @ basis.reshape(model.r_low, -1).T
    low = (coeff @ basis.reshape(model.r_low, -1)).reshape(
        amp.shape[0], 1, H, W
    )
    sparse = model.sparse_head(body)
    expected = low + sparse

    got = model.last_debug["F_log"]
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, rtol=1e-5, atol=1e-6), (
        f"{name}: the structure operator is not E44-A2's low + sparse: "
        f"max_abs_diff={(got - expected).abs().max().item()}"
    )


@pytest.mark.parametrize("name", _ARMS)
def test_both_modes_forward_and_honour_the_output_contract(name):
    """The evaluator must not learn a new shape, and rain is >= 0."""
    torch.manual_seed(_SEED)
    model = build_model(name, **_config(name))
    model.eval()

    x_radar, x_nwp = _inputs(seed=13)
    with torch.no_grad():
        out = model(x_radar, x_nwp)

    assert out.shape == (2, 10, 1, 64, 64)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0


@pytest.mark.parametrize("name", _ARMS)
def test_the_shape_field_has_a_unit_spatial_mean(name):
    """``S`` is the unit-mean shape field the mechanism is built on."""
    model = _armed(name)

    x_radar, x_nwp = _inputs(seed=17)
    with torch.no_grad():
        model(x_radar, x_nwp)

    debug = model.last_debug
    S = debug["S"]
    assert S.shape == (2 * 10, 1, 64, 64)
    assert torch.allclose(S.mean(dim=(2, 3)).reshape(-1),
                           torch.ones(2 * 10), rtol=1e-5, atol=1e-6), (
        f"{name}: the shape field's mean is not one"
    )
    assert float(debug["F_mm"].min()) >= 0.0, (
        f"{name}: the millimetre field is not non-negative"
    )


# ------------------------------------------------------- the two modes


def test_the_own_mode_has_no_amount_head():
    """The ``own`` mode carries the amount with the ordinary decoder.

    It must not own a scalar head: a head nobody reads would be a
    head whose weights are in the checkpoint and whose contribution to
    the output is nil, and that is indistinguishable from a bug.
    """
    torch.manual_seed(_SEED)
    own = build_model("scaled_structure_own", **_config("scaled_structure_own"))
    assert not [k for k in own.state_dict() if k.startswith("scale_head.")], (
        "the own mode has an amount head"
    )
    assert [k for k in own.state_dict() if k.startswith("sparse_head.")], (
        "the own mode has no sparse term"
    )

    torch.manual_seed(_SEED)
    headed = build_model("scaled_structure", **_config("scaled_structure"))
    assert [k for k in headed.state_dict() if k.startswith("scale_head.")], (
        "the head mode has no amount head"
    )


def test_the_two_modes_carry_the_amount_differently():
    """The amount of one mode must not be the amount of the other.

    The modes are two arms, and an arm whose amount is a copy of the
    other's is not an arm.
    """
    head = _armed("scaled_structure")
    own = _armed("scaled_structure_own")

    x_radar, x_nwp = _inputs(seed=19)
    with torch.no_grad():
        head(x_radar, x_nwp)
        own(x_radar, x_nwp)

    assert not torch.allclose(head.last_debug["T"], own.last_debug["T"]), (
        "the two modes carry the same amount"
    )


def test_both_arms_are_registered_under_their_own_names():
    """The mode is part of the model's identity, not a runtime switch.

    A switch does not change the state dict, so an evaluation run that
    forgot to pass it would silently evaluate the other mechanism.
    """
    names = list_models()
    for name in _ARMS:
        assert name in names, f"{name} is not registered"
    assert names == sorted(names)


@pytest.mark.parametrize("name", _ARMS)
def test_an_unknown_scale_mode_is_rejected(name):
    """A typo in the mode must not fall back to a default."""
    with pytest.raises(ValueError):
        build_model(name, scale_mode="nonsense")


# ------------------------------------------------------------- the contract


@pytest.mark.parametrize("name", _ARMS)
def test_the_debug_tensors_are_published(name):
    """The diagnostics the read-outs need are on the model.

    ``last_debug`` holds detached tensors: the training loop calls the
    model over and over, and a live graph would be kept alive for
    nothing.  They stay on the model's device, so no host copy is paid
    for on every step.
    """
    torch.manual_seed(_SEED)
    model = build_model(name, **_config(name))
    model.eval()

    x_radar, x_nwp = _inputs(seed=23)
    with torch.no_grad():
        model(x_radar, x_nwp)

    debug = model.last_debug
    assert set(debug) == {"amp", "F_log", "F_mm", "S", "T"}
    for key, tensor in debug.items():
        assert isinstance(tensor, torch.Tensor), key
        assert not tensor.requires_grad, f"{key} still carries a graph"
        assert tensor.device == x_radar.device, key

    assert debug["amp"].shape == (2 * 10, 1, 64, 64)
    assert debug["F_log"].shape == (2 * 10, 1, 64, 64)
    assert debug["F_mm"].shape == (2 * 10, 1, 64, 64)
    assert debug["S"].shape == (2 * 10, 1, 64, 64)
    assert debug["T"].shape == (2 * 10,)


@pytest.mark.parametrize("name", _ARMS)
def test_the_s0_path_publishes_the_same_keys(name):
    """The switch must not change what the diagnostics look like.

    A read-out that reads ``last_debug`` has to work on both paths, or
    the probe has to know which path it is on -- and it usually does
    not.
    """
    kwargs = _config(name)
    kwargs["use_scaled"] = False
    torch.manual_seed(_SEED)
    model = build_model(name, **kwargs)
    model.eval()

    x_radar, x_nwp = _inputs(seed=29)
    with torch.no_grad():
        model(x_radar, x_nwp)

    debug = model.last_debug
    assert set(debug) == {"amp", "F_log", "F_mm", "S", "T"}
    assert float(debug["F_log"].abs().sum()) == 0.0
    assert float(debug["F_mm"].abs().sum()) == 0.0
    assert float(debug["S"].abs().sum()) == 0.0
    assert float(debug["T"].abs().sum()) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
