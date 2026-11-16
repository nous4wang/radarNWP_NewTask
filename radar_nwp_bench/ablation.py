"""NWP ablation: switch the NWP input off, or blank it out.

Two different questions are asked with the same switch, and the difference
matters for what the numbers mean:

``no_nwp`` (retrain without NWP)
    The model never receives NWP information *during training*. The NWP
    pathway still exists -- the parameter count, the tensor shapes and the
    checkpoint layout are identical to the multimodal run -- but the input it
    sees carries no information about the atmosphere. This isolates the
    *information* NWP contributes from the *capacity* it would otherwise
    spend, which is the only comparison that answers "how much does NWP buy
    this architecture?".

``zero_nwp`` (inference-time ablation)
    The model was trained with the real NWP fields and is then evaluated with
    them blanked. Any skill it loses was skill it was actually using; skill
    it keeps was skill it had found in the radar alone.

Why the switch lives here and not in the two scripts
----------------------------------------------------
An earlier version of the harness had ``--slide-stride`` in ``train.py``
only, and the evaluation script crashed with an ``AttributeError`` the first
time anybody used it. The same failure mode -- two scripts, one behaviour,
discovered months later -- is why the ablation switch is a single function
that both scripts call, and why the tests compare the two parsers directly.
"""
from __future__ import annotations

import torch

#: The NWP input is used as it comes out of the dataset.
MODE_FULL = "full"
#: The NWP input is withheld: the model trains and predicts without it.
MODE_NO_NWP = "no_nwp"
#: The NWP input is blanked out at inference time.
MODE_ZERO_NWP = "zero_nwp"

#: Every mode the ablation switch understands.
MODES: tuple[str, ...] = (MODE_FULL, MODE_NO_NWP, MODE_ZERO_NWP)


def resolve_mode(no_nwp: bool = False, zero_nwp: bool = False) -> str:
    """Turn the two command-line switches into one mode.

    Args:
        no_nwp: the ``--no-nwp`` flag was given.
        zero_nwp: the ``--zero-nwp`` flag was given.

    Returns:
        One of :data:`MODES`.

    Raises:
        ValueError: when both switches are given. They mean different
            things -- one removes NWP from the training, the other removes it
            from the inference -- so accepting both would produce a run whose
            intent nobody can reconstruct afterwards.
    """
    if no_nwp and zero_nwp:
        raise ValueError(
            "--no-nwp and --zero-nwp are different experiments and cannot be "
            "combined: --no-nwp withholds NWP from the training, --zero-nwp "
            "blanks it at inference"
        )
    if no_nwp:
        return MODE_NO_NWP
    if zero_nwp:
        return MODE_ZERO_NWP
    return MODE_FULL


def apply_ablation(x_nwp: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply the ablation to one NWP tensor.

    Args:
        x_nwp: the NWP tensor straight out of the loader,
            ``(B, T_nwp, C_nwp, H, W)``.
        mode: one of :data:`MODES`.

    Returns:
        ``x_nwp`` unchanged for :data:`MODE_FULL`; a zero tensor of the same
        shape and device for the two ablation modes.

    Notes:
        The shape is left untouched on purpose. Keeping the channel count
        identical is what makes the ablation a controlled experiment: the
        multimodal run and the ablated run differ only in whether the NWP
        input carries information, not in how many weights the model has.
    """
    if mode == MODE_FULL:
        return x_nwp
    if mode not in MODES:
        raise ValueError(f"unknown ablation mode {mode!r}; expected one of {MODES}")
    return torch.zeros_like(x_nwp)


def describe(mode: str) -> str:
    """A one-line description of a mode, for the run log and the report."""
    if mode == MODE_NO_NWP:
        return (
            "NWP withheld from the model (pure radar); the NWP pathway keeps "
            "its shape and parameter count but receives no information"
        )
    if mode == MODE_ZERO_NWP:
        return (
            "NWP blanked to zero at inference; the model was trained with the "
            "real NWP fields"
        )
    return "NWP used as provided"


__all__ = [
    "MODES",
    "MODE_FULL",
    "MODE_NO_NWP",
    "MODE_ZERO_NWP",
    "apply_ablation",
    "describe",
    "resolve_mode",
]
