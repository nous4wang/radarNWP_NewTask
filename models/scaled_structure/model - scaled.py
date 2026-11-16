"""E45 / MASTER 4.47 -- A4: the amount and the structure, separated.

E44 ran four branches and every one of them came back negative, but two
of them produced a fact worth keeping: the arms that improved the
STRUCTURE of the field made the AMOUNT worse.  The additive
decomposition (A2, ``lr_sparse_add``) and the FNO decoder (A4) both
DROPPED ``BIAS@2.0`` (to 0.4192 and 0.3845) while their shape metrics
moved the right way.

That is the coupling this arm targets.  "Getting the field right" and
"getting the amount right" were two statements about the SAME tensor,
so a better shape could only be bought with a worse total.  Here they
are two different objects::

    body  = dec[:-1](future_feats)              # the shared trunk
    amp   = dec[-1](body)                       # (B*T, 1, H, W), log-mm
    low   = P_low(amp)                          # E44-A2's low-rank term
    sparse = sparse_head(body)                  # E44-A2's sparse term
    F_log = low + sparse                         # E44-A2's decomposition
    F_mm  = floor_at_zero(expm1(F_log.clamp(-10.0, 6.0)))
    S     = F_mm / (mean(F_mm) + eps)           # a unit-mean shape field
    T     = <the amount: one scalar per frame>
    out   = T.view(-1, 1, 1, 1) * S             # the output's mean IS T

The structure operator is E44-A2's additive decomposition, reused
VERBATIM -- the low-rank projection onto the ``r_low`` lowest-frequency
DCT modes plus the independently predicted sparse head, summed.  It is
not a new head, and that is the whole point of the arm: A2 was the arm
that MEASURED a structural improvement (``tv_z`` -27 %, ``lap_z``
-36 %, ``kernel_px`` up, ``tail_max`` up) while its ``BIAS@2.0`` fell to
0.4192.  A4 has to take that structure improvement and PAIR it with an
amount that does not move.  If the structure operator were a head
invented here, the arm would measure "a brand-new head plus an
explicit total" -- it would say nothing about whether the KNOWN
structural improvement survives the decoupling.

The guarantee is arithmetic rather than empirical.  ``S`` is divided by
its own spatial mean, so ``mean(S) = 1`` whatever the structure
operator outputs, and therefore::

    mean(out) = T * mean(S) = T

for EVERY choice of the structure operator.  The structure operator may
therefore be rewritten, sharpened or replaced without being able to
move the total: the total is carried by ``T`` alone.  This is the
statement the E44 arms could not make -- there one tensor carried both
roles, so a better shape was paid for with a worse amount.

The two modes differ in WHO carries ``T``:

* ``"head"`` -- ``T = softplus(scale_head(gpool(body)))``, where
  ``gpool(body) = body.mean(dim=(2, 3))`` is the globally pooled body.
  A new scalar head predicts the amount from the pool, so the amount
  has a channel of its own.
* ``"own"`` -- ``T`` comes from the model's OWN ordinary decode head:
  ``plain_mm = floor_at_zero(expm1(amp.clamp(-10.0, 6.0)))`` and ``T``
  is that field's spatial mean -- the mean rain intensity the decoder
  would itself have returned.  No new scalar head is added: the
  structure operator carries the shape and the existing decoder carries
  the amount.  The decoder's field is NOT the output field; the
  structure operator reaches it only through the low-rank projection,
  which is where E44-A2's decomposition takes the decoder's output, and
  the decoder's MEAN is what the amount is.

  The straight-through floor at zero is the project's own, and it is
  what makes the amount a RAIN amount: the mean of a signed
  millimetre field is negative wherever the decoder predicts no rain,
  and a negative amount would make the output negative and would break
  ``mean(out) = T``.  The floor changes no gradient, so the amount
  learns exactly as the signed mean would have.

What is NOT changed: the backbone -- the module names ``enc``,
``in_time_proj``, ``translator``, ``out_time_proj`` and ``dec`` are
simvp_v2's, so a ``simvp_v2`` state dict loads into this model -- the
loss, the number of loss terms, the output contract, and the dependency
set: only ``torch`` is used.  The S0 gate checks that with
``use_scaled`` off this module is bit-equal to ``simvp_v2``.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from radar_nwp_bench.models import register_model
from radar_nwp_bench.models.base import BasePrecipitationModel
from radar_nwp_bench.models.common import ConvNeXtBlock, floor_at_zero

from models.lr_sparse.model import _lowest_dct_modes

#: The guard in the shape normalisation.  It keeps the division finite
#: when the millimetre field is identically zero, and it is far below
#: the resolution of a millimetre field that carries any rain at all,
#: so ``mean(S) = 1`` holds to float precision for every non-degenerate
#: field and the guarantee ``mean(out) = T`` is not weakened.
_SHAPE_EPS = 1e-12


@register_model("scaled_structure")
class ScaledStructureModel(BasePrecipitationModel):
    """SimVP-v2 whose output is a total times a shape.

    The output field is the product of an amount and a shape::

        out = T * S,   mean(S) = 1,   mean(out) = T

    so the amount is decided by ``T`` alone and the structure operator
    cannot move it.  The structure operator is E44-A2's additive
    decomposition, reused verbatim: the low-rank projection of the
    decoder's output map plus the independently predicted sparse head.

    Args:
        in_radar_c: number of radar channels.
        in_nwp_c: number of NWP channels.
        in_len: number of radar history frames.
        out_len: number of forecast frames.
        hid_c: channel width.
        num_blocks: translator depth.
        r_low: dimension of the spatial subspace the low-rank term is
            confined to.  It is E44-A2's ``r_low``, unchanged.
        scale_mode: who carries the amount.  ``"head"`` gives it to a
            new scalar head on the globally pooled body; ``"own"`` takes
            it from the model's own ordinary decode head, so no scalar
            head is added at all.
        use_scaled: when False the model IS simvp_v2 -- the plain
            decoder is returned unchanged -- which is what the S0 gate
            checks.

    Raises:
        ValueError: when ``scale_mode`` is neither ``"head"`` nor
            ``"own"``.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
        r_low: int = 8,
        scale_mode: str = "head",
        use_scaled: bool = True,
    ) -> None:
        super().__init__()
        if scale_mode not in ("head", "own"):
            raise ValueError(
                f"scale_mode must be 'head' or 'own', got {scale_mode!r}"
            )
        self.in_len = in_len
        self.out_len = out_len
        self.r_low = int(r_low)
        self.scale_mode = scale_mode
        self.use_scaled = bool(use_scaled)

        #: The diagnostic tensors of the last forward pass.  They are
        #: detached: the training loop calls this model over and over,
        #: so a live graph would be kept alive for nothing.  They stay
        #: on the model's device, so no host copy is paid for.
        self.last_debug: Dict[str, torch.Tensor] = {}

        total_in = in_radar_c + in_nwp_c * 2

        # --- identical to simvp_v2 (module names matter for load_state_dict) ---
        self.enc = nn.Sequential(
            nn.Conv2d(total_in, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hid_c * 2),
            nn.GELU(),
        )

        self.in_time_proj = nn.Conv2d(in_len * hid_c * 2, hid_c * 4, kernel_size=1)
        self.translator = nn.Sequential(
            *[ConvNeXtBlock(hid_c * 4) for _ in range(num_blocks)]
        )
        self.out_time_proj = nn.Conv2d(hid_c * 4, out_len * hid_c * 2, kernel_size=1)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hid_c * 2, hid_c, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, hid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_c),
            nn.GELU(),
            nn.Conv2d(hid_c, 1, kernel_size=1),
        )

        # The sparse term of E44-A2's decomposition: an independently
        # predicted component that never passes through the projection.
        # It is here verbatim, with E44-A2's shape -- a one-by-one
        # convolution on the decoder's body.
        self.sparse_head = nn.Conv2d(hid_c, 1, kernel_size=1)

        # The amount head.  It exists only in the "head" mode: in the
        # "own" mode the amount comes from the ordinary decoder, so a
        # scalar head would be a head nobody reads.
        self.scale_head = (
            nn.Sequential(
                nn.Linear(hid_c, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            if scale_mode == "head"
            else None
        )

    def forward(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        B, T_in, _, H, W = x_radar.shape
        nwp_flat = x_nwp.view(B, 1, -1, H, W).expand(-1, T_in, -1, -1, -1)
        x_concat = torch.cat([x_radar, nwp_flat], dim=2)

        feats = self.enc(x_concat.view(B * T_in, -1, H, W))
        _, C_f, H_f, W_f = feats.shape

        feats_time = feats.view(B, T_in * C_f, H_f, W_f)
        h = self.in_time_proj(feats_time)
        h = self.translator(h)
        future_feats = self.out_time_proj(h).view(B * self.out_len, C_f, H_f, W_f)

        body = self.dec[:-1](future_feats)
        amp = self.dec[-1](body)                      # (B*T, 1, H, W)

        if not self.use_scaled:
            # The S0 path: exactly simvp_v2's decoder on exactly
            # simvp_v2's features.  Neither new head is read, so
            # nothing but the shared trunk can reach the output.
            self.last_debug = {
                "amp": amp.detach(),
                "F_log": torch.zeros_like(amp),
                "F_mm": torch.zeros_like(amp),
                "S": torch.zeros_like(amp),
                "T": torch.zeros(amp.shape[0], device=amp.device,
                                 dtype=amp.dtype),
            }
            log_mm = amp.view(B, self.out_len, 1, H, W)
            return floor_at_zero(torch.expm1(log_mm.clamp(-10.0, 6.0)))

        # --- the structure: E44-A2's additive decomposition, verbatim ---
        # The low-rank term: the decoder's output map projected onto the
        # r_low lowest-frequency DCT modes.  The projection is onto a
        # FIXED basis, so the background cannot take an arbitrary shape.
        basis = _lowest_dct_modes(H, W, self.r_low).to(
            device=amp.device, dtype=amp.dtype
        )
        flat = amp.reshape(amp.shape[0], -1)                 # (B*T, H*W)
        coeff = flat @ basis.reshape(self.r_low, -1).T      # (B*T, r_low)
        low = (coeff @ basis.reshape(self.r_low, -1)).reshape(
            amp.shape[0], 1, H, W
        )
        # The sparse term: predicted independently, so it never passes
        # through the projection and the low-rank term cannot flatten it.
        sparse = self.sparse_head(body)                     # (B*T, 1, H, W)
        F_log = low + sparse
        F_mm = floor_at_zero(torch.expm1(F_log.clamp(-10.0, 6.0)))

        # --- the shape: a non-negative field with a unit spatial mean ---
        S = F_mm / (F_mm.mean(dim=(2, 3), keepdim=True) + _SHAPE_EPS)

        # --- the amount: one scalar per (window, lead time) ---
        if self.scale_mode == "head":
            pooled = body.mean(dim=(2, 3))            # (B*T, hid_c)
            T = F.softplus(self.scale_head(pooled)).reshape(-1)
        else:
            # The ordinary decode head's own rain intensity, and
            # nothing else: the decoder's field is not the output
            # field's shape -- the shape comes from the decomposition
            # above -- and the amount is this field's mean.
            #
            # The mean is taken over that head's MILLIMETRE field, i.e.
            # over ``floor_at_zero(expm1(clamp(plain_log)))`` -- the very
            # field the decoder would have returned.  A signed mean
            # would be negative wherever the decoder predicts no rain,
            # which is everywhere at the start of a run, and a negative
            # amount would make the output negative and would break
            # ``mean(out) = T``.  The floor is the project's own
            # straight-through one, so the gradient is the same as the
            # signed mean's.
            plain_mm = floor_at_zero(torch.expm1(amp.clamp(-10.0, 6.0)))
            T = plain_mm.mean(dim=(2, 3)).reshape(-1)

        out = T.view(-1, 1, 1, 1) * S                 # (B*T, 1, H, W)

        self.last_debug = {
            "amp": amp.detach(),
            "F_log": F_log.detach(),
            "F_mm": F_mm.detach(),
            "S": S.detach(),
            "T": T.detach(),
        }

        return floor_at_zero(out.view(B, self.out_len, 1, H, W))


@register_model("scaled_structure_own")
class ScaledStructureOwnModel(ScaledStructureModel):
    """The ``own`` variant of the arm, registered under its own name.

    The two modes are DIFFERENT EXPERIMENTS, so they are DIFFERENT
    MODEL NAMES.  A runtime switch would not do: it does not change the
    state dict, so an evaluation run that forgot to pass it would
    silently evaluate the mechanism that was supposed to be switched
    off.  Registering ``"scaled_structure_own"`` next to
    ``"scaled_structure"`` makes the mode part of the checkpoint's
    identity -- the name is in the training command, in the evaluation
    command and in the file names -- so it cannot be dropped on the way
    to the evaluation.

    The class is a thin subclass of :class:`ScaledStructureModel` in
    the same module, so the mechanism lives in exactly one place and
    there is no second copy of it to drift.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        hid_c: int = 64,
        num_blocks: int = 4,
        r_low: int = 8,
        scale_mode: str = "own",
        use_scaled: bool = True,
    ) -> None:
        super().__init__(
            in_radar_c=in_radar_c,
            in_nwp_c=in_nwp_c,
            in_len=in_len,
            out_len=out_len,
            hid_c=hid_c,
            num_blocks=num_blocks,
            r_low=r_low,
            scale_mode=scale_mode,
            use_scaled=use_scaled,
        )
