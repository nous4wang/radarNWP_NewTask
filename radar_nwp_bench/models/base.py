"""The model seam.

Every baseline in the benchmark is an **adapter** behind one interface, so the
training loop, the metrics and the data pipeline never learn a model's
particularities. That is what makes an 8-model comparison meaningful: the
models differ, the harness does not.

Interface
---------
``forward(x_radar, x_nwp) -> y_rain``

    x_radar : ``(B, T_in, C_radar, H, W)`` float32
        Radar history. ``T_in`` frames of 6 minutes, ``C_radar`` physical channels.
    x_nwp   : ``(B, T_nwp, C_nwp, H, W)`` float32
        NWP fields already resampled onto the radar grid and aligned by valid
        time. May be ``None``, which means "no NWP at all" (see
        :mod:`radar_nwp_bench.ablation`).
    returns : ``(B, T_out, 1, H, W)`` float32
        Predicted 6-minute accumulated ground rainfall in millimetres.

The base class fixes the tensor conventions and leaves the network body to the
adapter. Adapters that need a different internal layout reshape internally and
still honour this contract.

Withholding the NWP
-------------------
``x_nwp=None`` is a first-class input, not an error. It is what the
``--no-nwp`` ablation passes, and it is resolved **here**, at the seam, rather
than in each of the fifteen adapters:

    * six of the adapters never call :meth:`_check_shapes` at all and go
      straight to ``x_nwp.view(...)``, so a check inside that method cannot
      protect them;
    * the substitution has to happen *before* the adapter body runs, or the
      adapter still sees ``None``.

:meth:`BasePrecipitationModel.__call__` therefore normalises the input once,
for every model, and the adapters can keep assuming a well-formed tensor.
The materialised tensor is all zeros: it carries no information about the
atmosphere, while the parameter count and the tensor shapes stay exactly what
the multimodal run used. That is what makes the ablation a controlled
experiment -- the two runs differ in whether the NWP input carries
information, not in how much capacity the model has.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class BasePrecipitationModel(nn.Module, ABC):
    """Common interface and tensor bookkeeping for every baseline.

    Args:
        in_radar_c: number of radar channels fed to the model.
        in_nwp_c: number of NWP channels fed to the model.
        in_len: number of radar history frames.
        out_len: number of forecast frames to produce.
        out_channels: number of output channels per frame (1 = ground rainfall).
    """

    #: Number of NWP slices the model expects. The benchmark's default
    #: alignment (``forecast``) hands over two: the analysis hour and the
    #: forecast hour that covers the end of the target window.
    #:
    #: It is a class attribute rather than a constructor argument because it
    #: is a property of the *data alignment*, not of any one adapter, and
    #: adding a constructor argument would mean touching all fifteen of them.
    #: The harness overwrites it with the value the dataset actually
    #: produces, so the ``--no-nwp`` zero tensor has the right shape even
    #: under an alignment that yields a different number of slices.
    nwp_time_len: int = 2

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.in_radar_c = in_radar_c
        self.in_nwp_c = in_nwp_c
        self.in_len = in_len
        self.out_len = out_len
        self.out_channels = out_channels

    #: Whether the baseline has a trainable path. Analytical baselines
    #: (advection, persistence) set this to ``False``: they are evaluated,
    #: not fitted, and carry no gradient.
    is_learnable: bool = True

    @abstractmethod
    def forward(
        self,
        x_radar: torch.Tensor,
        x_nwp: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Predict future accumulated rainfall.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp:   ``(B, T_nwp, C_nwp, H, W)`` or ``None`` when ablated.
        Returns:
            pred_rain: ``(B, T_out, 1, H, W)`` in millimetres

        """
        raise NotImplementedError

    # -- the seam -----------------------------------------------------------
    def __call__(
        self,
        x_radar: torch.Tensor,
        x_nwp: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Run the model, resolving a missing NWP tensor first.

        This is the single point where ``x_nwp=None`` becomes a real tensor.
        Doing it here rather than in the adapters means the fifteen of them
        do not each have to know about the ablation, and the six that never
        call :meth:`_check_shapes` are protected too.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)``, or ``None`` for a pure-radar
                run.
            **kwargs: forwarded to the adapter (teacher forcing, for example).

        Returns:
            The adapter's output, ``(B, T_out, 1, H, W)``.
        """
        return super().__call__(x_radar, self.resolve_nwp(x_radar, x_nwp), **kwargs)

    def resolve_nwp(
        self,
        x_radar: torch.Tensor,
        x_nwp: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Turn a missing NWP tensor into an all-zero one of the right shape.

        Args:
            x_radar: the radar input, used for the batch size, the spatial
                extent and the device.
            x_nwp: the NWP tensor, or ``None``.

        Returns:
            ``x_nwp`` unchanged, or a zero tensor of shape
            ``(B, T_nwp, C_nwp, H, W)``.
        """
        if x_nwp is not None:
            return x_nwp
        b = x_radar.shape[0]
        h, w = x_radar.shape[-2:]
        return torch.zeros(
            b,
            self.nwp_time_len,
            self.in_nwp_c,
            h,
            w,
            device=x_radar.device,
            dtype=x_radar.dtype,
        )

    # -- helpers available to adapters ---------------------------------------
    def _check_shapes(self, x_radar: torch.Tensor, x_nwp: torch.Tensor) -> torch.Tensor:
        """Validate the incoming tensors against the declared contract.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            x_nwp: ``(B, T_nwp, C_nwp, H, W)``; ``None`` is accepted and
                resolved to zeros.

        Returns:
            The NWP tensor the adapter should use -- the one that was passed
            in, or the zero substitute.

        Raises:
            ValueError: when a shape disagrees with the declared contract.
        """
        if x_radar.dim() != 5:
            raise ValueError(
                f"x_radar must be (B, T, C, H, W), got {tuple(x_radar.shape)}"
            )
        x_nwp = self.resolve_nwp(x_radar, x_nwp)
        if x_nwp.dim() != 5:
            raise ValueError(
                f"x_nwp must be (B, T, C, H, W), got {tuple(x_nwp.shape)}"
            )
        b = x_radar.shape[0]

        if x_nwp.shape[0] != b:
            raise ValueError(
                f"batch mismatch: radar {b} vs nwp {x_nwp.shape[0]}"
            )
        if x_radar.shape[1] != self.in_len:
            raise ValueError(
                f"expected {self.in_len} radar frames, got {x_radar.shape[1]}"
            )
        if x_radar.shape[2] != self.in_radar_c:
            raise ValueError(
                f"expected {self.in_radar_c} radar channels, got {x_radar.shape[2]}"
            )
        if x_nwp.shape[2] != self.in_nwp_c:
            raise ValueError(
                f"expected {self.in_nwp_c} nwp channels, got {x_nwp.shape[2]}"
            )
        if x_radar.shape[-2:] != x_nwp.shape[-2:]:
            raise ValueError(
                "radar and nwp must share the spatial grid, got "
                f"{tuple(x_radar.shape[-2:])} vs {tuple(x_nwp.shape[-2:])}"
            )
        return x_nwp

    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        """Flatten a ``(B, T, C, H, W)`` tensor to ``(B*T, C, H, W)``."""
        b, t, c, h, w = x.shape
        return x.reshape(b * t, c, h, w)

    @staticmethod
    def _unflatten_sequence(x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        """Inverse of :meth:`_as_sequence`."""
        return x.reshape(b, t, *x.shape[-3:])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
