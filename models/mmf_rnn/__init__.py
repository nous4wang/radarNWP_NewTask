"""MMF-RNN baseline adapter.

Paper: "MMF-RNN: A Multimodal Fusion Model for Precipitation Nowcasting",
IEEE Transactions on Geoscience and Remote Sensing, 2025.

Note: the paper's second modality is a ground-station gauge network.  This
benchmark has no gauges, so the NWP fields take that role; the dual-branch
encoder, the bidirectional cross-attention fusion and the two auxiliary losses
are unchanged.
"""
from __future__ import annotations

from .model import MMFRNNAdapter

__all__ = ["MMFRNNAdapter"]
