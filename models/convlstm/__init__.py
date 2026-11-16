"""ConvLSTM baseline.

A convolutional LSTM nowcasting baseline: the recurrence is carried by a
ConvLSTM cell over the 10-channel radar stack.
"""
from __future__ import annotations

from .model import ConvLSTMModel
from .model import ConvLSTMCell

__all__ = ['ConvLSTMModel', 'ConvLSTMCell']
