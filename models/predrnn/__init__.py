"""PredRNN / ST-LSTM baseline.

A deep recurrent baseline with a scheduled/predicted cell state and a
spatio-temporal LSTM cell.
"""
from __future__ import annotations

from .model import PredRNNModel
from .model import SpatioTemporalLSTMCell

__all__ = ['PredRNNModel', 'SpatioTemporalLSTMCell']
