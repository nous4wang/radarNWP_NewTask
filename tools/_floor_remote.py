"""Diagnose why a baseline's loss does not move.

Two questions:

1. What is the *floor*? The loss of a constant predictor that always says
   "no rain" is the score a model gets for predicting nothing. A model
   whose training loss sits at that floor has learned nothing, and its
   verification score is zero for a reason that has nothing to do with the
   weather.

2. Where is the gradient going? A layer that is dead -- a hard ReLU over
   an all-negative field, for instance -- shows up as a parameter that
   never moves.
"""
import sys

import torch

sys.path.insert(0, "/home/wf/radarNWP")

from radar_nwp_bench.data import RadarNWPDataset  # noqa: E402
from radar_nwp_bench.metrics import PrecipitationLoss  # noqa: E402

DATA = "/home/wf/data/radarNWP/BJ_mmap"

dataset = RadarNWPDataset(
    DATA, packed=True, in_len=11, out_len=10, patch_size=64,
    split="train", max_samples=64, random_crop=False,
)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
batch = next(iter(loader))
truth = batch["y_rain"]

criterion = PrecipitationLoss()

# The floor: predict "no rain" everywhere.
zero = torch.zeros_like(truth)
loss_zero, _ = criterion(zero, truth)
print(f"loss of a constant 'no rain' forecast : {loss_zero.item():.4f}")

# The floor: predict the mean of the training set.
mean = truth.mean().expand_as(truth)
loss_mean, _ = criterion(mean, truth)
print(f"loss of a constant 'mean rain' forecast: {loss_mean.item():.4f}")

# What a perfect forecast would score.
perfect = truth.clone()
loss_perfect, _ = criterion(perfect, truth)
print(f"loss of a perfect forecast             : {loss_perfect.item():.4f}")
print()
print("A model that sits at the 'no rain' floor has learned nothing.")
print(f"Observed training losses: mocast 0.1410, convlstm 0.1502,")
print(f"                        fusioncast 0.1255, mmf_rnn 0.1054")
