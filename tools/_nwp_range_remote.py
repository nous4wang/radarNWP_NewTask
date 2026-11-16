"""Report the physical range of every NWP variable.

The radar fields are normalised before they reach the network
(log1p(dBZ)/6 and log1p(VIL)/4). The NWP fields are not, and their
ranges differ by four orders of magnitude between variables -- a
precipitable-water value in millimetres and a CAPE value in J/kg have
nothing in common numerically.

This prints the range of each variable so the normalisation can be
chosen from measurements rather than from a guess.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/wf/radarNWP")

from radar_nwp_bench.data import NWP_VARS, RadarNWPDataset  # noqa: E402
from radar_nwp_bench.data.pack import load_packed_mmap  # noqa: E402

DATA = "/home/wf/data/radarNWP/BJ_mmap"

dataset = RadarNWPDataset(DATA, packed=True, split="test", max_samples=1)
case_dir = dataset.samples[0].case_path
loaded = load_packed_mmap(case_dir)
nwp = loaded["nwp"]              # (T, C, 167, 167)
names = list(loaded["nwp_vars"])
print(f"{'variable':10s} {'min':>12s} {'max':>12s} {'mean':>12s} {'std':>12s}")
print("-" * 62)
for index, name in enumerate(names):
    field = nwp[:, index]
    print(f"{name:10s} {field.min():12.3f} {field.max():12.3f} "
          f"{field.mean():12.3f} {field.std():12.3f}")
