"""Validate the fixed pysteps_adv against the real archive.

The synthetic checks prove the units and the sign are right.  This one
answers the question that actually matters for the paper: on a real storm,
does the baseline produce rain of a plausible magnitude, and does it track
the storm rather than running away from it?

The direction test compares the forecast against the truth *and* against
persistence (the same field held still).  If the advection runs the storm
backwards, the forecast decorrelates from the truth faster than persistence
does; if it runs the right way, it holds up at least as well.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from radar_nwp_bench.data import RadarNWPDataset
from radar_nwp_bench.models import build_model

DATA_ROOT = "R:/1radarNWPprediction/data/BJ"

model = build_model("pysteps_adv").eval()

ds = RadarNWPDataset(
    DATA_ROOT, packed=False, in_len=11, out_len=10, patch_size=64,
    random_crop=False, split="test",
)

print(f"test windows: {len(ds)}")
cr_index = list(ds.radar_vars).index("CR")
print(f"CR channel index: {cr_index}\n")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


n = min(len(ds), 60)
pred_max, truth_max = [], []
fc_corr, pers_corr = [], []

for i in range(n):
    item = ds[i]
    x = item["x_radar"]
    y = item["y_rain"]
    with torch.no_grad():
        pred = model(x.unsqueeze(0), torch.zeros(1, 2, 27, 64, 64))
    p = pred[0, :, 0].numpy()
    t = y[:, 0].numpy()
    # persistence: the last observed radar frame, held still
    last = x[-1, cr_index].numpy()

    pred_max.append(float(p.max()))
    truth_max.append(float(t.max()))

    if t.max() < 0.5:          # only windows with real rain can test direction
        continue
    for k in range(min(6, p.shape[0])):
        fc_corr.append(corr(p[k], t[k]))
        pers_corr.append(corr(last, t[k]))

pred_max = np.array(pred_max)
truth_max = np.array(truth_max)
print("=== magnitudes over %d real test windows ===" % n)
print(f"  predicted max rain: median={np.median(pred_max):.3f} mm/6min "
      f"(p90={np.percentile(pred_max, 90):.3f})")
print(f"  truth    max rain: median={np.median(truth_max):.3f} mm/6min "
      f"(p90={np.percentile(truth_max, 90):.3f})")
print(f"  windows whose forecast never reaches 0.1 mm: "
      f"{int((pred_max < 0.1).sum())}/{n}")
print()

fc = np.array(fc_corr)
pe = np.array(pers_corr)
print("=== direction test: correlation with truth, by lead ===")
print(f"  {'lead':<8}{'forecast':>12}{'persistence':>14}{'forecast-pers':>16}")
for k in range(6):
    f_all = fc[k::6]
    p_all = pe[k::6]
    ok = ~np.isnan(f_all) & ~np.isnan(p_all)
    f = f_all[ok]
    p_ = p_all[ok]
    if len(f) == 0:
        continue
    print(f"  T+{k+1:<6}{f.mean():>12.4f}{p_.mean():>14.4f}"
          f"{f.mean() - p_.mean():>+16.4f}")
print()
print(f"  overall: forecast {np.nanmean(fc):.4f} vs persistence {np.nanmean(pe):.4f}")
print("  (a forecast that tracks the storm beats holding it still)")
