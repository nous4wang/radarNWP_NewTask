"""Measure the true per-variable NWP statistics, with the fill pixels
handled the way the pipeline handles them.

``clean_nwp_field`` maps the fill code -9999 to 0.0 before the field is
stored. So a fill pixel and a genuine zero are indistinguishable downstream.
The statistics have to be measured on the *cleaned* field, otherwise the
mean is dragged towards zero by the fill pixels -- and the LFC level, which
has 38 % fill pixels, is the extreme case.

Two candidate conventions are measured:

  filled-in   the fill pixels keep their 0.0, so they are part of the
              distribution the network sees. This is what the pipeline
              actually feeds in, so it is the one the constants must match.
  excluded    the fill pixels are dropped from the statistics and the
              valid pixels are centred on their own mean.

The first is what the network sees; the second is what the physics says.
The difference is reported so the choice is visible.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from radar_nwp_bench.data.index import NWP_VARS
from radar_nwp_bench.data.transforms import NWP_NORM, clean_nwp_field

DATA = Path("/home/wf/data/radarNWP/BJ")

sums = {v: 0.0 for v in NWP_VARS}
sumsq = {v: 0.0 for v in NWP_VARS}
count = {v: 0 for v in NWP_VARS}
valid_sum = {v: 0.0 for v in NWP_VARS}
valid_sumsq = {v: 0.0 for v in NWP_VARS}
valid_count = {v: 0 for v in NWP_VARS}
fill_count = {v: 0 for v in NWP_VARS}
total_count = {v: 0 for v in NWP_VARS}

for case_dir in sorted(p for p in DATA.glob("CP_*") if p.is_dir()):
    nwp_dir = case_dir / "NWP"
    if not nwp_dir.is_dir():
        continue
    for var_dir in sorted(nwp_dir.iterdir()):
        tag = var_dir.name
        if tag not in sums:
            continue
        for npy in sorted(var_dir.glob("*.npy")):
            raw = np.load(npy).astype(np.float64).ravel()
            is_fill = ~np.isfinite(raw) | (raw <= -9999.0)
            cleaned = np.where(is_fill, 0.0, raw)

            sums[tag] += float(cleaned.sum())
            sumsq[tag] += float((cleaned * cleaned).sum())
            count[tag] += int(cleaned.size)

            valid = cleaned[~is_fill]
            valid_sum[tag] += float(valid.sum())
            valid_sumsq[tag] += float((valid * valid).sum())
            valid_count[tag] += int(valid.size)

            fill_count[tag] += int(is_fill.sum())
            total_count[tag] += int(raw.size)

out = {}
print(f"{'var':10s} {'filled mean':>13s} {'filled std':>12s} "
      f"{'valid mean':>13s} {'valid std':>12s} {'fill%':>7s} "
      f"{'shipped mean':>13s} {'shipped std':>12s} {'shipped std ratio':>18s}")
print("-" * 130)

for v in NWP_VARS:
    if count[v] == 0:
        continue
    filled_mean = sums[v] / count[v]
    filled_var = sumsq[v] / count[v] - filled_mean * filled_mean
    filled_std = float(np.sqrt(max(filled_var, 0.0)))

    if valid_count[v]:
        valid_mean = valid_sum[v] / valid_count[v]
        valid_var = valid_sumsq[v] / valid_count[v] - valid_mean * valid_mean
        valid_std = float(np.sqrt(max(valid_var, 0.0)))
    else:
        valid_mean = valid_std = float("nan")

    ship_mean, ship_std = NWP_NORM.get(v, (float("nan"), float("nan")))
    fill_pct = 100.0 * fill_count[v] / max(total_count[v], 1)
    ratio = ship_std / filled_std if filled_std > 0 else float("nan")

    print(f"{v:10s} {filled_mean:13.6g} {filled_std:12.6g} "
          f"{valid_mean:13.6g} {valid_std:12.6g} {fill_pct:7.2f} "
          f"{ship_mean:13.6g} {ship_std:12.6g} {ratio:18.2f}")

    out[v] = {
        "filled_mean": filled_mean,
        "filled_std": filled_std,
        "valid_mean": valid_mean,
        "valid_std": valid_std,
        "fill_fraction": fill_pct / 100.0,
        "shipped_mean": ship_mean,
        "shipped_std": ship_std,
    }

Path("/tmp/nwp_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

print()
print("=" * 130)
print("The constants the pipeline should use: measured on the cleaned field,")
print("which is exactly what the network sees (fill -> 0.0).")
print("=" * 130)
print("NWP_NORM = {")
for v in NWP_VARS:
    if v not in out:
        continue
    d = out[v]
    # A standard deviation of zero would divide by zero.
    std = max(d["filled_std"], 1e-12)
    print(f'    "{v}": ({d["filled_mean"]:.6g}, {std:.6g}),')
print("}")
