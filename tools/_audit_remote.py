"""Audit the archive: which cases really have missing frames, and what are
the true per-variable NWP statistics."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from radar_nwp_bench.data.index import NWP_VARS, scan_case
from radar_nwp_bench.data.transforms import NWP_NORM

DATA = Path("/home/wf/data/radarNWP/BJ")

print("=" * 78)
print("PART 1: which cases actually have a RADAR/LABEL frame mismatch")
print("=" * 78)

mismatched = []
for case_dir in sorted(DATA.glob("CP_*")):
    if not case_dir.is_dir():
        continue
    index = scan_case(case_dir)
    radar_n = len(index.radar_times)
    label_n = len(index.label_times)
    nwp_n = len(index.nwp_times)
    if radar_n != label_n:
        mismatched.append((case_dir.name, radar_n, label_n, nwp_n))

print(f"cases with radar != label frame count: {len(mismatched)}")
for name, r, l, n in mismatched:
    print(f"  {name}  radar={r} label={l} nwp={n}")

# Which of those would the naive split put into train/val/test?
from radar_nwp_bench.data.index import split_cases

all_ids = sorted(p.name for p in DATA.glob("CP_*") if p.is_dir())
parts = split_cases(all_ids)
mismatch_names = {m[0] for m in mismatched}
print()
for split_name, ids in parts.items():
    inside = mismatch_names & set(ids)
    print(f"  {split_name:6s}: {len(ids)} cases, {len(inside)} of them irregular -> {sorted(inside)}")

print()
print("=" * 78)
print("PART 2: the timestamp arithmetic across midnight")
print("=" * 78)


def naive_minutes(t):
    return (t // 100) * 60 + (t % 100)


def proper_minutes(t):
    md, hm = divmod(t, 10000)
    month, day = divmod(md, 100)
    hh, mm = divmod(hm, 100)
    days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    doy = days[month - 1] + day
    return doy * 1440 + hh * 60 + mm


crossings = 0
total_steps = 0
for case_dir in sorted(DATA.glob("CP_*")):
    if not case_dir.is_dir():
        continue
    index = scan_case(case_dir)
    times = list(index.radar_times)
    for a, b in zip(times, times[1:]):
        total_steps += 1
        naive = naive_minutes(b) - naive_minutes(a)
        proper = proper_minutes(b) - proper_minutes(a)
        if naive != proper:
            crossings += 1

print(f"radar steps whose naive vs proper minute difference disagrees: {crossings} / {total_steps}")

# how many 12-minute gaps does the naive rule detect vs the proper rule?
naive_gaps = 0
proper_gaps = 0
for case_dir in sorted(DATA.glob("CP_*")):
    if not case_dir.is_dir():
        continue
    index = scan_case(case_dir)
    times = list(index.radar_times)
    for a, b in zip(times, times[1:]):
        if naive_minutes(b) - naive_minutes(a) == 12:
            naive_gaps += 1
        if proper_minutes(b) - proper_minutes(a) == 12:
            proper_gaps += 1
print(f"12-minute gaps detected by the naive rule: {naive_gaps}")
print(f"12-minute gaps detected by the proper rule: {proper_gaps}")

print()
print("=" * 78)
print("PART 3: the true NWP statistics (fill pixels excluded)")
print("=" * 78)

# Accumulate over a sample of cases -- the whole archive if time allows.
sums = {v: 0.0 for v in NWP_VARS}
sumsq = {v: 0.0 for v in NWP_VARS}
count = {v: 0 for v in NWP_VARS}
fill = {v: 0 for v in NWP_VARS}
seen = set()

case_dirs = sorted(p for p in DATA.glob("CP_*") if p.is_dir())
for case_dir in case_dirs:
    nwp_dir = case_dir / "NWP"
    if not nwp_dir.is_dir():
        continue
    for var_dir in sorted(nwp_dir.iterdir()):
        tag = var_dir.name
        if tag not in sums:
            continue
        seen.add(tag)
        for npy in sorted(var_dir.glob("*.npy")):
            values = np.load(npy).astype(np.float64).ravel()
            valid = np.isfinite(values) & (values > -9999.0)
            fill[tag] += int((~valid).sum())
            v = values[valid]
            sums[tag] += float(v.sum())
            sumsq[tag] += float((v * v).sum())
            count[tag] += int(v.size)

print(f"variables found in the archive: {len(seen)} / {len(NWP_VARS)}")
print(f"{'var':10s} {'measured mean':>14s} {'measured std':>13s} "
      f"{'shipped mean':>13s} {'shipped std':>12s} {'mean/sig':>9s} {'std ratio':>10s}")
print("-" * 100)
bad = []
for v in NWP_VARS:
    if v not in seen or count[v] == 0:
        print(f"{v:10s}  (not found)")
        continue
    mean = sums[v] / count[v]
    var = sumsq[v] / count[v] - mean * mean
    std = float(np.sqrt(max(var, 0.0)))
    ship_mean, ship_std = NWP_NORM.get(v, (float("nan"), float("nan")))
    sigma_off = (ship_mean - mean) / std if std > 0 else float("nan")
    ratio = ship_std / std if std > 0 else float("nan")
    print(f"{v:10s} {mean:14.6g} {std:13.6g} {ship_mean:13.6g} {ship_std:12.6g} "
          f"{sigma_off:9.2f} {ratio:10.2f}")
    if abs(sigma_off) > 2.0 or (ratio < 0.67 or ratio > 1.5):
        bad.append((v, sigma_off, ratio))

print()
print(f"variables more than 2 sigma off in the mean, or with std ratio outside [0.67, 1.5]: {len(bad)}")
for v, s, r in bad:
    print(f"  {v:10s} mean offset {s:+.2f} sigma, std ratio {r:.2f}")
