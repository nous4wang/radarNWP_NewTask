"""Is the heavy-rain score bought with false alarms?

A score at a high threshold can be raised in two ways: by predicting
the heavy rain in the right places, or by predicting rain everywhere
until some of it lands. The second is a degenerate strategy and it
shows up in the counts: a large number of false alarms, and a hit rate
that is no better than the base rate of the rain.

Three numbers separate the two:

  * the hit rate among the pixels the model called heavy, which is the
    precision of the heavy-rain forecast;
  * the lift over the base rate -- the model's heavy-rain forecast has
    to beat "predict heavy everywhere" by this factor to be worth
    anything;
  * how the hits and the false alarms are distributed over the field,
    because a model that paints a blob over the whole storm and calls
    it all heavy gets a different spatial pattern from one that puts
    the heavy rain where it belongs.

The comparison is against the physical extrapolation, which is the
reference on this benchmark: it is a physics model with no free
parameters, and it scores 0.101 at 2 mm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/wf/radarNWP")

from radar_nwp_bench.data import RadarNWPDataset
from radar_nwp_bench.models import build_model

DATA = "/home/wf/data/radarNWP/BJ_mmap"
RUNS = Path("/home/wf/radarNWP/runs")
THRESHOLD = 2.0

# The run names are not model names.
MODEL_OF = {
    "scan_C1": "simvp_v2_hqma",
    "scan_C2": "simvp_v2_hqma",
}


def load(name: str, model_name: str | None = None):
    found = None
    for folder in ("hqma_scan3", "hqma_scan2", "hqma", "tgrs"):
        base = RUNS / folder / name
        if base.exists():
            for sub in sorted(base.iterdir()):
                if (sub / "last.pt").exists():
                    found = (sub / "last.pt", sub / "config.json")
                    break
        if found:
            break
    if found is None:
        return None, "no checkpoint"
    ckpt_path, cfg_path = found
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    kwargs = dict(in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10)
    if model_name is None:
        model_name = name
    if model_name == "simvp_v2_hqma":
        kwargs["head_bins"] = cfg.get("head_bins", 6)
        kwargs["level_mode"] = cfg.get("level_mode", "floor")

    weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = {k: v for k, v in weights.items() if isinstance(v, torch.Tensor)}
    model = build_model(model_name, **kwargs)
    try:
        model.load_state_dict(weights)
    except RuntimeError as exc:
        return None, str(exc).splitlines()[0]
    return model.eval().cuda(), None


def main() -> None:
    dataset = RadarNWPDataset(
        DATA, packed=True, split="test", in_len=11, out_len=10,
        patch_size=64, slide_stride=64,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=4,
    )

    for name, label in (
        ("scan_C1", "C1 (HQMA, fixed)"),
        ("pysteps_adv", "pysteps_adv (physical)"),
        ("simvp_v2", "simvp_v2 (backbone)"),
    ):
        model, problem = load(name, MODEL_OF.get(name))
        if problem:
            print(f"{label}: skipped ({problem})")
            continue

        hits = 0
        false_alarms = 0
        truth_heavy = 0
        truth_total = 0
        # How much of the heavy rain the model's heavy-rain area covers.
        # A model that paints a large blob scores a high recall on a low
        # precision; one that finds the core scores a high precision.
        per_window_precision = []
        with torch.no_grad():
            for batch in loader:
                pred = model(batch["x_radar"].cuda(), batch["x_nwp"].cuda())
                pred_heavy = (pred[:, :, 0] >= THRESHOLD)
                truth = batch["y_rain"][:, :, 0].cuda() >= THRESHOLD
                h = int((pred_heavy & truth).sum())
                f = int((pred_heavy & ~truth).sum())
                hits += h
                false_alarms += f
                truth_heavy += int(truth.sum())
                truth_total += truth.numel()
                for b in range(truth.shape[0]):
                    t = truth[b]
                    p = pred_heavy[b]
                    if p.any():
                        per_window_precision.append(
                            float((p & t).sum()) / float(p.sum())
                        )

        base_rate = truth_heavy / max(truth_total, 1)
        precision = hits / max(hits + false_alarms, 1)
        lift = precision / max(base_rate, 1e-12)
        print(f"--- {label} @ {THRESHOLD} mm ---")
        print(f"    heavy rain in the truth: {truth_heavy:,} of "
              f"{truth_total:,} pixels ({100 * base_rate:.3f} %)")
        print(f"    hits {hits:,}   false alarms {false_alarms:,}")
        print(f"    precision of the heavy-rain forecast: {precision:.4f}")
        print(f"    the base rate of heavy rain:          {base_rate:.6f}")
        print(f"    lift over predicting heavy everywhere: {lift:.1f}x")
        if per_window_precision:
            arr = np.array(per_window_precision)
            print(f"    per-window precision: median "
                  f"{np.median(arr):.4f}, 10th percentile "
                  f"{np.percentile(arr, 10):.4f}")
        print()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
