"""Report the output statistics of a trained baseline on the server.

The question this answers: is the model predicting a *constant* field, and
if so, which layer is the bottleneck?
"""
import sys

import torch

sys.path.insert(0, "/home/wf/radarNWP")

from radar_nwp_bench.config import load_config, model_kwargs  # noqa: E402
from radar_nwp_bench.data import RadarNWPDataset  # noqa: E402
from radar_nwp_bench.models import build_model  # noqa: E402

DATA = "/home/wf/data/radarNWP/BJ_mmap"
RUNS = "/home/wf/radarNWP/runs/sweep"

dataset = RadarNWPDataset(
    DATA, packed=True, in_len=11, out_len=10, patch_size=64,
    split="test", max_samples=4, random_crop=False,
)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))

for name in sys.argv[1:]:
    print("=" * 70)
    print(name)
    print("=" * 70)

    try:
        config = load_config(name)
    except FileNotFoundError:
        config = {}
    kwargs = model_kwargs(config, shapes={
        "in_radar_c": len(dataset.radar_vars),
        "in_nwp_c": len(dataset.nwp_vars),
        "in_len": 11,
        "out_len": 10,
    })
    model = build_model(name, **kwargs)
    model.nwp_time_len = dataset.nwp_frames

    import glob
    import os
    candidates = sorted(glob.glob(f"{RUNS}/{name}/*/last.pt"), key=os.path.getmtime)
    if not candidates:
        print("  no checkpoint")
        continue
    state = torch.load(candidates[-1], map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  MISSING keys: {missing[:5]}")
    if unexpected:
        print(f"  UNEXPECTED keys: {unexpected[:5]}")
    model.eval()

    x_radar = batch["x_radar"]
    x_nwp = batch["x_nwp"]
    with torch.no_grad():
        out = model(x_radar, x_nwp)

    truth = batch["y_rain"]
    print(f"  prediction: shape={tuple(out.shape)} "
          f"min={out.min():.4f} max={out.max():.4f} "
          f"mean={out.mean():.4f} std={out.std():.4f}")
    print(f"  truth     : min={truth.min():.4f} max={truth.max():.4f} "
          f"mean={truth.mean():.4f} std={truth.std():.4f}")
    print(f"  unique prediction values (first frame): "
          f"{out[0, 0].unique().numel()} of {out[0, 0].numel()}")
    print(f"  fraction of prediction above 0.5mm: "
          f"{(out >= 0.5).float().mean():.4f}")
    print(f"  fraction of truth above 0.5mm:     "
          f"{(truth >= 0.5).float().mean():.4f}")
