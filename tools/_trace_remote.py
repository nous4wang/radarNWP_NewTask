"""Trace where a baseline's output magnitude is lost.

The question: the model predicts 0-0.35 mm where the truth reaches
2.5 mm. Is the *state* it reads already too small (the recurrent update
damps it), or is the state fine and the *head* too small (it saturates)?

The script prints the magnitude at each stage of the forward pass.
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
    split="test", max_samples=8, random_crop=False,
)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
batch = next(iter(loader))
truth = batch["y_rain"]

name = sys.argv[1] if len(sys.argv) > 1 else "mocast"
print("=" * 72)
print(f"{name}: where is the magnitude lost?")
print("=" * 72)

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
if candidates:
    model.load_state_dict(torch.load(candidates[-1], map_location="cpu"),
                          strict=False)
model.eval()

x_radar = batch["x_radar"]
x_nwp = batch["x_nwp"]
print(f"radar input : min={x_radar.min():.4f} max={x_radar.max():.4f} "
      f"mean={x_radar.mean():.4f}")
print(f"nwp input   : min={x_nwp.min():.4f} max={x_nwp.max():.4f} "
      f"mean={x_nwp.mean():.4f}")
print(f"truth (mm)  : min={truth.min():.4f} max={truth.max():.4f} "
      f"mean={truth.mean():.4f}")
print(f"log1p(truth): max={torch.log1p(truth.clamp_min(0)).max():.4f}")

# Hook the head's last convolution to see what it produces.
if hasattr(model, "rain_head"):
    head = model.rain_head
    conv = head.net[-1]
    captured = {}

    def hook(_module, inputs, output):
        captured["logit"] = output.detach()

    conv.register_forward_hook(hook)

with torch.no_grad():
    out = model(x_radar, x_nwp)

if "logit" in captured:
    logit = captured["logit"]
    print(f"head logit  : min={logit.min():.4f} max={logit.max():.4f} "
          f"mean={logit.mean():.4f} std={logit.std():.4f}")
    print(f"  -> the head would need a logit around {torch.log1p(truth.clamp_min(0)).max():.3f} "
          f"to reach the heaviest truth")
else:
    print("no rain_head on this model")

print(f"prediction  : min={out.min():.4f} max={out.max():.4f} "
      f"mean={out.mean():.4f} std={out.std():.4f}")
