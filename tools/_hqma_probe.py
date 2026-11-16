"""A short probe of the hybrid model and the balanced loss.

Two things have to be checked before a full run is worth starting:

  * the losses have to be on a comparable scale. The balanced loss adds a
    soft CSI term, whose value starts near one, to a regression term
    whose value is about a tenth. If the soft term dominates by an order
    of magnitude, the alpha has to come down or the regression is doing
    nothing.

  * the direction has to be right. The whole point of the hierarchical
    head and the balanced loss is to raise the frequency bias: the models
    trained under the standard loss predicted between two and a third of
    the rain that fell. A few epochs are enough to see whether the bias
    moves in the right direction, even if the absolute numbers are far
    from the final ones.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch

from radar_nwp_bench.data import RadarNWPDataset
from radar_nwp_bench.metrics import PrecipitationVerifier
from radar_nwp_bench.models import build_model
from radar_nwp_bench.models.common import BalancedExtremePrecipLoss

DATA = Path("/home/wf/data/radarNWP/BJ_mmap")
EPOCHS = 3


def run(loss_name: str, alpha_csi: float = 0.5) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)

    ds = RadarNWPDataset(
        DATA, packed=True, split="train",
        in_len=11, out_len=10, patch_size=64, random_crop=True,
        slide_stride=64, max_samples=3000,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=True, num_workers=2, drop_last=True
    )

    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=64,
    ).cuda()

    if loss_name == "balanced":
        criterion = BalancedExtremePrecipLoss(alpha_csi=alpha_csi)
    else:
        from radar_nwp_bench.metrics import PrecipitationLoss
        criterion = PrecipitationLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    history = []
    for epoch in range(EPOCHS):
        model.train()
        running = []
        for batch in loader:
            x_radar = batch["x_radar"].cuda(non_blocking=True)
            x_nwp = batch["x_nwp"].cuda(non_blocking=True)
            y = batch["y_rain"].cuda(non_blocking=True)
            pred = model(x_radar, x_nwp)
            if loss_name == "balanced":
                loss = criterion(pred, y)
            else:
                loss, _ = criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running.append(float(loss.detach()))
        history.append(float(np.mean(running)))
        print(f"  [{loss_name}] epoch {epoch}: loss={history[-1]:.4f}")

    # The score that matters: the frequency bias on the training windows.
    model.eval()
    verifier = PrecipitationVerifier(thresholds=(0.5,))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 30:
                break
            pred = model(
                batch["x_radar"].cuda(), batch["x_nwp"].cuda()
            ).clamp_min(0.0)
            verifier.update(pred.cpu().numpy(), batch["y_rain"].numpy())
    report = verifier.report()["thresholds"]["0.5"]
    print(
        f"  [{loss_name}] CSI@0.5={report['csi']:.3f} POD={report['pod']:.3f} "
        f"FAR={report['far']:.3f} BIAS={report['bias']:.3f}"
    )
    return {
        "loss": loss_name,
        "alpha_csi": alpha_csi,
        "history": history,
        "csi": report["csi"],
        "pod": report["pod"],
        "far": report["far"],
        "bias": report["bias"],
    }


def main() -> None:
    print("=" * 74)
    print("the loss scales")
    print("=" * 74)
    # A prediction that is always zero and a prediction that is right:
    # the two terms of the balanced loss measured separately.
    target = torch.full((2, 4, 1, 16, 16), 3.0)
    zero = torch.zeros(2, 4, 1, 16, 16)
    good = torch.full((2, 4, 1, 16, 16), 3.0)

    loss_fn = BalancedExtremePrecipLoss(alpha_csi=0.5)
    for name, pred in (("all dry", zero), ("perfect", good)):
        value = loss_fn(pred, target)
        reg = torch.mean(
            (1.0 + 2.0 * torch.tanh(target / 2.0))
            * torch.nn.functional.smooth_l1_loss(
                torch.log1p(pred.clamp_min(0.0)), torch.log1p(target),
                reduction="none",
            )
        )
        print(f"  {name:10s} total={float(value):.4f}  regression={float(reg):.4f}")

    print()
    print("=" * 74)
    print("three epochs each, on the training split")
    print("=" * 74)
    results = []
    results.append(run("standard"))
    results.append(run("balanced", alpha_csi=0.5))
    results.append(run("balanced", alpha_csi=0.1))

    print()
    print("=" * 74)
    print("summary")
    print("=" * 74)
    print(f"{'loss':>16s}{'alpha':>8s}{'final':>10s}{'CSI@0.5':>10s}"
          f"{'POD':>8s}{'FAR':>8s}{'BIAS':>8s}")
    for r in results:
        print(f"{r['loss']:>16s}{r['alpha_csi']:8.2f}{r['history'][-1]:10.4f}"
              f"{r['csi']:10.3f}{r['pod']:8.3f}{r['far']:8.3f}{r['bias']:8.3f}")

    Path("/tmp/hqma_probe.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
