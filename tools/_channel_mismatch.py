"""How much of the heavy-rain collapse was the channel misalignment?

The claim is that the run was supervising P(R >= 1.0) against the truth's
"at or above 2 mm" indicator and P(R >= 2.0) against "at or above
4 mm", because the probability channels were matched to the thresholds
by position.

This measures the claim directly rather than by reasoning about it. For
a trained model, it asks two questions about each tier:

  * what does the head's channel actually answer, and
  * how well does that channel separate the truth's indicator for the
    threshold the loss was supposed to use, compared with the threshold
    the channel really answers.

The separation is the area under the curve, computed over the pixels of
the test split. A channel that answers P(R >= 1.0) separates the
truth's 1.0 mm indicator well and its 2.0 mm indicator badly. If the
loss was told to treat it as the 2.0 mm channel, it was optimising the
wrong separation, and the amount by which it is wrong is the size of
the error.
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
THRESHOLDS = (0.1, 0.5, 1.0, 2.0, 4.0)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """The area under the ROC curve, by the rank method."""
    positives = labels > 0.5
    n_pos = int(positives.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, labels.size + 1)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load(name: str):
    found = None
    for folder in ("hqma_scan2", "hqma"):
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
    weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = {k: v for k, v in weights.items() if isinstance(v, torch.Tensor)}
    model = build_model(
        "simvp_v2_hqma", in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10,
        head_bins=cfg.get("head_bins", 6),
    )
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

    model, problem = load("scan_B2")
    if problem:
        print(f"skipped: {problem}")
        return

    # The channels of the head, each with the truth's indicator for
    # every threshold.
    channel_scores: list[np.ndarray] = [[] for _ in THRESHOLDS]
    truth_indicators: list[np.ndarray] = [[] for _ in THRESHOLDS]
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 12:
                break
            model(batch["x_radar"].cuda(), batch["x_nwp"].cuda())
            cum = model.quantum_povm_head.last_cum_probs
            if cum is None:
                continue
            cum = cum.reshape(-1, cum.shape[-3], *cum.shape[-2:])
            truth = batch["y_rain"].reshape(-1, *batch["y_rain"].shape[-2:])
            for c in range(cum.shape[1]):
                channel_scores[c].append(cum[:, c].flatten().cpu().numpy())
            for t, thr in enumerate(THRESHOLDS):
                truth_indicators[t].append(
                    (truth >= thr).flatten().cpu().numpy().astype(np.float32)
                )

    scores = [np.concatenate(s) for s in channel_scores]
    indicators = [np.concatenate(t) for t in truth_indicators]

    print("The head's channels, and how well each separates the truth's")
    print("indicator for each threshold. The channel that answers a")
    print("threshold is the one with the highest area under the curve")
    print("for that threshold's indicator.\n")

    header = "            " + "".join(f"{thr:>9.1f}" for thr in THRESHOLDS)
    print("AUC of channel 0 against each truth indicator:")
    print(header)
    for c in range(len(THRESHOLDS)):
        row = "".join(f"{auc(scores[c], indicators[t]):9.3f}"
                       for t in range(len(THRESHOLDS)))
        print(f"  channel {c}: {row}")

    print()
    print("For each truth threshold, the channel that separates it best:")
    for t, thr in enumerate(THRESHOLDS):
        best = max(
            range(len(THRESHOLDS)),
            key=lambda c: auc(scores[c], indicators[t]),
        )
        by_position = t          # what the position-based match would pick
        print(f"  {thr:4.1f} mm -> channel {best}"
              + ("   (position matching agrees)"
                 if best == by_position else
                 f"   (position matching would pick channel {by_position})"))

    print()
    print("The last column is the error: a channel that separates the")
    print("1.0 mm indicator but not the 2.0 mm one cannot be used to")
    print("supervise the 2.0 mm tier.")


if __name__ == "__main__":
    main()
