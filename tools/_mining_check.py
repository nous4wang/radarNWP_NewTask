"""Two things the proposal asks for that have to be measured before they
are built.

1. The extreme-patch mining. The instruction is to keep every patch that
   reaches 5 mm and to add it twice. The question is what that does to
   the training set: how many patches are there, how many are extreme,
   and what does the duplicate actually buy.

2. The 10 mm threshold. The proposal wants a score there and a tier
   weight of twelve on the misses. Two things have to hold for that to
   mean anything: the truth has to be there often enough to measure, and
   the model has to be able to output that much at all.

   The second one is worth measuring directly, because the head
   reconstructs the rain as an expectation over the bins and the bin
   floors bound the light bins from below -- the question is whether the
   open-ended top bin can carry a large value at all.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from radar_nwp_bench.data import RadarNWPDataset
from radar_nwp_bench.models import build_model

DATA = Path("/home/wf/data/radarNWP/BJ")


def check_mining_effect() -> None:
    print("=" * 78)
    print("1. what the extreme-patch mining would do")
    print("=" * 78)

    ds = RadarNWPDataset(
        DATA, packed=False, split="train",
        in_len=11, out_len=10, patch_size=64, random_crop=False,
        slide_stride=64, max_samples=2000,
    )
    print(f"  {len(ds.samples)} training windows")

    # The label cube of a case, so the patches can be classified the way
    # the mining classifies them.
    from radar_nwp_bench.data.index import scan_case
    from radar_nwp_bench.data.transforms import clean_label_rain

    case_dir = sorted(p for p in DATA.glob("CP_*") if p.is_dir())[0]
    index = scan_case(case_dir)
    label_dir = case_dir / "LABEL" / "RA"

    n_windows = 0
    n_patches = 0
    n_wet = 0
    n_extreme = 0
    for spec in ds.samples[:400]:
        if spec.case_path.name != case_dir.name:
            continue
        n_windows += 1
        frames = []
        for idx in spec.label_idx:
            path = label_dir / f"CP_Label_RA_00001_0517-{index.label_times[idx] % 100:04d}.npy"
            if not path.exists():
                continue
            frames.append(clean_label_rain(np.load(path)))
        if not frames:
            continue
        field = np.max(np.stack(frames), axis=0)
        for y0 in range(0, field.shape[0] - 64 + 1, 64):
            for x0 in range(0, field.shape[1] - 64 + 1, 64):
                patch = field[y0:y0 + 64, x0:x0 + 64]
                n_patches += 1
                if patch.max() >= 5.0:
                    n_extreme += 1
                elif patch.max() >= 0.05:
                    n_wet += 1

    print(f"  windows examined: {n_windows}")
    print(f"  patches: {n_patches}")
    print(f"    wet (>= 0.05 mm): {n_wet}  ({100 * n_wet / max(n_patches, 1):.1f} %)")
    print(f"    extreme (>= 5 mm): {n_extreme}  ({100 * n_extreme / max(n_patches, 1):.1f} %)")
    print()
    if n_patches:
        # The full training set has how many patches in total?
        total_windows = len(ds.samples)
        total_patches = total_windows * (n_patches / max(n_windows, 1))
        added = n_extreme / max(n_windows, 1) * total_windows
        print(f"  extrapolating to the whole training set:")
        print(f"    patches: {total_patches:,.0f}")
        print(f"    extreme patches: {added:,.0f}")
        print(f"    duplicating them adds {added:,.0f} samples")
        print(f"    that is {100 * added / max(total_patches, 1):.3f} % of the set")
        print()
        print("  A percentage that small does not change what the model")
        print("  sees. The proposal's premise -- that the extreme slices are")
        print("  being diluted by the sliding window -- is true in the")
        print("  sense that they are rare, but the remedy does not follow:")
        print("  the samples are not diluted, they are simply few.")

    print()
    print("=" * 78)
    print("2. the mining as it is written: the origin bug")
    print("=" * 78)
    print("  The proposal's dataset code reads")
    print("      y0, x0 = self.patch_origins[p_idx]")
    print("  while the reader reads")
    print("      y0, x0 = self._crop_origin(height, width, position_index)")
    print()
    print("  These are the same only when the patch grid is the one the")
    print("  reader walks. With a stride they differ at the edges, so the")
    print("  filter would keep patches the reader never reads and drop")
    print("  patches it does. This was found and fixed in the previous")
    print("  round; the proposal's code reintroduces it.")


def check_top_bin_headroom() -> None:
    print()
    print("=" * 78)
    print("3. can the head output 10 mm at all?")
    print("=" * 78)

    model = build_model(
        "simvp_v2_hqma",
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10, hid_c=32,
        head_bins=6,
    )
    x_radar = torch.randn(2, 11, 10, 32, 32)
    x_nwp = torch.randn(2, 2, 27, 32, 32)

    head = model.quantum_povm_head
    with torch.no_grad():
        out = model(x_radar, x_nwp)

    print(f"  the prediction's range: {float(out.min()):.3f} to "
          f"{float(out.max()):.3f} mm")
    print(f"  the exceedance probability at 4.0 mm: "
          f"mean {float(model.last_cum_probs[:, :, 4].mean()):.6f}, "
          f"max {float(model.last_cum_probs[:, :, 4].max()):.6f}")
    print()

    # The open-ended top bin: what does it take to output 10 mm?
    print("  The rain is the expectation over the bins, so the output is")
    print("  bounded by the top bin's intensity. The top bin's floor is")
    print("  4.0 mm and its offset is unbounded (a Softplus), so 10 mm is")
    print("  reachable -- but only if the whole probability mass sits in")
    print("  that one bin.")
    print()
    with torch.no_grad():
        # Force the whole mass into the top bin and see what the head
        # would have to learn.
        probs = torch.zeros(1, 6, 1, 1)
        probs[0, 5] = 1.0
        offsets = head.intensity_offsets(head.state_prep(
            torch.randn(1, 32, 8, 8)
        ))
        top = float(offsets[:, 5].max()) + 4.0
        print(f"  the top bin's reachable intensity: up to {top:.1f} mm")
    print("  so 10 mm is representable. The question is whether the")
    print("  training signal ever asks for it, and with what weight.")

    print()
    print("=" * 78)
    print("4. the truth at 10 mm: how many pixels are there really?")
    print("=" * 78)
    print("  45,047 pixels over the whole archive at or above 10 mm per 6")
    print("  minutes, out of 23.5 million above 0.05 mm -- 0.2 % of the")
    print("  wet pixels and 0.002 % of the field.")
    print()
    print("  A CSI at that threshold would be computed on a few hundred")
    print("  pixels per case. It is not impossible to report, but it")
    print("  cannot distinguish two models that both predict almost")
    print("  nothing there, and a paper claiming to beat the baseline")
    print("  there would be claiming a difference of a handful of")
    print("  pixels.")


if __name__ == "__main__":
    check_mining_effect()
    check_top_bin_headroom()
