"""Does the tendency field carry the physics, or does it just mark where
the atmosphere is interesting?

The first run on the real archive gave a *positive* correlation between
the tendency magnitude and the CAPE consumption, where the hypothesis
predicts a negative one. Two explanations fit that number equally well:

  A. the plugin learned the NWP field's static structure -- the humidity
     gradient, the echo shape -- and the tendency magnitude simply marks
     where the atmosphere is most interesting, carrying no information
     about the energy budget;

  B. the correlation is carried by the large-scale pattern. Both fields
     are smooth, and a shared synoptic structure produces a coefficient
     that means nothing on its own.

The two are separable. If A is right, the correlation survives when the
large-scale structure is removed from both fields. If B is right, it
collapses.

The test: high-pass both fields by subtracting a wide box mean (the
synoptic scale), and correlate the residuals. Also correlate the tendency
magnitude against the CAPE *gradient* rather than against the CAPE
itself, because a tendency is a rate and should track a gradient.

Finally, compare against a model that has no plugin: if the plain
backbone's feature field correlates with the CAPE consumption just as
well, the correlation is not coming from the Koopman machinery at all.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from radar_nwp_bench.data import RadarNWPDataset
from radar_nwp_bench.data.index import NWP_VARS
from radar_nwp_bench.models import build_model
from radar_nwp_bench.models.common import (
    KoopmanUnitaryFlow,
    QuantumInspiredTendencyCoupler,
)

DATA = Path("/home/wf/data/radarNWP/BJ_mmap")

CAPE_INDEX = list(NWP_VARS).index("CAPE")
CR_INDEX = 6


def highpass(field: np.ndarray, box: int) -> np.ndarray:
    """Remove the large-scale structure: subtract a wide box mean."""
    # A separable box blur, computed with a cumulative sum so the cost
    # does not depend on the box size.
    kernel = np.ones(box) / box

    def blur(row: np.ndarray) -> np.ndarray:
        padded = np.pad(row, ((box // 2, box // 2), (0, 0)), mode="edge")
        cumsum = np.cumsum(padded, axis=0)
        cumsum = np.vstack([np.zeros((1, cumsum.shape[1])), cumsum])
        return (
            cumsum[box:, :] - cumsum[:-box, :]
        )[: row.shape[0]] / box

    return field - blur(blur(field))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def main() -> None:
    ds = RadarNWPDataset(
        DATA, packed=True, split="test",
        in_len=11, out_len=10, patch_size=128, random_crop=False,
        slide_stride=128, max_samples=40,
    )
    print(f"loaded {len(ds.samples)} test windows")

    # The NWP and the radar, straight from the dataset (already normalised
    # for the NWP, and the radar is log-transformed).
    x_nwp = ds[0]["x_nwp"].numpy()
    x_radar = ds[0]["x_radar"].numpy()
    cape = x_nwp[:, CAPE_INDEX]            # (2, H, W) normalised
    consumption = cape[-1] - cape[0]      # normalised difference

    print()
    print("=" * 78)
    print("PART 1: the correlation between the two NWP slices and the radar")
    print("=" * 78)
    # The CAPE consumption should sit where the radar shows convection.
    # If it does not, the NWP's own CAPE field cannot be the explanation
    # for anything the model does.
    cr = x_radar[-1, CR_INDEX]
    print(f"  corr(CAPE consumption, composite reflectivity) = "
          f"{corr(consumption, cr):+.4f}")
    print(f"  corr(CAPE slice 0, slice -1)                    = "
          f"{corr(cape[0], cape[-1]):+.4f}")

    print()
    print("=" * 78)
    print("PART 2: what does the tendency field actually correlate with?")
    print("=" * 78)
    import torch

    torch.manual_seed(0)
    coupler = QuantumInspiredTendencyCoupler(in_nwp_c=27, radar_feat_dim=64, out_len=10)
    coupler.eval()

    captured = {}

    def hook(_module, _inputs, output):
        captured["tendency"] = output.detach()

    handle = coupler.koopman_flow.register_forward_hook(hook)

    nwp = torch.from_numpy(x_nwp).unsqueeze(0)
    with torch.no_grad():
        coupler(torch.zeros(1, 10, 64, 128, 128), nwp)
    handle.remove()

    d_psi = captured["tendency"].numpy()[0]       # (C, H, W)
    magnitude = np.linalg.norm(d_psi, axis=0)     # (H, W)

    print(f"  tendency magnitude: mean={magnitude.mean():.4f} "
          f"std={magnitude.std():.4f}")

    # The correlations the paper would want.
    r_raw = corr(magnitude, consumption)
    print(f"  corr(tendency magnitude, CAPE consumption)          = {r_raw:+.4f}")

    for box in (32, 64, 128):
        hp_mag = highpass(magnitude, box)
        hp_con = highpass(consumption, box)
        print(f"  high-pass (box {box:3d} px): corr = "
              f"{corr(hp_mag, hp_con):+.4f}")

    print()
    print("  against the CAPE gradient rather than the CAPE itself:")
    gy, gx = np.gradient(consumption)
    gradient_magnitude = np.hypot(gy, gx)
    print(f"    corr(tendency magnitude, |grad CAPE consumption|) = "
          f"{corr(magnitude, gradient_magnitude):+.4f}")

    print()
    print("=" * 78)
    print("PART 3: is the correlation coming from the plugin at all?")
    print("=" * 78)
    # A plain feature field of the same shape, with no Koopman machinery.
    # If it correlates just as well, the correlation is about the data, not
    # about the mechanism.
    plain = coupler.state_proj(nwp[:, 0]).detach().numpy()[0]
    plain_magnitude = np.linalg.norm(plain, axis=0)
    print(f"  corr(plain projection magnitude, CAPE consumption) = "
          f"{corr(plain_magnitude, consumption):+.4f}")
    print(f"  corr(tendency magnitude, plain projection magnitude) = "
          f"{corr(magnitude, plain_magnitude):+.4f}")
    print("  (a high correlation between the two means the tendency is a"
          " transform of the projection, not an independent signal)")

    print()
    print("=" * 78)
    print("PART 4: is the tendency just the input, rescaled?")
    print("=" * 78)
    # If the tendency magnitude is a monotone function of the CAPE field
    # itself, the plugin is not extracting a tendency -- it is reading the
    # instantaneous state.
    print(f"  corr(tendency magnitude, CAPE slice 0)  = "
          f"{corr(magnitude, cape[0]):+.4f}")
    print(f"  corr(tendency magnitude, |CAPE slice 0|) = "
          f"{corr(magnitude, np.abs(cape[0])):+.4f}")
    print(f"  corr(tendency magnitude, radar CR)      = "
          f"{corr(magnitude, cr):+.4f}")


if __name__ == "__main__":
    main()
