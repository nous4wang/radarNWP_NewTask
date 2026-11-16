#!/usr/bin/env bash
# ZERO-COMPUTE mechanism probe for the Gemini D4 SlicedWassersteinPrecipLoss.
# No dataset, no model, no GPU.  Pure numpy on a synthetic 32x32 rain blob.
#
# Question (Gate 1, MASTER 4.20.1): is the doc's "SWD" a spatial-displacement
# penalty, or is it dominated by total-mass bias (=> same family as 4.17/4.18)?
cd /home/wf/radarNWP || exit 1
/home/wf/.conda/envs/radar_nwp/bin/python - <<'PY'
import numpy as np

H = W = 32
K = 16

yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
coords = np.stack([xx.flatten(), yy.flatten()], axis=0)          # (2, H*W)
theta = np.linspace(0.0, np.pi, K)
directions = np.stack([np.cos(theta), np.sin(theta)], axis=0)     # (2, K)
proj = directions.T @ coords                                       # (K, H*W)
sort_idx = [np.argsort(proj[k]) for k in range(K)]                 # CONSTANT permutation


def blob(cx, cy, amp=8.0, sigma=2.2):
    y, x = np.mgrid[0:H, 0:W]
    return amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def doc_swd(pred, tgt):
    """Exactly the Gemini code: cumsum of RAW mass along fixed sorted coords, L1, /K."""
    p = pred.reshape(1, -1)
    t = tgt.reshape(1, -1)
    s = 0.0
    for k in range(K):
        i = sort_idx[k]
        p_cdf = np.cumsum(p[:, i], axis=-1)
        t_cdf = np.cumsum(t[:, i], axis=-1)
        s += np.mean(np.abs(p_cdf - t_cdf))
    return s / K


def norm_swd(pred, tgt):
    """Proper-ish W1: mass-normalised CDF, weighted by projection spacing ds."""
    p = pred.reshape(1, -1)
    t = tgt.reshape(1, -1)
    p = p / (p.sum() + 1e-12)
    t = t / (t.sum() + 1e-12)
    s = 0.0
    for k in range(K):
        i = sort_idx[k]
        ds = np.diff(proj[k][i], prepend=proj[k][i][0])
        p_cdf = np.cumsum(p[:, i], axis=-1)
        t_cdf = np.cumsum(t[:, i], axis=-1)
        s += np.sum(np.abs(p_cdf - t_cdf) * ds[None, :])
    return s / K


def log_huber(pred, tgt):
    """The regression base used by BOTH the doc's SWD and BalancedExtremePrecipLoss."""
    p = np.log1p(np.maximum(pred, 0))
    t = np.log1p(np.maximum(tgt, 0))
    d = p - t
    return np.mean(np.where(np.abs(d) < 1, 0.5 * d ** 2, np.abs(d) - 0.5))


def soft_fss_like(pred, tgt):
    """Neighbourhood-smoothed MSE, the 4.16.1 SoftFSS family."""
    def blur(a):
        k = np.ones((5, 5)) / 25.0
        out = np.zeros_like(a)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                out += k[dy + 2, dx + 2] * np.roll(np.roll(a, dy, 0), dx, 1)
        return out
    return np.mean((blur(pred) - blur(tgt)) ** 2)


tgt = blob(16, 16)
cases = {
    "identical (control)":      tgt.copy(),
    "shift 1 px (pure disp)":   blob(17, 16),
    "shift 2 px (pure disp)":   blob(18, 16),
    "shift 4 px (pure disp)":   blob(20, 16),
    "amplitude x1.5 (pure amp)": blob(16, 16, amp=12.0),
    "amplitude x0.5 (pure amp)": blob(16, 16, amp=4.0),
    "amplitude x1.1":           blob(16, 16, amp=8.8),
}

print("=" * 96)
print("D4 mechanism probe: synthetic 32x32 Gaussian rain blob (peak 8 mm/6min), n=1")
print("target total mass = %.2f" % tgt.sum())
print("=" * 96)
print("%-28s %12s %12s %12s %12s" % ("case", "doc_swd", "norm_swd", "log_huber", "softFSSlike"))
print("-" * 96)
rows = {}
for name, pred in cases.items():
    d, n, lh, sf = doc_swd(pred, tgt), norm_swd(pred, tgt), log_huber(pred, tgt), soft_fss_like(pred, tgt)
    rows[name] = (d, n, lh, sf)
    print("%-28s %12.4f %12.4f %12.4f %12.4f" % (name, d, n, lh, sf))
print("-" * 96)

# Decisive comparison: a 2-px displacement vs a +10 % amplitude error.
d_shift2 = rows["shift 2 px (pure disp)"][0]
d_amp11 = rows["amplitude x1.1"][0]
d_amp15 = rows["amplitude x1.5 (pure amp)"][0]
print("doc_swd(shift 2px)          = %.4f" % d_shift2)
print("doc_swd(amp x1.1, +10%% bias)= %.4f" % d_amp11)
print("doc_swd(amp x1.5, +50%% bias)= %.4f" % d_amp15)
print("ratio amp(x1.1)/shift(2px)  = %.2f" % (d_amp11 / max(d_shift2, 1e-12)))
print("ratio amp(x1.5)/shift(2px)  = %.2f" % (d_amp15 / max(d_shift2, 1e-12)))
print()
print("VERDICT: doc_swd is mass-normalised? NO -> raw cumsum dominated by total-mass bias.")
print("         A modest amplitude bias outranks a genuine 2-px spatial displacement" if d_amp11 > d_shift2 else
      "         Spatial displacement dominates (amplitude bias is secondary).")
print("=" * 96)
PY
