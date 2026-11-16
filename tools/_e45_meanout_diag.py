"""Measure the A4/A4b `mean(out)=T` failure DIRECTLY, per frame.

The claim under test: the guarantee fails because the shape field's
unit-mean normalisation `S = F_mm / (mean(F_mm) + 1e-12)` degenerates
when the decomposition field F_mm's spatial mean m is at or below the
guard's scale.  Then mean(S) = m/(m+eps) != 1 and, since
`out = T * S`, mean(out) = T * mean(S) != T.

This measures m, mean(S), T and mean(out) on the real checkpoints, per
frame, and reports the frames that break the guarantee.  CPU only.

Run on the remote:
    /home/wf/.conda/envs/radar_nwp/bin/python tools/_e45_meanout_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/home/wf/radarNWP")
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from radar_nwp_bench.data.dataset import RadarNWPDataset, build_dataloader  # noqa: E402
from radar_nwp_bench.models.registry import build_model  # noqa: E402

EPS = 1e-12
# Every arm-seed of the amount channel, so the localisation is measured
# per seed and not only on seed 0.
ARMS = {
    "A4_s0": ("scaled_structure", dict(
        hid_c=64, num_blocks=4, r_low=8, scale_mode="head", use_scaled=True,
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10)),
    "A4_s1": ("scaled_structure", dict(
        hid_c=64, num_blocks=4, r_low=8, scale_mode="head", use_scaled=True,
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10)),
    "A4_s2": ("scaled_structure", dict(
        hid_c=64, num_blocks=4, r_low=8, scale_mode="head", use_scaled=True,
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10)),
    "A4b_s0": ("scaled_structure_own", dict(
        hid_c=64, num_blocks=4, r_low=8, scale_mode="own", use_scaled=True,
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10)),
    "A4b_s1": ("scaled_structure_own", dict(
        hid_c=64, num_blocks=4, r_low=8, scale_mode="own", use_scaled=True,
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10)),
    "A4b_s2": ("scaled_structure_own", dict(
        hid_c=64, num_blocks=4, r_low=8, scale_mode="own", use_scaled=True,
        in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10)),
}
CKPT_GLOB = "runs/e45/{arm}/train/{model}_*/epoch_00.pt"


def main() -> int:
    ds = RadarNWPDataset(
        "/home/wf/data/radarNWP/BJ_mmap",
        packed=True, patch_size=64, random_crop=False,
        split="test", slide_stride=64,
    )
    loader = build_dataloader(ds, batch_size=8, shuffle=False,
                               num_workers=2, drop_last=False, seed=0)
    for arm, (model_name, kwargs) in ARMS.items():
        cands = sorted(REPO.glob(CKPT_GLOB.format(arm=arm, model=model_name)))
        if not cands:
            print(f"=== {arm}: NO CHECKPOINT FOUND ===")
            continue
        ckpt = cands[0]
        print(f"=== {arm} ({model_name})  {ckpt} ===")
        model = build_model(model_name, **kwargs)
        obj = torch.load(ckpt, map_location="cpu")
        state = obj.get("model", obj.get("state_dict", obj)) if isinstance(obj, dict) else obj
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("  missing:", sorted(missing)[:4], "unexpected:", sorted(unexpected)[:4])
        model.eval()

        rows = []
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if bi >= 400:
                    break
                xr = batch["x_radar"]
                xn = batch["nwp"] if "nwp" in batch else batch["x_nwp"]
                out = model(xr, xn)
                d = model.last_debug
                T = d["T"].detach().double()              # (B*T,)
                S = d["S"].detach().double()              # (B*T,1,H,W)
                F = d["F_mm"].detach().double()            # the decomposition field
                out_mean = out.double().mean(dim=(2, 3, 4)).reshape(-1)
                s_mean = S.mean(dim=(2, 3)).reshape(-1)
                m = F.mean(dim=(2, 3)).reshape(-1)        # the spatial mean of F_mm
                for i in range(T.numel()):
                    rows.append((float(T[i]), float(out_mean[i]),
                                 float(s_mean[i]), float(m[i])))
        T = np.array([r[0] for r in rows])
        O = np.array([r[1] for r in rows])
        S = np.array([r[2] for r in rows])
        M = np.array([r[3] for r in rows])
        err = np.abs(O - T)
        scale = float(np.mean(np.abs(T)))
        rel = float(np.max(err) / scale) if scale > 0 else float("nan")
        print(f"  frames={len(rows)}  rel_err(max_abs_err/scale)={rel:.6f}")
        print(f"  mean(S) over frames: min={S.min():.12f} max={S.max():.12f}")
        print(f"  |mean(S)-1| max    : {np.max(np.abs(S-1)):.6e}   <-- the TRUE guard deviation")
        print(f"  m = mean(F_mm) over frames: min={M.min():.3e} max={M.max():.3e}")
        print(f"  frames with m == 0 exactly: {int(np.sum(M==0))} / {len(M)}")
        # the closed form: out = T*S  =>  |out_mean - T| = T*eps/(m+eps)
        pred = np.abs(T) * EPS / (M + EPS)
        j = int(np.argmax(err))
        print(f"  worst frame: T={T[j]:.6f} out_mean={O[j]:.6f} err={err[j]:.6e}")
        print(f"     its mean(S)={S[j]:.12f}  m={M[j]:.3e}")
        print(f"     predicted |out_mean-T| = T*eps/(m+eps) = {pred[j]:.6e}")
        print(f"     actual    |out_mean-T|                  = {err[j]:.6e}")
        print(f"     ratio actual/predicted = {err[j]/pred[j] if pred[j]>0 else float('nan'):.6f}")
        good = pred > 0
        print(f"  over all frames: max |actual - predicted| = {np.max(np.abs(err-good*0 - np.where(good,pred,err))[good]):.3e}"
              if good.any() else "  (no frames with m>0)")
        # correlation between predicted and actual error
        sel = good & (pred > 0)
        if sel.sum() > 1:
            print(f"  corr(predicted, actual) over {int(sel.sum())} frames with m>0: "
                  f"{np.corrcoef(pred[sel], err[sel])[0,1]:.6f}")
        bad = np.where(err > 1e-5)[0]
        print(f"  frames breaking the guarantee (err > 1e-5): {len(bad)}")
        for i in bad[:8]:
            print(f"     T={T[i]:.6f} out_mean={O[i]:.6f} err={err[i]:.3e} "
                  f"m={M[i]:.3e} mean(S)={S[i]:.9f}")
        print()
    print("=> the guard is broken on the frames whose shape mean collapsed:")
    print("   mean(S) = m/(m+1e-12) != 1  =>  mean(out) = T*mean(S) != T")
    return 0


if __name__ == "__main__":
    sys.exit(main())
