"""Diagnose WHY the gate arms' maxratio/tail read-outs are NaN (E45).

Read-only.  CPU only, sixty batches per arm-seed.  It prints, for each
intermediate the gate read-out is built from, the log-millimetre range and
the millimetre range, so the empty-list branch of E44's ``_max_ratio`` /
``_tail_ratio`` can be attributed to a specific tensor instead of being
guessed at.  For every arm-seed it also prints what the probe REPORTS
(``np.median`` over the per-batch values) against what it SHOULD report
(``np.nanmedian`` over the same list) -- the difference between the two is
the whole defect: one dry batch makes E44's ``_max_ratio`` return NaN, and
``np.median`` is not NaN-aware, so a single NaN poisons the whole read-out.

Sweeps all six gate arm-seeds (A1/A2 x s0/s1/s2), because the read-out is
empty in every one of them.

Run on the remote with:
    /home/wf/.conda/envs/radar_nwp/bin/python tools/_e45_maxratio_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/home/wf/radarNWP")
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from radar_nwp_bench.data.dataset import RadarNWPDataset, build_dataloader  # noqa: E402
from radar_nwp_bench.models.registry import build_model  # noqa: E402
from radar_nwp_bench.models.common import floor_at_zero  # noqa: E402

CKPT_GLOB = "runs/e45/{arm}/train/{model}_*/epoch_00.pt"
# Every gate arm-seed: the maxratio read-out is empty for A1 and A2 in
# all three seeds, so the localisation is measured per seed.
ARMS = {
    "A1_s0": ("amp_monotone_gate", dict(monotone_gate=True)),
    "A1_s1": ("amp_monotone_gate", dict(monotone_gate=True)),
    "A1_s2": ("amp_monotone_gate", dict(monotone_gate=True)),
    "A2_s0": ("amp_hard_support", dict(hard_gate=True)),
    "A2_s1": ("amp_hard_support", dict(hard_gate=True)),
    "A2_s2": ("amp_hard_support", dict(hard_gate=True)),
}
KWARGS = dict(
    hid_c=64, num_blocks=4, r_low=8, gate_temp=1.0,
    low_rank=True,
    in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10,
)


def mm(x):
    """The project's own log-mm -> mm decode (E44's _mm)."""
    return floor_at_zero(torch.expm1(x.clamp(-10.0, 6.0)))


def main() -> int:
    ds = RadarNWPDataset(
        "/home/wf/data/radarNWP/BJ_mmap",
        packed=True, patch_size=64, random_crop=False,
        split="test", slide_stride=64,
    )
    loader = build_dataloader(ds, batch_size=8, shuffle=False,
                               num_workers=2, drop_last=False, seed=0)

    for arm, (model_name, extra) in ARMS.items():
        print(f"=== {arm} ({model_name}) ===")
        kwargs = dict(KWARGS)
        kwargs.update(extra)
        cands = sorted(REPO.glob(CKPT_GLOB.format(arm=arm, model=model_name)))
        if not cands:
            print(f"  NO CHECKPOINT FOUND for {arm}")
            print()
            continue
        ckpt = cands[0]
        print("  checkpoint:", ckpt)
        model = build_model(model_name, **kwargs)
        obj = torch.load(ckpt, map_location="cpu")
        state = obj.get("model", obj.get("state_dict", obj)) if isinstance(obj, dict) else obj
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("  checkpoint missing:", sorted(missing)[:6],
              "unexpected:", sorted(unexpected)[:6])
        model.eval()

        n_frames = 0
        usable = 0
        worst = None
        per_batch_maxratio = []          # what the probe records, per batch
        per_batch_maxratio_nanmedian = []
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if bi >= 60:
                    break
                model(batch["x_radar"],
                      batch["nwp"] if "nwp" in batch else batch["x_nwp"])
                d = model.last_debug
                amp, mask, low, resid = d["amp"], d["mask"], d["low"], d["resid"]
                as_trained = low + mask * resid
                full_bypass = low + resid
                n = amp.shape[0]
                n_frames += n
                if bi < 3:
                    for t in ("amp", "low", "resid", "as_trained", "full_bypass"):
                        z = {"amp": amp, "low": low, "resid": resid,
                             "as_trained": as_trained,
                             "full_bypass": full_bypass}[t]
                        print(f"  b{bi} {t:14s} log-mm "
                              f"[{float(z.min()):+.4f}, {float(z.max()):+.4f}]"
                              f"  mm [{float(mm(z).min()):.4f}, "
                              f"{float(mm(z).max()):.4f}]")
                # --- reproduce E44's _max_ratio exactly, per batch ---
                ratios = []
                fa_t = mm(as_trained).reshape(n, -1).float()
                fb_t = mm(full_bypass).reshape(n, -1).float()
                for a, b in zip(fa_t, fb_t):
                    if a.numel() == 0 or b.numel() == 0:
                        continue
                    fa, fb = a[torch.isfinite(a)], b[torch.isfinite(b)]
                    if fa.numel() == 0 or fb.numel() == 0:
                        continue
                    mb = float(fb.max())
                    if mb <= 0.0:
                        continue
                    ratios.append(float(fa.max()) / mb)
                batch_val = float("nan") if not ratios else float(
                    __import__("numpy").median(ratios))
                per_batch_maxratio.append(batch_val)
                if ratios:
                    per_batch_maxratio_nanmedian.append(batch_val)
                    usable += len(ratios)
                    for r in ratios:
                        if worst is None or r < worst:
                            worst = r
        import numpy as np
        print()
        print("frames seen                  :", n_frames)
        print("batches                      :", len(per_batch_maxratio))
        print("batches with NO usable frame :",
              sum(1 for v in per_batch_maxratio if v != v))
        print("batches with a value         :", len(per_batch_maxratio_nanmedian))
        print("usable frames overall        :", usable)
        print("worst ratio over usable frames:", worst)
        print()
        print("WHAT THE PROBE REPORTS  np.median(all per-batch values) =",
              float(np.median(np.asarray(per_batch_maxratio, dtype=np.float64))))
        print("WHAT IT SHOULD REPORT  np.nanmedian(same list)          =",
              float(np.nanmedian(np.asarray(per_batch_maxratio, dtype=np.float64))))
        print()
        print("=> np.median is NOT nan-aware: one dry batch (NaN) poisons "
              "the whole read-out")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
