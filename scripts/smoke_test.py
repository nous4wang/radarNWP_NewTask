#!/usr/bin/env python3
"""End-to-end smoke test for the radar-NWP benchmark.

Exercises the whole chain against the real archive:

    scan -> pack -> index -> sample -> model forward -> loss -> metrics

and asserts the properties the rewrite was supposed to guarantee, in particular
that NWP is aligned by valid time and carries a *forecast* slice.

Usage:
    python scripts/smoke_test.py --data-root R:/1radarNWPprediction/data/BJ
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_nwp_bench.data import (
    NWPAlignment,
    RadarNWPDataset,
    build_dataloader,
    floor_hour,
    minutes_between,
    scan_case,
)
from radar_nwp_bench.data.index import discover_cases
from radar_nwp_bench.metrics import PrecipitationLoss, PrecipitationVerifier
from radar_nwp_bench.models import build_model, list_models

PASS = "PASS"
FAIL = "FAIL"


class Checker:
    """Collects check results and prints a verdict table."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, PASS if ok else FAIL, detail))
        print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
        return ok

    @property
    def failed(self) -> list[str]:
        return [r[0] for r in self.rows if r[1] == FAIL]

    def summary(self) -> str:
        n_fail = len(self.failed)
        n = len(self.rows)
        return f"{n - n_fail}/{n} checks passed" + (
            "" if n_fail == 0 else f" -- FAILED: {', '.join(self.failed)}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--in-len", type=int, default=11)
    ap.add_argument("--out-len", type=int, default=10)
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of baselines to exercise")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--keep-pack", default=None,
                    help="write the packed containers here instead of a temp dir")
    args = ap.parse_args()

    ck = Checker()
    models_to_test = args.models or list_models()

    print("=" * 74)
    print("radar-nwp-bench :: smoke test")
    print("=" * 74)

    # ---------------------------------------------------------------- scan
    print("\n[1] archive scan")
    cases = discover_cases(args.data_root)
    ck.check("cases discovered", len(cases) > 0, f"{len(cases)} cases")

    idx = scan_case(cases[0])
    ck.check("radar time axis", len(idx.radar_times) > 0, f"{len(idx.radar_times)} frames")
    ck.check("label time axis", len(idx.label_times) > 0, f"{len(idx.label_times)} frames")
    ck.check("nwp time axis", len(idx.nwp_times) > 0, f"{len(idx.nwp_times)} frames")
    ck.check(
        "nwp is hourly",
        all(
            (b % 100) == 0 and minutes_between(b, a) == 60
            for a, b in zip(idx.nwp_times, idx.nwp_times[1:])
        ),
        "all NWP stamps on whole hours, one hour apart",
    )
    ck.check(
        "nwp window brackets radar window",
        idx.nwp_times[0] <= idx.radar_times[0] and idx.nwp_times[-1] >= idx.radar_times[-1],
        f"nwp {idx.nwp_times[0]}..{idx.nwp_times[-1]} vs radar "
        f"{idx.radar_times[0]}..{idx.radar_times[-1]}",
    )

    # ------------------------------------------------------------ alignment
    print("\n[2] NWP valid-time alignment")
    forecast_ds = RadarNWPDataset(
        args.data_root, in_len=args.in_len, out_len=args.out_len,
        patch_size=args.patch_size, alignment=NWPAlignment.FORECAST,
        random_crop=False,
    )
    bracket_ds = RadarNWPDataset(
        args.data_root, in_len=args.in_len, out_len=args.out_len,
        patch_size=args.patch_size, alignment=NWPAlignment.BRACKET,
        random_crop=False, require_forecast_nwp=False,
    )
    ck.check("forecast alignment produces samples", len(forecast_ds) > 0,
             f"{len(forecast_ds)} samples")
    ck.check("bracket alignment produces samples", len(bracket_ds) > 0,
             f"{len(bracket_ds)} samples")

    leads = [lead for s in forecast_ds.samples for lead in s.nwp_lead_minutes]
    ck.check(
        "every forecast sample carries a future NWP slice",
        all(s.has_forecast_nwp for s in forecast_ds.samples),
        f"leads observed: {sorted(set(leads))} minutes from issue time",
    )
    # Reproduce the original defect directly: a naive per-radar-frame lookup
    # of the NWP hour collapses to a single static slice when the radar window
    # is shorter than an hour.
    cache: dict[Path, object] = {}

    def case_index(path: Path):
        if path not in cache:
            cache[path] = scan_case(path)
        return cache[path]

    def naive_static_fraction() -> tuple[int, int]:
        """Fraction of samples whose radar window collapses to one NWP hour.

        This is the original defect: with a 10-frame (54-minute) window, a
        naive per-frame lookup of the NWP hour returns the same slice for
        every step, so the model's NWP input is a static repetition.
        """
        static = total = 0
        for s in forecast_ds.samples[:300]:
            case = case_index(s.case_path)
            window = case.radar_times[s.radar_idx[0] : s.radar_idx[0] + 10]
            hours = {floor_hour(t) for t in window}
            total += 1
            if len(hours) == 1:
                static += 1
        return static, total

    static_old, total_old = naive_static_fraction()
    ck.check(
        "the original defect is real: a 54-minute window collapses to one NWP hour",
        static_old > 0,
        f"{static_old}/{total_old} samples resolve the whole 10-frame radar window to a single NWP hour",
    )
    distinct = sum(1 for s in forecast_ds.samples[:200] if len(set(s.nwp_idx)) == 2)
    ck.check(
        "forecast alignment always uses two distinct NWP slices",
        distinct == min(200, len(forecast_ds.samples)),
        f"{distinct}/{min(200, len(forecast_ds.samples))} use two distinct slices",
    )
    ck.check(
        "forecast and bracket select different NWP slices",
        any(
            set(a.nwp_idx) != set(b.nwp_idx)
            for a, b in zip(forecast_ds.samples[:200], bracket_ds.samples[:200])
        ),
        "the forecast policy reaches past the issue time, the bracket policy does not",
    )

    # ----------------------------------------------------------------- pack
    print("\n[3] case packing")
    pack_dir = Path(args.keep_pack) if args.keep_pack else Path(tempfile.mkdtemp(prefix="rnb_pack_"))
    pack_dir.mkdir(parents=True, exist_ok=True)
    from radar_nwp_bench.data import CasePacker

    packer = CasePacker()
    t0 = time.time()
    stats = packer.pack_case(cases[0], pack_dir)
    print(f"  packed {stats.case_id}: radar={stats.n_radar} label={stats.n_label} "
          f"nwp={stats.n_nwp} -> {stats.bytes_written/1e6:.1f} MB "
          f"({time.time()-t0:.1f}s)")
    ck.check("packed container written", stats.bytes_written > 0,
             f"{stats.bytes_written/1e6:.1f} MB")

    from radar_nwp_bench.data import load_packed
    packed = load_packed(pack_dir / f"{cases[0].name}.npz")
    ck.check("packed radar shape", packed["radar"].shape[1:] == (10, 301, 301),
             str(packed["radar"].shape))
    ck.check("packed label shape", packed["label"].shape[1:] == (1, 301, 301),
             str(packed["label"].shape))
    ck.check("packed nwp shape", packed["nwp"].shape[1:] == (27, 167, 167),
             str(packed["nwp"].shape))
    ck.check("packed label is non-negative", bool((packed["label"] >= 0).all()))
    ck.check("packed radar has no no-echo sentinel", bool((packed["radar"] != -128.0).all()))
    ck.check("packed radar has no nodata sentinel",
             bool((packed["radar"] > -100.0).all()),
             "dBZ fields may be negative but never below the -30 dBZ floor")
    ck.check("packed nwp has no -9999 fill", bool((packed["nwp"] > -9999.0).all()))

    # -------------------------------------------------------------- sampling
    print("\n[4] dataset sampling")
    packed_ds = RadarNWPDataset(
        pack_dir, packed=True, in_len=args.in_len, out_len=args.out_len,
        patch_size=args.patch_size, alignment=NWPAlignment.FORECAST,
        random_crop=False,
    )
    item = packed_ds[0]
    ck.check("x_radar shape", tuple(item["x_radar"].shape) ==
             (args.in_len, 10, args.patch_size, args.patch_size),
             str(tuple(item["x_radar"].shape)))
    ck.check("x_nwp shape", tuple(item["x_nwp"].shape) ==
             (2, 27, args.patch_size, args.patch_size),
             str(tuple(item["x_nwp"].shape)))
    ck.check("y_rain shape", tuple(item["y_rain"].shape) ==
             (args.out_len, 1, args.patch_size, args.patch_size),
             str(tuple(item["y_rain"].shape)))
    ck.check("x_radar finite", bool(torch.isfinite(item["x_radar"]).all()))
    ck.check("x_nwp finite", bool(torch.isfinite(item["x_nwp"]).all()))
    ck.check("y_rain finite", bool(torch.isfinite(item["y_rain"]).all()))

    # NWP and radar must live on the same grid -- verify a strong signal lines up
    print("\n[5] cross-modal co-registration (PWAT vs rainfall)")
    from radar_nwp_bench.data.coordinates import bilinear_upsample_center_crop

    corr_ok = True
    details = []
    for case_dir in cases[:3]:
        cidx = scan_case(case_dir)
        rtimes = cidx.radar_times
        ntimes = cidx.nwp_times
        if len(rtimes) < 20 or len(ntimes) < 2:
            continue
        mid = len(rtimes) // 2
        stamp = rtimes[mid]
        hour = (stamp // 10000) * 10000 + ((stamp % 10000) // 100) * 100
        if hour not in ntimes:
            continue
        ra = np.load(cidx.label_file(stamp)).astype(np.float32)
        ra = np.where(ra < 0, 0, ra) * 0.1
        pw = bilinear_upsample_center_crop(
            np.load(cidx.nwp_file("PWAT", hour)).astype(np.float32)
        )
        if ra.std() < 1e-6:
            continue
        a = ra - ra.mean()
        b = pw - pw.mean()
        c = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
        details.append(f"{case_dir.name}: r={c:+.3f}")
        if c <= 0:
            corr_ok = False
    ck.check("NWP is co-registered with the radar grid (PWAT-RA corr > 0)",
             corr_ok, "; ".join(details))

    # ------------------------------------------------------------ dataloader
    print("\n[6] dataloader")
    loader = build_dataloader(
        packed_ds, batch_size=2, shuffle=False, num_workers=0, drop_last=False
    )
    batch = next(iter(loader))
    ck.check("batch x_radar", tuple(batch["x_radar"].shape) ==
             (2, args.in_len, 10, args.patch_size, args.patch_size),
             str(tuple(batch["x_radar"].shape)))
    ck.check("batch x_nwp", tuple(batch["x_nwp"].shape) ==
             (2, 2, 27, args.patch_size, args.patch_size),
             str(tuple(batch["x_nwp"].shape)))
    ck.check("batch y_rain", tuple(batch["y_rain"].shape) ==
             (2, args.out_len, 1, args.patch_size, args.patch_size),
             str(tuple(batch["y_rain"].shape)))

    # --------------------------------------------------------------- models
    print("\n[7] model adapters")
    criterion = PrecipitationLoss()
    model_rows = []
    for name in models_to_test:
        t0 = time.time()
        model = build_model(
            name, in_radar_c=10, in_nwp_c=27,
            in_len=args.in_len, out_len=args.out_len,
        )
        x_r = torch.randn(2, args.in_len, 10, args.patch_size, args.patch_size)
        x_n = torch.randn(2, 2, 27, args.patch_size, args.patch_size)
        y = torch.rand(2, args.out_len, 1, args.patch_size, args.patch_size)
        pred = model(x_r, x_n)
        ok_shape = tuple(pred.shape) == (2, args.out_len, 1, args.patch_size, args.patch_size)
        ok_finite = bool(torch.isfinite(pred).all())
        ok_nonneg = bool((pred >= 0).all())
        loss, comp = criterion(pred, y, model=model)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        ok_grad = len(grads) > 0 and all(bool(torch.isfinite(g).all()) for g in grads)
        dt = time.time() - t0
        passed = ok_shape and ok_finite and ok_nonneg and ok_grad
        model_rows.append({
            "model": name,
            "params": model.num_parameters(),
            "shape_ok": ok_shape,
            "finite": ok_finite,
            "non_negative": ok_nonneg,
            "grads_ok": ok_grad,
            "loss": comp["total"],
            "seconds": round(dt, 2),
        })
        print(f"  [{PASS if passed else FAIL}] {name:12s} "
              f"params={model.num_parameters():>9,}  "
              f"loss={comp['total']:.4f}  {dt:.2f}s")
        ck.check(f"adapter {name}", passed,
                 f"params={model.num_parameters():,}")

    # -------------------------------------------------------------- metrics
    print("\n[8] verification metrics")
    rng = np.random.default_rng(0)
    truth = (rng.random((4, args.out_len, 1, 32, 32)) > 0.8).astype(np.float32) * 5.0
    perfect = truth.copy()
    ver = PrecipitationVerifier()
    ver.update(perfect, truth)
    rep = ver.report()
    csi_perfect = rep["thresholds"]["0.5"]["csi"]
    ck.check("CSI of a perfect forecast is 1.0", abs(csi_perfect - 1.0) < 1e-6,
             f"CSI={csi_perfect:.4f}")
    ck.check("ETS/HSS/FSS present",
             all(k in rep["thresholds"]["0.5"] for k in ("ets", "hss", "bias", "fss")))
    ver2 = PrecipitationVerifier()
    ver2.update(np.zeros_like(truth), truth)
    csi_none = ver2.report()["thresholds"]["0.5"]["csi"]
    ck.check("CSI of a dry forecast is 0.0", csi_none < 1e-6, f"CSI={csi_none:.4f}")

    # The verification is done on the whole field, so the tiles have to be
    # stitched back together. Check that the stitch restores the field and
    # that an incomplete walk is an error rather than silent zero rain.
    from radar_nwp_bench.metrics import PatchStitcher
    field = rng.random((2, 40, 40)).astype(np.float32)
    stitcher = PatchStitcher((40, 40), n_frames=2, n_positions=4)
    for i, (y0, x0) in enumerate([(0, 0), (0, 20), (20, 0), (20, 20)]):
        tile = field[:, y0:y0 + 20, x0:x0 + 20]
        stitcher.add(0, i, (y0, x0), tile, tile)
    s_pred, s_truth = stitcher.finalize(0)
    ck.check("the stitched field restores the original exactly",
             bool(np.array_equal(s_pred, field)),
             f"max |diff| = {float(np.abs(s_pred - field).max()):.3e}")
    ck.check("the stitched truth restores the original exactly",
             bool(np.array_equal(s_truth, field)),
             f"max |diff| = {float(np.abs(s_truth - field).max()):.3e}")
    partial = PatchStitcher((40, 40), n_frames=1, n_positions=4)
    tile1 = np.zeros((1, 20, 20), dtype=np.float32)
    partial.add(0, 0, (0, 0), tile1, tile1)
    partial.add(0, 1, (5, 5), tile1, tile1)
    try:
        partial.finalize(0)
        uncovered = False
    except ValueError:
        uncovered = True
    ck.check("an incomplete walk is an error, not silent dry pixels",
             uncovered, "uncovered pixels must not be scored as no rain")

    # ------------------------------------------------------------- verdict
    print("\n" + "=" * 74)
    verdict = ck.summary()
    print(verdict)
    print("=" * 74)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({
                "checks": [{"name": n, "status": s, "detail": d}
                           for n, s, d in ck.rows],
                "models": model_rows,
                "summary": verdict,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"report written to {args.json_out}")

    if not args.keep_pack:
        import shutil
        shutil.rmtree(pack_dir, ignore_errors=True)

    return 0 if not ck.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
