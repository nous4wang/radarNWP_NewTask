#!/usr/bin/env python3
"""Pack the raw archive into one container file per case.

Usage:
    python scripts/pack_data.py --data-root R:/1radarNWPprediction/data/BJ \
                                --out-dir  R:/1radarNWPprediction/data/BJ_packed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_nwp_bench.data import CasePacker, NWP_VARS, RADAR_VARS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="directory holding CP_* cases")
    ap.add_argument("--out-dir", required=True, help="destination for the .npz containers")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-compress", action="store_true", help="store uncompressed (faster)")
    ap.add_argument(
        "--layout", choices=["npz", "mmap"], default="npz",
        help="npz: one compressed archive per case (small on disk, but every "
             "access has to inflate the whole member -- 1.4 s per case). "
             "mmap: one directory of .npy files per case, which the OS can "
             "page in on demand and share between workers. Use mmap for "
             "training runs that shuffle across many cases.",
    )
    ap.add_argument(
        "--radar-vars", nargs="*", default=list(RADAR_VARS),
        help="subset of radar variables to store",
    )
    ap.add_argument(
        "--nwp-vars", nargs="*", default=list(NWP_VARS),
        help="subset of NWP variables to store",
    )
    args = ap.parse_args()

    packer = CasePacker(
        radar_vars=tuple(args.radar_vars),
        nwp_vars=tuple(args.nwp_vars),
        compress=not args.no_compress,
    )
    print(f"packing {args.data_root} -> {args.out_dir} ({args.layout})")
    if args.layout == "mmap":
        stats = packer.pack_all_mmap(
            args.data_root, args.out_dir, workers=args.workers
        )
    else:
        stats = packer.pack_all(args.data_root, args.out_dir, workers=args.workers)

    total_bytes = sum(s.bytes_written for s in stats)
    total_frames = sum(s.n_radar for s in stats)
    print(
        f"\ndone: {len(stats)} cases, {total_frames} radar frames, "
        f"{total_bytes / 1e9:.2f} GB"
    )
    print(f"mean frame count per case: {total_frames / max(len(stats), 1):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
