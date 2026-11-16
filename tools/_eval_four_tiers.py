"""Score every kept epoch of a set of runs, several at a time.

The verification is bound by the processor rather than by the card: the
Hannings stitching runs on the processor and the GPU sits at zero per
cent while it happens. One evaluation at a time therefore leaves the
machine idle, and the measurement says eight at a time is the best of
the three that were tried -- four would take longer, and three longer
still.

Each run directory carries the configuration it was trained with, so the
model and its extra keyword arguments are read from there rather than
guessed from the directory name. A run whose configuration cannot be
read is skipped and reported, not scored with the wrong architecture.

The judge is the number of reports written, not the exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def find_runs(root: Path) -> list[tuple[str, Path]]:
    """Every run directory under the root, with the arm it belongs to.

    The training writes into ``<out-dir>/<model>_<timestamp>/``, so a
    directory that carries a configuration is one level below the name
    the caller gave. The arm is the name of that level -- ``fss35_s0``,
    ``full_s3`` -- and it is what tells one run from another.

    The timestamp is not a unique name. ``train.py`` stamps the run
    directory with the wall clock at launch, so two arms that start in
    the same second carry the same stamp, and the ten seeds of one arm
    fall into a handful of names. The name is therefore not used to
    identify the run anywhere here; the directory path is.
    """
    runs: list[tuple[str, Path]] = []
    for config in sorted(root.glob("*/*/config.json")):
        run_dir = config.parent
        runs.append((run_dir.parent.name, run_dir))
    if not runs:
        for config in sorted(root.glob("*/config.json")):
            run_dir = config.parent
            runs.append((run_dir.name, run_dir))
    return runs


def read_meta(run_dir: Path) -> tuple[str, dict] | None:
    """The model and its extra keyword arguments, from the run itself."""
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model = config.get("model")
    if not model:
        return None
    kwargs = config.get("model_kwargs") or {}
    if isinstance(kwargs, str):
        try:
            kwargs = json.loads(kwargs)
        except json.JSONDecodeError:
            kwargs = {}
    return str(model), dict(kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tiers", nargs="*", default=["0.1", "0.5", "2.0", "4.0"])
    ap.add_argument("--data-root", default="/home/wf/data/radarNWP/BJ_mmap")
    ap.add_argument("--repo", default="/home/wf/radarNWP")
    ap.add_argument("--python", default="/home/wf/.conda/envs/radar_nwp/bin/python")
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--no-window-tables",
        action="store_true",
        help="do not write the per-window contingency tables. They are "
             "what a bootstrap resamples to put an interval on the score, "
             "so they are written by default",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out_root = root / "eval"
    out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, Path, Path, dict]] = []
    for arm, run_dir in find_runs(root):
        meta = read_meta(run_dir)
        if meta is None:
            print(f"  {run_dir}: no readable configuration, skipped")
            continue
        model, kwargs = meta
        # The output directory is named after the arm, which carries the
        # seed: ``fss35_s0``, ``full_s3``, ``lag11_s2``. That name is
        # unique, because the caller gave it.
        #
        # The run directory's own name is not. The training stamps it with
        # the wall clock at launch, so the seeds that start in the same
        # second share a name, and keying on it makes one seed's report
        # overwrite another's. It happened: sixteen evaluations of the
        # second stage came back as six reports, and the six were whichever
        # seed finished last. The run directory is still what identifies
        # the checkpoint; it is only the output path that has to be keyed
        # on something unique.
        out_dir = out_root / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        for ck in sorted(run_dir.glob("epoch_*.pt")):
            stem = ck.stem
            try:
                epoch = int(stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if epoch >= args.epochs:
                continue
            out = out_dir / f"{stem}.eval.json"
            if out.is_file():
                continue
            jobs.append((f"{arm}/{stem}", model, ck, out, kwargs))

    print(f"  {len(jobs)} evaluations to run, {args.parallel} at a time")
    if not jobs:
        return 0

    # The card each evaluation runs on is chosen from a counter rather
    # than from the job's index: the pool does not take the jobs in
    # order, so the index would send several evaluations to one card.
    card_lock = threading.Lock()
    card_next = [0]

    def score(item: tuple[str, str, Path, Path, dict]) -> tuple[str, bool]:
        name, model, ck, out, kwargs = item
        with card_lock:
            slot = card_next[0]
            card_next[0] = (card_next[0] + 1) % 8
        cmd = [
            args.python, "scripts/evaluate.py",
            "--model", model,
            "--checkpoint", str(ck),
            "--data-root", args.data_root,
            "--packed", "--split", args.split,
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.workers),
            "--patch-size", "64", "--slide-stride", "64",
            "--thresholds", *args.tiers,
            "--device", "cuda",
            "--out-json", str(out),
        ]
        if kwargs:
            cmd += ["--model-kwargs", json.dumps(kwargs)]
        if not args.no_window_tables:
            cmd += ["--per-window-out", f"{out}.windows.json"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot))
        with open(f"{out}.log", "w", encoding="utf-8") as handle:
            subprocess.run(
                cmd, cwd=args.repo, env=env, stdout=handle, stderr=handle,
                check=False,
            )
        return name, out.is_file()

    done = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for name, ok in pool.map(score, jobs):
            done += 1
            print(f"  {'done' if ok else 'FAILED'}: {name}  ({done}/{len(jobs)})")

    n = len(list(out_root.rglob("*.eval.json")))
    print(f"  reports written: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
