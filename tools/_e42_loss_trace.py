"""E42 -- read the A2' training log and trace the two loss components.

READ-ONLY.  It prints, per epoch, the first and last logged step of the
training run, with the mass_gap and uot_loss components, so the direction
of each term over training can be read off directly.

The question it answers: the fixed UOT term no longer diverges, but the
forecast came out far too small (field maximum 0.18 mm against B0's
2.24 mm).  Which component pushed it there -- the transport cost (which
compares shapes) or the mass_gap log ratio (which compares amounts)?
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

STEP = re.compile(
    r"epoch (\d+) step (\d+) loss=([0-9.eE+-]+) .*?"
    r"uot_loss=([0-9.eE+-]+) uot_cost=([0-9.eE+-]+) mass_gap=([0-9.eE+-]+)"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--per-epoch", type=int, default=3,
                    help="how many steps to show per epoch (first and last)")
    args = ap.parse_args()

    rows = []
    for line in Path(args.log).read_text(encoding="utf-8").splitlines():
        m = STEP.search(line)
        if m:
            rows.append((
                int(m.group(1)), int(m.group(2)), float(m.group(3)),
                float(m.group(4)), float(m.group(5)), float(m.group(6)),
            ))
    if not rows:
        print("no parsed step lines")
        return 1

    print(f"{'epoch':>5s} {'step':>6s} {'loss':>10s} {'uot_loss':>10s} "
          f"{'uot_cost':>10s} {'mass_gap':>10s}")
    epochs = sorted({r[0] for r in rows})
    for e in epochs:
        sub = [r for r in rows if r[0] == e]
        picks = sub[:args.per_epoch] + (
            sub[-args.per_epoch:] if len(sub) > 2 * args.per_epoch else []
        )
        for r in picks:
            print(f"{r[0]:5d} {r[1]:6d} {r[2]:10.4f} {r[3]:10.4f} "
                  f"{r[4]:10.4f} {r[5]:10.4f}")
        print("  ...")

    print()
    print("per-epoch means:")
    for e in epochs:
        sub = [r for r in rows if r[0] == e]
        n = len(sub)
        mean = lambda i: sum(r[i] for r in sub) / n  # noqa: E731
        print(f"  epoch {e}: loss={mean(2):.4f} uot_loss={mean(3):.4f} "
              f"uot_cost={mean(4):.4f} mass_gap={mean(5):.4f}  (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
