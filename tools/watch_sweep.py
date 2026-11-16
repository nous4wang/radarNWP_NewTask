"""Watch the remote sweep until every baseline has been evaluated.

The training runs take hours. Rather than poll them by hand, this watches
the remote run directories and reports when the last evaluation lands, then
prints the combined ranking table.

    python tools/watch_sweep.py --every 300
"""
from __future__ import annotations

import argparse
import json
import posixpath
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from remote import REMOTE_ROOT, capture, connect, run  # noqa: E402

RUNS = "/home/wf/radarNWP/runs/sweep"

# The baselines the two sweeps between them cover.
EXPECTED = [
    "cascast_nwp", "convlstm", "diffcast", "earthformer", "furecast",
    "fusioncast", "metnet3", "mmf_rnn", "mocast", "predrnn", "pw_foucast",
    "pysteps_adv", "simvp_early_concat", "simvp_radar_only", "simvp_v2",
]

#: The ranking table, computed on the remote where the reports live.
RANK_COMMAND = r"""cd /home/wf/radarNWP && /home/wf/.conda/envs/radar_nwp/bin/python - <<'PY'
import json
from pathlib import Path

out = Path('/home/wf/radarNWP/runs/sweep')
rows = []
for path in sorted(out.glob('*.eval.json')):
    try:
        report = json.loads(path.read_text())
    except Exception:
        continue
    thresholds = report.get('thresholds', {})
    a = thresholds.get('0.5', {})
    b = thresholds.get('2.0', {})
    rows.append((
        path.stem.replace('.eval', ''),
        a.get('csi', float('nan')),
        b.get('csi', float('nan')),
        a.get('ets', float('nan')),
        a.get('hss', float('nan')),
        a.get('fss', float('nan')),
        report.get('n_samples', 0),
    ))

rows.sort(key=lambda r: (-r[1] if r[1] == r[1] else 0))
header = f"{'model':22s}{'CSI@0.5':>9s}{'CSI@2.0':>9s}{'ETS':>8s}{'HSS':>8s}{'FSS':>8s}{'n':>8s}"
print(header)
print('-' * len(header))
for name, c1, c2, ets, hss, fss, n in rows:
    print(f"{name:22s}{c1:9.3f}{c2:9.3f}{ets:8.3f}{hss:8.3f}{fss:8.3f}{n:8d}")
print(f"\n{len(rows)} baselines evaluated")
PY"""


def snapshot(client) -> dict:
    """Read the remote state: which models are done, evaluated, running.

    A checkpoint appears after every epoch, so "has a checkpoint" does not
    mean "finished". The log's completion marker does.
    """
    out = capture(
        client,
        "cd /home/wf/radarNWP && "
        f"for m in {' '.join(EXPECTED)}; do "
        # grep -c prints a count and a newline; when the file is missing it
        # prints nothing and the || echo supplies the zero. Taking the
        # first line only keeps the record on one line.
        "  done=$(grep -c 'artifacts written to' runs/sweep/$m.log 2>/dev/null | head -1); "
        "  done=${done:-0}; "
        "  ev=$(ls runs/sweep/$m.eval.json 2>/dev/null | head -1); "
        "  ep=$(grep -E '^epoch [0-9]+:' runs/sweep/$m.log 2>/dev/null | tail -1); "
        '  echo "$m|${done}|${ev:+yes}|${ep}"; '
        "done; echo '---RUNNING---'; "
        "ps -eo args | grep '[t]rain.py' | grep -oE '\\-\\-model [a-z_0-9]+' "
        "| cut -d' ' -f2 | sort -u | tr '\\n' ' '",
    )
    trained, evaluated, running, epochs = [], [], [], {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("---RUNNING---"):
            running = line.replace("---RUNNING---", "").split()
            continue
        if "|" not in line:
            continue
        # The epoch line may itself contain a "|", so split from the left
        # and keep the remainder as the epoch text.
        name, _, rest = line.partition("|")
        tr, _, rest = rest.partition("|")
        ev, _, ep = rest.partition("|")
        try:
            finished = int(tr.strip() or 0) > 0
        except ValueError:
            finished = False
        if finished:
            trained.append(name)
        if ev:
            evaluated.append(name)
        if ep:
            epochs[name] = ep.strip()
    return {
        "trained": trained,
        "evaluated": evaluated,
        "running": running,
        "epochs": epochs,
        "raw": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--every", type=int, default=300, help="seconds between polls")
    ap.add_argument("--timeout", type=int, default=21600, help="give up after this")
    args = ap.parse_args()

    client = connect()
    start = time.time()
    try:
        while True:
            state = snapshot(client)
            trained = state["trained"]
            evaluated = state["evaluated"]
            missing = [m for m in EXPECTED if m not in trained]
            done = [m for m in EXPECTED if m not in missing]

            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"trained {len(trained)}/{len(EXPECTED)}  "
                  f"evaluated {len(evaluated)}/{len(EXPECTED)}  "
                  f"running: {' '.join(state['running']) or 'none'}")
            for name in done:
                ep = state["epochs"].get(name, "")
                if ep:
                    print(f"    {name:20s} {ep}")
            if not missing:
                print("\nall baselines trained. waiting for the evaluations...")
                break
            if time.time() - start > args.timeout:
                print("timed out")
                return 1
            time.sleep(args.every)

        # The evaluations lag the training by a few minutes each.
        while True:
            state = snapshot(client)
            evaluated = state["evaluated"]
            pending = [m for m in EXPECTED if m not in evaluated]
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"evaluated {len(evaluated)}/{len(EXPECTED)}"
                  + (f"  pending: {' '.join(pending)}" if pending else ""))
            if not pending:
                break
            if time.time() - start > args.timeout:
                print("timed out waiting for the evaluations")
                return 1
            time.sleep(args.every)

        print("\n" + "=" * 78)
        print("  ranking (test split)")
        print("=" * 78)
        print(capture(client, RANK_COMMAND))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
