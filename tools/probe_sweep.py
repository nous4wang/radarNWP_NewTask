"""One-shot remote status probe for the radar-nwp-bench sweep.

Connects with retries (the server occasionally drops the SSH banner), runs the
status probe over the exec channel, and prints stdout. Read-only: nothing on
the remote side is started, stopped, or modified.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, r"R:\1radarNWPprediction\radar-nwp-bench\tools")

from remote import capture, connect  # noqa: E402

STATUS_CMD = r"""cd /home/wf/radarNWP && for m in mocast cascast_nwp convlstm earthformer predrnn simvp_early_concat simvp_radar_only simvp_v2 diffcast furecast fusioncast metnet3 mmf_rnn pw_foucast pysteps_adv; do printf '%-20s ' $m; grep -E '^epoch [0-9]+:' runs/sweep/$m.log 2>/dev/null | tail -1 || echo '(no log)'; done; echo 'eval json:'; ls runs/sweep/*.eval.json 2>/dev/null | wc -l; ps -eo args | grep -E '[t]rain.py|[e]valuate.py' | grep -oE '\-\-model [a-z_0-9]+' | cut -d' ' -f2 | sort -u | tr '\n' ' '; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"""

# Second probe: artifacts written markers + driver state.
ARTIFACT_CMD = r"""cd /home/wf/radarNWP && for m in mocast cascast_nwp convlstm earthformer predrnn simvp_early_concat simvp_radar_only simvp_v2 diffcast furecast fusioncast metnet3 mmf_rnn pw_foucast pysteps_adv; do printf '%-20s ' $m; grep -c 'artifacts written to' runs/sweep/$m.log 2>/dev/null || echo 0; done; echo '--- drivers ---'; ps -eo pid,etime,args | grep -E '[r]un_sweep.sh|[r]un_extra_baselines.sh' ; echo '--- eval json files ---'; ls -1 runs/sweep/*.eval.json 2>/dev/null"""


def main() -> int:
    client = None
    last = None
    for attempt in range(1, 9):
        try:
            client = connect()
            print(f"[connected on attempt {attempt}]", file=sys.stderr)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[attempt {attempt} failed: {exc}]", file=sys.stderr)
            time.sleep(5)
    if client is None:
        print(f"CONNECT_FAILED: {last}", file=sys.stderr)
        return 2

    try:
        print("===== STATUS =====")
        print(capture(client, STATUS_CMD, timeout=180))
        print("===== ARTIFACTS =====")
        print(capture(client, ARTIFACT_CMD, timeout=180))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
