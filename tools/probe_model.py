"""Probe a trained baseline: is its output a constant field?

A model that scores near zero on CSI but whose training loss barely moves
is usually predicting a constant. This reads the checkpoint and reports
the statistics of the prediction, so the cause is visible rather than
inferred.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from remote import capture, connect  # noqa: E402

PROBE = r"""cd /home/wf/radarNWP && /home/wf/.conda/envs/radar_nwp/bin/python /tmp/_probe.py"""


def main() -> int:
    models = sys.argv[1:] or ["mocast", "convlstm", "fusioncast"]
    client = connect()
    try:
        payload = Path(__file__).resolve().parent / "_probe_remote.py"
        capture(client, "cat > /tmp/_probe.py <<'PROBEOF'\n"
                 + payload.read_text(encoding="utf-8") + "\nPROBEOF")
        print(capture(client, PROBE + " " + " ".join(models)))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
