"""Print the ranking table from the evaluation reports on the server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from remote import capture, connect  # noqa: E402

RANK = r"""cd /home/wf/radarNWP && /home/wf/.conda/envs/radar_nwp/bin/python /tmp/_rank.py"""


def main() -> int:
    client = connect()
    try:
        script = Path(__file__).resolve().parent / "_rank_remote.py"
        # The ranking script is small enough to send as a file rather than
        # as a nested heredoc, which the shell quoting mangles.
        payload = script.read_text(encoding="utf-8")
        capture(client, "cat > /tmp/_rank.py <<'RANKEOF'\n" + payload + "\nRANKEOF")
        print(capture(client, RANK))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
