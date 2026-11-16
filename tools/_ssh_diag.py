"""Diagnose the SSH connection: raw banner, auth attempts, timing."""
from __future__ import annotations

import socket
import sys
import time

HOST = "183.136.129.105"
PORT = 1003


def banner_probe() -> str:
    try:
        with socket.create_connection((HOST, PORT), timeout=30) as s:
            s.settimeout(30)
            data = s.recv(256)
            return repr(data)
    except Exception as exc:  # noqa: BLE001
        return f"<ERROR {exc}>"


def attempt(allow_agent: bool, look: bool) -> str:
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    t0 = time.time()
    try:
        client.connect(
            HOST, port=PORT, username="wf", password="wf12345",
            timeout=30, banner_timeout=60, auth_timeout=60,
            allow_agent=allow_agent, look_for_keys=look,
        )
        chan = client.get_transport().open_session()
        chan.exec_command("echo OK; uptime; date")
        out = b""
        while True:
            if chan.recv_ready():
                chunk = chan.recv(65536)
                if not chunk:
                    break
                out += chunk
            elif chan.exit_status_ready():
                break
        return f"SUCCESS in {time.time()-t0:.1f}s: {out.decode('utf-8','replace').strip()}"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL in {time.time()-t0:.1f}s [{type(exc).__name__}]: {exc}"
    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    print(f"banner probe: {banner_probe()}")
    for i, (agent, look) in enumerate(
        [(True, True), (False, False), (False, False), (True, True)], start=1
    ):
        print(f"attempt {i} allow_agent={agent} look_for_keys={look}: {attempt(agent, look)}")
        if i < 4:
            time.sleep(15)
    sys.exit(0)
