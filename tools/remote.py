"""Run a command on the benchmark server over SSH.

The server takes a password rather than a key, and no ``sshpass`` is
installed on this workstation, so paramiko does the work.

    python tools/remote.py "nvidia-smi"
    python tools/remote.py --upload radar_nwp_bench/ablation.py
    python tools/remote.py --download runs/x.json ./x.json
    python tools/remote.py --script deploy.sh
    python tools/remote.py --sync radar_nwp_bench/data/dataset.py
    python tools/remote.py --sync tools/x.py -- "python scripts/train.py ..."
    python tools/remote.py --doctor          # diagnose, change nothing

The failures this server has produced are not the same kind of failure
and they need different responses: a dropped banner is worth retrying, a
rejected password is not, and retrying a rejected password can lock the
account. :func:`classify` tells them apart and :func:`connect` retries
only what a retry can fix.

When the failure needs a human -- a changed password, a locked account, a
full disk -- :func:`escalation` says which, and prints the command to run
by hand. The escalation covers the whole call, not only the connect: a
transfer that fails halfway is the failure most likely to need a person,
and it is the one a connect-only guard would swallow.
"""
from __future__ import annotations

import argparse
import errno
import os
import posixpath
import socket
import sys
import time
from enum import Enum
from pathlib import Path

import paramiko

HOST = os.environ.get("RADAR_HOST", "183.136.129.105")
PORT = int(os.environ.get("RADAR_PORT", "1003"))
USER = os.environ.get("RADAR_USER", "wf")
PASSWORD = os.environ.get("RADAR_PASSWORD", "Wf84120116")
REMOTE_ROOT = os.environ.get("RADAR_ROOT", "/home/wf/radarNWP")


class RemoteFailure(str, Enum):
    """What kind of failure happened, and therefore who has to act.

    The distinction is the point of this module. A dropped banner is
    worth retrying because the server does it under load and the next
    attempt works. A rejected password is not, because the server counts
    the attempts and may lock the account, and because no amount of
    retrying produces a different answer.
    """

    TRANSIENT = "transient"
    AUTH = "auth"
    NETWORK = "network"
    PERMISSION = "permission"
    STORAGE = "storage"
    UNKNOWN = "unknown"


def classify(exc: BaseException) -> RemoteFailure:
    """Decide what kind of failure this is.

    Args:
        exc: the exception that was raised.

    Returns:
        The failure class. Never raises: an unknown exception is a
        class of its own rather than an error in the classifier.
    """
    if isinstance(exc, paramiko.AuthenticationException):
        return RemoteFailure.AUTH

    # SFTP failures arrive as plain operating-system errors, not as
    # dedicated parameter classes: the client converts the protocol
    # status into an OSError carrying the errno. Permission denied is
    # EACCES and a full disk is ENOSPC or EDQUOT.
    if isinstance(exc, PermissionError):
        return RemoteFailure.PERMISSION
    if isinstance(exc, OSError):
        if exc.errno in (errno.ENOSPC, errno.EDQUOT):
            return RemoteFailure.STORAGE
        if exc.errno in (errno.EACCES, errno.EPERM):
            return RemoteFailure.PERMISSION
        if isinstance(exc, socket.gaierror):
            return RemoteFailure.NETWORK
        if isinstance(exc, (ConnectionError, socket.timeout, TimeoutError)):
            return RemoteFailure.NETWORK
        # An OSError with no errno is usually a socket error from the
        # transport, which is worth another attempt.
        return RemoteFailure.TRANSIENT

    if isinstance(exc, paramiko.SSHException):
        text = str(exc).lower()
        if "banner" in text:
            return RemoteFailure.TRANSIENT
        if "authentication" in text or "publickey" in text:
            return RemoteFailure.AUTH
        return RemoteFailure.TRANSIENT

    return RemoteFailure.UNKNOWN


def connect(retries: int = 8) -> paramiko.SSHClient:
    """Open a connection, retrying only the failures a retry can fix.

    A dropped banner is worth retrying: the server does it under load
    and the connection works on a later attempt. A rejected password is
    not worth retrying -- the server may count the attempts and lock the
    account -- so it is raised immediately with the failure class
    attached.

    Args:
        retries: how many attempts to make for a transient failure.

    Returns:
        A connected client.

    Raises:
        The exception that ended the attempts, with ``remote_failure``
        set to the :class:`RemoteFailure` it belongs to.
    """
    last: BaseException | None = None
    for attempt in range(retries):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                HOST, port=PORT, username=USER, password=PASSWORD,
                timeout=30, banner_timeout=60, auth_timeout=60,
                look_for_keys=False, allow_agent=False,
            )
            return client
        except Exception as exc:  # noqa: BLE001 - classified below
            last = exc
            kind = classify(exc)
            if kind is not RemoteFailure.TRANSIENT:
                setattr(exc, "remote_failure", kind)
                raise
            time.sleep(min(5 * (attempt + 1), 30))
    if last is None:
        last = RuntimeError("could not connect")
    setattr(last, "remote_failure", classify(last))
    raise last


def _failure_of(exc: BaseException) -> RemoteFailure:
    """The class of a failure, preferring the one already attached."""
    tagged = getattr(exc, "remote_failure", None)
    if isinstance(tagged, RemoteFailure):
        return tagged
    return classify(exc)


def escalation(exc: BaseException) -> str:
    """The manual steps for a failure that a retry cannot fix.

    The commands are given for Windows and for a POSIX shell, because
    this workstation is Windows and the server is not.

    Args:
        exc: the exception that ended the attempt.

    Returns:
        A block of text naming the failure and what to do about it.
    """
    kind = _failure_of(exc)
    lines = [
        "",
        "=" * 70,
        " 远端连接失败，需要人工处理 / REMOTE ACCESS NEEDS A HUMAN",
        "=" * 70,
        f"失败类型 / failure class : {kind.value}",
        f"目标 / target            : {USER}@{HOST}:{PORT}",
        f"原始异常 / exception     : {type(exc).__name__}: {exc}",
        "",
    ]

    if kind is RemoteFailure.AUTH:
        lines += [
            "服务端拒绝了认证：banner 正常、TCP 可达，但密码被拒。",
            "这是服务端侧的问题（改过密码、锁定账户或换了密钥），本机无法绕过。",
            "",
            "请任选其一处理：",
            "  1) 拿到新密码后，设置环境变量再重试（本机是 Windows）：",
            "       PowerShell : $env:RADAR_PASSWORD='<新密码>'",
            "       cmd        : set RADAR_PASSWORD=<新密码>",
            "       bash       : export RADAR_PASSWORD='<新密码>'",
            "       然后执行   : python tools/remote.py --doctor",
            "  2) 若已配置公钥，用原生 ssh 登录（paramiko 这条路只走密码）：",
            f"       ssh -p {PORT} {USER}@{HOST}",
            "  3) 恢复后把密码写进仓库根目录的 .env（注意：代码目前只读环境变量，",
            "     .env 需要你自行 export，或告诉我加上 dotenv 支持）。",
            "",
        ]
    elif kind is RemoteFailure.NETWORK:
        lines += [
            "网络层不可达：连不到端口，问题在网络或服务器本身。",
            "",
            "请在服务器控制台确认它是否开机，然后在本机测试端口：",
            f"       PowerShell : Test-NetConnection -Port {PORT} {HOST}",
            f"       bash       : nc -zv {HOST} {PORT}",
            "",
        ]
    elif kind is RemoteFailure.TRANSIENT:
        lines += [
            "连接在握手阶段被反复中断，重试次数已用尽。",
            "服务器在高负载时会丢弃连接；若刚跑完大批训练，等负载降下来再试。",
            "",
            "本机也可以直接用原生 ssh 验证（它与 paramiko 的握手路径不同）：",
            f"       ssh -p {PORT} {USER}@{HOST}",
            "",
        ]
    elif kind is RemoteFailure.PERMISSION:
        lines += [
            "认证通过，但写不到目标目录。",
            "",
            "请在服务器上确认属主与权限：",
            f"       ls -ld {REMOTE_ROOT} {REMOTE_ROOT}/runs",
            "",
        ]
    elif kind is RemoteFailure.STORAGE:
        lines += [
            "远端磁盘满或 inode 耗尽。",
            "",
            "请在服务器上清理后重试：",
            "       df -h /home/wf && df -i /home/wf",
            "       du -sh /home/wf/radarNWP/runs/* | sort -h | tail -20",
            "",
        ]
    else:
        lines += [
            "未归类的失败。上面的原始异常是全部线索。",
            "",
        ]

    lines += [
        "本地改动未受影响：代码在本地仓库里，同步在连接恢复后重跑即可。",
        "=" * 70,
        "",
    ]
    return "\n".join(lines)


def doctor() -> int:
    """Report what the connection looks like, without changing anything.

    Walks the layers in order -- name resolution, the TCP port, the SSH
    banner, the authentication -- and says which one breaks. A single
    failed connection does not say whether the server is down or the
    password is wrong, and that difference decides who has to act.

    Returns:
        0 when the connection works, 1 otherwise.
    """
    print("=" * 70)
    print(" 远端连接诊断 / REMOTE ACCESS DIAGNOSIS")
    print("=" * 70)
    print(f"target : {USER}@{HOST}:{PORT}")
    print(f"root   : {REMOTE_ROOT}")
    print()

    # 1. name resolution
    try:
        address = socket.gethostbyname(HOST)
        print(f"[1/4] DNS        : ok  ({HOST} -> {address})")
    except OSError as exc:
        print(f"[1/4] DNS        : FAIL  {exc}")
        print(escalation(exc))
        return 1

    # 2. the TCP port
    try:
        with socket.create_connection((HOST, PORT), timeout=15):
            print(f"[2/4] TCP {PORT}   : ok  (port open)")
    except OSError as exc:
        print(f"[2/4] TCP {PORT}   : FAIL  {exc}")
        print(escalation(exc))
        return 1

    # 3 and 4. banner and authentication, through the same path the
    # real commands take, so the diagnosis cannot disagree with reality.
    print("[3/4] SSH banner : ", end="", flush=True)
    try:
        client = connect(retries=4)
    except Exception as exc:  # noqa: BLE001 - reported below
        kind = _failure_of(exc)
        if kind is RemoteFailure.AUTH:
            print("ok")
            print("[4/4] auth       : FAIL  (password refused)")
        else:
            print(f"FAIL  ({type(exc).__name__}: {exc})")
        print(escalation(exc))
        return 1

    print("ok")
    print("[4/4] auth       : ok  (password accepted)")
    try:
        out = capture(
            client,
            f"cd {REMOTE_ROOT} && echo PWD=$(pwd) && "
            f"ls -d runs 2>/dev/null && df -h . | tail -1",
        )
        print()
        print(out.rstrip())
    except Exception as exc:  # noqa: BLE001
        print(f"(connected, but the probe failed: {exc})")
    finally:
        client.close()
    print("=" * 70)
    print(" 结论 / verdict : 远端连接正常 / remote access is working")
    print("=" * 70)
    return 0


def run(client: paramiko.SSHClient, command: str, timeout: float | None = None):
    """Run a command, streaming its output. Returns the exit status."""
    transport = client.get_transport()
    channel = transport.open_session()
    channel.settimeout(timeout)
    channel.get_pty(width=200, height=50)
    channel.exec_command(command)
    while True:
        if channel.recv_ready():
            data = channel.recv(65536)
            if not data:
                break
            sys.stdout.write(data.decode("utf-8", "replace"))
            sys.stdout.flush()
        elif channel.exit_status_ready():
            break
    return channel.recv_exit_status()


def capture(client: paramiko.SSHClient, command: str, timeout: float | None = None) -> str:
    """Run a command and return its output instead of printing it.

    No PTY is allocated: a pseudo-terminal rewrites the line endings, which
    corrupts the output of anything that parses it.
    """
    transport = client.get_transport()
    channel = transport.open_session()
    channel.settimeout(timeout)
    channel.exec_command(command)
    chunks: list[bytes] = []
    while True:
        if channel.recv_ready():
            data = channel.recv(65536)
            if not data:
                break
            chunks.append(data)
        elif channel.exit_status_ready():
            break
    return b"".join(chunks).decode("utf-8", "replace")


def upload(client: paramiko.SSHClient, local: Path, remote: str) -> None:
    sftp = client.open_sftp()
    try:
        parent = posixpath.dirname(remote)
        if parent:
            try:
                sftp.stat(parent)
            except OSError:
                run(client, f"mkdir -p {parent}")
        sftp.put(str(local), remote)
        print(f"uploaded {local} -> {remote}")
    finally:
        sftp.close()


def sync(client: paramiko.SSHClient, files: list[Path], root: Path) -> None:
    """Upload several files, keeping their paths under ``root``.

    One connection, one directory tree: the remote mirror of the local
    repository. The files are uploaded to a temporary name and then moved
    into place, so a file that is being read on the far side is never
    observed half-written.

    A shell script is converted to LF on the way out. The repository is
    edited on Windows, so its scripts carry CRLF line endings, and bash
    reads ``do\\r`` as a syntax error. The conversion happens here rather
    than in the repository, so the local copies stay usable on Windows.

    The upload goes to a temporary name and is then moved into place. That
    matters: bash reads a running script by byte offset, so rewriting the
    file under it can make it execute the wrong bytes. ``mv`` replaces the
    directory entry, and a shell that already has the file open keeps
    reading the original inode -- it is unaffected by the replacement.
    ``tar xzf`` over the same file would rewrite the inode in place and
    could corrupt the running script.
    """
    sftp = client.open_sftp()
    try:
        for local in files:
            relative = local.relative_to(root).as_posix()
            remote = posixpath.join(REMOTE_ROOT, relative)
            parent = posixpath.dirname(remote)
            if parent:
                try:
                    sftp.stat(parent)
                except OSError:
                    run(client, f"mkdir -p {parent}")
            staging = remote + ".uploading"
            # Both ``.sh`` and ``.sbatch`` are shell scripts, and bash
            # refuses a CRLF line ending: "case ${IDX} in\r" is a syntax
            # error, not a warning.  The files are edited on Windows, so
            # they arrive with CRLF unless they are converted here.
            if local.suffix in (".sh", ".sbatch"):
                text = local.read_text(encoding="utf-8")
                with sftp.open(staging, "w") as handle:
                    handle.write(text.replace("\r\n", "\n"))
            else:
                sftp.put(str(local), staging)
            run(client, f"mv -f {staging!r} {remote!r}")
            print(f"synced {relative}")
    finally:
        sftp.close()


def download(client: paramiko.SSHClient, remote: str, local: Path) -> None:
    sftp = client.open_sftp()
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, str(local))
        print(f"downloaded {remote} -> {local}")
    finally:
        sftp.close()


#: The interpreter on the server. The benchmark image has one conda
#: environment, and a script that is run by the wrong interpreter will
#: fail on the imports rather than on the science.
REMOTE_PYTHON = os.environ.get(
    "RADAR_PYTHON", "/home/wf/.conda/envs/radar_nwp/bin/python"
)


def _is_python(script_path: str) -> bool:
    """Whether a script is Python rather than shell."""
    return script_path.endswith(".py")


def run_python(
    client: paramiko.SSHClient, script: str, timeout: float | None
) -> int:
    """Run a Python script on the far side, streaming its output.

    The script is written to a file and run by the server's interpreter
    rather than piped into bash, which would read the docstring as a
    command.
    """
    remote_script = "/tmp/_rnb_script.py"
    sftp = client.open_sftp()
    try:
        with sftp.open(remote_script, "w") as handle:
            handle.write(script.replace("\r\n", "\n"))
    finally:
        sftp.close()
    return run(
        client, f"{REMOTE_PYTHON} {remote_script}", timeout=timeout
    )


def run_script(client: paramiko.SSHClient, script: str, timeout: float | None) -> int:
    """Feed a script to the remote bash over stdin."""
    transport = client.get_transport()
    channel = transport.open_session()
    channel.settimeout(timeout)
    channel.get_pty(width=200, height=50)
    channel.exec_command("bash -s")
    channel.sendall(script.encode("utf-8"))
    channel.shutdown_write()
    while True:
        if channel.recv_ready():
            data = channel.recv(65536)
            if not data:
                break
            sys.stdout.write(data.decode("utf-8", "replace"))
            sys.stdout.flush()
        elif channel.exit_status_ready():
            break
    return channel.recv_exit_status()


def run_detached(
    client: paramiko.SSHClient, script: str, log: str, python: bool = False
) -> int:
    """Start a script on the far side and let it outlive the connection.

    A training run takes hours and the SSH channel has a limit of about
    an hour, so a run started through the channel is cut off when the
    channel closes. Starting it inside its own session with the output
    redirected to a file means the run continues whether or not anyone is
    still connected.

    The script is written to a file first and then run, rather than
    being quoted into the command line. A multi-line script quoted with
    Python's ``repr`` arrives at the far side with its newlines turned
    into the two characters ``\\`` and ``n``, which bash does not read as
    line breaks.

    Args:
        client: the connection.
        script: the shell script to run.
        log: the file the output goes to on the far side.

    Returns:
        The exit status of the launcher, not of the run itself.
    """
    # The name carries the time and the process id. A fixed name is a
    # trap: bash reads a script lazily, by byte offset, so a second
    # launch that overwrites the file makes the first launch resume in
    # the middle of the new file and die at a random place. That is how
    # the route launcher lost its second half on 2026-11-02.
    remote_script = f"/tmp/_rnb_detached_{int(time.time())}_{os.getpid()}.sh"
    sftp = client.open_sftp()
    try:
        with sftp.open(remote_script, "w") as handle:
            handle.write(script.replace("\r\n", "\n"))
    finally:
        sftp.close()

    runner = f"{REMOTE_PYTHON} {remote_script}" if python else f"bash {remote_script}"
    payload = (
        f"cd {REMOTE_ROOT} && setsid {runner} "
        f"> {log} 2>&1 < /dev/null & echo LAUNCHED"
    )
    out = capture(client, payload)
    print(out.strip())
    if "LAUNCHED" in out:
        print("the run continues on the server; watch it with:")
        print(f"    tail -f {log}")
    else:
        print("the launch did not report success; check the log above")
    return 0 if "LAUNCHED" in out else 1


def _resolve_files(entries: list[str], root: Path) -> list[Path]:
    """Turn the command-line paths into the files to upload."""
    files: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            files.append(path)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", nargs="?", help="the command to run")
    ap.add_argument("--upload", metavar="LOCAL", help="upload a file")
    ap.add_argument("--to", metavar="REMOTE", help="destination for --upload")
    ap.add_argument("--download", metavar="REMOTE", help="download a file")
    ap.add_argument("--out", metavar="LOCAL", help="destination for --download")
    ap.add_argument("--script", metavar="LOCAL", help="run a local script remotely")
    ap.add_argument("--detached", metavar="LOG", default=None,
                    help="run --script detached, with its output in LOG; the "
                         "run outlives the connection")
    ap.add_argument("--sync", action="append", nargs="+", metavar="PATH",
                    help="upload these files, keeping their paths under the "
                         "repository root; may be combined with a command, "
                         "which then runs on the far side afterwards.  The "
                         "option may be given more than once: each "
                         "occurrence adds its own files")
    ap.add_argument("--doctor", action="store_true",
                    help="diagnose the connection and exit")
    ap.add_argument("--timeout", type=float, default=None)
    args = ap.parse_args()

    if args.doctor:
        return doctor()

    # The guard covers the whole call, not only the connect. A transfer
    # that fails halfway is the failure most likely to need a person, and
    # it is the one a connect-only guard would let through as a
    # traceback.
    try:
        client = connect()
    except Exception as exc:  # noqa: BLE001 - escalated below
        print(escalation(exc), file=sys.stderr)
        return 2

    try:
        if args.sync:
            root = Path(__file__).resolve().parent.parent
            # The option may be given several times, so the groups are
            # flattened before they are resolved.  With a single
            # nargs="+" option the second "--sync" would have been
            # swallowed as another value of the first, and everything
            # after it would silently not have been uploaded.
            entries = [p for group in args.sync for p in group]
            sync(client, _resolve_files(entries, root), root)
            # The command runs after the sync, in the same connection:
            # the local edit, the upload and the execution are one step,
            # which is how the work is actually done.
            if args.command:
                return run(client, args.command, timeout=args.timeout)
            return 0
        if args.upload:
            remote = args.to
            if remote is None:
                remote = posixpath.join(REMOTE_ROOT, Path(args.upload).name)
            upload(client, Path(args.upload), remote)
            return 0
        if args.download:
            local = Path(args.out) if args.out else Path(Path(args.download).name)
            download(client, args.download, local)
            return 0
        if args.script:
            path = Path(args.script)
            script = path.read_text(encoding="utf-8")
            python = _is_python(str(path))
            if args.detached:
                return run_detached(client, script, args.detached, python=python)
            if python:
                # A Python script is uploaded and run by the server's
                # interpreter rather than fed to bash, which would try to
                # read the docstring as a command.
                return run_python(client, script, args.timeout)
            return run_script(client, script, args.timeout)
        if args.command:
            return run(client, args.command, timeout=args.timeout)
        ap.error("nothing to do")
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - escalated below
        print(escalation(exc), file=sys.stderr)
        return 2
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
