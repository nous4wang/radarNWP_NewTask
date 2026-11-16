"""The remote transport's failure classification.

This is the part of the transport that decides whether a failure is worth
retrying, and it is worth testing because getting it wrong is expensive
in both directions: treating a rejected password as transient burns the
retry budget and can lock the account, while treating a dropped banner as
fatal throws away a connection that would have worked.

The classifier once referenced ``paramiko.SFTPPermissionError`` and
``paramiko.SFTPOSError``, which do not exist in any released paramiko.
The reference raised ``AttributeError`` inside the ``except`` block, so
every failure that was not an authentication failure produced a
traceback instead of a diagnosis. The test below pins the exception
types that actually exist, because that class of bug is invisible until
the failure happens.
"""
from __future__ import annotations

import errno
import importlib.util
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _pytest_compat import pytest  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tools"))
from remote import RemoteFailure, classify, escalation  # noqa: E402


def _load_remote():
    """The transport module, loaded the way the scripts load it."""
    path = REPO_ROOT / "tools" / "remote.py"
    spec = importlib.util.spec_from_file_location("rnb_remote", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rnb_remote"] = module
    spec.loader.exec_module(module)
    return module


def test_the_exception_types_the_classifier_names_actually_exist():
    """A name that does not exist raises AttributeError at the worst moment.

    The classifier used to name two SFTP exceptions that paramiko has
    never exported. The reference raised inside an ``except`` block, so
    the diagnosis was replaced by a traceback exactly when it was needed.
    """
    import paramiko

    for name in ("AuthenticationException", "SSHException", "SFTPError"):
        assert hasattr(paramiko, name), (
            f"paramiko has no {name}; the classifier must not name it"
        )

    module = _load_remote()
    # Every branch of the classifier must be reachable without raising.
    for exc in (
        paramiko.AuthenticationException("no"),
        paramiko.SSHException("Error reading SSH protocol banner"),
        PermissionError(errno.EACCES, "denied"),
        OSError(errno.ENOSPC, "full"),
        OSError("no errno"),
        socket.gaierror("no name"),
        RuntimeError("something else"),
    ):
        assert isinstance(module.classify(exc), module.RemoteFailure)


def test_a_rejected_password_is_not_retried():
    """Retrying it can lock the account."""
    import paramiko

    assert classify(paramiko.AuthenticationException("no")) is RemoteFailure.AUTH


def test_a_dropped_banner_is_worth_retrying():
    """The server does this under load and the next attempt works."""
    import paramiko

    exc = paramiko.SSHException("Error reading SSH protocol banner")
    assert classify(exc) is RemoteFailure.TRANSIENT


def test_permission_and_storage_are_told_apart():
    """They need different instructions, so they cannot share a class."""
    assert classify(PermissionError(errno.EACCES, "denied")) is (
        RemoteFailure.PERMISSION
    )
    assert classify(OSError(errno.ENOSPC, "full")) is RemoteFailure.STORAGE
    assert classify(OSError(errno.EDQUOT, "quota")) is RemoteFailure.STORAGE


def test_a_socket_failure_is_a_network_failure():
    assert classify(ConnectionRefusedError()) is RemoteFailure.NETWORK
    assert classify(socket.gaierror("no name")) is RemoteFailure.NETWORK


def test_an_unknown_failure_is_its_own_class():
    """Not every failure has to be one of the known kinds."""
    assert classify(RuntimeError("odd")) is RemoteFailure.UNKNOWN


def _an_exception_of(kind: RemoteFailure):
    """An exception that classifies as the given kind."""
    import paramiko

    if kind is RemoteFailure.AUTH:
        return paramiko.AuthenticationException("no")
    if kind is RemoteFailure.TRANSIENT:
        return paramiko.SSHException("Error reading SSH protocol banner")
    if kind is RemoteFailure.PERMISSION:
        return PermissionError(errno.EACCES, "denied")
    if kind is RemoteFailure.STORAGE:
        return OSError(errno.ENOSPC, "full")
    if kind is RemoteFailure.NETWORK:
        return ConnectionRefusedError()
    return RuntimeError("odd")


def test_every_class_has_manual_instructions():
    """The escalation is the whole point of the module."""
    for kind in RemoteFailure:
        text = escalation(_an_exception_of(kind))
        assert "REMOTE ACCESS NEEDS A HUMAN" in text
        assert kind.value in text
        # Every card tells the reader that the local work is safe.
        assert "本地改动未受影响" in text


def test_the_escalation_says_how_to_supply_a_new_password():
    """The action for an authentication failure is to set the variable."""
    import paramiko

    text = escalation(paramiko.AuthenticationException("no"))
    assert "RADAR_PASSWORD" in text


def test_the_escalation_gives_commands_for_this_workstation():
    """This workstation is Windows; the server is not."""
    text = escalation(ConnectionRefusedError())
    assert "PowerShell" in text
    assert "Test-NetConnection" in text
    # A POSIX-only command would not run here.
    assert "ping -p" not in text
    assert "telnet" not in text


def test_the_transport_still_exports_what_the_repository_imports():
    """Fourteen scripts import from this module; they must all resolve."""
    module = _load_remote()
    for name in ("connect", "capture", "run", "upload", "download", "sync"):
        assert hasattr(module, name), f"remote.{name} is gone"
