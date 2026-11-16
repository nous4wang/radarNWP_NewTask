"""Shared bootstrap for the test suite.

Two jobs:

1. Put the repository root on ``sys.path`` so ``import radar_nwp_bench``
   works no matter how the suite is started (plain ``pytest``, the bundled
   ``tests/run_tests.py`` runner, or an interactive session).
2. Provide a ``pytest`` object.  The benchmark image does not ship pytest, so
   when it is missing a tiny stand-in implements the slice of the API this
   suite uses (``pytest.mark.parametrize``, ``pytest.raises``, ``pytest.approx``,
   ``pytest.skip``, ``pytest.fail``).  When pytest *is* installed the real
   module is used and the extra bookkeeping is added on top, so the files stay
   ordinary pytest files either way.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

# --- 1. make the package importable -----------------------------------------
_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
for _path in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


# --- 2. pytest, real or stood in for ----------------------------------------
try:  # pragma: no cover - depends on the environment
    import pytest as _real_pytest
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    _real_pytest = None


class _Skipped(Exception):
    """Raised by the stand-in's ``pytest.skip``."""


class _RaisesContext:
    """Context manager emulating ``pytest.raises``."""

    def __init__(self, expected: Any) -> None:
        self.expected = expected
        self.value: BaseException | None = None

    def __enter__(self) -> "_RaisesContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected!r}")
        if not issubclass(exc_type, self.expected):
            return False
        self.value = exc
        return True


class _Approx:
    """Minimal ``pytest.approx`` replacement."""

    def __init__(self, expected: float, rel: float = 1e-6, abs: float = 1e-12) -> None:
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other: Any) -> bool:
        return abs(other - self.expected) <= max(
            self.abs, self.rel * abs(self.expected)
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"approx({self.expected!r})"


def _record_parametrize(
    func: Callable[..., Any],
    argnames: str | Sequence[str],
    argvalues: Sequence[Any],
) -> None:
    """Remember a parametrisation so the bundled runner can expand it."""
    specs = list(getattr(func, "_parametrize_specs", []))
    specs.append((argnames, list(argvalues)))
    func._parametrize_specs = specs  # type: ignore[attr-defined]


class _Mark:
    """The ``pytest.mark`` namespace of the stand-in."""

    def parametrize(
        self,
        argnames: str | Sequence[str],
        argvalues: Sequence[Any],
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        real_mark = (
            _real_pytest.mark.parametrize if _real_pytest is not None else None
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            _record_parametrize(func, argnames, argvalues)
            if real_mark is not None:
                # Let real pytest own the collection as well.
                func = real_mark(argnames, argvalues, **kwargs)(func)
            return func

        return decorator

    def __getattr__(self, name: str) -> Callable[..., Any]:
        def decorator(*args: Any, **kwargs: Any) -> Any:
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda func: func

        return decorator


class _PytestFacade:
    """The ``pytest`` object handed to the test modules."""

    def __init__(self) -> None:
        self.mark = _Mark()

    def raises(self, expected: Any, *args: Any, **kwargs: Any) -> Any:
        if _real_pytest is not None:
            return _real_pytest.raises(expected, *args, **kwargs)
        return _RaisesContext(expected)

    def approx(self, expected: float, rel: float = 1e-6, abs: float = 1e-12) -> Any:
        if _real_pytest is not None:
            return _real_pytest.approx(expected, rel=rel, abs=abs)
        return _Approx(expected, rel, abs)

    def skip(self, reason: str = "") -> None:
        if _real_pytest is not None:
            _real_pytest.skip(reason)
        raise _Skipped(reason)

    def fail(self, reason: str = "") -> None:
        if _real_pytest is not None:
            _real_pytest.fail(reason)
        raise AssertionError(reason)


pytest = _PytestFacade()

# Expose the facade under the canonical module name too, so a test module
# that says ``import pytest`` picks up the same object.
if _real_pytest is None:
    sys.modules.setdefault("pytest", pytest)


__all__ = ["pytest"]
