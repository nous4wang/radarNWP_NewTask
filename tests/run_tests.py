"""A dependency-free test runner for the radar-nwp-bench suite.

pytest is not installed in the benchmark image, so this script runs the
suite on its own:

    python tests/run_tests.py

It imports every ``tests/test_*.py`` module, discovers the ``test_*``
functions, expands the ``pytest.mark.parametrize`` cases, runs everything and
prints a summary.  The exit code is 0 when every test passed and 1 otherwise.

The test modules are ordinary pytest files, so ``pytest tests/`` works too
wherever pytest is available.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator


# --- make the package importable -------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
for _path in (str(REPO_ROOT), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


# --- import the test modules ------------------------------------------------
def _load_module(path: Path) -> Any:
    """Import a test module from its file path."""
    name = f"_bench_tests_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _discover_modules() -> list[tuple[str, Any]]:
    modules: list[tuple[str, Any]] = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        modules.append((path.stem, _load_module(path)))
    return modules


# --- parametrisation --------------------------------------------------------
def _parametrize_cases(func: Callable[..., Any]) -> list[tuple[dict[str, Any], str]]:
    """Expand the recorded parametrisation of ``func`` into keyword sets.

    Only the single-argument form is needed by this suite, but stacked
    decorators are handled as a Cartesian product for good measure.
    """
    specs = getattr(func, "_parametrize_specs", None)
    if not specs:
        return [({}, "")]

    per_spec: list[list[tuple[dict[str, Any], str]]] = []
    for argnames, argvalues in specs:
        names = (
            [n.strip() for n in argnames.split(",")]
            if isinstance(argnames, str)
            else list(argnames)
        )
        cases: list[tuple[dict[str, Any], str]] = []
        for value in argvalues:
            if len(names) == 1:
                kwargs = {names[0]: value}
            else:
                kwargs = dict(zip(names, value))
            label = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            cases.append((kwargs, label))
        per_spec.append(cases)

    combined: list[tuple[dict[str, Any], str]] = [({}, "")]
    for cases in per_spec:
        merged: list[tuple[dict[str, Any], str]] = []
        for kwargs_a, label_a in combined:
            for kwargs_b, label_b in cases:
                labels = [x for x in (label_a, label_b) if x]
                merged.append(
                    ({**kwargs_a, **kwargs_b}, ", ".join(labels))
                )
        combined = merged
    return combined


# --- running ----------------------------------------------------------------
class _Result:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[str] = []
        self.duration = 0.0

    @property
    def total(self) -> int:
        return len(self.passed) + len(self.failed) + len(self.skipped)


def _run_one(func: Callable[..., Any], kwargs: dict[str, Any], label: str, result: _Result) -> None:
    name = func.__name__
    if label:
        name = f"{name}[{label}]"

    try:
        func(**kwargs)
    except Exception as exc:  # noqa: BLE001 - a runner reports everything
        from _pytest_compat import _Skipped

        if isinstance(exc, _Skipped):
            result.skipped.append(name)
            print(f"  SKIP {name}")
            return
        result.failed.append((name, traceback.format_exc()))
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        return
    result.passed.append(name)
    print(f"  PASS {name}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    only = [a for a in argv if not a.startswith("-")]

    print("radar-nwp-bench test suite")
    print(f"repo root: {REPO_ROOT}")
    print("-" * 70)

    modules = _discover_modules()
    if not modules:
        print("no test modules found")
        return 1

    result = _Result()
    start = time.perf_counter()

    for mod_name, module in modules:
        print(f"\n{mod_name}")
        setup = getattr(module, "setup_module", None)
        teardown = getattr(module, "teardown_module", None)
        if callable(setup):
            setup()

        tests = [
            (name, getattr(module, name))
            for name in sorted(dir(module))
            if name.startswith("test_") and callable(getattr(module, name))
        ]
        if only:
            tests = [t for t in tests if any(o in t[0] for o in only)]
        if not tests:
            print("  (no tests)")
        for name, func in tests:
            for kwargs, label in _parametrize_cases(func):
                _run_one(func, kwargs, label, result)

        if callable(teardown):
            teardown()

    result.duration = time.perf_counter() - start

    print("-" * 70)
    print(f"total:    {result.total}")
    print(f"passed:   {len(result.passed)}")
    print(f"failed:   {len(result.failed)}")
    print(f"skipped:  {len(result.skipped)}")
    print(f"runtime:  {result.duration:.2f}s")

    if result.failed:
        print("\n=== FAILURES ===")
        for name, tb in result.failed:
            print(f"\n--- {name} ---")
            print(tb)
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
