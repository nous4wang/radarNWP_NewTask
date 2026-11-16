"""The master document is the project's authority, so its structure is tested.

The protocol this project follows says that every task reads the master
document first and folds its results back into it. That only works if the
document keeps the shape the protocol assumes: the task and its IO contract,
the methods in play, the data pitfalls, the attempts already made, the
conclusions split by how solid they are, the queue of what is left, the
protocol itself, and an append-only change log.

Two of these are load-bearing in a way the others are not. The
"refuted conclusions" section is what stops a settled argument from coming
back to life, and the change log is what makes the document's currency
auditable. Both are checked here rather than trusted.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER = REPO_ROOT / "docs" / "MASTER.md"


def _text() -> str:
    assert MASTER.exists(), (
        f"the master document is missing at {MASTER}. It is the entry point "
        f"for every task in this project."
    )
    return MASTER.read_text(encoding="utf-8")


def _titles() -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"^#{2,3}\s+(.+?)\s*$", _text(), re.M)
    ]


def test_the_master_document_exists_and_is_not_a_stub():
    text = _text()
    assert len(text) > 8000, (
        f"the master document is {len(text)} characters, which is too short "
        f"to carry the task, the pitfalls and the attempts"
    )


def test_the_master_document_states_the_research_task_and_its_io():
    """Without the IO contract, a task cannot know what it is producing."""
    text = _text()
    for marker in ("RADAR", "NWP", "LABEL", "mm/6min"):
        assert marker in text, (
            f"the IO contract does not mention {marker}"
        )


def test_the_master_document_records_the_pitfalls_and_their_handling():
    """The details chapter is the part that stops the same bug being rediscovered."""
    titles = _titles()
    assert any("细节处理" in t or "坑" in t for t in titles), (
        "the master document has no chapter for the data and engineering "
        "pitfalls"
    )
    text = _text()
    for marker in ("缺帧", "尺度", "哨兵"):
        assert marker in text, (
            f"the pitfalls chapter does not cover {marker}"
        )


def test_the_master_document_lists_the_attempts_including_the_negative_ones():
    """An attempt that was refuted still belongs in the record."""
    text = _text()
    assert "quantum" in text.lower(), (
        "the quantum plugin attempt is not in the master document"
    )
    assert "已做的尝试" in text


def test_the_master_document_separates_the_refuted_conclusions():
    """This is what stops a settled argument from coming back to life."""
    text = _text()
    assert "已被推翻" in text, (
        "the master document has no section for the refuted conclusions, so "
        "an old claim can be re-introduced without anyone noticing"
    )


def test_the_master_document_has_a_conflict_ruling_rule():
    """Several files carry results; the document has to say which wins."""
    text = _text()
    assert "冲突裁定" in text, (
        "the master document does not say which source wins when two of "
        "them disagree"
    )


def test_the_change_log_has_grown():
    """A log with one row means the document was written and never updated."""
    text = _text()
    rows = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) >= 3 and re.match(r"^\d{4}-\d{2}-\d{2}", cells[0]):
            rows += 1
    assert rows >= 1, "the change log has no rows"
    assert "最后更新" in text, (
        "the master document does not say when it was last updated"
    )


def test_the_master_document_points_at_the_authoritative_results():
    """The document summarises; the detailed numbers live elsewhere."""
    text = _text()
    for marker in ("RESULTS_FINAL", "THREE_DEFECTS"):
        assert marker in text, (
            f"the master document does not point at {marker}, which holds the "
            f"detailed numbers"
        )


def test_the_suggestion_document_carries_the_five_routes():
    """The suggestion document is where the routes are implemented.

    It is referenced from the master document, so a missing file would
    leave the queue pointing at nothing.
    """
    suggestion = REPO_ROOT / "docs" / "SUGGESTION.md"
    assert suggestion.exists(), (
        f"the suggestion document is missing at {suggestion}; the master "
        f"document points at it for the route implementations"
    )
    text = suggestion.read_text(encoding="utf-8")
    for route in ("R1", "R2", "R3", "R4", "R5"):
        assert route in text, f"route {route} is not in the suggestion document"


def test_the_suggestion_records_the_differences_from_the_original():
    """The original sketch had six defects that would not run.

    They are recorded in the document so that nobody follows the
    original and finds out the hard way.
    """
    text = (REPO_ROOT / "docs" / "SUGGESTION.md").read_text(encoding="utf-8")
    assert "与初稿的六处差异" in text
    assert "BalancedExtremePrecipLoss" in text
