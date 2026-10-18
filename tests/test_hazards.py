"""The hazard log (docs/HAZARDS.md) and the `hazard` markers in tests/ must agree exactly."""

from pathlib import Path

import pytest

from aerosplat.hazards import HAZARD_ID, load_hazards, marked_tests
from aerosplat.paths import PROJECT_DIR

TESTS = PROJECT_DIR / "tests"


def test_log_parses():
    hazards = load_hazards()
    assert len(hazards) >= 10
    for hid, h in hazards.items():
        assert HAZARD_ID.fullmatch(hid) and h.id == hid
        assert h.title and h.status in {"відкрита", "частково", "покрита", "знята"}, h
        if h.status in {"частково", "покрита"}:
            assert h.tests, f"{hid} is '{h.status}' but lists no tests"


def test_log_and_markers_agree():
    hazards = load_hazards()
    marked = marked_tests(TESTS)
    assert marked, "no hazard markers found at all: the scanner is broken"
    problems = []
    for node, ids in sorted(marked.items()):
        for hid in sorted(ids - hazards.keys()):
            problems.append(f"{node}: marker names {hid}, which is not in the log")
        for hid in sorted(ids & hazards.keys()):
            if node not in hazards[hid].tests:
                problems.append(f"{node}: marked {hid} but not listed under it in the log")
    for hid, h in hazards.items():
        for node in h.tests:
            if hid not in marked.get(node, set()):
                problems.append(f"{hid}: lists {node}, which has no hazard('{hid}') marker "
                                f"(or does not exist)")
    assert not problems, "\n".join(problems)


def test_scanner_reads_functions_methods_and_multiple_ids(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text(
        "import pytest\n"
        "from pytest import mark\n"
        "@pytest.mark.hazard('H-001', 'H-002')\n"
        "def test_a(): pass\n"
        "@pytest.mark.slow\n"
        "def test_b(): pass\n"
        "class TestC:\n"
        "    @mark.hazard('H-003')\n"
        "    def test_d(self): pass\n", encoding="utf-8")
    assert marked_tests(tests, root=tmp_path) == {
        "tests/test_x.py::test_a": {"H-001", "H-002"},
        "tests/test_x.py::TestC::test_d": {"H-003"},
    }


def _log(tmp_path: Path, *rows: str) -> Path:
    p = tmp_path / "HAZARDS.md"
    p.write_text("| ID | Небезпека | Причини | Заходи | Статус | Тести |\n|---|---|---|---|---|---|\n"
                 + "\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_log_rejects_duplicates_and_broken_rows(tmp_path):
    ok = "| H-001 | a | b | c | відкрита | `tests/test_x.py::test_a`, `tests/test_y.py::T::t` |"
    h = load_hazards(_log(tmp_path, ok))["H-001"]
    assert h.tests == ("tests/test_x.py::test_a", "tests/test_y.py::T::t")
    with pytest.raises(ValueError, match="duplicate"):
        load_hazards(_log(tmp_path, ok, ok))
    with pytest.raises(ValueError, match="columns"):
        load_hazards(_log(tmp_path, "| H-002 | a | b | відкрита | — |"))
