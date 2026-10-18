"""The hazard log (docs/HAZARDS.md) as data, and the tests that claim to cover each hazard.

Tests reference hazards with `@pytest.mark.hazard("H-001", ...)`. tests/conftest.py rejects
unknown ids at collection; tests/test_hazards.py keeps the log's "linked tests" column and the
markers in the test sources identical, so the log cannot silently drift from the suite.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from aerosplat.paths import PROJECT_DIR

HAZARDS_MD = PROJECT_DIR / "docs" / "HAZARDS.md"
HAZARD_ID = re.compile(r"H-\d{3}")
_TEST_REF = re.compile(r"`([^`]+\.py::[^`]+)`")


@dataclass(frozen=True)
class Hazard:
    id: str
    title: str
    status: str
    tests: tuple[str, ...]   # node ids relative to the repo root: tests/x.py::test_y


def load_hazards(path: Path = HAZARDS_MD) -> dict[str, Hazard]:
    """Rows of the hazard table: | ID | hazard | causes | mitigations | status | tests |."""
    hazards: dict[str, Hazard] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or not HAZARD_ID.fullmatch(cells[0].strip("*")):
            continue
        hid = cells[0].strip("*")
        if len(cells) != 6:
            raise ValueError(f"{path}:{n}: {hid} has {len(cells)} columns, expected 6")
        if hid in hazards:
            raise ValueError(f"{path}:{n}: duplicate hazard id {hid}")
        hazards[hid] = Hazard(hid, cells[1].strip("*"), cells[4], tuple(_TEST_REF.findall(cells[5])))
    return hazards


def marked_tests(tests_dir: Path, root: Path = PROJECT_DIR) -> dict[str, set[str]]:
    """Static scan of `@pytest.mark.hazard(...)` decorators: node id -> hazard ids.

    Independent of which tests a given run selects. Covers module-level test functions and
    methods of test classes (`file.py::Class::test`).
    """
    out: dict[str, set[str]] = {}
    for py in sorted(tests_dir.rglob("test_*.py")):
        rel = py.relative_to(root).as_posix()
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node, prefix in _test_functions(tree):
            ids: set[str] = set()
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and _is_hazard_mark(dec):
                    ids |= {a.value for a in dec.args
                            if isinstance(a, ast.Constant) and isinstance(a.value, str)}
            if ids:
                out[f"{rel}::{prefix}{node.name}"] = ids
    return out


def _test_functions(tree: ast.Module) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]]:
    found: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append((node, ""))
        elif isinstance(node, ast.ClassDef):
            found += [(f, f"{node.name}::") for f in node.body
                      if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return found


def _is_hazard_mark(dec: ast.Call) -> bool:
    """`pytest.mark.hazard(...)` or `mark.hazard(...)`."""
    if not (isinstance(dec.func, ast.Attribute) and dec.func.attr == "hazard"):
        return False
    owner = dec.func.value
    return (isinstance(owner, ast.Attribute) and owner.attr == "mark"
            or isinstance(owner, ast.Name) and owner.id == "mark")
