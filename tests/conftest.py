import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "src/reconstruction"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURES = Path(__file__).parent / "fixtures"
