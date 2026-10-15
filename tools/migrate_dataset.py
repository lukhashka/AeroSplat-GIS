#!/usr/bin/env python3
"""Migrate a legacy data/<dataset> layout into the project tree as a Site (ADR-0006).

    python tools/migrate_dataset.py dataset_v6 --dry-run     # show the plan
    python tools/migrate_dataset.py dataset_v6               # copy -> Site default/dataset_v6
    python tools/migrate_dataset.py dataset_v6 --move        # move (empties the legacy layout)

Same as `aerosplat catalog migrate-dataset ...`; logic in aerosplat/store/legacy.py.
"""

from aerosplat.logging_utils import fix_console_encoding
from aerosplat.store.legacy import main

if __name__ == "__main__":
    fix_console_encoding()
    raise SystemExit(main())
