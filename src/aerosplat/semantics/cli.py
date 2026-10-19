"""`aerosplat semantics ...` — exports and checks on a run's semantic layer.

    aerosplat semantics cityjson --site default/dataset_1910 [--run R] [--lifecycle verified]
                                 -o out.city.json
    aerosplat semantics match --site default/dataset_1910 --from-run R1 [--to-run R2]

Read-only on the project tree: `cityjson` writes only the output file, `match` only prints.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Optional

from aerosplat.authored.models import Lifecycle
from aerosplat.logging_utils import fix_console_encoding
from aerosplat.store.paths import default_projects_root
from aerosplat.store.catalog import Catalog


def _catalog(root: Optional[Path]) -> Catalog:
    return Catalog.open(root or default_projects_root())


def cmd_cityjson(args: argparse.Namespace) -> int:
    from aerosplat.semantics.cityjson import export_site

    cat = _catalog(args.projects_root)
    try:
        site = cat.find_site(args.site)
        doc, report, errors, warnings = export_site(
            cat, site, run_id=args.run, lifecycles=[Lifecycle(x) for x in args.lifecycle])
    finally:
        cat.close()
    for w in warnings:
        print(f"⚠️  {w}")
    if errors:
        print(f"❌ CityJSON не пройшов перевірку ({len(errors)} помилок), файл не записано:")
        for e in errors[:20]:
            print(f"   {e}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    r = report
    print(f"✅ {args.out}: {r.buildings} будівель (LoD1: {r.solids}, LoD3: {r.lod3}), "
          f"отворів на стінах {r.openings} (з інших run-ів {r.openings_carried}), "
          f"pad-зон на дахах {r.roof_pads}, окремих об'єктів {r.generic}; "
          f"контурів виправлено {len(r.repaired)}, пропущено {len(r.skipped)}; "
          f"{doc['metadata']['referenceSystem'].rsplit('/', 1)[-1]}, висоти еліпсоїдні")
    for s in r.skipped[:10]:
        print(f"   пропущено: {s}")
    return 0


def cmd_match(args: argparse.Namespace) -> int:
    from aerosplat.semantics.matching import match_facades
    from aerosplat.semantics.products import require_semantics

    cat = _catalog(args.projects_root)
    try:
        site = cat.find_site(args.site)
        a = require_semantics(cat, site, run_id=args.from_run)
        b = require_semantics(cat, site, run_id=args.to_run)
    finally:
        cat.close()
    facades = [f for f in a.result.facades if args.building is None or f.building_id == args.building]
    matches = match_facades(facades, b.result.facades)
    print(f"{a.run.id} → {b.run.id}: {len(matches)} фасадів, "
          + ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(m.status for m in matches).items())))
    for m in matches:
        if args.building is not None or m.status in ("ambiguous", "split"):
            print(f"  {m.a_id:12s} {m.status:9s} {','.join(m.b_ids):24s} {m.confidence:.2f}  {m.reason}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aerosplat semantics", description=__doc__.split("\n")[0])
    ap.add_argument("--projects-root", type=Path, default=None,
                    help="за замовчуванням $AEROSPLAT_PROJECTS або <repo>/projects")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cityjson", help="експорт будівель і точок доступу в CityJSON 2.0")
    c.add_argument("--site", required=True, help="project/site")
    c.add_argument("--run", default=None, help="run (за замовчуванням найновіший із семантикою)")
    c.add_argument("--lifecycle", nargs="+", default=["verified"],
                   choices=[x.value for x in Lifecycle], help="які точки доступу експортувати")
    c.add_argument("-o", "--out", type=Path, required=True)
    c.set_defaults(func=cmd_cityjson)
    m = sub.add_parser("match", help="зіставлення фасадів двох run-ів (ACCESS_POINTS §5.9)")
    m.add_argument("--site", required=True)
    m.add_argument("--from-run", required=True)
    m.add_argument("--to-run", default=None)
    m.add_argument("--building", default=None, help="лише фасади цієї будівлі run-а from")
    m.set_defaults(func=cmd_match)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    fix_console_encoding()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
