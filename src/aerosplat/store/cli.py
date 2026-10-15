"""`aerosplat catalog ...` — manage the project tree from the command line (until the UI, S2).

    aerosplat catalog list
    aerosplat catalog create-site --project demo --site pier-7 --origin 40.8910,-74.2163,233
    aerosplat catalog import-earthstudio --site demo/pier-7 --json export.json --footage footage/
    aerosplat catalog import-folder --site demo/pier-7 --images photos/ --capture phone-01 --source handheld
    aerosplat catalog migrate-dataset dataset_v6          # = tools/migrate_dataset.py
    aerosplat catalog upgrade                             # schema -> head (also done on open)
    aerosplat catalog revision -m "add jobs"              # dev: new Alembic revision
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from alembic import command

from aerosplat.geo.crs import GeoPoint
from aerosplat.store.catalog import Catalog, CatalogError, alembic_config
from aerosplat.store.models import BBox, CaptureSource, Site, SiteKind


def _bbox(text: str) -> BBox:
    w, s, e, n = (float(v) for v in text.split(","))
    return BBox(west=w, south=s, east=e, north=n)


def _print_tree(cat: Catalog) -> None:
    projects = cat.list_projects()
    if not projects:
        print(f"(порожньо) {cat.paths.root}")
        return
    for p in projects:
        print(f"{p.id}  {p.name}")
        for site in cat.list_sites(p.id):
            o = site.origin
            print(f"  └─ {site.key}  [{site.kind.value}]  origin=({o.lat:.8f}, {o.lon:.8f}, "
                  f"{o.h:.2f})")
            for c in cat.list_captures(site):
                n = len(cat.list_images(site, c.id))
                print(f"       capture {c.id}  {c.source.value}  {n} кадрів")
            for e in cat.list_epochs(site):
                print(f"       epoch {e.id}")
                for r in cat.list_runs(site, e.id):
                    flags = ("metric" if r.georeferenced else "relative") + f", {r.earth_model}"
                    print(f"         run {r.id}  {r.status.value}  ({flags})  "
                          f"products: {', '.join(sorted(r.products)) or '—'}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aerosplat catalog",
                                 description="Каталог Project/Site/Epoch/Capture (ADR-0006)")
    ap.add_argument("--projects-root", type=Path, default=None,
                    help="за замовчуванням $AEROSPLAT_PROJECTS або <repo>/projects")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="дерево проєктів, ділянок, захоплень і run-ів")

    cs = sub.add_parser("create-site", help="нова ділянка з фіксованим ENU-початком")
    cs.add_argument("--project", required=True)
    cs.add_argument("--site", required=True)
    cs.add_argument("--origin", type=GeoPoint.parse, required=True, metavar="LAT,LON,H",
                    help="WGS84, еліпсоїдальна висота; більше не змінюється")
    cs.add_argument("--name", default="")
    cs.add_argument("--kind", choices=[k.value for k in SiteKind], default="urban")
    cs.add_argument("--bbox", type=_bbox, default=None, metavar="W,S,E,N")

    ie = sub.add_parser("import-earthstudio", help="експорт Earth Studio → захоплення")
    ie.add_argument("--site", required=True, help="project/site")
    ie.add_argument("--json", type=Path, required=True)
    ie.add_argument("--footage", type=Path, required=True)
    ie.add_argument("--capture", default=None)
    ie.add_argument("--extra", type=Path, nargs="*", default=[],
                    help=".esp, ImagerySources.txt тощо → captures/<id>/source/")
    ie.add_argument("--move", action="store_true")

    ifo = sub.add_parser("import-folder", help="тека зображень без поз → захоплення")
    ifo.add_argument("--site", required=True)
    ifo.add_argument("--images", type=Path, required=True)
    ifo.add_argument("--capture", required=True)
    ifo.add_argument("--source", choices=[s.value for s in CaptureSource], required=True)
    ifo.add_argument("--license", default="")
    ifo.add_argument("--author", default="")
    ifo.add_argument("--move", action="store_true")

    sub.add_parser("migrate-dataset", help="legacy data/<ds> → Site (див. --help)",
                   add_help=False)
    sub.add_parser("upgrade", help="схема каталогу → head")
    rev = sub.add_parser("revision", help="(dev) нова міграція Alembic")
    rev.add_argument("-m", "--message", required=True)
    rev.add_argument("--autogenerate", action="store_true")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if "migrate-dataset" in argv:   # its own parser (tools/migrate_dataset.py)
        from aerosplat.store import legacy

        i = argv.index("migrate-dataset")
        pre = argv[:i]
        rest = argv[i + 1:]
        if "--projects-root" in pre:
            j = pre.index("--projects-root")
            rest += pre[j:j + 2]
        return legacy.main(rest)

    args = build_parser().parse_args(argv)
    cat = Catalog.open(args.projects_root)
    try:
        if args.cmd == "list":
            _print_tree(cat)
        elif args.cmd == "create-site":
            cat.ensure_project(args.project)
            site = cat.create_site(Site(id=args.site, project_id=args.project, name=args.name,
                                        kind=SiteKind(args.kind), origin=args.origin,
                                        bbox=args.bbox))
            cat.ensure_epoch(site)
            print(f"✅ {site.key}: {cat.site_dirs(site).root}")
        elif args.cmd == "import-earthstudio":
            from aerosplat.ingest.earthstudio import import_earthstudio

            site = cat.find_site(args.site)
            cap = import_earthstudio(cat, site, args.json, args.footage, capture_id=args.capture,
                                     extra_files=args.extra, move=args.move)
            print(f"✅ {site.key}: capture {cap.id}")
        elif args.cmd == "import-folder":
            from aerosplat.ingest.folder import import_folder

            site = cat.find_site(args.site)
            cap = import_folder(cat, site, args.images, capture_id=args.capture,
                                source=CaptureSource(args.source), license=args.license,
                                author=args.author, move=args.move)
            print(f"✅ {site.key}: capture {cap.id}")
        elif args.cmd == "upgrade":
            print(f"✅ {cat.paths.catalog_db}: схема актуальна")
        elif args.cmd == "revision":
            command.revision(alembic_config(cat.engine), message=args.message,
                             autogenerate=args.autogenerate)
    except (CatalogError, FileExistsError, FileNotFoundError) as e:
        print(f"❌ {e}")
        return 1
    finally:
        cat.close()
    return 0
