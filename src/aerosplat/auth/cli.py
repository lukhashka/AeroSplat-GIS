"""`aerosplat token ...` — API tokens for access from other machines.

    aerosplat token create alice          # prints the token ONCE; only its hash is stored
    aerosplat token list [--all]
    aerosplat token revoke alice          # by name or id; takes effect on the next request

Requests from this machine (loopback) need no token. A server bound to a non-loopback
address (`aerosplat up --host 0.0.0.0`) refuses to start until at least one token exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from aerosplat.auth.tokens import TokenError, TokenStore
from aerosplat.store.paths import default_projects_root


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aerosplat token",
                                 description="Токени доступу до API не з localhost")
    ap.add_argument("--projects-root", type=Path, default=None,
                    help="за замовчуванням $AEROSPLAT_PROJECTS або <repo>/projects")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="новий токен (показується один раз)")
    c.add_argument("name", help="хто ним користується; потрапляє в аудит як token:<name>")
    ls = sub.add_parser("list", help="токени (без секретів)")
    ls.add_argument("--all", action="store_true", help="разом із відкликаними")
    r = sub.add_parser("revoke", help="відкликати за іменем або id")
    r.add_argument("name_or_id")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    store = TokenStore.for_root(args.projects_root or default_projects_root())
    try:
        if args.cmd == "create":
            rec, secret = store.create(args.name)
            print(f"Токен '{rec.name}' (id {rec.id}) створено. Збережіть його зараз — "
                  f"повторно його не показати:\n\n    {secret}\n")
            print("У браузері: відкрийте UI з іншої машини й вставте токен у запит доступу.\n"
                  "Скрипти: заголовок `Authorization: Bearer <токен>`.")
            return 0
        if args.cmd == "list":
            records = store.load() if args.all else store.active()
            if not records:
                print(f"(немає {'жодного' if args.all else 'активного'} токена) {store.path}")
            for t in records:
                state = "активний" if t.active else f"відкликано {t.revoked_at:%Y-%m-%d %H:%M}"
                print(f"{t.id}  {t.name:<24} створено {t.created_at:%Y-%m-%d %H:%M} "
                      f"({t.created_by or '?'})  {state}")
            return 0
        if args.cmd == "revoke":
            for t in store.revoke(args.name_or_id):
                print(f"Відкликано: {t.name} (id {t.id})")
            return 0
    except TokenError as e:
        print(f"❌ {e}")
        return 1
    return 2
