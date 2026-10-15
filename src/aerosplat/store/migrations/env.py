"""Alembic environment for the catalog.

Driven programmatically by `aerosplat.store.catalog.upgrade` (no alembic.ini):
the caller passes an open connection in `config.attributes["connection"]`.
For a new revision during development:

    python -m aerosplat.store.catalog revision -m "what changed"
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import Connection

from aerosplat.store.db import Base, make_engine

config = context.config
target_metadata = Base.metadata


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,       # SQLite ALTER TABLE support for later revisions
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = make_engine(config.get_main_option("sqlalchemy.url") or "")
    with engine.connect() as conn:
        _run(conn)


if context.is_offline_mode():
    raise SystemExit("offline (SQL script) migrations are not supported for the catalog")
run_migrations_online()
