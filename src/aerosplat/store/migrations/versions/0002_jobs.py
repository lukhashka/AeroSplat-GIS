"""Job queue: jobs, job_events, workers (TARGET_ARCHITECTURE §8, ADR-0010).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-15

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("pk", sa.Integer(), nullable=False),
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("resource", sa.String(8), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("site_key", sa.String(130), nullable=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("worker_id", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("pk", name="pk_jobs"),
        sa.UniqueConstraint("id", name="uq_jobs_id"),
    )
    op.create_index("ix_jobs_resource", "jobs", ["resource"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_site_key", "jobs", ["site_key"])
    op.create_table(
        "job_events",
        sa.Column("pk", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.String(32), nullable=False),
        sa.Column("ts", sa.DateTime(), nullable=False),
        sa.Column("type", sa.String(16), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name="fk_job_events_job_id_jobs"),
        sa.PrimaryKeyConstraint("pk", name="pk_job_events"),
    )
    op.create_index("ix_job_events_job_id", "job_events", ["job_id"])
    op.create_table(
        "workers",
        sa.Column("id", sa.String(128), nullable=False),
        sa.Column("resource", sa.String(8), nullable=False),
        sa.Column("host", sa.String(128), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(), nullable=False),
        sa.Column("current_job", sa.String(32), nullable=True),
        sa.Column("stopped_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_workers"),
    )


def downgrade() -> None:
    for table in ("workers", "job_events", "jobs"):
        op.drop_table(table)
