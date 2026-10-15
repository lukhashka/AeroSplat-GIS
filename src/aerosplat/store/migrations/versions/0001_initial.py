"""Initial catalog: projects, sites, epochs, captures, images, runs, products.

Revision ID: 0001
Revises:
Create Date: 2026-10-15

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# ADR-0006: a Site's ENU origin is fixed at creation. Enforced in the ORM
# (SiteRow validator) and, independently of any client code, here.
SITE_ORIGIN_TRIGGER = """
CREATE TRIGGER site_origin_immutable
BEFORE UPDATE OF origin_lat, origin_lon, origin_h ON sites
WHEN NEW.origin_lat IS NOT OLD.origin_lat
  OR NEW.origin_lon IS NOT OLD.origin_lon
  OR NEW.origin_h IS NOT OLD.origin_h
BEGIN
    SELECT RAISE(ABORT, 'site origin is immutable');
END
"""


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_projects"),
    )
    op.create_table(
        "sites",
        sa.Column("pk", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.String(64), nullable=False),
        sa.Column("slug", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("origin_lat", sa.Float(), nullable=False),
        sa.Column("origin_lon", sa.Float(), nullable=False),
        sa.Column("origin_h", sa.Float(), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=True),
        sa.Column("taxonomy_packs", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"],
                                name="fk_sites_project_id_projects"),
        sa.PrimaryKeyConstraint("pk", name="pk_sites"),
        sa.UniqueConstraint("project_id", "slug", name="uq_sites_project_slug"),
    )
    op.create_index("ix_sites_project_id", "sites", ["project_id"])
    op.create_table(
        "epochs",
        sa.Column("pk", sa.Integer(), nullable=False),
        sa.Column("site_pk", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(64), nullable=False),
        sa.Column("label", sa.String(200), nullable=False),
        sa.Column("valid_from", sa.DateTime(), nullable=True),
        sa.Column("valid_to", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["site_pk"], ["sites.pk"], name="fk_epochs_site_pk_sites"),
        sa.PrimaryKeyConstraint("pk", name="pk_epochs"),
        sa.UniqueConstraint("site_pk", "slug", name="uq_epochs_site_slug"),
    )
    op.create_index("ix_epochs_site_pk", "epochs", ["site_pk"])
    op.create_table(
        "captures",
        sa.Column("pk", sa.Integer(), nullable=False),
        sa.Column("site_pk", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(64), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("license", sa.Text(), nullable=False),
        sa.Column("author", sa.String(200), nullable=False),
        sa.Column("attribution", sa.Text(), nullable=False),
        sa.Column("captured_at", sa.DateTime(), nullable=True),
        sa.Column("pose_prior_sigma_m", sa.Float(), nullable=True),
        sa.Column("metadata_file", sa.String(255), nullable=True),
        sa.Column("frame_prefix", sa.String(128), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["site_pk"], ["sites.pk"], name="fk_captures_site_pk_sites"),
        sa.PrimaryKeyConstraint("pk", name="pk_captures"),
        sa.UniqueConstraint("site_pk", "slug", name="uq_captures_site_slug"),
    )
    op.create_index("ix_captures_site_pk", "captures", ["site_pk"])
    op.create_table(
        "images",
        sa.Column("pk", sa.Integer(), nullable=False),
        sa.Column("capture_pk", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("bytes", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("prior_lat", sa.Float(), nullable=True),
        sa.Column("prior_lon", sa.Float(), nullable=True),
        sa.Column("prior_h", sa.Float(), nullable=True),
        sa.Column("prior_sigma_m", sa.Float(), nullable=True),
        sa.Column("pose", sa.JSON(), nullable=True),
        sa.Column("registration_status", sa.String(16), nullable=False),
        sa.Column("privacy_flags", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["capture_pk"], ["captures.pk"],
                                name="fk_images_capture_pk_captures"),
        sa.PrimaryKeyConstraint("pk", name="pk_images"),
        sa.UniqueConstraint("capture_pk", "name", name="uq_images_capture_name"),
    )
    op.create_index("ix_images_capture_pk", "images", ["capture_pk"])
    op.create_index("ix_images_sha256", "images", ["sha256"])
    op.create_table(
        "runs",
        sa.Column("pk", sa.Integer(), nullable=False),
        sa.Column("site_pk", sa.Integer(), nullable=False),
        sa.Column("epoch_pk", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("earth_model", sa.String(32), nullable=False),
        sa.Column("georeferenced", sa.Boolean(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("git_commit", sa.String(40), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["site_pk"], ["sites.pk"], name="fk_runs_site_pk_sites"),
        sa.ForeignKeyConstraint(["epoch_pk"], ["epochs.pk"], name="fk_runs_epoch_pk_epochs"),
        sa.PrimaryKeyConstraint("pk", name="pk_runs"),
        sa.UniqueConstraint("epoch_pk", "slug", name="uq_runs_epoch_slug"),
    )
    op.create_index("ix_runs_site_pk", "runs", ["site_pk"])
    op.create_index("ix_runs_epoch_pk", "runs", ["epoch_pk"])
    op.create_table(
        "run_captures",
        sa.Column("run_pk", sa.Integer(), nullable=False),
        sa.Column("capture_pk", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["run_pk"], ["runs.pk"], name="fk_run_captures_run_pk_runs"),
        sa.ForeignKeyConstraint(["capture_pk"], ["captures.pk"],
                                name="fk_run_captures_capture_pk_captures"),
        sa.PrimaryKeyConstraint("run_pk", "capture_pk", name="pk_run_captures"),
    )
    op.create_table(
        "products",
        sa.Column("pk", sa.Integer(), nullable=False),
        sa.Column("run_pk", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("path", sa.String(512), nullable=False),
        sa.ForeignKeyConstraint(["run_pk"], ["runs.pk"], name="fk_products_run_pk_runs"),
        sa.PrimaryKeyConstraint("pk", name="pk_products"),
        sa.UniqueConstraint("run_pk", "kind", name="uq_products_run_kind"),
    )
    op.create_index("ix_products_run_pk", "products", ["run_pk"])
    if op.get_bind().dialect.name == "sqlite":
        op.execute(SITE_ORIGIN_TRIGGER)


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS site_origin_immutable")
    for table in ("products", "run_captures", "runs", "images", "captures", "epochs",
                  "sites", "projects"):
        op.drop_table(table)
