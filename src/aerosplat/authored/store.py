"""Reading and writing `sites/<site>/authored/` (ADR-0006, ACCESS_POINTS §3.5).

Two files, with different rules:

    access_points.json   the current state; rewritten atomically on every change
    audit.jsonl          append-only history: who changed what, and what it was before

The state file is what everything reads; the audit log is what makes a change
answerable for. Nothing here ever touches a run directory, and nothing in a run
directory is allowed to write here.

Concurrency: one API process owns a Site's authored files, so an in-process lock per
path is enough. Writes go to a temporary file and are moved into place, so a crash
mid-write leaves the previous state intact rather than half a file.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable, Optional

from aerosplat.authored.models import (AccessPoint, AuditEntry, AuthoredFile, Edit, Lifecycle,
                                       actor, utcnow)
from aerosplat.store.paths import ACCESS_POINTS_JSON, AUDIT_JSONL, SiteDirs

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve() if path.exists() else path.absolute())
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.RLock()
        return lock


class AuthoredError(ValueError):
    """A change the store refuses: unknown feature, duplicate id, unreadable file."""


class NotFound(AuthoredError, KeyError):
    def __str__(self) -> str:                   # KeyError repr()s its message otherwise
        return str(self.args[0]) if self.args else ""


class AuthoredStore:
    """The authored features of one Site."""

    def __init__(self, authored_dir: Path, site_key: str):
        self.dir = Path(authored_dir)
        self.site_key = site_key
        self._lock = _lock_for(self.dir)

    @classmethod
    def for_site(cls, site_dirs: SiteDirs, site_key: str) -> "AuthoredStore":
        return cls(site_dirs.authored, site_key)

    @property
    def state_path(self) -> Path:
        return self.dir / ACCESS_POINTS_JSON

    @property
    def audit_path(self) -> Path:
        return self.dir / AUDIT_JSONL

    # ── read ─────────────────────────────────────────────────────────────
    def load(self) -> AuthoredFile:
        """The whole file, or an empty one when a Site has nothing authored yet."""
        with self._lock:
            if not self.state_path.is_file():
                return AuthoredFile(site_key=self.site_key)
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                return AuthoredFile(**data)
            except (OSError, ValueError) as e:
                raise AuthoredError(f"{self.state_path}: {e}") from None

    def features(self, *, facade_id: Optional[str] = None,
                 lifecycle: Optional[Iterable[Lifecycle]] = None) -> list[AccessPoint]:
        keep = set(lifecycle) if lifecycle is not None else None
        out = [f for f in self.load().features
               if (facade_id is None or f.host.facade_id == facade_id)
               and (keep is None or f.lifecycle in keep)]
        return sorted(out, key=lambda f: f.id)

    def get(self, feature_id: str) -> AccessPoint:
        for f in self.load().features:
            if f.id == feature_id:
                return f
        raise NotFound(f"no access point '{feature_id}' on site {self.site_key}")

    def audit(self, limit: int = 200) -> list[AuditEntry]:
        """The last `limit` audit entries, oldest first. Unparsable lines are skipped."""
        with self._lock:
            if not self.audit_path.is_file():
                return []
            lines = self.audit_path.read_text(encoding="utf-8").splitlines()
        out: list[AuditEntry] = []
        for line in lines[-limit:]:
            if not line.strip():
                continue
            try:
                out.append(AuditEntry(**json.loads(line)))
            except ValueError:
                continue
        return out

    # ── write ────────────────────────────────────────────────────────────
    def _save(self, state: AuthoredFile) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        state.updated_at = utcnow()
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(state.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _append_audit(self, entry: AuditEntry) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json(by_alias=True) + "\n")

    def create(self, feature: AccessPoint, *, by: Optional[str] = None,
               what: str = "drew") -> AccessPoint:
        with self._lock:
            state = self.load()
            if any(f.id == feature.id for f in state.features):
                raise AuthoredError(f"access point '{feature.id}' already exists")
            state.features.append(feature)
            self._save(state)
            self._append_audit(AuditEntry(
                by=actor(by), op="create", feature_id=feature.id, site_key=self.site_key,
                what=f"{what} {feature.class_} on {feature.host.facade_id}",
                after=feature.model_dump(mode="json", by_alias=True)))
            return feature

    @staticmethod
    def _carry_over(new: AccessPoint, old: AccessPoint, who: str, what: str) -> None:
        """Keep the old record's history, and append this change to it.

        Who created a feature and when is not the caller's to rewrite, so it is taken
        from the stored record rather than from the incoming one.
        """
        new.provenance.created_by = old.provenance.created_by
        new.provenance.created_at = old.provenance.created_at
        new.provenance.edits = [*old.provenance.edits, Edit(by=who, at=utcnow(), what=what)]

    def replace(self, feature: AccessPoint, *, by: Optional[str] = None, what: str = "edited",
                op: str = "update") -> AccessPoint:
        """Overwrite an existing feature, recording the previous value in the audit log."""
        return self.replace_many([feature], by=by, what=what, op=op)[0]

    def replace_many(self, features: list[AccessPoint], *, by: Optional[str] = None,
                     what: str = "edited", op: str = "update") -> list[AccessPoint]:
        """Overwrite several features in one state write, one audit entry each."""
        who = actor(by)
        with self._lock:
            state = self.load()
            index = {f.id: i for i, f in enumerate(state.features)}
            missing = [f.id for f in features if f.id not in index]
            if missing:
                raise NotFound(f"no access point '{missing[0]}' on site {self.site_key}")
            olds = [state.features[index[f.id]] for f in features]
            for new, old in zip(features, olds):
                self._carry_over(new, old, who, what)
                state.features[index[new.id]] = new
            self._save(state)
            for new, old in zip(features, olds):
                self._append_audit(AuditEntry(
                    by=who, op=op, feature_id=new.id, site_key=self.site_key, what=what,
                    before=old.model_dump(mode="json", by_alias=True),
                    after=new.model_dump(mode="json", by_alias=True)))
            return features

    def delete(self, feature_id: str, *, by: Optional[str] = None) -> AccessPoint:
        """Remove a feature outright.

        Note that rejecting is not deleting: a wrong candidate should go to
        `lifecycle = rejected` and stay (§3.4). Deleting is for a mistake in the
        drawing itself; the audit log keeps the record either way.
        """
        with self._lock:
            state = self.load()
            for i, f in enumerate(state.features):
                if f.id == feature_id:
                    break
            else:
                raise NotFound(f"no access point '{feature_id}' on site {self.site_key}")
            del state.features[i]
            self._save(state)
            self._append_audit(AuditEntry(
                by=actor(by), op="delete", feature_id=feature_id, site_key=self.site_key,
                what=f"deleted {f.class_} on {f.host.facade_id}",
                before=f.model_dump(mode="json", by_alias=True)))
            return f
