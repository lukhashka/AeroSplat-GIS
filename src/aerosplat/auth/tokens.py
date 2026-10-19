"""API tokens: `<projects_root>/auth/tokens.json`, hashes only.

    store = TokenStore.for_root(default_projects_root())
    record, secret = store.create("alice")      # the secret is shown once, never stored
    store.verify(secret)                        # -> TokenRecord, or None
    store.revoke("alice")

Why a file and not a catalog table: tokens are configuration, and a new Alembic
revision would make the catalog unreadable for older checkouts that the owner still
runs against the same `projects/` (Alembic refuses an unknown head). The file holds
SHA-256 hashes of 256-bit random secrets, so reading it reveals nothing usable;
a slow KDF is not needed for secrets that cannot be guessed.

Writes (CLI only — the server never writes this file) are atomic: a temporary file
and `os.replace`, under a cross-process lock. A corrupt or unreadable file is an
error, never "no tokens configured" silently turned into "everything allowed":
the middleware then denies every non-local request.
"""

from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from aerosplat.store.locking import file_lock

TOKEN_PREFIX = "asg_"
TOKEN_BYTES = 32                       # 256 bits of entropy
MAX_TOKEN_LEN = 256
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,63}$")
FILE_VERSION = 1


class TokenError(RuntimeError):
    """Bad name, unknown token, unreadable token file."""


class TokenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str                            # short public id (list/revoke), not secret
    name: str                          # who uses it; becomes `token:<name>` in the audit trail
    sha256: str                        # hex digest of the full secret
    created_at: AwareDatetime
    created_by: str = ""
    revoked_at: Optional[AwareDatetime] = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


def hash_token(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _os_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return ""


class TokenStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._cache_key: Optional[tuple[int, int]] = None
        self._cache: list[TokenRecord] = []
        self._cache_lock = threading.Lock()

    @classmethod
    def for_root(cls, projects_root: Path) -> "TokenStore":
        return cls(Path(projects_root) / "auth" / "tokens.json")

    # ── reading ──────────────────────────────────────────────────────────
    def _read(self) -> list[TokenRecord]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != FILE_VERSION:
                raise TokenError(f"{self.path}: unknown format (expected version {FILE_VERSION})")
            return [TokenRecord(**t) for t in data.get("tokens", [])]
        except TokenError:
            raise
        except (OSError, ValueError, TypeError, ValidationError) as e:
            raise TokenError(f"{self.path}: cannot read token file: {type(e).__name__}: {e}") from None

    def load(self) -> list[TokenRecord]:
        """All records (revoked too). Re-reads the file only when it changed on disk."""
        try:
            st = self.path.stat()
            key: Optional[tuple[int, int]] = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            key = None
        with self._cache_lock:
            if key is None:
                self._cache_key, self._cache = None, []
                return []
            if key != self._cache_key:
                self._cache = self._read()       # raises on a corrupt file: never cache that
                self._cache_key = key
            return list(self._cache)

    def active(self) -> list[TokenRecord]:
        return [t for t in self.load() if t.active]

    def verify(self, secret: Optional[str]) -> Optional[TokenRecord]:
        """The active record whose hash matches `secret`, or None.

        Every active hash is compared with `hmac.compare_digest` and the loop never
        exits early, so the time taken does not depend on which token (if any) matched.
        """
        if not secret or len(secret) > MAX_TOKEN_LEN or not secret.startswith(TOKEN_PREFIX):
            return None
        digest = hash_token(secret)
        found: Optional[TokenRecord] = None
        for rec in self.active():
            if hmac.compare_digest(rec.sha256, digest):
                found = rec
        return found

    # ── writing (CLI) ────────────────────────────────────────────────────
    def _write(self, records: list[TokenRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        body = {"version": FILE_VERSION,
                "note": "SHA-256 hashes of API tokens; manage with `aerosplat token`",
                "tokens": [r.model_dump(mode="json") for r in records]}
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # On Windows a reader holding the file open (the server, for a few microseconds)
        # makes the replace fail with a sharing violation; retry briefly.
        for attempt in range(50):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if attempt == 49:
                    tmp.unlink(missing_ok=True)
                    raise
                time.sleep(0.1)

    def _lock(self):
        return file_lock(self.path.with_name(self.path.name + ".lock"), timeout=30.0)

    def create(self, name: str) -> tuple[TokenRecord, str]:
        if not NAME_RE.fullmatch(name or ""):
            raise TokenError(f"bad token name '{name}': letters, digits, _ . @ - (up to 64)")
        with self._lock():
            records = self._read()
            if any(r.name == name and r.active for r in records):
                raise TokenError(f"an active token named '{name}' already exists "
                                 f"(revoke it first or choose another name)")
            ids = {r.id for r in records}
            rid = secrets.token_hex(4)
            while rid in ids:
                rid = secrets.token_hex(4)
            secret = TOKEN_PREFIX + secrets.token_urlsafe(TOKEN_BYTES)
            rec = TokenRecord(id=rid, name=name, sha256=hash_token(secret), created_at=_now(),
                              created_by=_os_user())
            self._write([*records, rec])
        return rec, secret

    def revoke(self, name_or_id: str) -> list[TokenRecord]:
        """Revoke the active token(s) with this id or name; returns what was revoked."""
        with self._lock():
            records = self._read()
            hit = [r for r in records if r.active and name_or_id in (r.id, r.name)]
            if not hit:
                raise TokenError(f"no active token with id or name '{name_or_id}'")
            now = _now()
            out = []
            for i, r in enumerate(records):
                if r in hit:
                    records[i] = r.model_copy(update={"revoked_at": now})
                    out.append(records[i])
            self._write(records)
        return out
