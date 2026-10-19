"""Who is making the current request (set by the auth middleware, read by audit code).

A context variable rather than a FastAPI dependency: the audit trail is written deep
inside the stores (`aerosplat.authored.models.actor`), and threading a request object
through every store call would be exactly the per-route sprinkling the middleware
exists to avoid. Starlette copies the context into the threadpool that runs sync
endpoints, so the value set in the middleware is visible there.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class Principal:
    via: Literal["loopback", "token"]
    name: Optional[str] = None        # token name (via="token")
    token_id: Optional[str] = None

    @property
    def actor(self) -> Optional[str]:
        """Audit name for changes made by this principal; None = keep the old rule."""
        return f"token:{self.name}" if self.via == "token" and self.name else None


_current: ContextVar[Optional[Principal]] = ContextVar("aerosplat_principal", default=None)


def current_principal() -> Optional[Principal]:
    return _current.get()


def set_principal(p: Optional[Principal]) -> Token:
    return _current.set(p)


def reset_principal(token: Token) -> None:
    _current.reset(token)
