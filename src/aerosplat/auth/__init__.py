"""Access tokens for requests that do not come from this machine (Phase 6, ADR-0010 п. 6).

Requests from loopback stay open, exactly as before; anything else needs a token
created with `aerosplat token create <name>`. The HTTP side (one ASGI middleware for
REST, WebSocket and static files) is `aerosplat.server.auth`.
"""

# Light on purpose: `aerosplat.authored` imports the context; the token store (and its
# store/locking import) is imported explicitly from aerosplat.auth.tokens where needed.
from aerosplat.auth.context import Principal, current_principal

__all__ = ["Principal", "current_principal"]
