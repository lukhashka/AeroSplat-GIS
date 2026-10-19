"""Who may talk to the API: loopback stays open, everyone else needs a token.

One pure-ASGI middleware in front of the whole app — REST, WebSocket, the scene
endpoints and the static mounts alike — instead of per-route dependencies, so a new
router cannot be forgotten. Rules, in order:

1. **Loopback trust** is granted only when *all* hold: the TCP peer (`scope["client"]`,
   set by the server from the socket, never from a header) is a loopback address; the
   request carries no proxy header (`X-Forwarded-For`, `Forwarded`, `X-Real-IP`, ...),
   because a reverse proxy or tunnel on this machine makes remote traffic arrive from
   127.0.0.1; `Host` is a loopback name (DNS rebinding); `Origin`, if sent, is a
   loopback origin (another web site open in the owner's browser must not drive the
   API). Headers can only *take away* loopback trust, never grant it.
   `AEROSPLAT_REQUIRE_AUTH=1` (`aerosplat up --require-auth`) disables it entirely.
2. Otherwise a **token**: `Authorization: Bearer <token>` (scripts), or the
   `aerosplat_token` cookie that `POST /api/v2/auth/login` sets for the browser UI.
   The cookie is what makes `<img>` thumbnails, the classic viewer's iframe and the
   job WebSocket work: browsers cannot attach an Authorization header to those, and
   putting the token in a URL would leak it into logs and history. The cookie is
   HttpOnly + SameSite=Strict; cookie-authenticated unsafe requests and WebSocket
   handshakes must also come from the same origin (CSRF / cross-site WebSocket).
3. A few **public paths** need no auth at all: the UI shell and static assets
   (`/`, `/app/...`, `/web/...`) — code, not data — and `/api/v2/auth/*`, so the
   UI can ask for a token in the first place.

Denied HTTP requests get 401 JSON (`{"detail", "auth": "token-required"}`); a denied
WebSocket handshake is closed before accept (HTTP 403). Tokens are never logged.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.requests import cookie_parser
from starlette.types import ASGIApp, Receive, Scope, Send

from aerosplat.auth.context import Principal, reset_principal, set_principal
from aerosplat.auth.tokens import TokenError, TokenRecord, TokenStore
from aerosplat.store.paths import default_projects_root

COOKIE = "aerosplat_token"
COOKIE_MAX_AGE_S = 30 * 24 * 3600
REQUIRE_AUTH_ENV = "AEROSPLAT_REQUIRE_AUTH"

# Any of these means the request was relayed by a proxy/tunnel: the TCP peer is then the
# proxy, not the client, so a loopback peer proves nothing.
PROXY_HEADERS = ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
                 "x-forwarded-port", "x-forwarded-server", "x-real-ip", "x-client-ip",
                 "x-cluster-client-ip", "x-original-forwarded-for", "cf-connecting-ip",
                 "true-client-ip", "fastly-client-ip", "fly-client-ip", "via")
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
PUBLIC_PREFIXES = ("/app/", "/web/", "/api/v2/auth/")
PUBLIC_EXACT = ("/", "/app", "/web", "/favicon.ico")


# ── address helpers ──────────────────────────────────────────────────────
def is_loopback_ip(text: str) -> bool:
    try:
        ip = ipaddress.ip_address(text.strip("[]").split("%")[0])
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return bool((mapped or ip).is_loopback)


def is_loopback_hostname(host: str) -> bool:
    h = host.strip().lower().rstrip(".")
    return h == "localhost" or h.endswith(".localhost") or is_loopback_ip(h)


def host_only(host_header: str) -> str:
    """'127.0.0.1:8000' -> '127.0.0.1', '[::1]:8000' -> '::1', 'localhost' -> 'localhost'."""
    h = host_header.strip()
    if h.startswith("["):
        return h[1:h.find("]")] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def is_loopback_bind(host: str) -> bool:
    """Is a server bound to `host` reachable from this machine only?"""
    return is_loopback_hostname(host) if host else False


def require_auth() -> bool:
    return os.environ.get(REQUIRE_AUTH_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def loopback_trust(scope: Scope, headers: Headers) -> tuple[bool, str]:
    """(trusted, reason it is not) for this request — see the module docstring, rule 1."""
    if require_auth():
        return False, f"{REQUIRE_AUTH_ENV} is set: every request needs a token"
    client = scope.get("client")
    peer = client[0] if client else None
    if not peer or not is_loopback_ip(peer):
        return False, f"request from {peer or 'an unknown peer'}, not from this machine"
    for name in PROXY_HEADERS:
        if name in headers:
            return False, f"request relayed by a proxy ({name} header)"
    host = headers.get("host")
    if host is not None and not is_loopback_hostname(host_only(host)):
        return False, f"Host '{host}' is not a loopback name"
    origin = headers.get("origin")
    if origin is not None:
        o = urlsplit(origin)
        if not o.hostname or not is_loopback_hostname(o.hostname):
            return False, f"Origin '{origin}' is not this machine"
    return True, ""


def same_origin(headers: Headers) -> bool:
    """For cookie auth: the browser says the request comes from our own pages."""
    if headers.get("sec-fetch-site") == "cross-site":
        return False
    origin = headers.get("origin")
    if origin is None:
        return True        # browsers send Origin on every unsafe/WebSocket request; scripts may not
    host = headers.get("host", "")
    return bool(host) and urlsplit(origin).netloc.lower() == host.lower()


def is_public(path: str) -> bool:
    if ".." in path.split("/") or "\\" in path:
        return False
    return path in PUBLIC_EXACT or path.startswith(PUBLIC_PREFIXES)


def _bearer(headers: Headers) -> Optional[str]:
    auth = headers.get("authorization")
    if not auth:
        return None
    scheme, _, value = auth.partition(" ")
    return value.strip() if scheme.lower() == "bearer" and value.strip() else None


def _cookie(headers: Headers) -> Optional[str]:
    raw = headers.get("cookie")
    if not raw:
        return None
    # Starlette's lenient parser: another app's odd cookie on the same host must not hide ours.
    return cookie_parser(raw).get(COOKIE) or None


# ── token stores (one per projects root, cached; re-read when the file changes) ──
_STORES: dict[str, TokenStore] = {}


def token_store(root: Optional[Path] = None) -> TokenStore:
    path = TokenStore.for_root(root or default_projects_root()).path
    key = str(path)
    store = _STORES.get(key)
    if store is None:
        store = _STORES[key] = TokenStore(path)
    return store


# ── decision ─────────────────────────────────────────────────────────────
class Decision:
    __slots__ = ("principal", "local", "why_not_local", "status", "detail")

    def __init__(self, principal: Optional[Principal], local: bool, why_not_local: str,
                 status: int = 200, detail: str = ""):
        self.principal = principal
        self.local = local
        self.why_not_local = why_not_local
        self.status = status
        self.detail = detail


def decide(scope: Scope, store: TokenStore) -> Decision:
    headers = Headers(scope=scope)
    local, why = loopback_trust(scope, headers)
    principal: Optional[Principal] = Principal(via="loopback") if local else None

    bearer = _bearer(headers)
    cookie = None if bearer else _cookie(headers)
    secret = bearer or cookie
    refusal: tuple[int, str] = (401, f"authentication required: {why}")
    if secret is not None:
        try:
            rec: Optional[TokenRecord] = store.verify(secret)
        except TokenError:
            rec = None              # unreadable token file: no token is valid (fail closed)
        unsafe = scope["type"] == "websocket" or scope.get("method", "GET") not in SAFE_METHODS
        if rec is None:
            refusal = (401, "invalid or revoked token")
        elif cookie is not None and unsafe and not same_origin(headers):
            refusal = (403, "cross-site request with the session cookie refused "
                            "(send the token in the Authorization header instead)")
        else:
            principal = Principal(via="token", name=rec.name, token_id=rec.id)
    # A stale cookie on this machine changes nothing: loopback stays open. Public paths
    # (the UI shell, /api/v2/auth/*) load even with a stale cookie, so the UI can ask again.
    if principal is None and not is_public(scope.get("path", "")):
        return Decision(None, local, why, *refusal)
    return Decision(principal, local, why)


class AuthMiddleware:
    """See the module docstring. `projects_root` is resolved per request (tests switch it)."""

    def __init__(self, app: ASGIApp, projects_root: Callable[[], Path] = default_projects_root):
        self.app = app
        self.projects_root = projects_root

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        d = decide(scope, token_store(self.projects_root()))
        if d.status != 200:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            resp = JSONResponse({"detail": d.detail, "auth": "token-required"}, status_code=d.status,
                                headers={"WWW-Authenticate": 'Bearer realm="aerosplat"',
                                         "Cache-Control": "no-store"})
            await resp(scope, receive, send)
            return
        scope.setdefault("state", {})["auth"] = d
        ctx = set_principal(d.principal)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_principal(ctx)


# ── startup check ────────────────────────────────────────────────────────
def startup_problem(host: str, root: Optional[Path] = None) -> Optional[str]:
    """Why a server on `host` must not start (None = fine). Loopback binds always start."""
    needs = not is_loopback_bind(host) or require_auth()
    if not needs:
        return None
    store = token_store(root)
    try:
        active = store.active()
    except TokenError as e:
        return f"файл токенів пошкоджено — {e}"
    if active:
        return None
    why = (f"сервер на {host or '(усі інтерфейси)'} буде доступний з мережі"
           if not is_loopback_bind(host) else f"{REQUIRE_AUTH_ENV} вимагає токен для всіх запитів")
    return (f"{why}, а жодного токена доступу немає ({store.path}).\n"
            f"   Створіть токен:  aerosplat token create <ім'я>"
            f"{'' if root is None else f' --projects-root {root}'}\n"
            f"   або слухайте лише цю машину: --host 127.0.0.1 (без автентифікації, як раніше).")


# ── /api/v2/auth: what the UI needs to ask for a token ──────────────────
router = APIRouter(prefix="/api/v2/auth", tags=["auth"])


class WhoAmI(BaseModel):
    authenticated: bool
    via: Optional[str] = None                 # loopback | token
    name: Optional[str] = None                # token name
    local: bool = False                       # request trusted as coming from this machine
    reason: Optional[str] = Field(None, description="why the request is not trusted as local")


class LoginBody(BaseModel):
    token: str = Field(min_length=1, max_length=512)


def _whoami(d: Optional[Decision]) -> WhoAmI:
    p = d.principal if d else None
    return WhoAmI(authenticated=p is not None, via=p.via if p else None,
                  name=p.name if p else None, local=bool(d and d.local),
                  reason=(d.why_not_local or None) if d else None)


@router.get("/whoami", response_model=WhoAmI)
def whoami(request: Request) -> WhoAmI:
    return _whoami(request.scope.get("state", {}).get("auth"))


@router.post("/login", response_model=WhoAmI)
def login(body: LoginBody, request: Request, response: Response) -> Any:
    """Check a token and keep it in an HttpOnly cookie for this browser."""
    try:
        rec = token_store().verify(body.token.strip())
    except TokenError:
        rec = None
    if rec is None:
        return JSONResponse({"detail": "invalid or revoked token", "auth": "token-required"},
                            status_code=401, headers={"Cache-Control": "no-store"})
    response.set_cookie(COOKIE, body.token.strip(), max_age=COOKIE_MAX_AGE_S, httponly=True,
                        samesite="strict", secure=request.url.scheme == "https", path="/")
    response.headers["Cache-Control"] = "no-store"
    d: Optional[Decision] = request.scope.get("state", {}).get("auth")
    return WhoAmI(authenticated=True, via="token", name=rec.name, local=bool(d and d.local),
                  reason=(d.why_not_local or None) if d else None)


@router.post("/logout", response_model=WhoAmI)
def logout(request: Request, response: Response) -> WhoAmI:
    response.delete_cookie(COOKIE, path="/", httponly=True, samesite="strict")
    d: Optional[Decision] = request.scope.get("state", {}).get("auth")
    local = bool(d and d.local)
    return WhoAmI(authenticated=local, via="loopback" if local else None, local=local,
                  reason=(d.why_not_local or None) if d else None)
