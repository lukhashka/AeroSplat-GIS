"""Authentication before non-localhost access (aerosplat.auth, aerosplat.server.auth).

Loopback stays open with zero configuration; everything else needs a token. Most tests
are negative on purpose: a missing, wrong or revoked token, a spoofed header, a
proxied request, a cross-site page — none of them may get through.
"""

import json

import pytest

from conftest import local_client

pytest.importorskip("httpx")
from fastapi import FastAPI, WebSocket  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from aerosplat.auth import cli as token_cli  # noqa: E402
from aerosplat.auth.tokens import TOKEN_PREFIX, TokenError, TokenStore, hash_token  # noqa: E402
from aerosplat.authored.models import actor  # noqa: E402
from aerosplat.server import auth  # noqa: E402

REMOTE = ("192.168.1.20", 40000)


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "projects"
    r.mkdir()
    monkeypatch.setenv("AEROSPLAT_PROJECTS", str(r))
    monkeypatch.delenv(auth.REQUIRE_AUTH_ENV, raising=False)
    auth._STORES.clear()
    yield r
    auth._STORES.clear()


@pytest.fixture
def store(root):
    return TokenStore.for_root(root)


@pytest.fixture
def app(tmp_path, root):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<p>viewer</p>", encoding="utf-8")
    a = FastAPI()
    a.add_middleware(auth.AuthMiddleware)
    a.include_router(auth.router)

    @a.get("/api/v2/ping")
    def ping(by: str = ""):                 # sync: runs in the threadpool
        return {"actor": actor(by or None)}

    @a.get("/api/v2/aping")
    async def aping():
        return {"actor": actor(None)}

    @a.post("/api/v2/thing")
    def thing():
        return {"ok": True, "actor": actor("mallory")}

    @a.websocket("/api/v2/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        await sock.send_json({"actor": actor(None)})
        await sock.close()

    @a.get("/app/{path:path}")
    def shell(path: str = ""):
        return {"shell": True}

    a.mount("/web", StaticFiles(directory=static, html=True), name="web")
    return a


def ws_url(client, path):
    """TestClient.websocket_connect ignores base_url (always ws://testserver); keep the host."""
    return str(client.base_url.copy_with(scheme="ws", path=path))


def remote(app, **kw):
    return TestClient(app, base_url="http://aerosplat.lan:8000", client=REMOTE, **kw)


# ── token store ──────────────────────────────────────────────────────────
def test_store_create_verify_revoke(store):
    rec, secret = store.create("alice")
    assert secret.startswith(TOKEN_PREFIX) and len(secret) > 40
    text = store.path.read_text(encoding="utf-8")
    assert secret not in text and hash_token(secret) in text      # only the hash is on disk
    assert store.verify(secret).name == "alice"
    for bad in (None, "", "nope", secret[:-1], secret + "x", TOKEN_PREFIX, "x" * 10_000):
        assert store.verify(bad) is None
    with pytest.raises(TokenError):
        store.create("alice")                                     # active names are unique
    with pytest.raises(TokenError):
        store.create("bad name!")
    store.revoke("alice")
    assert store.verify(secret) is None
    assert store.active() == [] and len(store.load()) == 1
    rec2, secret2 = store.create("alice")                         # the name is free again
    assert rec2.id != rec.id and store.verify(secret2).id == rec2.id
    store.revoke(rec2.id)                                         # by id too
    with pytest.raises(TokenError):
        store.revoke("alice")


def test_store_corrupt_file_is_an_error_not_an_empty_list(store):
    store.path.parent.mkdir(parents=True)
    store.path.write_text("{not json", encoding="utf-8")
    with pytest.raises(TokenError):
        store.load()
    store.path.write_text(json.dumps({"version": 99, "tokens": []}), encoding="utf-8")
    with pytest.raises(TokenError):
        store.active()


def test_cli_prints_the_secret_once_and_never_the_hash(root, capsys):
    assert token_cli.main(["--projects-root", str(root), "create", "ci-bot"]) == 0
    out = capsys.readouterr().out
    secret = next(w for w in out.split() if w.startswith(TOKEN_PREFIX))
    assert TokenStore.for_root(root).verify(secret).name == "ci-bot"
    assert token_cli.main(["--projects-root", str(root), "list"]) == 0
    listing = capsys.readouterr().out
    assert "ci-bot" in listing and secret not in listing and hash_token(secret) not in listing
    assert token_cli.main(["--projects-root", str(root), "revoke", "ci-bot"]) == 0
    assert token_cli.main(["--projects-root", str(root), "revoke", "ci-bot"]) == 1
    assert TokenStore.for_root(root).verify(secret) is None


# ── address helpers ──────────────────────────────────────────────────────
@pytest.mark.parametrize("ip,ok", [("127.0.0.1", True), ("127.8.9.10", True), ("::1", True),
                                   ("::ffff:127.0.0.1", True), ("[::1]", True),
                                   ("10.0.0.1", False), ("192.168.1.20", False), ("0.0.0.0", False),
                                   ("::", False), ("localhost", False), ("testclient", False),
                                   ("", False)])
def test_is_loopback_ip(ip, ok):
    assert auth.is_loopback_ip(ip) is ok


@pytest.mark.parametrize("host,ok", [("127.0.0.1", True), ("localhost", True), ("::1", True),
                                     ("LOCALHOST.", True), ("app.localhost", True),
                                     ("0.0.0.0", False), ("::", False), ("", False),
                                     ("192.168.1.5", False), ("mypc", False),
                                     ("localhost.evil.com", False)])
def test_is_loopback_bind(host, ok):
    assert auth.is_loopback_bind(host) is ok


def test_host_only_and_public_paths():
    assert auth.host_only("127.0.0.1:8000") == "127.0.0.1"
    assert auth.host_only("[::1]:8000") == "::1"
    assert auth.host_only("localhost") == "localhost"
    assert auth.is_public("/app/") and auth.is_public("/web/index.html") and auth.is_public("/")
    assert auth.is_public("/api/v2/auth/login")
    for p in ("/api/v2/sites", "/api/v1/scene-meta", "/docs", "/openapi.json",
              "/app/../api/v2/sites", "/web/..\\x", "/applesauce", "/api/v2/authx"):
        assert not auth.is_public(p), p


# ── loopback: open as before ─────────────────────────────────────────────
def test_loopback_needs_no_configuration(app, monkeypatch):
    monkeypatch.setenv("AEROSPLAT_OPERATOR", "owner")
    with local_client(app) as c:
        r = c.get("/api/v2/ping")
        assert r.status_code == 200 and r.json()["actor"] == "op:owner"   # unchanged rule
        assert c.get("/api/v2/ping", params={"by": "bob"}).json()["actor"] == "op:bob"
        assert c.post("/api/v2/thing").status_code == 200
        with c.websocket_connect(ws_url(c, "/api/v2/ws")) as ws:
            assert ws.receive_json()["actor"] == "op:owner"
        me = c.get("/api/v2/auth/whoami").json()
        assert me == {"authenticated": True, "via": "loopback", "name": None, "local": True,
                      "reason": None}
    for host in ("localhost:8000", "[::1]:8000", "127.0.0.1"):
        with local_client(app, headers={"Host": host}) as c:
            assert c.get("/api/v2/ping").status_code == 200
    with local_client(app, client=("::1", 1)) as c:
        assert c.get("/api/v2/ping").status_code == 200
    with local_client(app, headers={"Origin": "http://localhost:5173"}) as c:   # vite dev server
        assert c.post("/api/v2/thing").status_code == 200


# ── remote: token required ───────────────────────────────────────────────
def test_remote_without_token_is_refused_everywhere_but_the_ui_shell(app):
    with remote(app) as c:
        r = c.get("/api/v2/ping")
        assert r.status_code == 401 and r.json()["auth"] == "token-required"
        assert r.headers["www-authenticate"].startswith("Bearer")
        assert "192.168.1.20" in r.json()["detail"]
        assert c.post("/api/v2/thing").status_code == 401
        assert c.get("/docs").status_code == 401
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect(ws_url(c, "/api/v2/ws")) as ws:
                ws.receive_json()
        # the UI shell and static files are code, not data: they load, so the UI can ask
        assert c.get("/app/sites").status_code == 200
        assert c.get("/web/").status_code == 200
        assert c.get("/api/v2/auth/whoami").json()["authenticated"] is False


@pytest.mark.parametrize("header", ["Bearer nope", f"Bearer {TOKEN_PREFIX}forged", "Bearer ",
                                    "Basic dXNlcjpwYXNz", "bearer"])
def test_remote_with_an_invalid_token_is_refused(app, store, header):
    store.create("alice")
    with remote(app, headers={"Authorization": header}) as c:
        assert c.get("/api/v2/ping").status_code == 401
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect(ws_url(c, "/api/v2/ws")) as ws:
                ws.receive_json()


def test_remote_with_a_token_and_audit_takes_the_token_name(app, store):
    _, secret = store.create("alice")
    with remote(app, headers={"Authorization": f"Bearer {secret}"}) as c:
        # a claimed name is ignored: the token is what was verified
        assert c.get("/api/v2/ping", params={"by": "mallory"}).json()["actor"] == "token:alice"
        assert c.get("/api/v2/aping").json()["actor"] == "token:alice"
        assert c.post("/api/v2/thing").json()["actor"] == "token:alice"
        with c.websocket_connect(ws_url(c, "/api/v2/ws")) as ws:
            assert ws.receive_json()["actor"] == "token:alice"
        assert c.get("/api/v2/auth/whoami").json()["name"] == "alice"


def test_revocation_takes_effect_on_the_next_request(app, store):
    _, secret = store.create("alice")
    with remote(app, headers={"Authorization": f"Bearer {secret}"}) as c:
        assert c.get("/api/v2/ping").status_code == 200
        store.revoke("alice")
        r = c.get("/api/v2/ping")
        assert r.status_code == 401 and "revoked" in r.json()["detail"]


def test_corrupt_token_file_fails_closed_for_remote_but_keeps_loopback_open(app, store):
    _, secret = store.create("alice")
    store.path.write_text("garbage", encoding="utf-8")
    with remote(app, headers={"Authorization": f"Bearer {secret}"}) as c:
        assert c.get("/api/v2/ping").status_code == 401
    with local_client(app) as c:
        assert c.get("/api/v2/ping").status_code == 200


def test_principal_does_not_leak_between_requests(app, store):
    _, secret = store.create("alice")
    with remote(app, headers={"Authorization": f"Bearer {secret}"}) as c:
        assert c.get("/api/v2/ping").json()["actor"] == "token:alice"
    with local_client(app) as c:
        assert c.get("/api/v2/ping", params={"by": "bob"}).json()["actor"] == "op:bob"
    assert actor("carol") == "op:carol"           # outside any request: the old rule


# ── header spoofing must never grant loopback trust ──────────────────────
@pytest.mark.parametrize("headers", [
    {"X-Forwarded-For": "127.0.0.1"}, {"X-Real-IP": "127.0.0.1"},
    {"Forwarded": "for=127.0.0.1"}, {"Host": "127.0.0.1:8000"}, {"Origin": "http://127.0.0.1:8000"},
])
def test_spoofed_headers_from_a_remote_peer_do_not_help(app, headers):
    with remote(app, headers=headers) as c:
        assert c.get("/api/v2/ping").status_code == 401


@pytest.mark.parametrize("headers,why", [
    ({"X-Forwarded-For": "203.0.113.7"}, "proxy"),          # reverse proxy / tunnel on this machine
    ({"X-Forwarded-For": "127.0.0.1"}, "proxy"),
    ({"Forwarded": "for=203.0.113.7"}, "proxy"),
    ({"CF-Connecting-IP": "203.0.113.7"}, "proxy"),          # cloudflared
    ({"Via": "1.1 nginx"}, "proxy"),
    ({"Host": "evil.example:8000"}, "Host"),                 # DNS rebinding
    ({"Origin": "https://evil.example"}, "Origin"),          # another site in the owner's browser
    ({"Origin": "null"}, "Origin"),
])
def test_loopback_peer_loses_trust_when_headers_say_it_is_not_local(app, headers, why):
    with local_client(app, headers=headers) as c:
        r = c.get("/api/v2/ping")
        assert r.status_code == 401 and why in r.json()["detail"]
        assert c.post("/api/v2/thing").status_code == 401


def test_proxied_request_with_a_token_is_fine(app, store):
    _, secret = store.create("alice")
    with local_client(app, headers={"X-Forwarded-For": "203.0.113.7",
                                    "Authorization": f"Bearer {secret}"}) as c:
        assert c.get("/api/v2/ping").json()["actor"] == "token:alice"


def test_require_auth_disables_loopback_trust(app, store, monkeypatch):
    monkeypatch.setenv(auth.REQUIRE_AUTH_ENV, "1")
    with local_client(app) as c:
        assert c.get("/api/v2/ping").status_code == 401
    _, secret = store.create("alice")
    with local_client(app, headers={"Authorization": f"Bearer {secret}"}) as c:
        assert c.get("/api/v2/ping").status_code == 200


# ── browser session cookie ───────────────────────────────────────────────
def test_login_sets_an_httponly_strict_cookie_that_works_for_ws_too(app, store):
    _, secret = store.create("alice")
    with remote(app) as c:
        bad = c.post("/api/v2/auth/login", json={"token": "asg_wrong"})
        assert bad.status_code == 401 and auth.COOKIE not in bad.headers.get("set-cookie", "")
        r = c.post("/api/v2/auth/login", json={"token": secret})
        assert r.status_code == 200 and r.json()["name"] == "alice"
        cookie = r.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
        assert c.get("/api/v2/ping").json()["actor"] == "token:alice"
        with c.websocket_connect(ws_url(c, "/api/v2/ws")) as ws:
            assert ws.receive_json()["actor"] == "token:alice"
        # same-origin POST from our own page is fine, a cross-site one is not (CSRF)
        own = {"Origin": "http://aerosplat.lan:8000"}
        assert c.post("/api/v2/thing", headers=own).status_code == 200
        assert c.post("/api/v2/thing", headers={"Origin": "https://evil.example"}).status_code == 403
        assert c.post("/api/v2/thing", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect(ws_url(c, "/api/v2/ws"), headers={"Origin": "https://evil.example"}) as ws:
                ws.receive_json()
        c.post("/api/v2/auth/logout")
        assert c.get("/api/v2/ping").status_code == 401


def test_stale_cookie_still_loads_the_shell_and_login(app, store):
    _, secret = store.create("alice")
    with remote(app) as c:
        assert c.post("/api/v2/auth/login", json={"token": secret}).status_code == 200
        store.revoke("alice")
        assert c.get("/api/v2/ping").status_code == 401
        assert c.get("/app/").status_code == 200
        _, secret2 = store.create("alice2")
        assert c.post("/api/v2/auth/login", json={"token": secret2}).status_code == 200
        assert c.get("/api/v2/ping").json()["actor"] == "token:alice2"


# ── startup: a network-facing server without tokens does not start ───────
def test_startup_problem(root, store, monkeypatch):
    assert auth.startup_problem("127.0.0.1") is None
    assert auth.startup_problem("localhost") is None
    for host in ("0.0.0.0", "::", "192.168.1.5", ""):
        msg = auth.startup_problem(host)
        assert msg and "aerosplat token create" in msg
    store.create("alice")
    assert auth.startup_problem("0.0.0.0") is None
    store.path.write_text("garbage", encoding="utf-8")
    assert "пошкоджено" in auth.startup_problem("0.0.0.0")
    assert auth.startup_problem("127.0.0.1") is None             # loopback never blocked
    monkeypatch.setenv(auth.REQUIRE_AUTH_ENV, "1")
    store.path.unlink()
    assert auth.startup_problem("127.0.0.1")


def test_up_refuses_a_network_bind_without_tokens(root, monkeypatch, capsys):
    from aerosplat.server import up

    def boom(*a, **kw):
        raise AssertionError("nothing may start")

    monkeypatch.setattr(up.WorkerSupervisor, "start_all", boom)
    assert up.main(["--host", "0.0.0.0", "--port", "1", "--projects-root", str(root)]) == 2
    assert "aerosplat token create" in capsys.readouterr().out
    monkeypatch.delenv(auth.REQUIRE_AUTH_ENV, raising=False)
    assert up.main(["--require-auth", "--port", "1", "--projects-root", str(root)]) == 2


def test_render_service_refuses_a_network_bind(capsys):
    from aerosplat.render import service

    assert service.main(["--host", "0.0.0.0", "--port", "1"]) == 2
    assert "127.0.0.1" in capsys.readouterr().out
