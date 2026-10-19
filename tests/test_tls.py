"""TLS for network binds of `aerosplat up` (aerosplat.server.tls)."""

from __future__ import annotations

import datetime as dt

import pytest

from aerosplat.server import tls


def test_network_bind_needs_tls_or_an_explicit_insecure_flag():
    assert tls.transport_problem(True, False, False) is None          # loopback: as before
    assert "--tls-self-signed" in tls.transport_problem(False, False, False)
    assert tls.transport_problem(False, True, False) is None
    assert tls.transport_problem(False, False, True) is None


def test_self_signed_is_made_once_kept_and_renewed_near_expiry(tmp_path):
    a = tls.self_signed(tmp_path)
    assert a.self_signed and a.cert.is_file() and a.key.is_file() and len(a.fingerprint) == 95
    assert tls.self_signed(tmp_path).fingerprint == a.fingerprint     # kept
    later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=tls.SELF_SIGNED_DAYS - 10)
    assert tls.self_signed(tmp_path, now=later).fingerprint != a.fingerprint   # renewed
    assert tls.check_pair(a.cert, a.key).fingerprint == tls.fingerprint(a.cert.read_bytes())


def test_self_signed_names_this_machine():
    names, ips = tls.machine_names()
    assert "localhost" in names and "127.0.0.1" in ips


def test_a_key_of_another_certificate_is_refused(tmp_path):
    a = tls.self_signed(tmp_path / "a")
    b = tls.self_signed(tmp_path / "b")
    with pytest.raises(tls.TlsError, match="не від сертифіката"):
        tls.check_pair(a.cert, b.key)
    with pytest.raises(tls.TlsError, match="не знайдено"):
        tls.check_pair(tmp_path / "none.pem", a.key)


def test_up_refuses_plain_http_on_the_network(tmp_path, monkeypatch, capsys):
    from aerosplat.auth.tokens import TokenStore
    from aerosplat.server import up

    root = tmp_path / "projects"
    TokenStore.for_root(root).create("t")

    def boom(*a, **kw):
        raise AssertionError("nothing may start")

    monkeypatch.setattr(up.WorkerSupervisor, "start_all", boom)
    assert up.main(["--host", "0.0.0.0", "--port", "1", "--projects-root", str(root)]) == 2
    assert "--tls-self-signed" in capsys.readouterr().out
    assert up.main(["--host", "0.0.0.0", "--port", "1", "--projects-root", str(root),
                    "--tls-cert", str(tmp_path / "c.pem")]) == 2       # key missing
    assert "разом" in capsys.readouterr().out
