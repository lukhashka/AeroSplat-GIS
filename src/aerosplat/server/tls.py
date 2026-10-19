"""TLS for `aerosplat up` on the network (Phase 6, owner delegated the choice 2026-10-19).

A token sent over plain HTTP is readable by anyone on the network, so a non-loopback bind
serves HTTPS:

* `--tls-cert/--tls-key` — the operator's own certificate (a company CA, or one issued for a
  domain name). Checked before start: both files exist and the key belongs to the certificate.
* `--tls-self-signed` — a certificate this tool makes once and keeps in `<projects>/auth/tls/`,
  valid for this machine's names and addresses. Browsers warn on first visit; the SHA-256
  fingerprint printed at start is what the operator compares to accept it knowingly.
* `--insecure-http` — plain HTTP on the network, only when asked for by name (e.g. behind a
  reverse proxy that does TLS on the same machine).

Loopback binds are unchanged: `aerosplat up` on 127.0.0.1 stays plain HTTP with no setup.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SELF_SIGNED_DAYS = 365
RENEW_BEFORE_DAYS = 30          # a self-signed certificate closer to expiry is made anew


class TlsError(RuntimeError):
    """A TLS setup that must not start the server; the message says what to fix."""


@dataclass(frozen=True)
class TlsFiles:
    cert: Path
    key: Path
    fingerprint: str            # SHA-256 of the certificate, AA:BB:… as browsers show it
    self_signed: bool


def transport_problem(host_is_loopback: bool, tls: bool, insecure_http: bool) -> Optional[str]:
    """Why the server must not start on this transport (None = fine)."""
    if host_is_loopback or tls or insecure_http:
        return None
    return ("мережевий доступ без TLS: токен ішов би відкритим текстом.\n"
            "   Власний сертифікат:  --tls-cert cert.pem --tls-key key.pem\n"
            "   Самопідписаний:      --tls-self-signed  (браузер попередить; звірте відбиток)\n"
            "   Свідомо HTTP:        --insecure-http    (напр. за reverse proxy з TLS на цій машині)")


def _x509():
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError as e:            # pragma: no cover - cryptography ships with the deps
        raise TlsError("для TLS потрібен пакет cryptography (pip install cryptography)") from e
    return x509, hashes, serialization, ec, NameOID


def fingerprint(cert_pem: bytes) -> str:
    x509, hashes, *_ = _x509()
    digest = x509.load_pem_x509_certificate(cert_pem).fingerprint(hashes.SHA256())
    return ":".join(f"{b:02X}" for b in digest)


def check_pair(cert: Path, key: Path) -> TlsFiles:
    """The operator's certificate: both files there, the key is the certificate's own."""
    x509, _, serialization, _, _ = _x509()
    for what, p in (("сертифікат", cert), ("ключ", key)):
        if not Path(p).is_file():
            raise TlsError(f"{what} {p} не знайдено")
    try:
        c = x509.load_pem_x509_certificate(Path(cert).read_bytes())
        k = serialization.load_pem_private_key(Path(key).read_bytes(), password=None)
    except (ValueError, TypeError) as e:
        raise TlsError(f"не вдалося прочитати PEM ({e}); ключ має бути без пароля") from None
    pub = serialization.PublicFormat.SubjectPublicKeyInfo
    enc = serialization.Encoding.PEM
    if c.public_key().public_bytes(enc, pub) != k.public_key().public_bytes(enc, pub):
        raise TlsError(f"ключ {key} не від сертифіката {cert}")
    if c.not_valid_after_utc <= dt.datetime.now(dt.timezone.utc):
        raise TlsError(f"сертифікат {cert} прострочено ({c.not_valid_after_utc:%Y-%m-%d})")
    return TlsFiles(Path(cert), Path(key), fingerprint(Path(cert).read_bytes()), self_signed=False)


def machine_names() -> tuple[list[str], list[str]]:
    """DNS names and IP addresses this machine answers to (for the certificate's SANs)."""
    names = {"localhost", socket.gethostname()}
    try:
        names.add(socket.getfqdn())
    except OSError:
        pass
    ips = {"127.0.0.1", "::1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ips.add(info[4][0].split("%")[0])
    except OSError:
        pass
    return sorted(n for n in names if n), sorted(ips)


def self_signed(folder: Path, *, now: Optional[dt.datetime] = None) -> TlsFiles:
    """The kept self-signed certificate, made (or remade near expiry) when needed."""
    x509, hashes, serialization, ec, NameOID = _x509()
    folder = Path(folder)
    cert_p, key_p = folder / "cert.pem", folder / "key.pem"
    now = now or dt.datetime.now(dt.timezone.utc)
    if cert_p.is_file() and key_p.is_file():
        c = x509.load_pem_x509_certificate(cert_p.read_bytes())
        if c.not_valid_after_utc - now > dt.timedelta(days=RENEW_BEFORE_DAYS):
            check_pair(cert_p, key_p)
            return TlsFiles(cert_p, key_p, fingerprint(cert_p.read_bytes()), self_signed=True)

    names, ips = machine_names()
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"AeroSplat-GIS {names[0]}")])
    san = [x509.DNSName(n) for n in names] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    cert = (x509.CertificateBuilder()
            .subject_name(subject).issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=SELF_SIGNED_DAYS))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    folder.mkdir(parents=True, exist_ok=True)
    pem_key = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    pem_cert = cert.public_bytes(serialization.Encoding.PEM)
    for path, data in ((key_p, pem_key), (cert_p, pem_cert)):      # atomic: never half a pair
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    try:
        key_p.chmod(0o600)              # POSIX; on Windows the projects folder's ACL applies
    except OSError:
        pass
    return TlsFiles(cert_p, key_p, fingerprint(pem_cert), self_signed=True)
