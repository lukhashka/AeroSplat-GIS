"""L0 tests of the SITL harness: pinned config, home resolution, launch commands, opt-in marker.
No WSL, SITL or pymavlink needed."""

import sys
from types import SimpleNamespace

import pytest
import yaml

from aerosplat.geo.crs import GeoPoint
from aerosplat.sim import sitl as sitl_mod
from aerosplat.sim.sitl import (CONFIG_PATH, ENDPOINT_ENV, Home, Sitl, SitlConfig, SitlError,
                                load_config, parse_session, resolve_home, site_origin, to_wsl_path)
from aerosplat.store import Site

V6 = GeoPoint(lat=40.89104597075257, lon=-74.21634205070951, h=233.0)


@pytest.mark.hazard("H-003")
def test_config_pins_release_commit():
    cfg = load_config()
    assert cfg.ardupilot.tag.startswith("Copter-")
    assert len(cfg.ardupilot.commit) == 40
    assert cfg.ardupilot.version == (4, 7, 2)
    # A short or symbolic pin is rejected: the pin must name exactly one build.
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for bad in ("fb773567", "Copter-4.7.2", "FB773567896014DA007BCA58358704D4A0D8516F"):
        raw["ardupilot"]["commit"] = bad
        with pytest.raises(ValueError):
            SitlConfig.model_validate(raw)


def test_default_home_is_dataset_v6_origin(tmp_path):
    cfg = load_config()
    assert cfg.home.site == "default/dataset_v6"
    assert cfg.home.fallback == V6
    # No catalog at all: the configured snapshot of the dataset_v6 origin.
    home = resolve_home(cfg, projects_root=tmp_path / "none")
    assert (home.lat, home.lon, home.alt) == (V6.lat, V6.lon, V6.h)
    assert "fallback" in home.source


def test_home_parse_and_arg():
    h = Home.parse("40.8910,-74.2163,233,90")
    assert (h.lat, h.lon, h.alt, h.heading) == (40.891, -74.2163, 233.0, 90.0)
    assert Home.parse(h.arg()).arg() == h.arg()
    assert Home.parse("1,2,3").heading == 0.0
    for bad in ("40.89,-74.21", "1,2,3,4,5", "91,0,0", "0,181,0", "a,b,c", ""):
        with pytest.raises(ValueError):
            Home.parse(bad)


def test_resolve_home_order(catalog, tmp_path):
    cfg = load_config()
    root = catalog.paths.root
    catalog.ensure_project("demo")
    origin = GeoPoint(lat=50.45, lon=30.52, h=180.0)
    catalog.create_site(Site(id="yard", project_id="demo", origin=origin))

    assert resolve_home(cfg, location="1,2,3,4").source == "--location"
    home = resolve_home(cfg, site="demo/yard", projects_root=root)
    assert (home.lat, home.lon, home.alt, home.source) == (50.45, 30.52, 180.0, "catalog:demo/yard")

    # Without site.yaml the origin comes from the database, opened read-only.
    catalog.site_dirs(catalog.get_site("demo", "yard")).site_yaml.unlink()
    assert site_origin("demo/yard", root) == origin

    # A site named explicitly must exist; only the configured default falls back.
    with pytest.raises(SitlError):
        resolve_home(cfg, site="demo/nope", projects_root=root)
    assert "fallback" in resolve_home(cfg, projects_root=root).source
    with pytest.raises(ValueError):
        site_origin("dataset_v6", root)


def test_site_origin_creates_nothing(tmp_path):
    root = tmp_path / "projects"
    assert site_origin("default/dataset_v6", root) is None
    assert not root.exists()


def test_to_wsl_path():
    assert to_wsl_path(r"C:\coding\a b\run_sitl.sh") == "/mnt/c/coding/a b/run_sitl.sh"
    assert to_wsl_path("D:/x/y.sh") == "/mnt/d/x/y.sh"
    for bad in ("relative/run.sh", "/home/user/run.sh"):
        with pytest.raises(ValueError):
            to_wsl_path(bad)


def test_commands_windows_and_linux(monkeypatch):
    cfg = load_config()
    s = Sitl(cfg, instance=1)
    assert (s.test_port, s.gcs_port) == (cfg.ports.test + 10, cfg.ports.gcs + 10)
    env = s.env(Home.parse("1,2,3"), speedup=2)
    assert env["SITL_HOME"] == "1.000000000,2.000000000,3.000,0"
    assert env["SITL_SPEEDUP"] == "2" and env["SITL_INSTANCE"] == "1"
    assert env["AP_COMMIT"] == cfg.ardupilot.commit

    monkeypatch.setattr(sys, "platform", "win32")
    cmd = s.command("run_sitl.sh", env)
    # --exec: arguments reach Linux verbatim, without a second round of shell parsing.
    assert cmd[:5] == ["wsl.exe", "-d", cfg.wsl.distro, "--exec", "env"]
    assert cmd[-2] == "bash" and cmd[-1].startswith("/mnt/") and cmd[-1].endswith("/tools/sitl/run_sitl.sh")
    assigns = cmd[5:-2]
    assert assigns == sorted(assigns) and f"AP_COMMIT={cfg.ardupilot.commit}" in assigns

    monkeypatch.setattr(sys, "platform", "linux")
    cmd = s.command("stop_sitl.sh", {"SITL_RUN_DIR": "~/x"})
    assert cmd == ["env", "SITL_RUN_DIR=~/x", "bash", str(sitl_mod.SCRIPTS_DIR / "stop_sitl.sh")]


def test_boot_defaults_follow_the_configured_ones(monkeypatch, tmp_path):
    """SIM-2n: a GNSS-free run boots with an extra parameter file, after copter.parm."""
    cfg = load_config()
    s = Sitl(cfg, instance=1)
    boot = tmp_path / "boot.parm"
    assert s.env(Home.parse("1,2,3"))["SITL_DEFAULTS"] == ",".join(cfg.vehicle.defaults)
    monkeypatch.setattr(sys, "platform", "win32")
    if not str(tmp_path.resolve())[1:2] == ":":
        pytest.skip("needs a drive path")
    parts = s.env(Home.parse("1,2,3"), boot_defaults=(boot,))["SITL_DEFAULTS"].split(",")
    assert parts[:-1] == list(cfg.vehicle.defaults)
    assert parts[-1].startswith("/mnt/") and parts[-1].endswith("/boot.parm")
    with pytest.raises(SitlError, match="comma"):
        Sitl.host_path(tmp_path / "a,b.parm")


def test_endpoint_override_and_linux(monkeypatch):
    s = Sitl(load_config())
    monkeypatch.setenv(ENDPOINT_ENV, "udpout:10.0.0.5:14560")
    assert s.endpoint() == "udpout:10.0.0.5:14560"
    monkeypatch.delenv(ENDPOINT_ENV)
    monkeypatch.setattr(sys, "platform", "linux")
    assert s.endpoint() == f"udpout:127.0.0.1:{s.test_port}"


def test_parse_session():
    text = "pid=399\nhome=40.89,-74.21,233.000,0\ngcs=udp:172.30.192.1:14550\nrun_sitl: noise\n"
    assert parse_session(text) == {"pid": "399", "home": "40.89,-74.21,233.000,0",
                                   "gcs": "udp:172.30.192.1:14550"}


def test_sitl_tests_are_opt_in(monkeypatch):
    """A plain `pytest` must never fly a SITL vehicle, even if one is running."""
    from conftest import sitl_requested

    def config(markexpr="", flag=False):
        return SimpleNamespace(getoption=lambda name: {"--sitl": flag, "markexpr": markexpr}[name])

    monkeypatch.delenv("AEROSPLAT_SITL", raising=False)
    assert not sitl_requested(config())
    assert not sitl_requested(config("not sitl"))
    assert not sitl_requested(config("slow and not sitl"))
    assert sitl_requested(config("sitl"))
    assert sitl_requested(config("sitl and not slow"))
    assert sitl_requested(config(flag=True))
    monkeypatch.setenv("AEROSPLAT_SITL", "1")
    assert sitl_requested(config())
