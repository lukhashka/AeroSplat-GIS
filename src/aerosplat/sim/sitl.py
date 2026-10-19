"""ArduCopter SITL harness (SIM-0): the pinned build from configs/sitl.yaml, run in WSL2 or Linux.

    python -m aerosplat.sim.sitl setup        # pinned ArduPilot worktree + `waf copter` (once)
    python -m aerosplat.sim.sitl start        # home = origin of default/dataset_v6
    python -m aerosplat.sim.sitl start --site default/dataset_splater --speedup 5
    python -m aerosplat.sim.sitl start --location 40.8910,-74.2163,233,90
    python -m aerosplat.sim.sitl status
    python -m aerosplat.sim.sitl endpoint     # pymavlink connection string from this host
    python -m aerosplat.sim.sitl stop

On Windows every command runs a script from tools/sitl/ through `wsl.exe --exec` (no shell
re-parsing of arguments). `start` leaves a detached wsl.exe running the supervisor script: SITL
exits when its parent process goes away, so something on the Windows side has to hold it.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from aerosplat.geo.crs import GeoPoint
from aerosplat.paths import PROJECT_DIR

CONFIG_PATH = PROJECT_DIR / "configs" / "sitl.yaml"
SCRIPTS_DIR = PROJECT_DIR / "tools" / "sitl"
ENDPOINT_ENV = "AEROSPLAT_SITL_ENDPOINT"


class SitlError(RuntimeError):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArduPilotPin(_Model):
    tag: str
    commit: str
    remote: str
    source_clone: str
    worktree: str

    @field_validator("commit")
    @classmethod
    def _full_sha(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", v):
            raise ValueError("commit must be a full 40-character lowercase SHA-1")
        return v

    @property
    def version(self) -> tuple[int, int, int]:
        """(major, minor, patch) from a release tag such as 'Copter-4.7.2'."""
        m = re.search(r"(\d+)\.(\d+)\.(\d+)$", self.tag)
        if m is None:
            raise ValueError(f"tag '{self.tag}' does not end in a version number")
        return int(m[1]), int(m[2]), int(m[3])


class WslConfig(_Model):
    distro: str
    venv: str
    run_dir: str


class VehicleConfig(_Model):
    model: str = "+"
    defaults: tuple[str, ...]
    speedup: float = Field(1.0, gt=0)
    wipe: bool = True


class HomeConfig(_Model):
    site: Optional[str] = None
    heading_deg: float = 0.0
    fallback: GeoPoint


class PortsConfig(_Model):
    test: int = Field(14560, ge=1024, le=65000)
    gcs: int = Field(14550, ge=1024, le=65000)


class SitlConfig(_Model):
    ardupilot: ArduPilotPin
    tools: dict[str, str] = {}
    wsl: WslConfig
    vehicle: VehicleConfig
    home: HomeConfig
    ports: PortsConfig


def load_config(path: Path = CONFIG_PATH) -> SitlConfig:
    return SitlConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# ── home location ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Home:
    """SITL home (= sim_vehicle.py --custom-location). `alt` goes to SITL as metres AMSL."""
    lat: float
    lon: float
    alt: float
    heading: float
    source: str

    def arg(self) -> str:
        return f"{self.lat:.9f},{self.lon:.9f},{self.alt:.3f},{self.heading:g}"

    @classmethod
    def parse(cls, text: str, source: str = "--location") -> "Home":
        """'lat,lon,alt[,heading]' -> Home."""
        try:
            parts = [float(p) for p in text.split(",")]
        except ValueError:
            raise ValueError(f"location '{text}' is not lat,lon,alt[,heading]") from None
        if len(parts) not in (3, 4):
            raise ValueError(f"location '{text}' is not lat,lon,alt[,heading]")
        lat, lon, alt = parts[:3]
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"location '{text}': lat/lon out of range")
        heading = parts[3] if len(parts) == 4 else 0.0
        return cls(lat, lon, alt, heading, source)


def site_origin(site_ref: str, projects_root: Optional[Path] = None) -> Optional[GeoPoint]:
    """Origin of 'project/site' from the catalog, or None when there is no such catalog or site.

    Strictly read-only: site.yaml when it exists, else the catalog database opened with
    `mode=ro`. (Catalog.open would create and migrate the catalog, which a simulator launch
    must never do to someone's project tree.)
    """
    from aerosplat.store.paths import StorePaths, default_projects_root

    if site_ref.count("/") != 1:
        raise ValueError(f"site '{site_ref}' must be 'project/site'")
    project_id, site_id = site_ref.split("/")
    paths = StorePaths(projects_root if projects_root is not None else default_projects_root())
    site_yaml = paths.site(project_id, site_id).site_yaml
    if site_yaml.is_file():
        origin = (yaml.safe_load(site_yaml.read_text(encoding="utf-8")) or {}).get("origin")
        if origin:
            return GeoPoint(**origin)
    db = paths.catalog_db
    if not db.is_file():
        return None
    con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT origin_lat, origin_lon, origin_h FROM sites "
                          "WHERE project_id = ? AND slug = ?", (project_id, site_id)).fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        con.close()
    return GeoPoint(lat=row[0], lon=row[1], h=row[2]) if row else None


def resolve_home(cfg: SitlConfig, location: Optional[str] = None, site: Optional[str] = None,
                 projects_root: Optional[Path] = None) -> Home:
    """--location, else the site's origin from the catalog, else the configured fallback.

    A site named explicitly must exist; only the configured default site may fall back.
    The origin height is used as is: Earth Studio altitudes are not verified yet (§4.8), so
    missions fly heights above home and SITL's AMSL is nominal.
    """
    if location:
        return Home.parse(location)
    heading = cfg.home.heading_deg
    ref = site or cfg.home.site
    if ref:
        origin = site_origin(ref, projects_root)
        if origin is not None:
            return Home(origin.lat, origin.lon, origin.h, heading, f"catalog:{ref}")
        if site:
            raise SitlError(f"site '{site}' not found in the catalog")
    fb = cfg.home.fallback
    return Home(fb.lat, fb.lon, fb.h, heading,
                f"config fallback ({ref} not in catalog)" if ref else "config fallback")


# ── launcher ─────────────────────────────────────────────────────────────────

def to_wsl_path(path: Path | str) -> str:
    """C:\\a\\b -> /mnt/c/a/b (WSL's default automount root)."""
    p = PureWindowsPath(path)
    if not p.drive or not p.drive.endswith(":"):
        raise ValueError(f"{path} is not an absolute drive path")
    rest = "/".join(p.parts[1:])
    return f"/mnt/{p.drive[0].lower()}/{rest}"


def parse_session(text: str) -> dict[str, str]:
    """key=value lines (session.txt written by run_sitl.sh) -> dict."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key and " " not in key:
            out[key.strip()] = value.strip()
    return out


@dataclass
class Sitl:
    """One SITL instance (ports shift by 10 per instance, like sim_vehicle.py -I)."""
    cfg: SitlConfig
    instance: int = 0

    @property
    def test_port(self) -> int:
        return self.cfg.ports.test + 10 * self.instance

    @property
    def gcs_port(self) -> int:
        return self.cfg.ports.gcs + 10 * self.instance

    @property
    def launcher_log(self) -> Path:
        return Path(tempfile.gettempdir()) / f"aerosplat-sitl-i{self.instance}.log"

    def env(self, home: Optional[Home] = None, speedup: Optional[float] = None,
            boot_defaults: tuple[Path, ...] = ()) -> dict[str, str]:
        """Environment of the WSL scripts. `boot_defaults`: extra parameter files (host paths)
        loaded at boot after the configured ones — for what must hold from power-on, such as a
        vehicle without GPS (SIM-2n), which a PARAM_SET after boot cannot give."""
        ap, v = self.cfg.ardupilot, self.cfg.vehicle
        env = {
            "AP_TAG": ap.tag, "AP_COMMIT": ap.commit, "AP_REMOTE": ap.remote,
            "AP_SOURCE_CLONE": ap.source_clone, "AP_WORKTREE": ap.worktree,
            "AP_VENV": self.cfg.wsl.venv, "SITL_RUN_DIR": self.cfg.wsl.run_dir,
            "SITL_INSTANCE": str(self.instance), "SITL_MODEL": v.model,
            "SITL_SPEEDUP": f"{speedup if speedup is not None else v.speedup:g}",
            "SITL_WIPE": "1" if v.wipe else "0",
            "SITL_DEFAULTS": ",".join([*v.defaults, *(self.host_path(p) for p in boot_defaults)]),
            "SITL_TEST_PORT": str(self.cfg.ports.test), "SITL_GCS_PORT": str(self.cfg.ports.gcs),
        }
        if home is not None:
            env["SITL_HOME"] = home.arg()
        return env

    @staticmethod
    def host_path(path: Path) -> str:
        """A host file as the WSL scripts see it (/mnt/<drive>/...), or as is on Linux."""
        path = Path(path).resolve()
        if "," in str(path):
            raise SitlError(f"{path}: a path with a comma cannot go into SITL_DEFAULTS")
        return to_wsl_path(path) if sys.platform == "win32" else str(path)

    def command(self, script: str, env: dict[str, str]) -> list[str]:
        """argv running tools/sitl/<script> with `env`: through wsl.exe on Windows, else bash."""
        assigns = [f"{k}={v}" for k, v in sorted(env.items())]
        path = SCRIPTS_DIR / script
        if sys.platform == "win32":
            return ["wsl.exe", "-d", self.cfg.wsl.distro, "--exec", "env", *assigns,
                    "bash", to_wsl_path(path)]
        return ["env", *assigns, "bash", str(path)]

    def _run(self, script: str, env: dict[str, str], timeout: float = 60) -> subprocess.CompletedProcess:
        return subprocess.run(self.command(script, env), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)

    def setup(self) -> int:
        """Pinned worktree + build; streams the build output. Returns the exit code."""
        return subprocess.run(self.command("setup_ardupilot.sh", self.env())).returncode

    def status(self) -> Optional[dict[str, str]]:
        """session.txt of the running instance, or None when it is not running."""
        r = self._run("status_sitl.sh", self.env())
        if r.returncode == 3:
            return None
        if r.returncode != 0:
            raise SitlError(f"status failed ({r.returncode}): {r.stderr.strip() or r.stdout.strip()}")
        return parse_session(r.stdout)

    def start(self, home: Home, speedup: Optional[float] = None, timeout: float = 60,
              boot_defaults: tuple[Path, ...] = ()) -> dict[str, str]:
        """Start detached and wait until the supervisor reports both processes up."""
        if self.status() is not None:
            raise SitlError(f"SITL instance {self.instance} is already running; stop it first")
        log = self.launcher_log
        with open(log, "wb") as fh:
            cmd = self.command("run_sitl.sh", self.env(home, speedup, tuple(boot_defaults)))
            if sys.platform == "win32":
                # Own hidden console, survives the caller's console. Not DETACHED_PROCESS:
                # wsl.exe then relays no output at all.
                flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fh,
                                        stderr=subprocess.STDOUT, creationflags=flags)
            else:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fh,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        marker = f"run_sitl: instance {self.instance} up"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = log.read_text(encoding="utf-8", errors="replace")
            if marker in text:
                info = self.status()
                if info is None:
                    raise SitlError(f"SITL reported up but is not running; see {log}")
                return info
            if proc.poll() is not None:
                raise SitlError(f"SITL exited with {proc.returncode}:\n{text.strip()[-2000:]}")
            time.sleep(0.2)
        self.stop()
        raise SitlError(f"SITL did not come up in {timeout:.0f} s; see {log}")

    def stop(self) -> str:
        r = self._run("stop_sitl.sh", self.env())
        if r.returncode != 0:
            raise SitlError(f"stop failed ({r.returncode}): {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout.strip()

    def endpoint(self) -> str:
        """pymavlink connection string for the MAVProxy test port, as seen from this host.

        WSL2 NAT: MAVProxy listens (`udpin`) inside the VM and Windows sends to the VM's
        address; replies come back on that flow, so no inbound firewall rule on Windows is
        needed. Mirrored networking or native Linux: localhost.
        """
        override = os.environ.get(ENDPOINT_ENV)
        if override:
            return override
        host = "127.0.0.1"
        if sys.platform == "win32":
            r = subprocess.run(["wsl.exe", "-d", self.cfg.wsl.distro, "--exec", "sh", "-c",
                                "wslinfo --networking-mode 2>/dev/null; echo; hostname -I"],
                               capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                raise SitlError(f"wsl.exe failed ({r.returncode}): {r.stderr.strip()}")
            if "mirrored" not in r.stdout:
                ips = [t for t in r.stdout.split() if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", t)]
                if not ips:
                    raise SitlError(f"no IPv4 address from WSL: {r.stdout!r}")
                host = ips[0]
        return f"udpout:{host}:{self.test_port}"


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    from aerosplat.logging_utils import fix_console_encoding

    fix_console_encoding()
    ap = argparse.ArgumentParser(prog="python -m aerosplat.sim.sitl",
                                 description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--instance", "-I", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="create the pinned ArduPilot worktree and build SITL")
    st = sub.add_parser("start", help="start SITL + MAVProxy in the background")
    st.add_argument("--site", help="'project/site' whose origin becomes home (default: config)")
    st.add_argument("--location", help="lat,lon,alt[,heading] (overrides --site)")
    st.add_argument("--speedup", type=float, default=None)
    st.add_argument("--projects-root", type=Path, default=None)
    sub.add_parser("stop")
    sub.add_parser("status")
    sub.add_parser("endpoint", help="print the pymavlink connection string")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    sitl = Sitl(cfg, args.instance)
    try:
        if args.cmd == "setup":
            return sitl.setup()
        if args.cmd == "start":
            home = resolve_home(cfg, args.location, args.site, args.projects_root)
            info = sitl.start(home, args.speedup)
            print(f"SITL {cfg.ardupilot.tag} ({cfg.ardupilot.commit[:8]}) запущено, instance {args.instance}")
            print(f"  home     {home.arg()}  [{home.source}]")
            print(f"  pytest   {sitl.endpoint()}")
            print(f"  GCS      {info.get('gcs', '?')}  (Mission Planner: UDP, порт {sitl.gcs_port})")
            print(f"  логи     {cfg.wsl.run_dir}/i{args.instance}/  (WSL)")
            return 0
        if args.cmd == "stop":
            print(sitl.stop())
            return 0
        if args.cmd == "status":
            running = sitl.status()
            if running is None:
                print("SITL не запущено")
                return 3
            for k, v in running.items():
                print(f"{k:10} {v}")
            return 0
        if args.cmd == "endpoint":
            print(sitl.endpoint())
            return 0
    except (SitlError, ValueError) as e:
        print(f"помилка: {e}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
