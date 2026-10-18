"""The parameter table of the pinned firmware, as data: names, defaults, units, allowed values.

Parameter names and units change between ArduPilot releases (Copter 4.7 renamed RTL_ALT [cm]
to RTL_ALT_M [m] and WPNAV_SPEED [cm/s] to WP_SPD [m/s], among others), so nothing here is
written from memory. The snapshot is produced from the pinned build itself:

    python -m aerosplat.sim.sitl start
    python -m aerosplat.sim.params snapshot     # -> configs/ardupilot/<tag>.params.json

* names and defaults: PARAM_REQUEST_LIST on the running SITL (wiped eeprom, copter.parm
  defaults: the same state every scenario starts from);
* units, ranges, enumerations: ArduPilot's own `param_parse.py` metadata of the pinned source
  (tools/sitl/param_metadata.sh), kept for the parameter families a mission package, a fault
  or a vehicle profile may touch.

The live table is authoritative at run time (the SITL runner refuses a `.parm` with a name the
vehicle does not have); the snapshot lets plain `pytest` check the exporter and the fault
mapping without SITL, and the SITL suite checks the snapshot against the live table.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from aerosplat.paths import PROJECT_DIR

SNAPSHOT_DIR = PROJECT_DIR / "configs" / "ardupilot"
# Families whose metadata (units, range, values) the snapshot keeps.
META_PREFIXES = ("FENCE_", "RTL_", "WP_", "FS_", "BATT_", "AVOID_", "RALLY_", "LAND_", "SRTL_",
                 "MAV_GCS_", "SIM_GPS1_", "SIM_RC_", "SIM_BATT_", "SIM_WIND_", "SIM_ENGINE_",
                 "SIM_SPEEDUP", "RC_FS_", "DISARM_", "AUTO_OPTIONS", "ARMING_", "EK3_SRC1_")


class ParameterError(ValueError):
    """A parameter the pinned firmware does not have, or a value it does not accept."""


@dataclass(frozen=True)
class ParamInfo:
    name: str
    default: float
    units: Optional[str] = None
    display: Optional[str] = None
    low: Optional[float] = None
    high: Optional[float] = None
    values: Optional[dict[int, str]] = None
    bitmask: Optional[dict[int, str]] = None

    def problem(self, value: float) -> Optional[str]:
        """Why the firmware would not accept `value` as meant, or None."""
        if not math.isfinite(value):
            return f"{self.name} = {value}: not a number"
        if self.low is not None and value < self.low - 1e-9 or \
                self.high is not None and value > self.high + 1e-9:
            return (f"{self.name} = {value:g} is outside the documented range "
                    f"[{self.low:g}, {self.high:g}] {self.units or ''}".rstrip())
        if self.values is not None and value == int(value) and int(value) not in self.values:
            return f"{self.name} = {value:g} is not one of {sorted(self.values)}"
        if self.bitmask is not None:
            if value != int(value) or value < 0:
                return f"{self.name} = {value:g} is not a bitmask"
            unknown = [b for b in range(32) if int(value) >> b & 1 and b not in self.bitmask]
            if unknown:
                return f"{self.name} = {value:g} sets undocumented bits {unknown}"
        return None

    def describe(self, value: float) -> str:
        """`value` in the firmware's words: '3 (SmartRTL or RTL)', '25 m'."""
        v = f"{value:g}"
        if self.values is not None and value == int(value) and int(value) in self.values:
            return f"{v} ({self.values[int(value)]})"
        if self.bitmask is not None and value == int(value) and value >= 0:
            bits = [self.bitmask.get(b, f"bit {b}") for b in range(32) if int(value) >> b & 1]
            return f"{v} ({', '.join(bits) or 'none'})"
        return f"{v} {self.units}" if self.units else v


@dataclass(frozen=True)
class FirmwareParams:
    tag: str
    commit: str
    params: dict[str, ParamInfo]
    source: str = ""

    def __contains__(self, name: object) -> bool:
        return name in self.params

    def __getitem__(self, name: str) -> ParamInfo:
        try:
            return self.params[name]
        except KeyError:
            raise ParameterError(unknown_message([name], self.params, self.tag)) from None

    def default(self, name: str) -> float:
        return self[name].default

    def unknown(self, names) -> list[str]:
        return [n for n in names if n not in self.params]

    def problems(self, values: Mapping[str, float]) -> list[str]:
        """Unknown names first (with the closest real names), then values out of range."""
        out: list[str] = []
        unknown = self.unknown(values)
        if unknown:
            out.append(unknown_message(unknown, self.params, self.tag))
        for name, value in values.items():
            if name in self.params:
                p = self.params[name].problem(float(value))
                if p:
                    out.append(p)
        return out

    def require(self, values: Mapping[str, float], what: str) -> None:
        problems = self.problems(values)
        if problems:
            raise ParameterError(f"{what}: " + "; ".join(problems))


def suggestions(name: str, known) -> list[str]:
    """Real names close to `name`: renamed families first (RTL_ALT -> RTL_ALT_M)."""
    known = list(known)
    # NAME + a unit suffix (RTL_ALT -> RTL_ALT_M) first, shortest first.
    prefixed = sorted((k for k in known if k.startswith(name + "_")), key=lambda k: (len(k), k))
    close = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
    out: list[str] = []
    for k in prefixed[:2] + close:
        if k not in out:
            out.append(k)
    return out[:3]


def unknown_message(names, known, tag: str) -> str:
    parts = []
    for n in names:
        s = suggestions(n, known)
        parts.append(f"{n}" + (f" (did you mean {' / '.join(s)}?)" if s else ""))
    return f"not parameters of {tag}: " + ", ".join(parts)


# ── snapshot file ────────────────────────────────────────────────────────
def snapshot_path(tag: str, folder: Path = SNAPSHOT_DIR) -> Path:
    return folder / f"{tag}.params.json"


def load_snapshot(tag: Optional[str] = None, commit: Optional[str] = None,
                  folder: Path = SNAPSHOT_DIR) -> FirmwareParams:
    """The snapshot of the pinned firmware (configs/sitl.yaml unless tag/commit are given).
    Refuses a snapshot taken from another commit than the pin."""
    if tag is None or commit is None:
        from aerosplat.sim.sitl import load_config

        pin = load_config().ardupilot
        tag, commit = tag or pin.tag, commit or pin.commit
    path = snapshot_path(tag, folder)
    if not path.is_file():
        raise ParameterError(f"no parameter snapshot {path}: python -m aerosplat.sim.params snapshot")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("commit") != commit:
        raise ParameterError(f"{path} was taken from {doc.get('commit')}, the pin is {commit}: "
                             f"regenerate it (python -m aerosplat.sim.params snapshot)")
    return from_document(doc)


def from_document(doc: dict) -> FirmwareParams:
    params = {}
    for name, d in doc["params"].items():
        params[name] = ParamInfo(
            name=name, default=float(d["default"]), units=d.get("units"), display=d.get("display"),
            low=d.get("range", [None, None])[0], high=d.get("range", [None, None])[1],
            values={int(k): v for k, v in d["values"].items()} if "values" in d else None,
            bitmask={int(k): v for k, v in d["bitmask"].items()} if "bitmask" in d else None)
    return FirmwareParams(tag=doc["tag"], commit=doc["commit"], params=params,
                          source=doc.get("source", ""))


def _flatten_metadata(pdef: dict) -> dict[str, dict]:
    flat: dict[str, dict] = {}
    for group in pdef.values():
        if isinstance(group, dict):
            for name, meta in group.items():
                if isinstance(meta, dict):
                    flat.setdefault(name, meta)
    return flat


def _int_keys(d: dict) -> Optional[dict[str, str]]:
    out = {}
    for k, v in d.items():
        try:
            if float(k) != int(float(k)):
                return None                     # e.g. FS_EKF_THRESH: real-valued presets
            out[str(int(float(k)))] = str(v)
        except ValueError:
            return None
    return out


def build_document(live: Mapping[str, float], pdef: dict, *, tag: str, commit: str,
                   source: str) -> dict:
    meta = _flatten_metadata(pdef)
    params: dict[str, dict[str, Any]] = {}
    for name in sorted(live):
        entry: dict[str, Any] = {"default": float(live[name])}
        m = meta.get(name)
        if m is not None and name.startswith(META_PREFIXES):
            if m.get("Units"):
                entry["units"] = m["Units"]
            if m.get("DisplayName"):
                entry["display"] = m["DisplayName"]
            rng = m.get("Range")
            if isinstance(rng, dict) and "low" in rng and "high" in rng:
                entry["range"] = [float(rng["low"]), float(rng["high"])]
            if isinstance(m.get("Values"), dict):
                vals = _int_keys(m["Values"])
                if vals is not None:
                    entry["values"] = vals
            if isinstance(m.get("Bitmask"), dict):
                bits = _int_keys(m["Bitmask"])
                if bits is not None:
                    entry["bitmask"] = bits
        params[name] = entry
    return {"tag": tag, "commit": commit, "source": source,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": len(params), "metadata_prefixes": list(META_PREFIXES), "params": params}


def main(argv: Optional[list[str]] = None) -> int:
    from aerosplat.logging_utils import fix_console_encoding
    from aerosplat.sim.sitl import Sitl, load_config
    from aerosplat.sim.vehicle import Vehicle

    fix_console_encoding()
    ap = argparse.ArgumentParser(prog="python -m aerosplat.sim.params",
                                 description="parameter snapshot of the pinned ArduCopter build")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sn = sub.add_parser("snapshot", help="live table of a freshly started SITL + source metadata")
    sn.add_argument("--endpoint", help="pymavlink endpoint (default: the running instance 0)")
    sn.add_argument("--metadata", type=Path, help="apm.pdef.json (default: generate in WSL)")
    sn.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    sitl = Sitl(cfg)
    if args.metadata:
        pdef = json.loads(args.metadata.read_text(encoding="utf-8"))
    else:
        env = {k: v for k, v in sitl.env().items() if k in ("AP_WORKTREE", "AP_COMMIT", "AP_VENV")}
        r = subprocess.run(sitl.command("param_metadata.sh", env), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=600)
        if r.returncode != 0:
            print(f"помилка: param_metadata.sh: {r.stderr.strip()}", file=sys.stderr)
            return 1
        pdef = json.loads(r.stdout)
    endpoint = args.endpoint or sitl.endpoint()
    with Vehicle.connect(endpoint) as v:
        fw = v.firmware()
        if fw.git_hash != cfg.ardupilot.commit[:8]:
            print(f"помилка: the vehicle runs {fw.git_hash}, the pin is {cfg.ardupilot.commit[:8]}",
                  file=sys.stderr)
            return 1
        live = v.fetch_params()
    doc = build_document(live, pdef, tag=cfg.ardupilot.tag, commit=cfg.ardupilot.commit,
                         source="PARAM_REQUEST_LIST on SITL (-w, copter.parm defaults) + "
                                "Tools/autotest/param_metadata/param_parse.py")
    out = args.out or snapshot_path(cfg.ardupilot.tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{doc['count']} parameters of {cfg.ardupilot.tag} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
