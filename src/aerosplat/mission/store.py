"""`sites/<site>/missions/<id>/`: computed missions and their exports.

    mission.json        the plan: target, access point, twin run, profile, legs, findings
    mission.waypoints   written on export only, and only for a mission that may fly
    mission.plan
    mission.parm

A mission is computed from authored data (the target) and derived data (a run's occupancy),
so it lives in neither place. It is kept because it is the record of what was checked: who
asked, against which twin, with which numbers, and what was found (DOMAIN §3 p. 6).

Exporting re-checks the target against the access point as it is NOW — a target revoked or an
access point edited after planning stops the export (ADR-0009 p. 5) — and appends who exported
what, with which risks accepted, to `authored/audit.jsonl`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from aerosplat.authored.models import AccessPoint, AuditEntry, actor
from aerosplat.authored.store import AuthoredError, NotFound
from aerosplat.geo.crs import GeoPoint
from aerosplat.mission.export import ExportRefused, render
from aerosplat.mission.models import MISSION_JSON, MissionPlan, MissionRefused, Verdict
from aerosplat.mission.targets import TargetStore, planning_findings

MISSION_ID = re.compile(r"^msn_[0-9A-HJKMNP-TV-Z]{26}$")


class MissionStore:
    def __init__(self, missions_dir: Path, site_key: str):
        self.dir = Path(missions_dir)
        self.site_key = site_key

    def folder(self, mission_id: str) -> Path:
        if not MISSION_ID.match(mission_id):          # also keeps ids from escaping the folder
            raise NotFound(f"'{mission_id}' is not a mission id")
        return self.dir / mission_id

    def save(self, plan: MissionPlan) -> Path:
        folder = self.folder(plan.id)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / (MISSION_JSON + ".tmp")
        tmp.write_text(plan.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
        os.replace(tmp, folder / MISSION_JSON)
        return folder

    def load(self, mission_id: str) -> MissionPlan:
        path = self.folder(mission_id) / MISSION_JSON
        if not path.is_file():
            raise NotFound(f"no mission '{mission_id}' on site {self.site_key}")
        try:
            return MissionPlan(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            raise AuthoredError(f"{path}: {e}") from None

    def ids(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.name for p in self.dir.iterdir()
                      if MISSION_ID.match(p.name) and (p / MISSION_JSON).is_file())

    def export(self, plan: MissionPlan, fmt: str, *, origin: GeoPoint, targets: TargetStore,
               by: Optional[str] = None, accept_risks: bool = False) -> tuple[Path, str, str]:
        """Write one flyable format next to mission.json; (path, media type, text)."""
        try:
            target = targets.get(plan.target.id)
        except NotFound:
            raise ExportRefused(f"target {plan.target.id} no longer exists") from None
        try:
            ap: Optional[AccessPoint] = targets.authored.get(target.ap_id)
        except NotFound:
            ap = None
        stopped = planning_findings(target, ap)
        if stopped:
            raise ExportRefused(str(MissionRefused(stopped)))
        name, media, text = render(plan, origin, fmt, accept_risks=accept_risks)
        path = self.folder(plan.id) / name
        path.write_text(text, encoding="utf-8")
        accepted = sorted({f.code.value for f in plan.risks})
        targets.authored.append_audit(AuditEntry(
            by=actor(by), op="mission_export", feature_id=target.ap_id, site_key=self.site_key,
            what=f"exported {plan.id} as {fmt} ({plan.verdict.value}"
                 + (f", risks accepted: {', '.join(accepted)}" if accepted else "") + ")",
            after={"mission_id": plan.id, "target_id": target.id, "format": fmt,
                   "verdict": plan.verdict.value, "risks_accepted": accepted
                   if plan.verdict is Verdict.feasible_with_risks else []}))
        return path, media, text
