"""Targets: a person's decision to deliver to one verified access point (ACCESS_POINTS §6, ADR-0009).

The system proposes, a person decides. A `MissionTarget` is created only from a `verified`
access point, and only an operator who is named can confirm it; the confirmation is
appended to `authored/audit.jsonl` with the target as it was confirmed. Confirming is
**blocked** (§6.3) when:

* the access point is not `verified` (a candidate, rejected, or `stale`);
* it is flagged `not_redetected`, or its address is `floor_ambiguous`;
* the σ of the target's centre (access point + georeferencing) is unknown, or above the
  threshold of the chosen handover mode — a number from the taxonomy, not from code.

Targets are authored data (ADR-0006): they live in `authored/targets.json`, next to the
access points, and outlive every run. Planning re-checks all of this, because the access
point may have changed between the confirmation and the plan (`target_changed`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Optional

import numpy as np

from aerosplat.authored.models import (AccessPoint, ApFlag, AuditEntry, Lifecycle, actor,
                                       new_ulid, utcnow)
from aerosplat.authored.store import AuthoredError, AuthoredStore, NotFound
from aerosplat.mission.models import (TARGET_PREFIX, TARGETS_JSON, Code, Finding, HandoverOverride,
                                      HandoverSpec, MissionRefused, MissionTarget, Severity,
                                      SigmaBudget, SigmaTerm, TargetsFile, TargetState)
from aerosplat.taxonomy import ApKind, HandoverMode, Taxonomy, TaxonomyError

SUPPORTED_KINDS = (ApKind.opening.value, ApKind.pad.value)


def _block(code: Code, detail: str = "") -> Finding:
    return Finding(code=code, severity=Severity.blocking, detail=detail)


def ap_fingerprint(ap: AccessPoint) -> str:
    """What a confirmation is a confirmation OF: class, lifecycle and where the access point is."""
    keep = {"id": ap.id, "class": ap.class_, "lifecycle": ap.lifecycle.value,
            "geometry": ap.geometry.model_dump(mode="json") if ap.geometry else None,
            "pad": ap.pad.model_dump(mode="json") if ap.pad else None,
            "host": ap.host.model_dump(mode="json") if ap.host else None}
    blob = json.dumps(keep, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sigma_of_ap(ap: AccessPoint) -> Optional[float]:
    """1σ of the access point's centre in the model: the largest axis of its covariance."""
    if ap.uncertainty is None:
        return None
    eig = np.linalg.eigvalsh(np.asarray(ap.uncertainty.center_cov, float))
    return float(math.sqrt(max(float(eig.max()), 0.0)))


# ── the gates ────────────────────────────────────────────────────────────
def ap_findings(ap: AccessPoint) -> list[Finding]:
    """Why this access point may not be a target, whatever the mode (§3.4, §5.9, §6.3)."""
    out: list[Finding] = []
    if ap.lifecycle is Lifecycle.stale:
        out.append(_block(Code.stale, f"{ap.id} is stale: it moved, a new epoch came, or it was "
                                      f"not redetected; somebody has to look at it again"))
    elif ap.lifecycle is not Lifecycle.verified:
        out.append(_block(Code.not_verified, f"{ap.id} is {ap.lifecycle.value}; only a verified "
                                             f"access point can become a target (§3.4)"))
    if ApFlag.not_redetected in ap.flags:
        out.append(_block(Code.not_redetected, f"the newest run did not find {ap.id} again"))
    if ap.address is not None and ap.address.floor_ambiguous:
        out.append(_block(Code.floor_ambiguous, f"the floor of {ap.id} is ambiguous: a wrong floor "
                                                f"is someone else's window (§4)"))
    if ap.kind not in SUPPORTED_KINDS:
        out.append(_block(Code.unsupported_kind, f"v0 approaches openings and pads, not "
                                                 f"'{ap.kind}'"))
    return out


def resolve_handover(tax: Taxonomy, ap: AccessPoint, mode: Optional[HandoverMode],
                     override: Optional[HandoverOverride] = None
                     ) -> tuple[Optional[HandoverSpec], list[Finding]]:
    """The class's approach numbers for `mode`, with the operator's overrides — if they are safer."""
    try:
        handover = tax.handover_of(ap.class_)
    except (KeyError, TaxonomyError):
        handover = None
    if handover is None:
        return None, [_block(Code.no_handover, f"class '{ap.class_}' has no handover numbers in "
                                               f"taxonomy {tax.version_string}")]
    mode = mode or handover.default
    findings: list[Finding] = []
    if mode not in handover.allowed:
        findings.append(_block(Code.mode_not_allowed, f"'{mode.value}' is not allowed for "
                                                      f"{ap.class_} (allowed: "
                                                      f"{', '.join(m.value for m in handover.allowed)})"))
    spec = tax.handover_mode(mode)
    if spec is None:
        findings.append(_block(Code.no_threshold, f"taxonomy {tax.version_string} has no σ "
                                                  f"threshold for '{mode.value}'"))
    if ap.kind == ApKind.opening.value and handover.entry_distance_m is None:
        findings.append(_block(Code.no_entry_distance, f"class '{ap.class_}' has no "
                                                       f"entry_distance_m"))
    if findings or spec is None:
        return None, findings

    base = HandoverSpec(mode=mode, standoff_m=handover.standoff_m,
                        min_clearance_m=handover.min_clearance_m,
                        approach_cone_deg=handover.approach_cone_deg,
                        entry_distance_m=handover.entry_distance_m,
                        hover_dh_m=handover.hover_dh_m, max_sigma_m=spec.max_sigma_m,
                        source=f"{tax.version_string} {ap.class_}")
    # Which way is "more careful" for each number: further off, more clearance, a narrower
    # cone, a longer straight approach. Anything else would let a target bypass the taxonomy.
    safer = {"standoff_m": 1, "min_clearance_m": 1, "approach_cone_deg": -1, "entry_distance_m": 1}
    changes: dict[str, float] = {}
    for name, value in (override.model_dump(exclude_none=True) if override else {}).items():
        was = getattr(base, name)
        if was is not None and (value - was) * safer[name] < 0:
            findings.append(_block(Code.override_relaxes_taxonomy,
                                   f"{name}={value} is less careful than the taxonomy's {was}"))
            continue
        changes[name] = float(value)
    out = base.model_copy(update={**changes, "overridden": sorted(changes)})
    if out.entry_distance_m is not None and out.entry_distance_m <= out.standoff_m:
        findings.append(_block(Code.override_relaxes_taxonomy,
                               f"entry_distance_m={out.entry_distance_m} must lie beyond "
                               f"standoff_m={out.standoff_m}"))
    if findings:
        return None, findings
    return out, []


def sigma_budget(ap: AccessPoint, *, sigma_ap_m: Optional[float], sigma_georef_m: Optional[float],
                 by: str) -> tuple[Optional[SigmaBudget], list[Finding]]:
    """σ of the access point (its own record, else the operator's word) and of georeferencing
    (the operator's word: no stage measures it yet). Unknown is a refusal, never a zero."""
    findings: list[Finding] = []
    stored = sigma_of_ap(ap)
    ap_term: Optional[SigmaTerm] = None
    if stored is not None:
        ap_term = SigmaTerm(value_m=stored, source=f"{ap.id}.uncertainty")
    elif sigma_ap_m is not None:
        ap_term = SigmaTerm(value_m=sigma_ap_m, source=f"stated by {by}")
    else:
        findings.append(_block(Code.sigma_unknown, f"{ap.id} carries no uncertainty and none was "
                                                   f"stated (sigma_ap_m)"))
    geo_term: Optional[SigmaTerm] = None
    if sigma_georef_m is not None:
        geo_term = SigmaTerm(value_m=sigma_georef_m, source=f"stated by {by}")
    else:
        findings.append(_block(Code.sigma_unknown, "the georeferencing σ of the twin was not "
                                                   "stated (sigma_georef_m)"))
    for term in (ap_term, geo_term):
        if term is not None and not (math.isfinite(term.value_m) and term.value_m >= 0):
            findings.append(_block(Code.sigma_unknown, f"σ {term.value_m} from {term.source}"))
    if findings or ap_term is None or geo_term is None:
        return None, findings
    return SigmaBudget(ap=ap_term, georef=geo_term), []


def sigma_findings(spec: HandoverSpec, sigma: SigmaBudget) -> list[Finding]:
    total = sigma.target_m
    if total > spec.max_sigma_m:
        return [_block(Code.sigma_too_large,
                       f"σ of the target centre is {total:.2f} m (access point "
                       f"{sigma.ap.value_m:.2f}, georeferencing {sigma.georef.value_m:.2f}); "
                       f"'{spec.mode.value}' allows {spec.max_sigma_m:.2f} m ({spec.source})")]
    return []


def planning_findings(target: MissionTarget, ap: Optional[AccessPoint]) -> list[Finding]:
    """Everything that stops a confirmed target from being planned or exported NOW."""
    if target.state is TargetState.revoked:
        return [_block(Code.target_revoked, f"{target.id} was revoked by {target.revoked_by}")]
    out: list[Finding] = []
    if target.state is not TargetState.confirmed:
        out.append(_block(Code.target_not_confirmed, f"{target.id} is {target.state.value}: an "
                                                     f"operator has to confirm it (ADR-0009)"))
    if ap is None:
        out.append(_block(Code.ap_missing, f"access point {target.ap_id} no longer exists"))
        return out
    out += ap_findings(ap)
    if ap_fingerprint(ap) != target.ap_fingerprint:
        out.append(_block(Code.target_changed, f"{ap.id} changed after {target.id} was confirmed; "
                                               f"confirm the target again"))
    out += sigma_findings(target.handover, target.sigma)
    return out


# ── the store ────────────────────────────────────────────────────────────
class TargetStore:
    """`authored/targets.json` of one Site; confirmations go to `authored/audit.jsonl`."""

    def __init__(self, authored: AuthoredStore):
        self.authored = authored

    @property
    def path(self):
        return self.authored.dir / TARGETS_JSON

    def load(self) -> TargetsFile:
        with self.authored.lock:
            if not self.path.is_file():
                return TargetsFile(site_key=self.authored.site_key)
            try:
                return TargetsFile(**json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as e:
                raise AuthoredError(f"{self.path}: {e}") from None

    def targets(self) -> list[MissionTarget]:
        return sorted(self.load().targets, key=lambda t: t.id)

    def get(self, target_id: str) -> MissionTarget:
        for t in self.load().targets:
            if t.id == target_id:
                return t
        raise NotFound(f"no target '{target_id}' on site {self.authored.site_key}")

    def _save(self, state: TargetsFile) -> None:
        self.authored.dir.mkdir(parents=True, exist_ok=True)
        state.updated_at = utcnow()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(state.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def _put(self, target: MissionTarget, *, op: str, by: str, what: str,
             before: Optional[MissionTarget]) -> MissionTarget:
        with self.authored.lock:
            state = self.load()
            state.targets = [t for t in state.targets if t.id != target.id] + [target]
            self._save(state)
            self.authored.append_audit(AuditEntry(
                by=by, op=op, feature_id=target.ap_id, site_key=self.authored.site_key, what=what,
                before=before.model_dump(mode="json", by_alias=True) if before else None,
                after=target.model_dump(mode="json", by_alias=True)))
        return target

    def create(self, tax: Taxonomy, ap_id: str, *, by: Optional[str],
               mode: Optional[HandoverMode] = None, override: Optional[HandoverOverride] = None,
               sigma_ap_m: Optional[float] = None, sigma_georef_m: Optional[float] = None,
               confirm: bool = False, notes: str = "") -> MissionTarget:
        """Create a target from an access point; with `confirm`, confirm it in the same step.

        Raises MissionRefused with every reason at once, so an operator sees the whole
        picture instead of fixing one complaint at a time. Nothing is written on refusal.
        """
        who = actor(by)
        try:
            ap = self.authored.get(ap_id)
        except NotFound:
            raise MissionRefused([_block(Code.ap_missing, f"no access point '{ap_id}'")]) from None
        findings = ap_findings(ap)
        spec, f_spec = resolve_handover(tax, ap, mode, override)
        sigma, f_sigma = sigma_budget(ap, sigma_ap_m=sigma_ap_m, sigma_georef_m=sigma_georef_m,
                                      by=who)
        findings += f_spec + f_sigma
        if spec is not None and sigma is not None:
            findings += sigma_findings(spec, sigma)
        if findings or spec is None or sigma is None:
            raise MissionRefused(findings)
        now = utcnow()
        target = MissionTarget(
            id=new_ulid(TARGET_PREFIX), site_key=self.authored.site_key, ap_id=ap.id,
            ap_class=ap.class_, ap_kind=ap.kind, ap_fingerprint=ap_fingerprint(ap),
            taxonomy=tax.version_string, handover=spec, sigma=sigma, address=ap.address,
            state=TargetState.confirmed if confirm else TargetState.draft, created_by=who,
            created_at=now, confirmed_by=who if confirm else None,
            confirmed_at=now if confirm else None, notes=notes)
        what = (f"confirmed target {target.id}: {ap.class_} on {ap.where}, {spec.mode.value}, "
                f"σ {sigma.target_m:.2f} m" if confirm else
                f"proposed target {target.id}: {ap.class_} on {ap.where}, {spec.mode.value}")
        return self._put(target, op="target_confirm" if confirm else "target_create", by=who,
                         what=what, before=None)

    def confirm(self, tax: Taxonomy, target_id: str, *, by: Optional[str]) -> MissionTarget:
        """Confirm a draft. The gates run again against the access point as it is now."""
        who = actor(by)
        with self.authored.lock:
            old = self.get(target_id)
            if old.state is not TargetState.draft:
                raise MissionRefused([_block(
                    Code.target_revoked if old.state is TargetState.revoked
                    else Code.target_not_confirmed, f"{target_id} is {old.state.value}, not a draft")])
            try:
                ap: Optional[AccessPoint] = self.authored.get(old.ap_id)
            except NotFound:
                ap = None
            if ap is None:
                raise MissionRefused([_block(Code.ap_missing, f"no access point '{old.ap_id}'")])
            findings = ap_findings(ap)
            if ap_fingerprint(ap) != old.ap_fingerprint:
                findings.append(_block(Code.target_changed, f"{ap.id} changed since the draft"))
            spec, f_spec = resolve_handover(tax, ap, old.handover.mode)
            findings += f_spec
            findings += sigma_findings(old.handover, old.sigma)
            if findings or spec is None:
                raise MissionRefused(findings)
            now = utcnow()
            new = old.model_copy(update={"state": TargetState.confirmed, "confirmed_by": who,
                                         "confirmed_at": now}, deep=True)
            return self._put(new, op="target_confirm", by=who, before=old,
                             what=f"confirmed target {new.id}: {ap.class_} on {ap.where}, "
                                  f"{new.handover.mode.value}, σ {new.sigma.target_m:.2f} m")

    def revoke(self, target_id: str, *, by: Optional[str], reason: str = "") -> MissionTarget:
        who = actor(by)
        with self.authored.lock:
            old = self.get(target_id)
            new = old.model_copy(update={"state": TargetState.revoked, "revoked_by": who,
                                         "revoked_at": utcnow()}, deep=True)
            return self._put(new, op="target_revoke", by=who, before=old,
                             what=f"revoked target {new.id}" + (f": {reason}" if reason else ""))
