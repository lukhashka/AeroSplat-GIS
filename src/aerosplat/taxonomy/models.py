"""Taxonomy of access-point classes: the pydantic shape of a `taxonomy/*.yaml` pack.

Classes are data, not code (ADR-0007): a new class is a YAML entry, not a release.
This module only describes and validates the format from ACCESS_POINTS §2.4; which
packs a Site uses is `Site.taxonomy_packs`.

Ids read `kind.class[.subclass]`. The leading segment is the `kind`, which fixes the
geometry a feature of that class carries (§3.2). The id is a naming convention; the
real hierarchy edge is `parent`, which may point outside the id prefix
(`opening.dock_door` has parent `opening.door`), so a model may answer with the
parent when it is unsure of the child and still be partly right (§2.1).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# `kind.class` or `kind.class.subclass`; the kind itself is checked against ApKind.
CLASS_ID = re.compile(r"^[a-z_]+(?:\.[a-z0-9_]+){1,2}$")
PACK_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
REQUIREMENT = re.compile(r"^>=\s*(\d+)\.(\d+)(?:\.(\d+))?$")


class ApKind(str, Enum):
    """What a feature of this class is, and therefore what geometry it carries (§3.2)."""

    opening = "opening"            # a hole in a surface: polygon on the host surface
    pad = "pad"                    # a usable horizontal patch: polygon + usable area
    marker = "marker"              # an artificial target of known geometry: 6DoF pose
    constraint = "constraint"      # not a target: an obstacle, hazard or forbidden volume


class Orientation(str, Enum):
    vertical = "vertical"
    horizontal = "horizontal"
    any = "any"


class HandoverMode(str, Enum):
    """How the drone gives up or takes the load (§1)."""

    hover_handoff = "hover_handoff"      # hover opposite the opening, within a person's reach
    tether_lower = "tether_lower"        # lower on a tether
    drop = "drop"
    land = "land"
    pickup = "pickup"
    enter = "enter"                      # fly into the volume (drone ports only, later)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Labels(_Model):
    uk: str
    en: str = ""


class SizePrior(_Model):
    """Plausible metric size of the class, used as a filter when fusing detections (§5.5)."""

    width: tuple[float, float]
    height: tuple[float, float]

    @model_validator(mode="after")
    def _ordered(self) -> "SizePrior":
        for name, (lo, hi) in (("width", self.width), ("height", self.height)):
            if not 0 < lo <= hi:
                raise ValueError(f"size_prior_m.{name} must be [lo, hi] with 0 < lo <= hi")
        return self

    def admits(self, width_m: float, height_m: float) -> bool:
        return (self.width[0] <= width_m <= self.width[1]
                and self.height[0] <= height_m <= self.height[1])


class Handover(_Model):
    """Default approach numbers for the class (§7.1). The operator may override them per target,
    but only towards caution (aerosplat.mission.targets)."""

    allowed: list[HandoverMode] = Field(min_length=1)
    default: HandoverMode
    # Openings: hover distance in front of the opening. Pads: hover height above the pad.
    standoff_m: float = Field(gt=0, le=100)
    min_clearance_m: float = Field(gt=0, le=100)
    # Full apex angle of the cone the approach direction may tilt within (half each side).
    approach_cone_deg: float = Field(gt=0, le=180)
    # Openings only: distance of the entry point E from the opening, along the approach
    # direction (§7.2). Unset = the class cannot be approached horizontally.
    entry_distance_m: Optional[float] = Field(None, gt=0, le=200)
    # Openings only: hover this much above the opening's centre (§7.2, `up·Δh`).
    hover_dh_m: float = Field(0.0, ge=-10, le=10)

    @model_validator(mode="after")
    def _default_is_allowed(self) -> "Handover":
        if self.default not in self.allowed:
            raise ValueError(f"handover.default '{self.default.value}' is not in allowed")
        if len(set(self.allowed)) != len(self.allowed):
            raise ValueError("handover.allowed has duplicates")
        if self.entry_distance_m is not None and self.entry_distance_m <= self.standoff_m:
            raise ValueError(f"handover.entry_distance_m ({self.entry_distance_m}) must be beyond "
                             f"standoff_m ({self.standoff_m})")
        return self


class HandoverModeSpec(_Model):
    """Numbers that belong to the way of handing over, whatever the class (§6.3, SIMULATION §5.4).

    `max_sigma_m` gates designation: a target whose centre is known worse than this (1σ, metres,
    access point and georeferencing together) cannot be confirmed for this mode, because the
    drone's own guidance would start from a prior too vague to find the right opening.
    """

    max_sigma_m: float = Field(gt=0, le=50)


class Attribute(_Model):
    """An attribute classes may carry (§2.3). Either an enum of `values` or `free_text`."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    labels: Labels
    values: list[str] = Field(default_factory=list)
    default: Optional[str] = None
    free_text: bool = False
    # Depends on the epoch (a window gets opened), so it does not carry over to a new
    # run without a fresh capture.
    dynamic: bool = False

    @model_validator(mode="after")
    def _check(self) -> "Attribute":
        if self.free_text:
            if self.values:
                raise ValueError(f"attribute '{self.id}': free_text and values are exclusive")
        elif not self.values:
            raise ValueError(f"attribute '{self.id}': give values, or set free_text")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"attribute '{self.id}': duplicate values")
        if self.default is not None and self.values and self.default not in self.values:
            raise ValueError(f"attribute '{self.id}': default '{self.default}' is not a value")
        return self

    def admits(self, value: str) -> bool:
        return self.free_text or value in self.values


class TaxonomyClass(_Model):
    """One class (§2.4). `pack` is filled in by the loader, not written in the YAML."""

    id: str = Field(pattern=CLASS_ID.pattern)
    parent: Optional[str] = None
    labels: Labels
    prompts: list[str] = Field(default_factory=list)      # open-vocabulary bootstrap (§5.4)
    orientation: Orientation = Orientation.any
    size_prior_m: Optional[SizePrior] = None
    attributes: list[str] = Field(default_factory=list)
    handover: Optional[Handover] = None
    citygml: Optional[str] = None                         # CityGML 3.0 LOD3 export (§2.6)
    pack: str = ""

    @property
    def kind(self) -> ApKind:
        return ApKind(self.id.split(".", 1)[0])

    @model_validator(mode="after")
    def _check(self) -> "TaxonomyClass":
        head = self.id.split(".", 1)[0]
        if head not in {k.value for k in ApKind}:
            raise ValueError(f"class '{self.id}': '{head}' is not a kind "
                             f"({', '.join(k.value for k in ApKind)})")
        if self.parent == self.id:
            raise ValueError(f"class '{self.id}' is its own parent")
        if self.parent is not None and self.parent.split(".", 1)[0] != head:
            raise ValueError(f"class '{self.id}': parent '{self.parent}' is a different kind")
        if len(set(self.attributes)) != len(self.attributes):
            raise ValueError(f"class '{self.id}': duplicate attributes")
        # A constraint is not a target, so approach numbers would be meaningless on one.
        if self.handover is not None and head == ApKind.constraint.value:
            raise ValueError(f"class '{self.id}': a constraint has no handover")
        return self


class TaxonomyPack(_Model):
    """One `taxonomy/<pack>.yaml` file as written on disk."""

    pack: str = Field(pattern=PACK_NAME.pattern)
    version: str = Field(pattern=SEMVER.pattern)
    requires: dict[str, str] = Field(default_factory=dict)
    attributes: list[Attribute] = Field(default_factory=list)
    classes: list[TaxonomyClass] = Field(default_factory=list)
    handover_modes: dict[HandoverMode, HandoverModeSpec] = Field(default_factory=dict)

    @property
    def short_version(self) -> str:
        """`major.minor`: the identity that matters for labels, since a patch adds no class (§2.5)."""
        major, minor, _ = self.version.split(".")
        return f"{major}.{minor}"

    @model_validator(mode="after")
    def _unique_ids(self) -> "TaxonomyPack":
        for what, ids in (("class", [c.id for c in self.classes]),
                          ("attribute", [a.id for a in self.attributes])):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                raise ValueError(f"pack '{self.pack}': duplicate {what} ids {dupes}")
        return self
