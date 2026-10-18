"""Load and cross-validate taxonomy packs into one resolved `Taxonomy`.

Packs live in `taxonomy/` next to `configs/` (override with `AEROSPLAT_TAXONOMY`).
A Site names the ones it uses in `Site.taxonomy_packs`, default `["core"]`.

Validation a single pack cannot do itself: that `parent` points at a class that
exists, that `attributes` name attributes somebody defined, that `requires` is
satisfied by the versions actually loaded, and that two packs do not define the
same id. Everything that can fail inside one pack fails in `models.py`.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable, Optional

import yaml

from aerosplat.paths import PROJECT_DIR
from aerosplat.taxonomy.models import (REQUIREMENT, ApKind, Attribute, Handover, HandoverMode,
                                       HandoverModeSpec, TaxonomyClass, TaxonomyPack)

CORE = "core"


class TaxonomyError(ValueError):
    """A pack that is missing, malformed, or inconsistent with the others."""


def taxonomy_root() -> Path:
    env = os.environ.get("AEROSPLAT_TAXONOMY")
    return Path(env) if env else PROJECT_DIR / "taxonomy"


def _version_ok(have: str, requirement: str) -> bool:
    """Is `have` (x.y.z) allowed by `requirement` (">=x.y[.z]", or "*"/empty for any)?"""
    req = requirement.strip()
    if req in ("", "*"):
        return True
    m = REQUIREMENT.match(req)
    if m is None:
        raise TaxonomyError(f"unsupported version requirement '{requirement}' (use '>=x.y')")
    want = (int(m[1]), int(m[2]), int(m[3] or 0))
    return tuple(int(p) for p in have.split(".")) >= want


class Taxonomy:
    """Several packs merged and cross-checked: the class set a Site actually works with."""

    def __init__(self, packs: list[TaxonomyPack]):
        self.packs = packs
        self.classes: dict[str, TaxonomyClass] = {}
        self.attributes: dict[str, Attribute] = {}
        self.handover_modes: dict[HandoverMode, HandoverModeSpec] = {}
        attr_pack: dict[str, str] = {}
        mode_pack: dict[HandoverMode, str] = {}
        for pack in packs:
            for mode, spec in pack.handover_modes.items():
                if mode in self.handover_modes:
                    raise TaxonomyError(f"handover mode '{mode.value}' is defined by both "
                                        f"'{mode_pack[mode]}' and '{pack.pack}'")
                self.handover_modes[mode] = spec
                mode_pack[mode] = pack.pack
            for attr in pack.attributes:
                if attr.id in self.attributes:
                    raise TaxonomyError(f"attribute '{attr.id}' is defined by both "
                                        f"'{attr_pack[attr.id]}' and '{pack.pack}'")
                self.attributes[attr.id] = attr
                attr_pack[attr.id] = pack.pack
            for cls in pack.classes:
                if cls.id in self.classes:
                    raise TaxonomyError(f"class '{cls.id}' is defined by both "
                                        f"'{self.classes[cls.id].pack}' and '{pack.pack}'")
                self.classes[cls.id] = cls.model_copy(update={"pack": pack.pack})
        self._check()

    # ── validation across packs ──────────────────────────────────────────
    def _check(self) -> None:
        names = {p.pack: p for p in self.packs}
        for pack in self.packs:
            for need, requirement in pack.requires.items():
                have = names.get(need)
                if have is None:
                    raise TaxonomyError(f"pack '{pack.pack}' requires '{need}', which is not loaded")
                if not _version_ok(have.version, requirement):
                    raise TaxonomyError(f"pack '{pack.pack}' requires {need}{requirement}, "
                                        f"but {need} is {have.version}")
        for cls in self.classes.values():
            if cls.parent is not None and cls.parent not in self.classes:
                raise TaxonomyError(f"class '{cls.id}': parent '{cls.parent}' does not exist "
                                    f"(is a pack missing?)")
            unknown = [a for a in cls.attributes if a not in self.attributes]
            if unknown:
                raise TaxonomyError(f"class '{cls.id}': unknown attributes {unknown}")
        for cls_id in self.classes:
            self.ancestors(cls_id)                      # raises on a cycle

    # ── queries ──────────────────────────────────────────────────────────
    def get(self, class_id: str) -> TaxonomyClass:
        try:
            return self.classes[class_id]
        except KeyError:
            raise TaxonomyError(f"unknown class '{class_id}' in taxonomy "
                                f"{self.version_string}") from None

    def ancestors(self, class_id: str) -> list[str]:
        """Parent chain, nearest first. Raises on a cycle."""
        out: list[str] = []
        seen = {class_id}
        node = self.classes[class_id].parent
        while node is not None:
            if node in seen:
                raise TaxonomyError(f"class '{class_id}': parent chain loops at '{node}'")
            seen.add(node)
            out.append(node)
            node = self.classes[node].parent if node in self.classes else None
        return out

    def attributes_of(self, class_id: str) -> list[Attribute]:
        """The class's own attributes plus every ancestor's, nearest first, deduplicated."""
        out: dict[str, Attribute] = {}
        for cid in [class_id, *self.ancestors(class_id)]:
            for name in self.classes[cid].attributes:
                out.setdefault(name, self.attributes[name])
        return list(out.values())

    def handover_of(self, class_id: str) -> Optional[Handover]:
        """The class's handover defaults, inherited from the nearest ancestor that sets them."""
        for cid in [class_id, *self.ancestors(class_id)]:
            cls = self.classes[cid]
            if cls.handover is not None:
                return cls.handover
        return None

    def handover_mode(self, mode: HandoverMode) -> Optional[HandoverModeSpec]:
        """Per-mode numbers (σ gate), or None when no loaded pack defines the mode."""
        return self.handover_modes.get(mode)

    def ids_of_kind(self, kind: ApKind) -> list[str]:
        return sorted(c.id for c in self.classes.values() if c.kind == kind)

    def validate_attributes(self, class_id: str, values: dict[str, str]) -> None:
        """Raise unless every name is an attribute of the class and every value is in its domain."""
        allowed = {a.id: a for a in self.attributes_of(class_id)}
        for name, value in values.items():
            attr = allowed.get(name)
            if attr is None:
                raise TaxonomyError(f"class '{class_id}' has no attribute '{name}' "
                                    f"(it has: {', '.join(sorted(allowed)) or 'none'})")
            if not attr.admits(value):
                raise TaxonomyError(f"attribute '{name}': '{value}' is not one of "
                                    f"{', '.join(attr.values)}")

    @property
    def version_string(self) -> str:
        """`core@1.0+logistics@1.0`: stored on every feature, dataset and model (§2.5)."""
        return "+".join(f"{p.pack}@{p.short_version}" for p in self.packs)


def load_pack(name: str, root: Optional[Path] = None) -> TaxonomyPack:
    path = (root or taxonomy_root()) / f"{name}.yaml"
    if not path.is_file():
        raise TaxonomyError(f"taxonomy pack '{name}' not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise TaxonomyError(f"{path}: {e}") from None
    if not isinstance(data, dict):
        raise TaxonomyError(f"{path}: expected a mapping at the top level")
    try:
        pack = TaxonomyPack(**data)
    except ValueError as e:
        raise TaxonomyError(f"{path}: {e}") from None
    if pack.pack != name:
        raise TaxonomyError(f"{path}: declares pack '{pack.pack}' but the file is '{name}.yaml'")
    return pack


def pack_names(packs: Iterable[str]) -> list[str]:
    """The pack names that will be loaded, in order: core first, then the rest, deduplicated."""
    names = [CORE]
    for name in packs:
        if name != CORE and name not in names:
            names.append(name)
    return names


def load_taxonomy(packs: Iterable[str] = (CORE,), root: Optional[Path] = None) -> Taxonomy:
    """Load the named packs (core first, always included) and cross-validate them."""
    return Taxonomy([load_pack(n, root) for n in pack_names(packs)])


_CACHE: dict[tuple[str, ...], tuple[tuple[float, ...], Taxonomy]] = {}
_CACHE_LOCK = threading.Lock()


def cached_taxonomy(packs: Iterable[str] = (CORE,), root: Optional[Path] = None) -> Taxonomy:
    """`load_taxonomy` memoised on the files' mtimes, for request handlers.

    Classes are data (ADR-0007), so editing a pack must take effect without a restart;
    re-parsing four small YAML files on every request would still be silly.
    """
    base = root or taxonomy_root()
    names = tuple(pack_names(packs))
    key = (str(base), *names)
    # Size as well as mtime: two edits inside one filesystem clock tick would otherwise
    # look like no edit at all, and the stale pack would be served until the next one.
    files = [base / f"{n}.yaml" for n in names]
    stamps: tuple[float, ...] = ()
    for f in files:
        st = f.stat() if f.is_file() else None
        stamps += (st.st_mtime, float(st.st_size)) if st else (-1.0, -1.0)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == stamps:
            return hit[1]
    tax = load_taxonomy(names, base)
    with _CACHE_LOCK:
        _CACHE[key] = (stamps, tax)
    return tax
