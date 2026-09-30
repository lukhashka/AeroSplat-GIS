# ═══════════════════════════════════════════════════════════════════
#  AeroSplat-GIS — Serving layer (FastAPI)
#
#  Design rationale lives in docs/decisions/ (ADR-0001 scene frame and
#  transform split, ADR-0002 activation contract, ADR-0003 wire format).
#  In short: the server only *centres*; the up-axis flip and scene scale
#  are applied by the viewer (scene-meta reports them), and dense/splat
#  share one scene frame.
#
#  Wire format — splat (stride 16 floats, "SPLAT16"):
#      [ 0: 3] xyz        (centred, metric units)
#      [ 3]    opacity    (activated, 0..1)
#      [ 4: 7] scale      (activated, metric, > 0)
#      [ 7]    pad
#      [ 8:12] rotation   (w, x, y, z)
#      [12:15] rgb        (0..1)
#      [15]    pad
#
#  The two pad floats are not waste: 16 floats = exactly 4 RGBA32F
#  texels, so the response body *is* a valid GPU texture. The viewer
#  uploads it with zero repacking and indexes splats by instance, which
#  is what makes the depth sort a 1-float-per-splat update instead of a
#  full reshuffle of the vertex buffers. The payload is padded with
#  zero splats up to a whole number of texture rows for the same reason.
#
#  Wire format — dense (stride 6 floats): [0:3] xyz, [3:6] rgb.
# ═══════════════════════════════════════════════════════════════════

from __future__ import annotations

import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


from aerosplat.logging_utils import fix_console_encoding as _fix_console_encoding


_fix_console_encoding()

app = FastAPI(title="AeroSplat-GIS Dual Engine", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # NOTE: wildcard origin + allow_credentials=True is invalid per the CORS
    # spec — browsers refuse to expose the response to JS when both are set.
    # This API doesn't use cookies/auth headers, so credentials aren't needed.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

# Default only — every endpoint takes ?dataset=... so two datasets can be
# compared without restarting uvicorn (review §3.4).
DEFAULT_DATASET = os.environ.get("AEROSPLAT_DATASET", "zaporizhzhia")

# NOTE (review-2 #4): 'mill19' currently points at the same two files as
# 'zaporizhzhia'. The whole point of ?dataset=... is A/B comparison
# (review §3.4), and a silent collision here reads as "the two datasets
# are identical" instead of as the placeholder it actually is. Left as
# real paths (not removed) so the server still starts and the collision
# check below fires instead of a 404 — replace mill19's paths with its
# real files when that dataset exists.
DATASET_PATHS: Dict[str, Dict[str, str]] = {
    "zaporizhzhia": {
        "cloud": os.path.join(project_dir, "data", "dataset_v6_dense.pt"),
        "splat": os.path.join(project_dir, "data", "dataset_v6_splat.pt"),
    },
    # Matches starter.py's Paths naming convention (data/<dataset>_dense.pt,
    # data/<dataset>_splat.pt) directly, so a starter.py run for this
    # dataset is reachable as ?dataset=dataset_v6 without editing this file
    # again or going through the unrelated 'mill19' placeholder below.
    "dataset_v6": {
        "cloud": os.path.join(project_dir, "data", "dataset_v6_dense.pt"),
        "splat": os.path.join(project_dir, "data", "dataset_v6_splat.pt"),
    },
    "mill19": {
        "cloud": os.path.join(project_dir, "data", "dataset_v6_dense.pt"),
        "splat": os.path.join(project_dir, "data", "dataset_v6_splat.pt"),
    },
}


def _warn_dataset_path_collisions() -> None:
    """Flag datasets that resolve to the same file on disk — comparing
    them would silently just compare a dataset against itself."""
    seen: Dict[str, list] = {}
    for name, paths in DATASET_PATHS.items():
        for mode, path in paths.items():
            seen.setdefault(os.path.abspath(path), []).append(f"{name}.{mode}")
    for path, users in seen.items():
        if len(users) > 1:
            print(f"   ⚠️  [Server] {', '.join(users)} all resolve to the same file: {path}")


# Runs at import time too (not just __main__), so it fires when uvicorn
# imports the app directly rather than running this file as a script.
_warn_dataset_path_collisions()

# ── Trainer contract (§A4/§S1, see ADR-0002) ───────────────────────
# The .pt declares an "activations" key (written by gs_trainer_v3._save);
# _activate_splat trusts it, and falls back LOUDLY to a min/max heuristic
# only for older files. The heuristic is a guess that fails silently on
# large-extent scenes (dark screen, "success" in the log), so use the
# overrides below only to defeat a specific known-bad file; leave both
# None to use file-contract-then-heuristic.
SPLAT_ACTIVATION_OVERRIDES = {
    "scale_is_log": None,
    "opacity_is_logit": None,
}

# ── Scene framing (review §C4) ─────────────────────────────────────
TARGET_SCENE_RADIUS = 100.0   # viewer-space radius every dataset is scaled to
EXTENT_PERCENTILE = 99.5      # radius percentile used as "the scene extent"
OUTLIER_CLIP_FACTOR = 3.0     # drop points beyond N× extent; None disables

# review-2 §2.6: the shader's near-plane cull (`z_c >= -0.1`) used to be a
# hardcoded constant that happened to match `camera.near` only because
# frameCamera() computes `max(r * 0.001, 0.01)` with r = TARGET_SCENE_RADIUS
# = 100, giving 0.1 by coincidence. Change TARGET_SCENE_RADIUS and the two
# values silently diverge. Declaring the *ratio* here and deriving both
# camera.near and the shader's uNearCull from the same scene radius on the
# viewer side keeps them locked together by construction instead of luck.
NEAR_CULL_FRACTION = 0.001    # matches frameCamera's r * 0.001

# ── Wire format ────────────────────────────────────────────────────
DENSE_STRIDE = 6
SPLAT_STRIDE = 16                                    # 4 RGBA32F texels
SPLAT_TEXTURE_WIDTH = 2048                           # texels per texture row
SPLATS_PER_TEXTURE_ROW = SPLAT_TEXTURE_WIDTH // 4    # 512

SPLAT_LAYOUT = [
    {"name": "xyz", "offset": 0, "size": 3},
    {"name": "opacity", "offset": 3, "size": 1},
    {"name": "scale", "offset": 4, "size": 3},
    {"name": "rotation", "offset": 8, "size": 4},
    {"name": "rgb", "offset": 12, "size": 3},
]

# The up-axis conversion the viewer must apply in its object matrix.
# (x, y, z)_colmap → (x, −z, y)_webgl is Rx(+90°); the viewer sets
# sceneRoot.rotation.x = +π/2, which is the same map applied to positions
# *and* covariances at once.
UP_AXIS_ROTATION = {"axis": "x", "degrees": 90.0}

# review-2 §4 (ADR-0001): COLMAP units are arbitrary up to a similarity
# transform — an "extent" of 3.7 doesn't mean 3.7 of anything until it's
# tied to a real-world scale. A per-dataset geo file
# (`data/<dataset_id>_geo.txt`) should carry exactly this. This resolver
# looks for it and reports what it found; it does NOT invent a scale by
# guessing at the file's format, because a wrong "confident" number is
# worse for a measurement tool than an honest None.
GEO_FILE_TEMPLATE = os.path.join(project_dir, "data", "{dataset_id}_geo.txt")


def load_metric_scale(dataset: str) -> Optional[Dict[str, object]]:
    """
    Best-effort metric-scale lookup for `dataset`.

    Returns None if no geo file is found. If a file IS found, returns
    what's known about it without parsing a format that hasn't been
    confirmed yet — replace the body of the `if path exists` branch once
    dataset_v6.json / *_geo.txt's real schema is settled (the highest-value unexplored input in the repo, see ADR-0001).
    """
    # dataset_v6_geo.txt is named after the RUN (dataset_v6), not the
    # logical dataset key ("zaporizhzhia") used elsewhere in this file —
    # those are different namespaces today. Try the logical name first,
    # then the one run this project currently has.
    candidates = [
        GEO_FILE_TEMPLATE.format(dataset_id=dataset),
        GEO_FILE_TEMPLATE.format(dataset_id="dataset_v6"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return {
                "units_to_metres": None,   # TODO: parse once the geo file's schema is confirmed
                "source": f"geo_file_present_unparsed:{os.path.basename(path)}",
                "path": path,
            }
    return None


# ═══════════════════════════════════════════════════════════════════
#  Tensor helpers
# ═══════════════════════════════════════════════════════════════════

def finite_mask(*tensors: torch.Tensor) -> torch.Tensor:
    """
    Row mask that is True only where every supplied attribute is finite.

    One NaN — a degenerate COLMAP triangulation, a diverged Gaussian —
    propagates through mean/median into every coordinate, and the browser
    then gets a buffer of NaNs, a NaN bounding sphere, a NaN camera, and a
    black screen with no error anywhere (review §B1). Filtering has to
    happen before any statistic is computed, and has to be applied to all
    attributes together or the rows desynchronise.
    """
    n = tensors[0].shape[0]
    mask = torch.ones(n, dtype=torch.bool)
    for t in tensors:
        mask &= torch.isfinite(t.reshape(n, -1)).all(dim=1)
    return mask


def robust_center_and_extent(
    xyz: torch.Tensor, trim_pct: float = EXTENT_PERCENTILE
) -> Tuple[torch.Tensor, float]:
    """
    Median centre + percentile radius, instead of mean + std.

    COLMAP sparse and dense clouds routinely contain a handful of points at
    10³–10⁶× the scene extent (bad triangulations, sky points). Under
    mean/std those strays both offset the centre and inflate the extent,
    which shrinks the real geometry toward a speck and parks the viewer's
    auto-framing camera kilometres outside it. Median and a high percentile
    are both insensitive to a small fraction of arbitrarily-distant
    outliers (review §C4).
    """
    center = xyz.median(dim=0).values
    radii = (xyz - center).norm(dim=1)

    # torch.quantile is capped at ~16M elements; subsample for the estimate
    # on very large clouds. A percentile of a uniform random subsample is an
    # unbiased estimate of the percentile of the whole.
    if radii.numel() > 8_000_000:
        idx = torch.randperm(radii.numel())[:8_000_000]
        sample = radii[idx]
    else:
        sample = radii

    extent = float(torch.quantile(sample, trim_pct / 100.0).item())
    return center, max(extent, 1e-6)


def _log_raw_stats(label: str, tensors: Dict[str, torch.Tensor]) -> None:
    """
    The §A4 diagnostic dump, run automatically at load instead of by hand.

    Read it as: scale.min < 0 → log-space; opacity outside [0,1] → logits;
    quaternion column 0 ≈ 1 with the rest ≈ 0 → (w,x,y,z) ordering, which
    is what the shader assumes.
    """
    print(f"   ── {label}: raw tensor stats (pre-activation) ──")
    for key, t in tensors.items():
        t = t.float()
        print(
            f"      {key:9s} shape={tuple(t.shape)} "
            f"min={t.min():+.4f} max={t.max():+.4f} mean={t.mean():+.4f}"
        )


def _check_quaternion_convention(rot: torch.Tensor) -> None:
    """
    The shader reads rotations as (w, x, y, z). THREE.Quaternion is
    (x, y, z, w). Nothing crashes if they disagree — you just get a
    silently misoriented scene — so assert the convention once, here,
    where it is cheap to notice (review §A1, final note).

    review-2 §8: the original check tested `rot.mean(0)`, which only
    catches the mix-up on an untrained/identity-init file. A trained
    scene has diverse rotations pointing every which way, so every raw
    column mean drifts toward 0 regardless of convention — the check
    then passes without having tested anything, silently. Test the
    magnitude ordering instead (`rot.abs().mean(0)`), which stays
    meaningful after training, and say so explicitly when even that is
    inconclusive rather than staying quiet either way.
    """
    norms = rot.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        print(
            f"   ⚠️  quaternions are not unit-norm "
            f"(min={norms.min():.4f}, max={norms.max():.4f}); "
            f"the shader normalises, so this is harmless but unexpected"
        )

    abs_col_means = rot.abs().mean(dim=0)
    w_col, last_col = abs_col_means[0].item(), abs_col_means[3].item()

    # A (w,x,y,z) file whose rotations stay reasonably close to identity
    # (true for most of an aerial scene, which is mostly upright) has its
    # largest |mean| at column 0. A (x,y,z,w) file has it at column 3.
    # Neither signal is reliable if rotations are wildly diverse (both
    # means end up small and close together) — say so instead of guessing.
    spread = abs(w_col - last_col)
    if spread < 0.05:
        print(
            f"   ℹ️  quaternion convention check inconclusive (|mean col0|="
            f"{w_col:.3f}, |mean col3|={last_col:.3f} — too close to call from "
            f"magnitude alone). Not a failure; just means this check can't help here."
        )
    elif last_col > w_col:
        print(
            "   ❌ rotation looks like (x,y,z,w) — |mean col3| > |mean col0|. "
            "The shader expects (w,x,y,z). Fix the trainer or reorder here."
        )


def _activate_splat(
    scale: torch.Tensor, opacity: torch.Tensor, contract: Optional[Dict] = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """
    Undo the trainer's storage parameterisation — if and only if it's
    actually stored that way (review §A4).

    Applying exp() to already-metric scales inflates every ~0.003-unit
    Gaussian to ~1 unit — screen-filling splats, billions of fragments,
    and a GPU timeout that presents as a black screen. Skipping exp() on
    genuine log-scales does the opposite: every Gaussian renders ~100x
    too small, opacity compensation drives towards zero, and the alpha
    cutoff discards nearly everything — also a black screen, and the one
    that actually bit this project (review-2 §S1).

    Resolution order, each field independently:
      1. `contract[field]`, if the .pt declares it (producer-authoritative).
      2. `SPLAT_ACTIVATION_OVERRIDES[field]`, if set (operator override).
      3. The min/max heuristic — LOUDLY flagged as a guess, because it is
         provably wrong for a scene whose COLMAP unit scale is large
         enough that every log-scale comes out positive.
    """
    ov = SPLAT_ACTIVATION_OVERRIDES
    contract = contract or {}
    sources: Dict[str, str] = {}

    def resolve(field: str, heuristic_fn) -> bool:
        if field in contract:
            sources[field] = "file contract"
            return bool(contract[field])
        if ov.get(field) is not None:
            sources[field] = "override"
            return bool(ov[field])
        sources[field] = "HEURISTIC (unverified — see review-2 §S1)"
        return heuristic_fn()

    # Log-space scales for sub-unit Gaussians are negative; a tensor with
    # no negatives LOOKS already-metric, but this is only true if the
    # scene's COLMAP unit scale keeps every Gaussian under 1 unit. It is
    # not a safe default — it is a guess, and it is reported as one.
    scale_is_log = resolve("scale_is_log", lambda: bool(scale.min() < 0.0))
    opacity_is_logit = resolve(
        "opacity_is_logit", lambda: bool(opacity.min() < 0.0 or opacity.max() > 1.0)
    )

    print(
        f"   🔍 Активації: scale_is_log={scale_is_log} ({sources['scale_is_log']}), "
        f"opacity_is_logit={opacity_is_logit} ({sources['opacity_is_logit']})"
    )
    if "HEURISTIC" in sources["scale_is_log"]:
        print(
            "   ⚠️  scale_is_log is GUESSED from scale.min() < 0, not declared by the "
            "trainer. This heuristic is wrong whenever the scene's COLMAP unit scale "
            "is large enough that every log-scale is positive — check "
            "'scale p50 ÷ extent' in the diagnostics panel (healthy ≈ 1e-4…1e-2) "
            "before trusting a render built on this guess."
        )

    if scale_is_log:
        scale = torch.exp(scale)
    if opacity_is_logit:
        opacity = torch.sigmoid(opacity)

    # Exactly-zero scales are legal in activated storage (a fully collapsed
    # Gaussian) but degenerate downstream: Σ loses rank and the shader's
    # determinant guard is doing the hiding. Clamp to a floor instead of
    # failing the whole load over invisible splats.
    zero_scales = int((scale <= 0).any(dim=1).sum())
    if zero_scales:
        print(f"   ⚠️  {zero_scales:,} сплатів з нульовим масштабом — clamped to 1e-7")
        scale = scale.clamp_min(1e-7)

    # Post-conditions. Raised, not asserted, so `python -O` can't switch
    # them off — a silent failure here is exactly what §A4 is about.
    if not scale.min() > 0:
        raise ValueError(f"scale must be positive after activation (min={scale.min():.6f})")
    if not (opacity.min() >= 0.0 and opacity.max() <= 1.0):
        raise ValueError(
            f"opacity out of [0,1] after activation "
            f"(min={opacity.min():.6f}, max={opacity.max():.6f})"
        )
    applied = {
        "scale_is_log": scale_is_log,
        "opacity_is_logit": opacity_is_logit,
        "scale_is_log_source": sources["scale_is_log"],
        "opacity_is_logit_source": sources["opacity_is_logit"],
    }
    return scale, opacity, applied


# ═══════════════════════════════════════════════════════════════════
#  Scene frame — one centre and one scale per dataset (review §A3/§B4)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SceneFrame:
    center: Tuple[float, float, float]
    extent: float
    scene_scale: float
    source: str          # which file the frame was measured from


@dataclass
class Payload:
    data: bytes
    count: int           # real splats/points
    padded_count: int    # including texture-row padding (splat mode only)
    stride: int
    etag: str
    dropped_nonfinite: int
    dropped_outliers: int
    activations: Optional[Dict[str, object]] = None   # what was actually applied (splat only)


_lock = threading.Lock()
_FRAMES: Dict[str, SceneFrame] = {}

# review-2 #2: `step` is accepted straight from the query string
# (`ge=1, le=64`) and used as part of the cache key. An unbounded dict
# keyed that way is a slow memory leak driven entirely by whatever the
# browser happens to ask for — an afternoon of `?step=` experimentation
# across two datasets and two modes can hold several GB of decoded
# payloads that never come back. OrderedDict + an explicit byte budget
# gives LRU eviction for ~8 lines.
_PAYLOADS: "OrderedDict[Tuple[str, str, int], Payload]" = OrderedDict()
_CACHE_BUDGET_BYTES = 1_500_000_000   # ~1.5 GB of decoded payloads, tune to taste


def _evict_if_needed() -> None:
    """Drop the least-recently-used payload(s) until under budget. Always
    keeps at least one entry, so a single huge payload can't thrash."""
    total = sum(len(p.data) for p in _PAYLOADS.values())
    while total > _CACHE_BUDGET_BYTES and len(_PAYLOADS) > 1:
        key, dropped = _PAYLOADS.popitem(last=False)
        total -= len(dropped.data)
        print(f"   ♻️  [Server] Evicted cached payload {key} ({len(dropped.data)/1e6:.0f} MB)")


def dataset_paths(dataset: str) -> Dict[str, str]:
    if dataset not in DATASET_PATHS:
        raise KeyError(
            f"unknown dataset '{dataset}'; known: {sorted(DATASET_PATHS)}"
        )
    return DATASET_PATHS[dataset]


def _path_for(dataset: str, mode: str) -> str:
    paths = dataset_paths(dataset)
    return paths["splat"] if mode == "splat" else paths["cloud"]


def scene_frame(dataset: str) -> SceneFrame:
    """
    Measure the dataset's centre and extent once, from a single source, and
    reuse it for every mode.

    Previously each mode normalised itself independently (dense with
    scale_factor=100, splat with 10), so the same geometry came out at two
    different world sizes: modes couldn't be A/B'd, rendered together, or
    share a camera bookmark (review §B4). Measuring once fixes that, and
    means a future TRIPS output lands in the same frame without a third
    magic constant.
    """
    cached = _FRAMES.get(dataset)
    if cached is not None:
        return cached

    # Prefer the splat file: it's the representation being tuned, and the
    # dense cloud is a superset of roughly the same geometry either way.
    for mode in ("splat", "dense"):
        path = _path_for(dataset, mode)
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError(f"no .pt files found for dataset '{dataset}'")

    data = torch.load(path, map_location="cpu", weights_only=False)
    if "xyz" not in data:
        raise KeyError(f"expected key 'xyz' in {path}, found {list(data.keys())}")

    xyz = data["xyz"].float()
    xyz = xyz[finite_mask(xyz)]
    if xyz.shape[0] == 0:
        raise ValueError(f"{path} contains no finite points")

    center, extent = robust_center_and_extent(xyz)
    frame = SceneFrame(
        center=(float(center[0]), float(center[1]), float(center[2])),
        extent=extent,
        scene_scale=TARGET_SCENE_RADIUS / extent,
        source=os.path.basename(path),
    )
    _FRAMES[dataset] = frame

    print(
        f"📐 [Server] Кадр сцени '{dataset}' з {frame.source}: "
        f"center=({frame.center[0]:.3f}, {frame.center[1]:.3f}, {frame.center[2]:.3f}) "
        f"extent={frame.extent:.3f} scene_scale={frame.scene_scale:.5f}"
    )

    # review-2 #9: the frame is measured from whichever file was found
    # first (splat preferred). _clip_outliers then applies THIS extent to
    # both modes, including the dense cloud, which is usually a superset
    # of the same geometry with a genuinely larger footprint (MVS reaches
    # further than the capped/subsampled splat init). If the two extents
    # disagree substantially, the dense cloud can get real geometry
    # clipped as "outliers" — worth knowing about, not necessarily wrong.
    other_mode = "dense" if mode == "splat" else "splat"
    other_path = _path_for(dataset, other_mode)
    if os.path.exists(other_path) and other_path != path:
        try:
            other_data = torch.load(other_path, map_location="cpu", weights_only=False)
            other_xyz = other_data["xyz"].float()
            other_xyz = other_xyz[finite_mask(other_xyz)]
            if other_xyz.shape[0] > 0:
                _, other_extent = robust_center_and_extent(other_xyz)
                if other_extent > 0 and abs(other_extent - extent) / extent > 0.2:
                    print(
                        f"   ⚠️  [Server] Frame measured from {frame.source} (extent="
                        f"{extent:.3f}) diverges >20% from {os.path.basename(other_path)} "
                        f"(extent={other_extent:.3f}). Outlier clipping in {other_mode} "
                        f"mode uses the {mode} extent and may drop real geometry."
                    )
        except Exception as e:  # best-effort diagnostic only, never fatal
            print(f"   ℹ️  [Server] Could not cross-check extent against {other_path}: {e}")
    return frame


def _clip_outliers(
    xyz: torch.Tensor, frame: SceneFrame
) -> Tuple[torch.Tensor, int]:
    """Drop points beyond OUTLIER_CLIP_FACTOR × extent, reporting the count."""
    if OUTLIER_CLIP_FACTOR is None:
        return torch.ones(xyz.shape[0], dtype=torch.bool), 0
    center = torch.tensor(frame.center)
    keep = (xyz - center).norm(dim=1) <= OUTLIER_CLIP_FACTOR * frame.extent
    return keep, int((~keep).sum())


# ═══════════════════════════════════════════════════════════════════
#  Payload builders
# ═══════════════════════════════════════════════════════════════════

def _etag_for(path: str, mode: str, step: int, count: int) -> str:
    mtime = os.path.getmtime(path)
    return f'"{mode}-{step}-{count}-{int(mtime)}"'


def _build_dense_payload(dataset: str, step: int) -> Payload:
    path = _path_for(dataset, "dense")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    print(f"📦 [Server] Завантаження щільної хмари: {os.path.basename(path)}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    missing = {"xyz", "rgb"} - set(data.keys())
    if missing:
        raise KeyError(f"missing keys {missing} in {path}")

    xyz = data["xyz"][::step].float()
    rgb = data["rgb"][::step].float()

    # §B1/§S2: filter BEFORE deciding anything from a global statistic.
    # rgb.max() used to run before this filter — one +inf in the colour
    # tensor makes max() == inf, the ">1.0" test trivially passes, and
    # every colour in the scene (including the finite ones) gets divided
    # by 255. Geometry renders perfectly and the whole scene is black,
    # with no error anywhere (review-2 §S2).
    mask = finite_mask(xyz, rgb)
    dropped_nonfinite = int((~mask).sum())
    xyz, rgb = xyz[mask], rgb[mask]

    if xyz.shape[0] == 0:
        raise ValueError(f"{path} contains no finite points after filtering")

    # 99th percentile, not max: one remaining bright outlier should not
    # decide the encoding for the whole tensor.
    if rgb.numel() and float(torch.quantile(rgb.flatten(), 0.99)) > 1.0:
        rgb = rgb / 255.0

    frame = scene_frame(dataset)
    keep, dropped_outliers = _clip_outliers(xyz, frame)
    xyz, rgb = xyz[keep], rgb[keep]

    # Centring only — no rotation, no rescale (review §A3). Centring is
    # still done here and is *necessary*: it keeps coordinates inside
    # float32's accurate range before they reach the browser, which will
    # matter as soon as real UTM/geographic coordinates replace the
    # Earth Studio stand-in.
    xyz = xyz - torch.tensor(frame.center)

    count = xyz.shape[0]
    blob = torch.cat([xyz, rgb], dim=1).contiguous().numpy().tobytes()

    if dropped_nonfinite or dropped_outliers:
        print(
            f"   ⚠️  Відкинуто {dropped_nonfinite:,} NaN/inf, "
            f"{dropped_outliers:,} викидів"
        )
    print(f"✅ [Server] Dense: {count:,} точок.")

    return Payload(
        data=blob,
        count=count,
        padded_count=count,
        stride=DENSE_STRIDE,
        etag=_etag_for(path, "dense", step, count),
        dropped_nonfinite=dropped_nonfinite,
        dropped_outliers=dropped_outliers,
    )


def _build_splat_payload(dataset: str, step: int) -> Payload:
    path = _path_for(dataset, "splat")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    print(f"🔮 [Server] Завантаження Гаусіанів: {os.path.basename(path)}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    missing = {"xyz", "scale", "rotation", "opacity", "rgb"} - set(data.keys())
    if missing:
        raise KeyError(f"missing keys {missing} in {path}")

    xyz = data["xyz"][::step].float()
    scale = data["scale"][::step].float()
    rot = data["rotation"][::step].float()
    opac = data["opacity"][::step].float()
    rgb = data["rgb"][::step].float()

    if opac.dim() == 1:
        opac = opac.unsqueeze(1)

    _log_raw_stats(os.path.basename(path), {
        "xyz": xyz, "scale": scale, "rotation": rot, "opacity": opac, "rgb": rgb,
    })
    _check_quaternion_convention(rot)

    # §B1: filter BEFORE activating, and drop the rows before checking the
    # post-conditions. exp(NaN) is NaN, and NaN fails `scale.min() > 0` —
    # so a single stray Gaussian used to raise here and take the whole
    # endpoint down with a 500 instead of being quietly dropped, which is
    # the exact failure this filter exists to prevent.
    mask = finite_mask(xyz, scale, rot, opac, rgb)
    dropped_nonfinite = int((~mask).sum())
    xyz, scale, rot, opac, rgb = (t[mask] for t in (xyz, scale, rot, opac, rgb))

    if xyz.shape[0] == 0:
        raise ValueError(f"{path} contains no finite splats after filtering")

    # §S2: decide the rgb encoding only AFTER filtering, and from a
    # percentile rather than the max — the same +inf-in-rgb mechanism
    # that blacked out the dense branch applies here too, and one
    # remaining bright outlier shouldn't flip the whole tensor's encoding.
    if rgb.numel() and float(torch.quantile(rgb.flatten(), 0.99)) > 1.0:
        rgb = rgb / 255.0

    # File-declared activation contract, if the trainer wrote one
    # (review-2 §S1). Absent on older files — _activate_splat falls back
    # to the heuristic and says so loudly.
    contract = data.get("activations")
    scale, opac, activations_applied = _activate_splat(scale, opac, contract)

    # exp() of a large finite log-scale can still overflow to inf.
    mask2 = finite_mask(scale, opac)
    overflowed = int((~mask2).sum())
    if overflowed:
        dropped_nonfinite += overflowed
        xyz, scale, rot, opac, rgb = (t[mask2] for t in (xyz, scale, rot, opac, rgb))

    frame = scene_frame(dataset)
    keep, dropped_outliers = _clip_outliers(xyz, frame)
    xyz, scale, rot, opac, rgb = (t[keep] for t in (xyz, scale, rot, opac, rgb))

    # Centre only. Scale stays in metric units and is *not* multiplied by
    # any k here: the viewer's sceneRoot carries the uniform scale, and the
    # shader's W·Σ·Wᵀ applies it to positions and covariances together.
    # That is what the old `s_scale * xyz_k` line was reaching for, one
    # attribute short of rotation (review §A3).
    xyz = xyz - torch.tensor(frame.center)

    count = xyz.shape[0]

    # review-2 §2.1: the vertex shader addresses TEXELS via
    # `aSplatIndex * 4.0`, so the real ceiling on splat count is
    # 2^24 / 4 = 4,194,304 — not the 2^24 the depth-sort worker used to
    # check for. Past this, the multiply stops being exact and splats
    # start reading each other's parameters: silent corruption, not a
    # crash. Warn here too, not just in the worker, since this is the
    # function that actually decides the count.
    max_addressable = (2 ** 24) // 4
    if count > max_addressable:
        print(
            f"   ⚠️  [Server] {count:,} splats exceeds the fp32 texel-addressing "
            f"ceiling ({max_addressable:,}). aSplatIndex * 4.0 will round and "
            f"splats will read each other's data. Reload with a larger ?step="
        )

    # Pad up to a whole number of texture rows so the response body can be
    # uploaded as an RGBA32F texture with no repacking on the main thread.
    # Padding splats are never drawn (instanceCount = real count) and are
    # zero-filled anyway.
    rows = max(1, math.ceil(count / SPLATS_PER_TEXTURE_ROW))
    padded = rows * SPLATS_PER_TEXTURE_ROW

    buf = torch.zeros(padded, SPLAT_STRIDE, dtype=torch.float32)
    buf[:count, 0:3] = xyz
    buf[:count, 3:4] = opac
    buf[:count, 4:7] = scale
    buf[:count, 8:12] = rot
    buf[:count, 12:15] = rgb

    if dropped_nonfinite or dropped_outliers:
        print(
            f"   ⚠️  Відкинуто {dropped_nonfinite:,} нескінченних/NaN сплатів, "
            f"{dropped_outliers:,} викидів"
        )
    print(
        f"✅ [Server] Splat: {count:,} Гаусіанів "
        f"({padded:,} з padding, {rows} рядків текстури)."
    )

    return Payload(
        data=buf.contiguous().numpy().tobytes(),
        count=count,
        padded_count=padded,
        stride=SPLAT_STRIDE,
        etag=_etag_for(path, "splat", step, count),
        dropped_nonfinite=dropped_nonfinite,
        dropped_outliers=dropped_outliers,
        activations=activations_applied,
    )


def get_payload(dataset: str, mode: str, step: int) -> Payload:
    """
    Lazy, cached, and re-loadable.

    Loading used to happen at import time into module globals, so a malformed
    .pt meant restarting uvicorn to retry — painful when you're iterating on
    trainer output all day (review §3.4).
    """
    key = (dataset, mode, step)
    with _lock:
        cached = _PAYLOADS.get(key)
        if cached is not None:
            _PAYLOADS.move_to_end(key)   # mark as most-recently-used
            return cached
        payload = (
            _build_splat_payload(dataset, step)
            if mode == "splat"
            else _build_dense_payload(dataset, step)
        )
        _PAYLOADS[key] = payload
        _evict_if_needed()
        return payload


# ═══════════════════════════════════════════════════════════════════
#  API
# ═══════════════════════════════════════════════════════════════════

def _error(status: int, message: str) -> Response:
    return Response(status_code=status, content=message, media_type="text/plain")


@app.get("/api/v1/spatial-chunk")
def stream_spatial_chunk(
    request: Request,
    mode: str = Query("splat"),
    dataset: Optional[str] = Query(None),
    step: int = Query(1, ge=1, le=64),
):
    start = time.time()
    dataset = dataset or DEFAULT_DATASET

    if mode not in ("dense", "splat"):
        return _error(400, f"Invalid mode '{mode}'; expected 'dense' or 'splat'")
    if dataset not in DATASET_PATHS:
        return _error(404, f"Unknown dataset '{dataset}'; known: {sorted(DATASET_PATHS)}")

    try:
        payload = get_payload(dataset, mode, step)
    except FileNotFoundError as e:
        return _error(404, f"{mode} data not available for '{dataset}': {e}")
    except Exception as e:  # malformed .pt, failed post-condition, …
        return _error(500, f"Failed to load {mode} data for '{dataset}': {e}")

    headers = {
        "ETag": payload.etag,
        # Mode toggles used to re-download the whole scene every time.
        "Cache-Control": "private, max-age=0, must-revalidate",
        "X-Processing-Time-Ms": f"{(time.time() - start) * 1000:.2f}",
        "X-Gaussians-Count": str(payload.count),
        "X-Padded-Count": str(payload.padded_count),
        "X-Stride-Floats": str(payload.stride),
        "X-Dataset": dataset,
        "Access-Control-Expose-Headers": (
            "ETag, X-Processing-Time-Ms, X-Gaussians-Count, "
            "X-Padded-Count, X-Stride-Floats, X-Dataset"
        ),
    }

    if request.headers.get("if-none-match") == payload.etag:
        return Response(status_code=304, headers=headers)

    return Response(
        content=payload.data,
        media_type="application/octet-stream",
        headers=headers,
    )


@app.get("/api/v1/scene-meta")
def scene_meta(
    mode: str = Query("splat"),
    dataset: Optional[str] = Query(None),
    step: int = Query(1, ge=1, le=64),
):
    """
    Everything the viewer needs to place the scene, as JSON rather than
    smuggled through response headers (review §3.3).

    `scene_scale` and `up_axis_rotation` are the transform the *viewer*
    applies in its object matrix — deliberately not applied here, because
    doing it here desynchronises positions from rotation quaternions
    (review §A3).
    """
    dataset = dataset or DEFAULT_DATASET

    if mode not in ("dense", "splat"):
        return _error(400, f"Invalid mode '{mode}'; expected 'dense' or 'splat'")
    if dataset not in DATASET_PATHS:
        return _error(404, f"Unknown dataset '{dataset}'; known: {sorted(DATASET_PATHS)}")

    try:
        frame = scene_frame(dataset)
        payload = get_payload(dataset, mode, step)
    except FileNotFoundError as e:
        return _error(404, f"{mode} data not available for '{dataset}': {e}")
    except Exception as e:
        return _error(500, f"Failed to load {mode} data for '{dataset}': {e}")

    # review-2 §2.1 / §3: the fp32 addressing ceiling, computed once here
    # so the viewer doesn't have to duplicate the constant. The shader
    # addresses TEXELS (aSplatIndex * 4.0), not splats, so the ceiling on
    # splat count is 2^24 / 4, not 2^24.
    max_addressable_splats = (2 ** 24) // 4

    return {
        "dataset": dataset,
        "mode": mode,
        "step": step,
        "count": payload.count,
        "padded_count": payload.padded_count,
        "stride_floats": payload.stride,
        "dropped_nonfinite": payload.dropped_nonfinite,
        "dropped_outliers": payload.dropped_outliers,

        # Scene frame — identical for every mode of this dataset.
        "center": list(frame.center),
        "extent": frame.extent,
        "scene_scale": frame.scene_scale,
        "target_scene_radius": TARGET_SCENE_RADIUS,
        "frame_source": frame.source,
        "up_axis_rotation": UP_AXIS_ROTATION,

        # review-2 §2.6: derive the shader's near-plane cull from the same
        # fraction frameCamera() uses for camera.near, so the two stay
        # locked together instead of matching by coincidence.
        "near_cull_fraction": NEAR_CULL_FRACTION,

        # Splat-only GPU layout contract.
        "splat_layout": SPLAT_LAYOUT if mode == "splat" else None,
        "texture_width": SPLAT_TEXTURE_WIDTH if mode == "splat" else None,
        "splats_per_row": SPLATS_PER_TEXTURE_ROW if mode == "splat" else None,
        "max_addressable_splats": max_addressable_splats if mode == "splat" else None,

        "activations": payload.activations,

        # review-2 §4a: None until a geo file's schema is confirmed and
        # parsed — see load_metric_scale(). The viewer should label
        # measurements as relative units whenever this is None, and as
        # metres only once units_to_metres is a real number.
        "metric": load_metric_scale(dataset),
    }


@app.get("/api/v1/status")
def status():
    return {
        "default_dataset": DEFAULT_DATASET,
        "datasets": {
            name: {
                mode: os.path.exists(paths[key])
                for mode, key in (("splat", "splat"), ("dense", "cloud"))
            }
            for name, paths in DATASET_PATHS.items()
        },
        "cached_payloads": [
            {"dataset": d, "mode": m, "step": s, "count": p.count}
            for (d, m, s), p in _PAYLOADS.items()
        ],
        "device": str(DEVICE),
    }


@app.post("/api/v1/reload")
def reload_caches(dataset: Optional[str] = Query(None)):
    """Drop cached payloads and scene frames so re-trained .pt files are picked up."""
    with _lock:
        if dataset is None:
            n = len(_PAYLOADS)
            _PAYLOADS.clear()
            _FRAMES.clear()
        else:
            keys = [k for k in _PAYLOADS if k[0] == dataset]
            for k in keys:
                del _PAYLOADS[k]
            _FRAMES.pop(dataset, None)
            n = len(keys)
    return {"reloaded": dataset or "all", "dropped_payloads": n}


app.mount(
    "/web",
    StaticFiles(directory=os.path.join(project_dir, "web"), html=True),
    name="web",
)

if __name__ == "__main__":
    import uvicorn

    print(f"🌍 [Server] Датасет за замовчуванням: {DEFAULT_DATASET}")
    for name, paths in DATASET_PATHS.items():
        print(f"   {name}: cloud={paths['cloud']}")
        print(f"   {' ' * len(name)}  splat={paths['splat']}")
    _warn_dataset_path_collisions()
    uvicorn.run(app, host="127.0.0.1", port=8000)