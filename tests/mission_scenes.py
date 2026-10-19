"""Synthetic scenes with known answers for the mission tests (step 8.1b).

A street in front of one box building, voxelised directly (no cloud), so every distance in
a test can be worked out by hand:

    ground       z < 0, occupied
    building     x in [-10, 10), y in [15, 30), z in [0, 20): a one-voxel shell of occupied
                 voxels with UNKNOWN inside (nobody saw the interior), flat roof at z = 20
    south wall   the plane y = 15, outward normal (0, -1, 0)
    window       centre (0, 15, 9) on that wall
    roof pad     x in [-5, 5], y in [18, 27] on the roof, z = 20

The grid spans x, y in [-40, 40) and z in [-1, 39) at 0.5 m.
"""

from __future__ import annotations

import numpy as np

from aerosplat.authored.geometry import make_pad_point
from aerosplat.authored.models import (AccessPoint, Geometry, HostFacade, Provenance, new_ap_id)
from aerosplat.authored.store import AuthoredStore
from aerosplat.mission.targets import TargetStore
from aerosplat.taxonomy import load_taxonomy
from aerosplat.geo.crs import GeoPoint
from aerosplat.mission.models import OccupancyRef
from aerosplat.surface.esdf import distance_field_cm
from aerosplat.surface.models import FREE, OCCUPIED, UNKNOWN, ColumnSource, GridSpec
from aerosplat.surface.occupancy import OccupancyGrid

ORIGIN = GeoPoint(lat=50.4501, lon=30.5234, h=180.0)
RES = 0.5
X0, Y0, Z0 = -40.0, -40.0, -1.0
SHAPE = (160, 160, 80)                       # rows (y), cols (x), layers (z)
WINDOW_C = (0.0, 15.0, 9.0)
TAXONOMY = "core@1.1"
SITE = "demo/street"


def idx(x0, x1, y0, y1, z0, z1):
    """Index slices of the voxels whose centres lie in [x0, x1) x [y0, y1) x [z0, z1)."""
    def s(lo, hi, o):
        a = int(np.ceil((lo - o) / RES - 0.5))
        b = int(np.ceil((hi - o) / RES - 0.5))
        return slice(a, b)
    return s(y0, y1, Y0), s(x0, x1, X0), s(z0, z1, Z0)


def base_state() -> np.ndarray:
    st = np.zeros(SHAPE, np.uint8)
    st[idx(-40, 40, -40, 40, -1, 0)] = OCCUPIED                    # ground
    st[idx(-10, 10, 15, 30, 0, 20)] = OCCUPIED                     # building, solid ...
    st[idx(-9.5, 9.5, 15.5, 29.5, 0, 19.5)] = UNKNOWN              # ... with an unseen inside
    return st


def put(state: np.ndarray, box, value: int) -> np.ndarray:
    """Fill an axis-aligned box (x0, x1, y0, y1, z0, z1) with a state, in place."""
    state[idx(*box)] = value
    return state


def grid_from_state(state: np.ndarray, *, assumed=None,
                    max_distance_m: float = 20.0) -> OccupancyGrid:
    """`assumed`: (x0, x1, y0, y1) whose columns' free air is assumed (SKYLINE), not measured."""
    rows, cols, layers = state.shape
    spec = GridSpec(res_m=RES, x0=X0, y0=Y0, z0=Z0, rows=rows, cols=cols, layers=layers)
    blocked = state != FREE
    any_b = blocked.any(axis=2)
    top = layers - 1 - np.argmax(blocked[:, :, ::-1], axis=2)
    column_top = np.where(any_b, Z0 + (top + 1) * RES, np.nan).astype(np.float32)
    free_from = np.where(any_b, top + 1, 0).astype(np.int16)
    source = np.full((rows, cols), ColumnSource.POINTS, np.uint8)
    if assumed is not None:
        i, j, _ = idx(*assumed, 0, 1)
        source[i, j] = ColumnSource.SKYLINE
    return OccupancyGrid(spec=spec, state=state, esdf_cm=distance_field_cm(state, RES, max_distance_m),
                         column_top=column_top, column_source=source, free_from=free_from,
                         max_distance_m=max_distance_m)


def street_grid(**kw) -> OccupancyGrid:
    return grid_from_state(base_state(), **kw)


def occupancy_ref() -> OccupancyRef:
    return OccupancyRef(run_id="r1", epoch_id="baseline", format="occupancy-v2", res_m=RES,
                        unknown_mode="skyline")


def window_ap(class_id: str = "opening.window", **over) -> AccessPoint:
    """A 1.2 x 1.5 m window centred on WINDOW_C, on the south wall.

    The observer outside looks along -n = +y, so right = (-n) x up = (1, 0, 0) = u."""
    host = HostFacade(facade_id="B01/S1", building_id="B01", origin=(0.0, 15.0, 0.0),
                      normal=(0.0, -1.0, 0.0), u_axis=(1.0, 0.0, 0.0), v_ground=0.0,
                      azimuth_deg=180.0)
    geom = Geometry(polygon_uv=[(-0.6, 8.25), (0.6, 8.25), (0.6, 9.75), (-0.6, 9.75)],
                    center=WINDOW_C, normal=(0.0, -1.0, 0.0), u_axis=(1.0, 0.0, 0.0),
                    width=1.2, height=1.5, u_center=0.0, v_center=9.0)
    data = dict(id=new_ap_id(), site_key=SITE, kind=class_id.split(".", 1)[0], taxonomy=TAXONOMY,
                host=host, geometry=geom, lifecycle="verified",
                provenance=Provenance(created_by="op:test"))
    data.update(over)
    return AccessPoint(**{"class": class_id}, **data)


def roof_pad_ap(**over) -> AccessPoint:
    kw = dict(site_key=SITE, class_id="pad.roof_flat", taxonomy=TAXONOMY,
              polygon_enu=[(-5, 18), (5, 18), (5, 27), (-5, 27)], z=20.0, building_id="B01",
              created_by="op:test")
    kw.update(over)
    return make_pad_point(**kw)


def stores(tmp_path) -> tuple[AuthoredStore, TargetStore]:
    authored = AuthoredStore(tmp_path / "authored", SITE)
    return authored, TargetStore(authored)


def confirmed(tmp_path, ap: AccessPoint, *, sigma_ap: float = 0.1, sigma_georef: float = 0.3,
              **kw):
    """Store the access point and confirm a target on it; (target, authored, targets)."""
    authored, targets = stores(tmp_path)
    authored.create(ap)
    target = targets.create(load_taxonomy(), ap.id, by="tester", sigma_ap_m=sigma_ap,
                            sigma_georef_m=sigma_georef, confirm=True, **kw)
    return target, authored, targets


# ── the ground the takeoff search reads (semantics terrain of the same "run") ──
def street_terrain():
    """The semantics raster of the street: 1 m cells over the grid, flat DTM at z = 0, the
    building's footprint labelled 1 (B01). (TerrainModel, dsm, dtm, labels)."""
    from aerosplat.semantics.models import TerrainModel

    rows, cols = int(SHAPE[0] * RES), int(SHAPE[1] * RES)
    model = TerrainModel(res_m=1.0, x0=X0, y0=Y0, rows=rows, cols=cols, n_points=rows * cols,
                         filled_cells=rows * cols)
    xc = X0 + np.arange(cols) + 0.5
    yc = Y0 + np.arange(rows) + 0.5
    inside = ((yc >= 15) & (yc < 30))[:, None] & ((xc >= -10) & (xc < 10))[None, :]
    labels = inside.astype(np.int32)
    dtm = np.zeros((rows, cols))
    dsm = np.where(inside, 20.0, 0.0)
    return model, dsm, dtm, labels


def street_ground(labels: bool = True):
    from aerosplat.mission.takeoff import Ground

    model, _, dtm, lab = street_terrain()
    return Ground(terrain=model, dtm=dtm, labels=lab if labels else None)


def write_street_semantics(folder) -> None:
    """semantics.json + terrain.npz for the street, as the semantics stage would write them."""
    import json as _json
    from pathlib import Path as _Path

    from aerosplat.semantics.models import Building, SemanticsResult

    model, dsm, dtm, labels = street_terrain()
    b = Building(id="B01", footprint=[(-10, 15), (10, 15), (10, 30), (-10, 30)], area_m2=300,
                 ground_z=0.0, roof_z=20.0, height_m=20.0)
    folder = _Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "semantics.json").write_text(
        SemanticsResult(terrain=model, buildings=[b], facades=[]).model_dump_json(),
        encoding="utf-8")
    np.savez_compressed(folder / "terrain.npz", dsm=dsm, dtm=dtm, labels=labels)
    assert _json.loads((folder / "semantics.json").read_text(encoding="utf-8"))["terrain"]
