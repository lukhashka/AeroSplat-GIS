"""CityJSON 2.0 export (ACCESS_POINTS §2.6) and the offline structural validator.

cjio / cjval are not dependencies, so the validator here is the check; it is itself tested
against documents broken on purpose, one rule at a time.
"""

import copy
import json

import numpy as np
import pytest
from pyproj import Transformer

from aerosplat.authored.geometry import make_access_point, make_pad_point
from aerosplat.authored.models import Address, Lifecycle
from aerosplat.geo.crs import EnuFrame, GeoPoint, SiteProjection, utm_epsg
from aerosplat.semantics.cityjson import export_cityjson, validate
from aerosplat.semantics.models import (Building, Facade, SemanticsResult, SurfaceKind,
                                        TerrainModel, azimuth_of, u_axis_for)

ORIGIN = GeoPoint(lat=40.8942, lon=-74.2173, h=233.0)
CITYGML = {"opening.window": "Window", "opening.door": "Door", "pad.roof_flat": "RoofSurface",
           "pad.roof_pitched": "RoofSurface", "pad.ground": None, "constraint.obstacle": None}


def facade(fid, normal, origin, u_min, u_max, height=12.0):
    n = np.asarray(normal, float)
    ux, uy = u_axis_for((float(n[0]), float(n[1])))
    return Facade(id=fid, building_id=fid.split("/")[0], kind=SurfaceKind.wall, origin=origin,
                  normal=(float(n[0]), float(n[1]), 0.0), u_axis=(ux, uy, 0.0),
                  azimuth_deg=azimuth_of((float(n[0]), float(n[1]))), u_min=u_min, u_max=u_max,
                  v_ground=origin[2], v_max=origin[2] + height, plane_rms_m=0.1, n_points=1000)


def run_result():
    """One 20 × 12 × 12 m building, its north wall B01/N1 (u runs west, 0 at x = 0)."""
    north = facade("B01/N1", (0.0, 1.0), (0.0, 0.0, 0.0), -10.0, 10.0)
    b = Building(id="B01", footprint=[(-10, -12), (10, -12), (10, 0), (-10, 0)], area_m2=240.0,
                 ground_z=0.0, roof_z=12.0, height_m=12.0, facade_ids=["B01/N1"])
    return SemanticsResult(terrain=TerrainModel(res_m=1.0, x0=-40, y0=-40, rows=80, cols=80,
                                                n_points=1, filled_cells=1),
                           buildings=[b], facades=[north]), north


def window(fc, u0, v0, u1, v1, cls="opening.window", **kw):
    return make_access_point(site_key="demo/pier", facade=fc, class_id=cls, taxonomy="core@1.1",
                             polygon_uv=[(u0, v0), (u1, v0), (u1, v1), (u0, v1)], **kw)


@pytest.fixture
def features():
    result, north = run_result()
    other_run = facade("B07/N3", (0.0, 1.0), (0.5, 0.3, 0.0), -10.0, 10.0)   # same wall, other run
    elsewhere = facade("B09/E1", (1.0, 0.0), (300.0, 0.0, 0.0), -5.0, 5.0)
    w = window(north, -8.0, 3.0, -6.6, 4.5)
    w = w.model_copy(update={"address": Address(floor_index=1, floor_label="2", column=1,
                                                canonical="pier/B01/N1/01/01")})
    return result, {
        "window": w,
        "door": window(north, -1.0, 0.0, 0.2, 2.2, cls="opening.door"),
        "candidate": window(north, 2.0, 3.0, 3.4, 4.5, lifecycle=Lifecycle.candidate),
        "carried": window(other_run, 4.5, 6.0, 5.9, 7.5),
        "lost": window(elsewhere, 0.0, 3.0, 1.0, 4.0),
        "obstacle": window(north, 6.0, 9.0, 7.0, 10.0, cls="constraint.obstacle"),
        "roof": make_pad_point(site_key="demo/pier", class_id="pad.roof_flat", taxonomy="core@1.1",
                               polygon_enu=[(-6, -9), (6, -9), (6, -3), (-6, -3)], z=12.0,
                               building_id="B77"),           # another run's id: placed by geometry
        "yard": make_pad_point(site_key="demo/pier", class_id="pad.ground", taxonomy="core@1.1",
                               polygon_enu=[(-5, 8), (5, 8), (5, 14), (-5, 14)], z=0.0),
    }


def export(result, feats, **kw):
    return export_cityjson("demo/pier", ORIGIN, "r1", "baseline", result, feats.values(),
                           citygml_of=CITYGML, **kw)


def test_the_export_is_valid_and_holds_what_it_should(features):
    result, feats = features
    doc, report = export(result, feats)
    errors, warnings = validate(doc)
    assert errors == [] and warnings == []
    assert doc["type"] == "CityJSON" and doc["version"] == "2.0"
    assert doc["metadata"]["referenceSystem"] == "https://www.opengis.net/def/crs/EPSG/0/32618"
    assert "ellipsoidal" in doc["metadata"]["title"]
    b = doc["CityObjects"]["B01"]
    assert b["type"] == "Building" and [g["lod"] for g in b["geometry"]] == ["1", "3"]
    assert b["attributes"]["measuredHeight"] == 12.0 and b["attributes"]["aerosplat_run"] == "r1"
    sem = b["geometry"][1]["semantics"]["surfaces"]
    types = [s["type"] for s in sem]
    assert types.count("WallSurface") == 1 and types.count("Window") == 2 and types.count("Door") == 1
    assert types.count("RoofSurface") == 1
    win = next(s for s in sem if s.get("aerosplat_id") == feats["window"].id)
    assert win["aerosplat_address"] == "pier/B01/N1/01/01" and win["aerosplat_lifecycle"] == "verified"
    assert sem[win["parent"]]["type"] == "WallSurface"
    # The verified openings are cut out of the wall as holes; the candidate stays out.
    wall_rings = b["geometry"][1]["boundaries"][0]
    assert len(wall_rings) == 1 + 2                       # the door touches the ground: not cut
    door = next(s for s in sem if s.get("aerosplat_id") == feats["door"].id)
    assert door["aerosplat_not_cut"] is True
    assert not any(s.get("aerosplat_id") == feats["candidate"].id for s in sem)
    assert report.openings == 3 and report.openings_carried == 1 and report.roof_pads == 1
    generic = {k: v for k, v in doc["CityObjects"].items() if v["type"] == "GenericCityObject"}
    reasons = {v["attributes"]["aerosplat_class"]: v["attributes"]["aerosplat_export_reason"]
               for v in generic.values()}
    assert reasons == {"opening.window": "host_none", "constraint.obstacle": "no_citygml_equivalent",
                       "pad.ground": "ground"}
    assert report.generic == 3


def test_candidates_go_out_only_when_asked(features):
    result, feats = features
    doc, report = export(result, feats, lifecycles=[Lifecycle.verified, Lifecycle.candidate])
    assert validate(doc)[0] == []
    assert report.openings == 4


def test_coordinates_are_the_sites_utm_with_ellipsoidal_heights(features):
    result, feats = features
    doc, _ = export(result, feats)
    scale, translate = np.array(doc["transform"]["scale"]), np.array(doc["transform"]["translate"])
    real = np.array(doc["vertices"]) * scale + translate
    # The building's north-east roof corner, ENU (10, 0, 12), done independently.
    lat, lon, h = (float(np.ravel(v)[0]) for v in EnuFrame(ORIGIN).to_geodetic([10.0, 0.0, 12.0]))
    e, n = Transformer.from_crs("EPSG:4979", "EPSG:32618", always_xy=True).transform(lon, lat)
    d = np.linalg.norm(real - np.array([e, n, h]), axis=1)
    assert d.min() < 0.0015                      # millimetre quantisation
    assert h == pytest.approx(ORIGIN.h + 12.0, abs=0.01)
    ext = doc["metadata"]["geographicalExtent"]
    assert ext[2] == pytest.approx(ORIGIN.h, abs=0.01) and ext[5] == pytest.approx(ORIGIN.h + 12, abs=0.01)


def test_a_pinched_footprint_keeps_its_largest_loop_and_a_crossed_one_is_refused():
    result, _ = run_result()
    # Two squares meeting only at the corner (5, 5): the raster outline visits it twice.
    pinched = Building(id="B02", footprint=[(0, 0), (5, 0), (5, 5), (9, 5), (9, 9), (5, 9), (5, 5), (0, 5)],
                       area_m2=41.0, ground_z=0.0, roof_z=6.0, height_m=6.0)
    crossed = Building(id="B03", footprint=[(20, 0), (30, 10), (30, 0), (20, 4)], area_m2=50.0,
                       ground_z=0.0, roof_z=6.0, height_m=6.0)
    result = result.model_copy(update={"buildings": [*result.buildings, pinched, crossed]})
    doc, report = export(result, {})
    assert validate(doc)[0] == []
    b2 = doc["CityObjects"]["B02"]
    assert [g["lod"] for g in b2["geometry"]] == ["1"] and "16 м²" in b2["attributes"]["aerosplat_footprint_note"]
    assert len(b2["geometry"][0]["boundaries"][0]) == 2 + 4          # the 5 × 5 square kept
    assert doc["CityObjects"]["B03"]["geometry"] == []
    assert any(s.startswith("B03: контур перетинає") for s in report.skipped)
    assert report.repaired and report.repaired[0].startswith("B02")


def test_utm_zones_and_the_projection():
    assert utm_epsg(40.89, -74.22) == 32618
    assert utm_epsg(-33.9, 18.4) == 32734
    assert utm_epsg(60.0, 4.0) == 32632                 # Norway: zone 32V, not 31
    assert utm_epsg(78.0, 10.0) == 32633                 # Svalbard
    p = SiteProjection(ORIGIN)
    o, north = p.from_enu([[0.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    assert o[2] == pytest.approx(ORIGIN.h, abs=1e-6)
    # 100 m north is ~100 m of northing: UTM scale factor and grid convergence only.
    assert np.hypot(*(north - o)[:2]) == pytest.approx(100.0, abs=0.1)


# ── the validator catches what it claims to ──────────────────────────────
@pytest.fixture
def good(features):
    result, feats = features
    doc, _ = export(result, feats)
    assert validate(doc)[0] == []
    return doc


def _solid(doc):
    return doc["CityObjects"]["B01"]["geometry"][0]


def _ms(doc):
    return doc["CityObjects"]["B01"]["geometry"][1]


def broken(good, how):
    doc = copy.deepcopy(good)
    how(doc)
    return validate(doc)[0]


def test_an_open_or_inside_out_solid_is_an_error(good):
    def drop_face(doc):
        _solid(doc)["boundaries"][0].pop()
        _solid(doc)["semantics"]["values"][0].pop()
    assert any("not closed" in e for e in broken(good, drop_face))

    def flip_all(doc):
        for face in _solid(doc)["boundaries"][0]:
            face[0].reverse()
    assert any("inwards" in e for e in broken(good, flip_all))

    def flip_one(doc):
        _solid(doc)["boundaries"][0][2][0].reverse()
    assert any("consistently oriented" in e for e in broken(good, flip_one))


def test_ring_rules(good):
    def out_of_range(doc):
        _ms(doc)["boundaries"][1][0][0] = len(doc["vertices"]) + 5
    assert any("out of range" in e for e in broken(good, out_of_range))

    def closed(doc):
        ring = _ms(doc)["boundaries"][1][0]
        ring.append(ring[0])
    assert any("closed" in e for e in broken(good, closed))

    def bow_tie(doc):
        ring = _ms(doc)["boundaries"][1][0]
        ring[1], ring[2] = ring[2], ring[1]
    assert any("crosses itself" in e for e in broken(good, bow_tie))

    def lift(doc):
        v = _ms(doc)["boundaries"][1][0][0]
        doc["vertices"][v][1] += 500                      # 0.5 m off the north wall
    assert any("not planar" in e for e in broken(good, lift))

    def hole_same_way(doc):
        _ms(doc)["boundaries"][0][1].reverse()
    assert any("inner ring oriented" in e for e in broken(good, hole_same_way))


def test_semantics_rules(good):
    def orphan_window(doc):
        sem = _ms(doc)["semantics"]["surfaces"]
        w = next(i for i, s in enumerate(sem) if s["type"] == "Window")
        del sem[w]["parent"]
    errs = broken(good, orphan_window)
    assert any("no parent wall" in e for e in errs) and any("does not name it" in e for e in errs)

    def bad_type(doc):
        _ms(doc)["semantics"]["surfaces"][0]["type"] = "Balcony"
    assert any("not one of a Building's" in e for e in broken(good, bad_type))

    def short_values(doc):
        _ms(doc)["semantics"]["values"].pop()
    assert any("semantic values for" in e or "no entry" in e for e in broken(good, short_values))


def test_document_level_rules(good):
    assert any("unused" in e for e in broken(good, lambda d: d["vertices"].append([0, 0, 0])))
    assert any("geographicalExtent" in e
               for e in broken(good, lambda d: d["metadata"]["geographicalExtent"].__setitem__(0, 0.0)))
    assert any("unknown type" in e
               for e in broken(good, lambda d: d["CityObjects"]["B01"].__setitem__("type", "House")))
    assert any("referenceSystem" in e
               for e in broken(good, lambda d: d["metadata"].__setitem__("referenceSystem", "EPSG:32618")))
    assert any("lod" in e for e in broken(good, lambda d: _solid(d).__setitem__("lod", 1)))
    json.dumps(good)                                       # plain JSON all the way down
