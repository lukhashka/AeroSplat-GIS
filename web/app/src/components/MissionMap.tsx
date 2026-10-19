import type { Feature, FeatureCollection, LineString, Polygon } from "geojson";
import {
  type ExpressionSpecification,
  type GeoJSONSource,
  LngLatBounds,
  Map as MapLibreMap,
  Marker,
  NavigationControl,
  ScaleControl,
} from "maplibre-gl";
import { useEffect, useRef, useState } from "react";

import type { FlightTrajectory, MissionGeometry } from "../api/types";
import { LAYER_COLORS, phaseColor } from "../lib/flight";
import { type Basemap, rasterStyle } from "./SiteMap";

type LonLat = [number, number];

// Metres -> pixels at any zoom: MapLibre's world is 512·2^z px around the equator.
const EARTH_CIRCUMFERENCE_M = 40075016.686;
function metricWidth(lat: number): ExpressionSpecification {
  const k0 = 512 / (EARTH_CIRCUMFERENCE_M * Math.cos((lat * Math.PI) / 180));
  return ["interpolate", ["exponential", 2], ["zoom"], 0, ["*", ["get", "width_m"], k0], 24, ["*", ["get", "width_m"], k0 * 2 ** 24]];
}

const line = (coords: LonLat[], props: Record<string, unknown> = {}): Feature<LineString> => ({
  type: "Feature",
  properties: props,
  geometry: { type: "LineString", coordinates: coords },
});
const collection = <T extends Feature>(features: T[]): FeatureCollection => ({ type: "FeatureCollection", features });

function trajectoryFeatures(tr: FlightTrajectory | null, upTo: number) {
  const out: Feature<LineString>[] = [];
  if (!tr) return collection(out);
  for (const s of tr.segments) {
    if (s.phase === "ground" || s.i0 >= upTo) continue;
    const coords = tr.lonlat.slice(s.i0, Math.min(s.i1, upTo) + 1) as LonLat[];
    if (coords.length > 1) out.push(line(coords, { color: phaseColor(s.phase, s.failsafe), failsafe: s.failsafe }));
  }
  return collection(out);
}

function addLayers(map: MapLibreMap, g: MissionGeometry, tr: FlightTrajectory | null, index: number) {
  const fenceRing = [...g.fence.lonlat, g.fence.lonlat[0]!] as LonLat[];
  const fence: Feature<Polygon> = { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [fenceRing] } };
  const legs = g.legs.map((l) => line(l.lonlat as LonLat[], { width_m: 2 * l.radius_m, risk: l.risk_only }));
  const plan = line([[g.home.lon, g.home.lat], ...g.waypoints.map((w): LonLat => [w.lon, w.lat])]);
  const target = g.target.lonlat.length ? line([...g.target.lonlat, g.target.lonlat[0]!] as LonLat[]) : null;

  const set = (id: string, data: FeatureCollection | Feature) => {
    const src = map.getSource(id) as GeoJSONSource | undefined;
    if (src) src.setData(data);
    else map.addSource(id, { type: "geojson", data });
  };
  set("fence", fence);
  set("legs", collection(legs));
  set("plan", plan);
  set("target", target ? target : collection([]));
  set("traj-all", trajectoryFeatures(tr, Infinity));
  set("traj-flown", trajectoryFeatures(tr, index));
  if (map.getLayer("fence-fill")) return;

  map.addLayer({ id: "fence-fill", type: "fill", source: "fence", paint: { "fill-color": LAYER_COLORS.fence, "fill-opacity": 0.05 } });
  map.addLayer({
    id: "fence-line",
    type: "line",
    source: "fence",
    paint: { "line-color": LAYER_COLORS.fence, "line-width": 1.5, "line-dasharray": [3, 2], "line-opacity": 0.9 },
  });
  // The corridor at its true width: 2·radius, round caps — the footprint of the tubes in 3D.
  map.addLayer({
    id: "legs",
    type: "line",
    source: "legs",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: {
      "line-color": ["case", ["get", "risk"], LAYER_COLORS.corridorRisk, LAYER_COLORS.corridor],
      "line-opacity": 0.16,
      "line-width": metricWidth(g.origin.lat),
    },
  });
  map.addLayer({
    id: "plan",
    type: "line",
    source: "plan",
    paint: { "line-color": LAYER_COLORS.plan, "line-width": 1.5, "line-dasharray": [1, 2] },
  });
  map.addLayer({ id: "target", type: "line", source: "target", paint: { "line-color": LAYER_COLORS.target, "line-width": 4 } });
  map.addLayer({
    id: "traj-all",
    type: "line",
    source: "traj-all",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": ["get", "color"], "line-width": 2, "line-opacity": 0.35 },
  });
  map.addLayer({
    id: "traj-flown",
    type: "line",
    source: "traj-flown",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": ["get", "color"], "line-width": 3 },
  });
}

function labelMarker(map: MapLibreMap, text: string, color: string, lon: number, lat: number, title: string) {
  const el = document.createElement("div");
  el.className = "map-label";
  el.style.setProperty("--c", color);
  el.textContent = text;
  el.title = title;
  return new Marker({ element: el, anchor: "bottom-left", offset: [4, -4] }).setLngLat([lon, lat]).addTo(map);
}

/** Top view on the map: geofence, corridor at its width, plan, target, the true track and the drone. */
export function MissionMap({
  geometry,
  trajectory,
  index,
  onSeek,
}: {
  geometry: MissionGeometry;
  trajectory: FlightTrajectory | null;
  index: number;
  onSeek?: (index: number) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const droneRef = useRef<Marker | null>(null);
  const data = useRef({ geometry, trajectory, index });
  const onSeekRef = useRef(onSeek);
  const [basemap, setBasemap] = useState<Basemap>("satellite");
  const [ready, setReady] = useState(false);
  data.current = { geometry, trajectory, index };
  onSeekRef.current = onSeek;

  useEffect(() => {
    if (!box.current) return;
    const g = data.current.geometry;
    const map = new MapLibreMap({
      container: box.current,
      style: rasterStyle("satellite"),
      center: [g.origin.lon, g.origin.lat],
      zoom: 17,
      attributionControl: { compact: true },
      cooperativeGestures: true,
      locale: {
        "CooperativeGesturesHandler.WindowsHelpText": "Ctrl + коліщатко — масштаб карти",
        "CooperativeGesturesHandler.MacHelpText": "⌘ + коліщатко — масштаб карти",
        "CooperativeGesturesHandler.MobileHelpText": "Двома пальцями — рух карти",
      },
    });
    map.addControl(new NavigationControl({ visualizePitch: true }), "top-right");
    map.addControl(new ScaleControl({ unit: "metric" }), "bottom-right");
    map.on("style.load", () => {
      const d = data.current;
      addLayers(map, d.geometry, d.trajectory, d.index);
    });
    map.on("load", () => setReady(true));
    // A click near the track seeks the replay to the nearest sample.
    map.on("click", (e) => {
      const tr = data.current.trajectory;
      if (!tr || !onSeekRef.current) return;
      let best = -1;
      let bestD = 14 * 14;
      tr.lonlat.forEach(([lon, lat], k) => {
        if (!tr.armed[k]) return;
        const p = map.project([lon, lat]);
        const d = (p.x - e.point.x) ** 2 + (p.y - e.point.y) ** 2;
        if (d < bestD) {
          bestD = d;
          best = k;
        }
      });
      if (best >= 0) onSeekRef.current(best);
    });
    mapRef.current = map;
    const ro = new ResizeObserver(() => map.resize());
    ro.observe(box.current);
    return () => {
      ro.disconnect();
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (ready) mapRef.current?.setStyle(rasterStyle(basemap));
  }, [basemap, ready]);

  // the mission: sources, labels, framing
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    if (map.isStyleLoaded()) addLayers(map, geometry, trajectory, data.current.index);
    const marks = [
      labelMarker(map, "дім", LAYER_COLORS.home, geometry.home.lon, geometry.home.lat, "Дім: точка зльоту й посадки"),
      labelMarker(map, "E", LAYER_COLORS.entry, geometry.target.entry.lon, geometry.target.entry.lat, "E: точка входу"),
      labelMarker(map, "H", LAYER_COLORS.hover, geometry.target.hover.lon, geometry.target.hover.lat, "H: точка зависання"),
    ];
    const b = new LngLatBounds();
    geometry.fence.lonlat.forEach((p) => b.extend(p));
    trajectory?.lonlat.forEach((p, k) => trajectory.armed[k] && b.extend(p));
    if (!b.isEmpty()) map.fitBounds(b, { padding: 40, maxZoom: 19, duration: 0 });
    return () => marks.forEach((m) => m.remove());
  }, [geometry, trajectory, ready]);

  // the drone and the flown part of the track
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !trajectory || !trajectory.lonlat.length) return;
    const k = Math.max(0, Math.min(index, trajectory.lonlat.length - 1));
    const [lon, lat] = trajectory.lonlat[k]!;
    if (!droneRef.current) {
      const el = document.createElement("div");
      el.className = "drone-marker";
      el.innerHTML = '<svg viewBox="0 0 24 24" width="26" height="26"><path d="M12 2 19 20 12 16 5 20Z"/></svg>';
      droneRef.current = new Marker({ element: el, rotationAlignment: "map" }).setLngLat([lon, lat]).addTo(map);
    }
    const m = droneRef.current;
    m.setLngLat([lon, lat]).setRotation(trajectory.yaw_deg[k] ?? 0);
    m.getElement().classList.toggle("failsafe", !!trajectory.failsafe[k]);
    const src = map.getSource("traj-flown") as GeoJSONSource | undefined;
    src?.setData(trajectoryFeatures(trajectory, k));
  }, [index, trajectory, ready]);

  useEffect(
    () => () => {
      droneRef.current?.remove();
      droneRef.current = null;
    },
    [trajectory],
  );

  return (
    <div className="map tall">
      <div ref={box} style={{ position: "absolute", inset: 0 }} />
      <div className="map-overlay">
        <div className="segmented">
          <button className={basemap === "scheme" ? "on" : ""} onClick={() => setBasemap("scheme")}>
            Схема
          </button>
          <button className={basemap === "satellite" ? "on" : ""} onClick={() => setBasemap("satellite")}>
            Супутник
          </button>
        </div>
      </div>
    </div>
  );
}
