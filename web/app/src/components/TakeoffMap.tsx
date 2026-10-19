import type { Feature, FeatureCollection, LineString, Point, Polygon } from "geojson";
import { type GeoJSONSource, LngLatBounds, Map as MapLibreMap, Marker, NavigationControl, ScaleControl } from "maplibre-gl";
import { useEffect, useRef, useState } from "react";

import type { GeoPoint, TakeoffCandidate, TakeoffSearch, Vec3 } from "../api/types";
import { LAYER_COLORS } from "../lib/flight";
import { enuToGeodetic } from "../lib/frames";
import { circleEnu, fmtEnu } from "../lib/takeoff";
import { type Basemap, rasterStyle } from "./SiteMap";

type LonLat = [number, number];

const MISS_COLOR = "#ffb84d";
const SHORT_COLOR = "#ff6b6b";

function toLonLat(origin: GeoPoint, p: readonly number[]): LonLat {
  const g = enuToGeodetic(origin, [p[0]!, p[1]!, p[2] ?? 0] as Vec3);
  return [g.lon, g.lat];
}

const ring = (origin: GeoPoint, c: readonly number[], r: number) => circleEnu(c[0]!, c[1]!, r).map((p) => toLonLat(origin, p));
const disk = (origin: GeoPoint, c: readonly number[], r: number, props: Record<string, unknown>): Feature<Polygon> => ({
  type: "Feature",
  properties: props,
  geometry: { type: "Polygon", coordinates: [ring(origin, c, r)] },
});
const circleLine = (origin: GeoPoint, c: readonly number[], r: number, props: Record<string, unknown>): Feature<LineString> => ({
  type: "Feature",
  properties: props,
  geometry: { type: "LineString", coordinates: ring(origin, c, r) },
});
const point = (origin: GeoPoint, p: readonly number[], props: Record<string, unknown>): Feature<Point> => ({
  type: "Feature",
  properties: props,
  geometry: { type: "Point", coordinates: toLonLat(origin, p) },
});
const collection = (features: Feature[]): FeatureCollection => ({ type: "FeatureCollection", features });

function setLayers(map: MapLibreMap, origin: GeoPoint, s: TakeoffSearch, selected: number | null) {
  // every candidate with the takeoff corridor r around it; the near miss with what it has (open) and needs (r)
  const corridors = s.candidates.map((c) => disk(origin, c.enu, s.radius_m, { on: c.rank === selected }));
  const miss: Feature[] = s.best_open
    ? [
        circleLine(origin, s.best_open.enu, Math.max(s.best_open.open_radius_m, 0.1), { kind: "open" }),
        circleLine(origin, s.best_open.enu, s.radius_m, { kind: "need" }),
      ]
    : [];
  const data: Record<string, FeatureCollection> = {
    "to-reach": collection([circleLine(origin, s.target, s.max_distance_m, {})]),
    "to-corridors": collection(corridors),
    "to-miss": collection(miss),
    // exact places under the labels: a candidate can sit on the target (the pins would hide them)
    "to-points": collection([point(origin, s.target, { kind: "target" }), point(origin, [0, 0, 0], { kind: "origin" })]),
  };
  for (const [id, fc] of Object.entries(data)) {
    const src = map.getSource(id) as GeoJSONSource | undefined;
    if (src) src.setData(fc);
    else map.addSource(id, { type: "geojson", data: fc });
  }
  if (map.getLayer("to-reach")) return;
  map.addLayer({
    id: "to-reach",
    type: "line",
    source: "to-reach",
    paint: { "line-color": LAYER_COLORS.waypoint, "line-width": 2, "line-dasharray": [3, 2], "line-opacity": 0.8 },
  });
  map.addLayer({
    id: "to-corridors",
    type: "fill",
    source: "to-corridors",
    paint: { "fill-color": LAYER_COLORS.corridor, "fill-opacity": ["case", ["get", "on"], 0.35, 0.14] },
  });
  map.addLayer({
    id: "to-corridors-line",
    type: "line",
    source: "to-corridors",
    paint: { "line-color": LAYER_COLORS.corridor, "line-width": ["case", ["get", "on"], 3, 1.5] },
  });
  map.addLayer({
    id: "to-miss",
    type: "line",
    source: "to-miss",
    paint: {
      "line-color": ["case", ["==", ["get", "kind"], "need"], SHORT_COLOR, MISS_COLOR],
      "line-width": 2,
      "line-dasharray": [2, 2],
    },
  });
  map.addLayer({
    id: "to-points",
    type: "circle",
    source: "to-points",
    paint: {
      "circle-radius": 5,
      "circle-color": ["case", ["==", ["get", "kind"], "target"], LAYER_COLORS.target, LAYER_COLORS.home],
      "circle-stroke-color": "#0a1017",
      "circle-stroke-width": 2,
    },
  });
}

function pin(text: string, cls: string, title: string, onClick?: () => void): HTMLDivElement {
  const el = document.createElement("div");
  el.className = `takeoff-pin ${cls}`;
  el.textContent = text;
  el.title = title;
  if (onClick) {
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      onClick();
    });
  }
  return el;
}

const candidateTitle = (c: TakeoffCandidate) =>
  `#${c.rank}: E/N ${fmtEnu(c.enu)} м · ${c.distance_m.toFixed(0)} м до цілі · ≈${c.open_radius_m.toFixed(1)} м відкрито` +
  (c.climb_m !== null ? ` · підйом ${c.climb_m.toFixed(1)} м` : "");

/** Top view of a takeoff search: the reach around the target, every candidate with its corridor r, the near miss. */
export function TakeoffMap({
  origin,
  search,
  selected,
  onSelect,
}: {
  origin: GeoPoint;
  search: TakeoffSearch;
  selected: number | null;
  onSelect: (rank: number) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const pins = useRef<Map<number, HTMLDivElement>>(new Map());
  const data = useRef({ origin, search, selected });
  const onSelectRef = useRef(onSelect);
  const [basemap, setBasemap] = useState<Basemap>("satellite");
  const [ready, setReady] = useState(false);
  data.current = { origin, search, selected };
  onSelectRef.current = onSelect;

  useEffect(() => {
    if (!box.current) return;
    const map = new MapLibreMap({
      container: box.current,
      style: rasterStyle("satellite"),
      center: toLonLat(data.current.origin, data.current.search.target),
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
      setLayers(map, d.origin, d.search, d.selected);
    });
    map.on("load", () => setReady(true));
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

  // the search: layers, pins, framing on the reach
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    if (map.isStyleLoaded()) setLayers(map, origin, search, data.current.selected);
    const add = (el: HTMLDivElement, p: readonly number[]) => new Marker({ element: el }).setLngLat(toLonLat(origin, p)).addTo(map);
    // labels beside their dots (to-points), below the candidate pins
    const label = (el: HTMLDivElement, p: readonly number[], offset: [number, number]) =>
      new Marker({ element: el, anchor: "top-right", offset }).setLngLat(toLonLat(origin, p)).addTo(map);
    const marks = [
      label(pin("ціль", "target", `Кінець транзиту (E або майданчик): E/N ${fmtEnu(search.target)} м`), search.target, [-6, 6]),
      label(pin("початок", "origin", "Початок ділянки (0, 0): типовий дім, якщо місце зльоту не задано"), [0, 0, 0], [-6, 6]),
    ];
    pins.current.clear();
    for (const c of search.candidates) {
      const el = pin(String(c.rank), "cand", candidateTitle(c), () => onSelectRef.current(c.rank));
      pins.current.set(c.rank, el);
      marks.push(add(el, c.enu));
    }
    if (search.best_open)
      marks.push(add(pin("?", "miss", `Найвідкритіше місце, не підходить: ${candidateTitle(search.best_open)}`), search.best_open.enu));
    // framed on what was found (with its corridor), the target and the origin; the reach is the legend's
    const b = new LngLatBounds();
    const around = Math.max(search.radius_m, 5);
    for (const c of [search.target, [0, 0, 0], ...search.candidates.map((c) => c.enu), ...(search.best_open ? [search.best_open.enu] : [])])
      for (const p of ring(origin, c, around)) b.extend(p);
    map.fitBounds(b, { padding: 48, maxZoom: 20, duration: 0 });
    return () => marks.forEach((m) => m.remove());
  }, [origin, search, ready]);

  useEffect(() => {
    const map = mapRef.current;
    pins.current.forEach((el, rank) => el.classList.toggle("on", rank === selected));
    if (map && ready && map.isStyleLoaded()) setLayers(map, origin, search, selected);
  }, [selected, origin, search, ready]);

  return (
    <div className="map">
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
      <div className="flight-legend takeoff-legend">
        {search.candidates.length > 0 && (
          <span>
            <i className="ring" style={{ borderColor: LAYER_COLORS.corridor }} />
            коридор зльоту r = {search.radius_m.toFixed(2)} м
          </span>
        )}
        <span>
          <i className="dash" style={{ borderColor: LAYER_COLORS.waypoint }} />
          пошук до {search.max_distance_m.toFixed(0)} м
        </span>
        {search.best_open && (
          <span>
            <i className="dash" style={{ borderColor: MISS_COLOR }} />
            відкрито ≈{search.best_open.open_radius_m.toFixed(1)} м
          </span>
        )}
        {search.best_open && (
          <span>
            <i className="dash" style={{ borderColor: SHORT_COLOR }} />
            треба r = {search.radius_m.toFixed(2)} м
          </span>
        )}
      </div>
    </div>
  );
}
