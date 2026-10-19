import type { FeatureCollection } from "geojson";
import {
  type GeoJSONSource,
  LngLatBounds,
  Map as MapLibreMap,
  Marker,
  NavigationControl,
  Popup,
  ScaleControl,
  setWorkerUrl,
  type StyleSpecification,
} from "maplibre-gl";
// MapLibre 6 resolves its worker at run time (new URL(name, import.meta.url)),
// which a bundler cannot see; ship the file as a hashed asset and point at it.
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef, useState } from "react";

export interface MapSite {
  key: string;
  name: string;
  lat: number;
  lon: number;
  busy?: boolean;
}

export interface MapPoint {
  lat: number;
  lon: number;
}

setWorkerUrl(maplibreWorkerUrl);

export type Basemap = "scheme" | "satellite";

const prefersLight = () => window.matchMedia?.("(prefers-color-scheme: light)").matches ?? false;

// Keyless tile sources only (the project is non-commercial; attribution shown).
const OSM = { tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], attribution: "© OpenStreetMap contributors", maxzoom: 19 };
const ESRI = {
  tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
  attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
  maxzoom: 19,
};

export function rasterStyle(basemap: Basemap): StyleSpecification {
  const src = basemap === "satellite" ? ESRI : OSM;
  // Dark theme: OSM with brightness inverted (min > max) and hue turned back,
  // so the map sits quietly behind the UI instead of glaring white.
  const dark = basemap === "scheme" && !prefersLight();
  return {
    version: 8,
    sources: { base: { type: "raster", tiles: src.tiles, tileSize: 256, maxzoom: src.maxzoom, attribution: src.attribution } },
    layers: [
      {
        id: "base",
        type: "raster",
        source: "base",
        paint: dark
          ? { "raster-brightness-min": 0.92, "raster-brightness-max": 0.08, "raster-hue-rotate": 180, "raster-saturation": -0.55, "raster-contrast": 0.1 }
          : {},
      },
    ],
  };
}

const UKRAINE: [number, number] = [31.2, 48.6];

function addPointsLayer(map: MapLibreMap, points: MapPoint[]) {
  const data: FeatureCollection = {
    type: "FeatureCollection",
    features: points.map((p) => ({
      type: "Feature",
      properties: {},
      geometry: { type: "Point", coordinates: [p.lon, p.lat] },
    })),
  };
  const src = map.getSource("points") as GeoJSONSource | undefined;
  if (src) {
    src.setData(data);
    return;
  }
  map.addSource("points", { type: "geojson", data });
  map.addLayer({
    id: "points-halo",
    type: "circle",
    source: "points",
    paint: { "circle-radius": 7, "circle-color": "#ffb84d", "circle-opacity": 0.18 },
  });
  map.addLayer({
    id: "points",
    type: "circle",
    source: "points",
    paint: {
      "circle-radius": 3.2,
      "circle-color": "#ffb84d",
      "circle-stroke-width": 1,
      "circle-stroke-color": "#0a1017",
    },
  });
}

/**
 * MapLibre map: site markers (click -> onSiteClick), camera-prior points, a
 * fixed Site origin, and an optional pick mode that reports clicked lat/lon.
 */
export function SiteMap({
  sites = [],
  points = [],
  origin,
  pick,
  onPick,
  onSiteClick,
  tall,
  hint,
}: {
  sites?: MapSite[];
  points?: MapPoint[];
  origin?: MapPoint | null;
  pick?: MapPoint | null;
  onPick?: (p: MapPoint) => void;
  onSiteClick?: (key: string) => void;
  tall?: boolean;
  hint?: string;
}) {
  const box = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markers = useRef<Marker[]>([]);
  const pickMarker = useRef<Marker | null>(null);
  const pointsRef = useRef(points);
  const fitted = useRef<string>("");
  const onPickRef = useRef(onPick);
  const onSiteRef = useRef(onSiteClick);
  const [basemap, setBasemap] = useState<Basemap>("scheme");
  const [ready, setReady] = useState(false);

  onPickRef.current = onPick;
  onSiteRef.current = onSiteClick;
  pointsRef.current = points;

  useEffect(() => {
    if (!box.current) return;
    const map = new MapLibreMap({
      container: box.current,
      style: rasterStyle("scheme"),
      center: UKRAINE,
      zoom: 4.6,
      attributionControl: { compact: true },
      // Maps sit inside scrolling pages: the wheel scrolls the page, Ctrl+wheel zooms.
      cooperativeGestures: true,
      locale: {
        "CooperativeGesturesHandler.WindowsHelpText": "Ctrl + коліщатко — масштаб карти",
        "CooperativeGesturesHandler.MacHelpText": "⌘ + коліщатко — масштаб карти",
        "CooperativeGesturesHandler.MobileHelpText": "Двома пальцями — рух карти",
      },
    });
    map.addControl(new NavigationControl({ visualizePitch: true }), "top-right");
    map.addControl(new ScaleControl({ unit: "metric" }), "bottom-right");
    map.on("click", (e) => onPickRef.current?.({ lat: e.lngLat.lat, lon: e.lngLat.lng }));
    map.on("style.load", () => addPointsLayer(map, pointsRef.current));
    map.on("load", () => setReady(true));
    mapRef.current = map;
    // Grids and dialogs settle their size after mount; keep the canvas in step.
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

  // site markers
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    markers.current.forEach((m) => m.remove());
    markers.current = sites.map((s) => {
      const el = document.createElement("div");
      el.className = `site-marker${s.busy ? " busy" : ""}`;
      el.title = s.name;
      el.addEventListener("click", (ev) => {
        ev.stopPropagation();
        onSiteRef.current?.(s.key);
      });
      const popup = new Popup({ offset: 14, closeButton: false }).setHTML(
        `<b>${escapeHtml(s.name)}</b><br/><span style="opacity:.7;font-family:var(--mono);font-size:12px">${escapeHtml(s.key)}</span>`,
      );
      el.addEventListener("mouseenter", () => popup.setLngLat([s.lon, s.lat]).addTo(map));
      el.addEventListener("mouseleave", () => popup.remove());
      return new Marker({ element: el }).setLngLat([s.lon, s.lat]).addTo(map);
    });
    if (origin) {
      const el = document.createElement("div");
      el.className = "site-marker origin";
      el.title = "ENU-початок ділянки (фіксований)";
      markers.current.push(new Marker({ element: el }).setLngLat([origin.lon, origin.lat]).addTo(map));
    }
  }, [sites, origin]);

  // camera-prior points
  useEffect(() => {
    const map = mapRef.current;
    if (map && ready && map.isStyleLoaded()) addPointsLayer(map, points);
  }, [points, ready]);

  // pick marker
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!pick) {
      pickMarker.current?.remove();
      pickMarker.current = null;
      return;
    }
    if (!pickMarker.current) {
      const el = document.createElement("div");
      el.className = "site-marker origin";
      pickMarker.current = new Marker({ element: el }).setLngLat([pick.lon, pick.lat]).addTo(map);
    } else pickMarker.current.setLngLat([pick.lon, pick.lat]);
  }, [pick]);

  // fit once per distinct set of things to show
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const all: MapPoint[] = [...sites, ...points, ...(origin ? [origin] : [])];
    const key = all.map((p) => `${p.lat.toFixed(5)},${p.lon.toFixed(5)}`).join("|");
    if (!all.length || key === fitted.current) return;
    fitted.current = key;
    if (all.length === 1) {
      map.flyTo({ center: [all[0]!.lon, all[0]!.lat], zoom: 15, speed: 1.6 });
      return;
    }
    const b = new LngLatBounds();
    all.forEach((p) => b.extend([p.lon, p.lat]));
    map.fitBounds(b, { padding: 70, maxZoom: 16.5, duration: 900 });
  }, [sites, points, origin]);

  return (
    <div className={`map${tall ? " tall" : ""}${onPick ? " picking" : ""}`}>
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
      {hint && <div className="map-hint">{hint}</div>}
    </div>
  );
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}
