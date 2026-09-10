'use client';

import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

export type CatchmentBoundary = {
  type: 'Feature';
  geometry: { type: string; coordinates: unknown };
  properties?: Record<string, unknown>;
} | null;

interface WatershedMapProps {
  lat: number;
  lng: number;
  catchmentBoundary?: CatchmentBoundary;
  elevationTileUrl?: string | null;
  baseLayer: 'satellite' | 'terrain' | 'hybrid';
  showBoundary: boolean;
  showElevation: boolean;
}

// Real satellite/terrain tile sources — no API key required.
const TILE_URLS = {
  satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  terrain: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
  labels: 'https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png',
};

export default function WatershedMap({
  lat,
  lng,
  catchmentBoundary,
  elevationTileUrl,
  baseLayer,
  showBoundary,
  showElevation,
}: WatershedMapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const baseLayerRef = useRef<L.TileLayer | L.LayerGroup | null>(null);
  const boundaryLayerRef = useRef<L.GeoJSON | null>(null);
  const elevationLayerRef = useRef<L.TileLayer | null>(null);
  const markerRef = useRef<L.CircleMarker | null>(null);

  // Init map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, { zoomControl: true, attributionControl: false }).setView([lat, lng], 13);
    mapRef.current = map;

    markerRef.current = L.circleMarker([lat, lng], {
      radius: 7,
      color: '#f97316',
      fillColor: '#f97316',
      fillOpacity: 0.9,
      weight: 2,
    }).addTo(map).bindPopup('Capture point');

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Recenter marker + view when coordinates change
  useEffect(() => {
    if (!mapRef.current) return;
    markerRef.current?.setLatLng([lat, lng]);
    if (!catchmentBoundary) mapRef.current.setView([lat, lng], 13);
  }, [lat, lng, catchmentBoundary]);

  // Base layer switching
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (baseLayerRef.current) {
      map.removeLayer(baseLayerRef.current);
      baseLayerRef.current = null;
    }

    if (baseLayer === 'satellite') {
      baseLayerRef.current = L.tileLayer(TILE_URLS.satellite, { maxZoom: 19 }).addTo(map);
    } else if (baseLayer === 'terrain') {
      baseLayerRef.current = L.tileLayer(TILE_URLS.terrain, { maxZoom: 17, subdomains: 'abc' }).addTo(map);
    } else {
      const group = L.layerGroup([
        L.tileLayer(TILE_URLS.satellite, { maxZoom: 19 }),
        L.tileLayer(TILE_URLS.labels, { maxZoom: 19, subdomains: 'abcd' }),
      ]).addTo(map);
      baseLayerRef.current = group;
    }
  }, [baseLayer]);

  // Real watershed boundary polygon (HydroBASINS from backend) — replaces the old fake fixed SVG shape
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (boundaryLayerRef.current) {
      map.removeLayer(boundaryLayerRef.current);
      boundaryLayerRef.current = null;
    }

    if (showBoundary && catchmentBoundary?.geometry) {
      const layer = L.geoJSON(catchmentBoundary as GeoJSON.Feature, {
        style: { color: '#38bdf8', weight: 2, fillColor: '#38bdf8', fillOpacity: 0.12 },
      }).addTo(map);
      boundaryLayerRef.current = layer;
      const bounds = layer.getBounds();
      if (bounds.isValid()) map.fitBounds(bounds.pad(0.3), { maxZoom: 15 });
    }
  }, [catchmentBoundary, showBoundary]);

  // Real elevation color raster from Earth Engine — replaces decorative gradient
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (elevationLayerRef.current) {
      map.removeLayer(elevationLayerRef.current);
      elevationLayerRef.current = null;
    }

    if (showElevation && elevationTileUrl) {
      elevationLayerRef.current = L.tileLayer(elevationTileUrl, { opacity: 0.55 }).addTo(map);
    }
  }, [elevationTileUrl, showElevation]);

  return <div ref={containerRef} className="absolute inset-0 w-full h-full" />;
}
