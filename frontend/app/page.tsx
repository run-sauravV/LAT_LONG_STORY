'use client';

import { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import {
  Droplet,
  Home as HomeIcon,
  Upload,
  Map as MapIcon,
  FileText,
  Bell,
  Leaf,
  Mountain,
  Droplets,
  Navigation,
  Compass,
  Calendar,
  Cloud,
  AlertTriangle,
  Waves,
} from 'lucide-react';

// Leaflet touches window/document at import time, so it must be client-only and loaded dynamically.
const WatershedMap = dynamic(() => import('./WatershedMap'), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

interface ExifData {
  latitude: number;
  longitude: number;
  altitude?: number | null;
  timestamp?: string | null;
}

interface WaterBody {
  confidence: number;
  area_sqm: number;
}

interface AnalysisResults {
  upload_id: string;
  status: 'processing' | 'complete' | 'failed';
  processing_time_seconds: number;
  error?: string;
  cv_output?: {
    vegetation_density: number;
    slope_estimate: number;
    water_bodies: WaterBody[];
    cv_model_status: string;
  };
  gis_output?: {
    elevation_m: number | null;
    outlet_elevation_m: number | null;
    max_elevation_m: number | null;
    dem_source: string;
    flow_direction: string;
    catchment_area_sqkm: number | null;
    catchment_boundary: {
      type: 'Feature';
      geometry: { type: string; coordinates: unknown };
      properties?: Record<string, unknown>;
    } | null;
    elevation_tile_url: string | null;
    dem_slope_deg: number | null;
    erosion_risk_category: string | null;
    relief_ratio: number | null;
    runoff_coefficient?: number;
    weather?: {
      temperature_c: number;
      humidity_percent: number;
      rainfall_mm: number;
      source: string;
    };
    satellite?: {
      ndvi_mean: number | null;
      true_color_base64?: string;
      source: string;
    };
  };
}

const NAV_ITEMS = [
  { label: 'Dashboard', icon: HomeIcon },
  { label: 'Uploads', icon: Upload },
  { label: 'Map', icon: MapIcon },
  { label: 'Reports', icon: FileText },
];

const MAP_LAYER_TABS: Array<{ key: 'satellite' | 'terrain' | 'hybrid'; label: string }> = [
  { key: 'satellite', label: 'Satellite' },
  { key: 'terrain', label: 'Terrain' },
  { key: 'hybrid', label: 'Hybrid' },
];

export default function Home() {
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [exif, setExif] = useState<ExifData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [activeNav, setActiveNav] = useState('Dashboard');
  const [mapLayer, setMapLayer] = useState<'satellite' | 'terrain' | 'hybrid'>('satellite');
  const [layers, setLayers] = useState({ boundary: true, elevation: false });

  useEffect(() => {
    if (!uploadId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/results/${uploadId}`);
        if (!res.ok) throw new Error('Failed to fetch analysis status');

        const data: AnalysisResults = await res.json();

        if (data.status === 'complete') {
          setResults(data);
          setLoading(false);
          clearInterval(interval);
        } else if (data.status === 'failed') {
          setError(data.error || 'Processing failed on the server.');
          setLoading(false);
          clearInterval(interval);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error polling results.');
        setLoading(false);
        clearInterval(interval);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [uploadId]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setError(null);
      setResults(null);
      setUploadId(null);
      setExif(null);
      handleUpload(e.target.files[0]);
    }
  };

  const handleUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('photo', file);

    try {
      const res = await fetch(`${API_BASE}/api/v1/upload`, { method: 'POST', body: formData });
      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || 'Upload failed');

      setExif(data.exif);
      setUploadId(data.upload_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during upload.');
      setLoading(false);
    }
  };

  const hasResult = results?.status === 'complete';
  const cv = results?.cv_output;
  const gis = results?.gis_output;

  const lat = exif?.latitude ?? null;
  const lng = exif?.longitude ?? null;
  const altitude = exif?.altitude ?? null;
  const captured = exif?.timestamp ? new Date(exif.timestamp).toLocaleString() : null;

  const vegPct = cv ? Math.round(cv.vegetation_density * 100) : null;
  const waterBodyCount = cv?.water_bodies?.length ?? 0;

  const riskColor = useMemo(() => {
    switch (gis?.erosion_risk_category) {
      case 'Severe':
      case 'High':
        return { badge: 'bg-red-50 text-red-600', dot: 'bg-red-500' };
      case 'Moderate':
        return { badge: 'bg-amber-50 text-amber-600', dot: 'bg-amber-400' };
      case 'Low':
        return { badge: 'bg-emerald-50 text-emerald-600', dot: 'bg-emerald-500' };
      default:
        return { badge: 'bg-slate-100 text-slate-500', dot: 'bg-slate-400' };
    }
  }, [gis?.erosion_risk_category]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-800 flex relative">
      <Starfield />

      {/* Sidebar */}
      <aside className="w-56 shrink-0 bg-white border-r border-slate-200 flex flex-col relative z-10">
        <div className="flex items-center gap-2 px-5 py-5">
          <div className="w-8 h-8 rounded-lg bg-sky-500 flex items-center justify-center">
            <Droplet className="w-4 h-4 text-white" fill="white" />
          </div>
          <span className="font-semibold text-slate-800 text-[15px] leading-tight">
            Watershed Monitoring System
          </span>
        </div>

        <nav className="mt-2 flex flex-col gap-1 px-3">
          {NAV_ITEMS.map(({ label, icon: Icon }) => {
            const active = activeNav === label;
            return (
              <button
                key={label}
                onClick={() => setActiveNav(label)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-left transition-colors ${
                  active ? 'bg-sky-50 text-sky-600' : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700'
                }`}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            );
          })}
        </nav>
      </aside>

      {/* Main content */}
      <div className="flex-1 min-w-0 relative z-10">
        <div
          className="relative h-48 md:h-56 bg-cover bg-center flex items-end"
          style={{
            backgroundImage:
              "linear-gradient(180deg, rgba(2,6,23,0.35) 0%, rgba(2,6,23,0.85) 100%), url('/images/hero-satellite.jpg')",
          }}
        >
          <div className="flex items-center justify-between w-full px-8 pb-5">
            <div>
              <h1 className="text-2xl font-bold text-white drop-shadow-sm">Dashboard</h1>
              <p className="text-sm text-slate-200/90 mt-0.5 max-w-md">
                Monitor watershed health and get real-time insights from field data.
              </p>
            </div>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 bg-sky-500 hover:bg-sky-600 text-white text-sm font-medium px-4 py-2 rounded-lg cursor-pointer transition-colors shadow-lg shadow-sky-950/40">
                <Upload className="w-4 h-4" />
                {loading ? 'Uploading…' : 'Upload Photo'}
                <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
              </label>
              <button className="text-slate-100/80 hover:text-white">
                <Bell className="w-5 h-5" />
              </button>
              <div className="w-9 h-9 rounded-full bg-white/10 backdrop-blur border border-white/30 text-white flex items-center justify-center text-sm font-semibold">
                SV
              </div>
            </div>
          </div>
        </div>

        <main className="p-8 space-y-6 bg-slate-50">
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
              <strong>Error:</strong> {error}
            </div>
          )}

          {!hasResult && !loading && !error && (
            <div className="p-4 bg-sky-50 border border-sky-200 rounded-lg text-sky-700 text-sm">
              No survey uploaded yet. Upload a geo-tagged field photo to see real elevation, vegetation, water and weather data.
            </div>
          )}

          {/* Stat cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            <StatCard
              icon={<Leaf className="w-5 h-5 text-emerald-500" />}
              iconBg="bg-emerald-50"
              label="Vegetation Density"
              value={vegPct !== null ? `${vegPct}%` : '—'}
            >
              {vegPct !== null && (
                <div className="w-full h-1.5 bg-slate-100 rounded-full mt-3 overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${vegPct}%` }} />
                </div>
              )}
            </StatCard>

            <StatCard
              icon={<Mountain className="w-5 h-5 text-sky-500" />}
              iconBg="bg-sky-50"
              label="Elevation"
              value={gis?.elevation_m != null ? `${gis.elevation_m} m` : '—'}
            />

            <StatCard
              icon={<AlertTriangle className="w-5 h-5 text-amber-500" />}
              iconBg="bg-amber-50"
              label="Erosion Risk"
              value={gis?.erosion_risk_category ?? '—'}
            >
              {gis?.erosion_risk_category && (
                <span className={`inline-flex items-center gap-1.5 text-xs mt-2 px-2 py-0.5 rounded-full ${riskColor.badge}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${riskColor.dot}`} />
                  Slope {gis.dem_slope_deg ?? '—'}°
                </span>
              )}
            </StatCard>

            <StatCard
              icon={<Navigation className="w-5 h-5 text-emerald-500" />}
              iconBg="bg-emerald-50"
              label="Flow Direction"
              value={gis?.flow_direction ?? '—'}
            />
          </div>

          {/* Map + right rail */}
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6 items-start">
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
              <div className="flex items-center gap-1 px-4 pt-3 border-b border-slate-200">
                {MAP_LAYER_TABS.map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => setMapLayer(tab.key)}
                    className={`px-3 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
                      mapLayer === tab.key ? 'border-sky-500 text-sky-600' : 'border-transparent text-slate-400 hover:text-slate-600'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="relative h-[420px] bg-slate-200 overflow-hidden">
                {lat != null && lng != null ? (
                  <WatershedMap
                    lat={lat}
                    lng={lng}
                    catchmentBoundary={gis?.catchment_boundary ?? null}
                    elevationTileUrl={gis?.elevation_tile_url ?? null}
                    baseLayer={mapLayer}
                    showBoundary={layers.boundary}
                    showElevation={layers.elevation}
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm text-center px-8">
                    Upload a geo-tagged photo to see its location and watershed boundary on the map.
                  </div>
                )}

                <div className="absolute top-3 right-3 w-8 h-8 rounded-full bg-black/30 backdrop-blur-sm flex items-center justify-center pointer-events-none z-[500]">
                  <Compass className="w-4 h-4 text-white" />
                </div>

                {lat != null && (
                  <div className="absolute top-14 right-3 w-48 bg-white rounded-lg shadow-lg border border-slate-200 p-3 text-xs z-[500]">
                    <p className="font-semibold text-slate-700 mb-2">Layers</p>
                    <LayerCheckbox
                      label="Watershed Boundary"
                      checked={layers.boundary}
                      onChange={() => setLayers((p) => ({ ...p, boundary: !p.boundary }))}
                      disabled={!gis?.catchment_boundary}
                    />
                    <LayerCheckbox
                      label="Elevation (Earth Engine)"
                      checked={layers.elevation}
                      onChange={() => setLayers((p) => ({ ...p, elevation: !p.elevation }))}
                      disabled={!gis?.elevation_tile_url}
                    />
                    {!gis?.catchment_boundary && !gis?.elevation_tile_url && hasResult && (
                      <p className="text-slate-400 mt-2 leading-snug">
                        No boundary or elevation layer returned for this location.
                      </p>
                    )}
                  </div>
                )}

                {loading && (
                  <div className="absolute inset-0 bg-slate-900/60 flex items-center justify-center gap-3 text-white text-sm z-[600]">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Fetching satellite, elevation, weather and CV data…
                  </div>
                )}
              </div>
            </div>

            {/* Right rail */}
            <div className="space-y-5">
              <SidePanel title="Location" icon={<Navigation className="w-4 h-4 text-sky-500" />}>
                {lat != null && lng != null ? (
                  <>
                    <p className="text-lg font-semibold text-slate-800">
                      {lat.toFixed(4)}° N, {lng.toFixed(4)}° E
                    </p>
                    <div className="grid grid-cols-2 gap-3 mt-3 text-xs text-slate-500">
                      <div>
                        <p className="text-[11px]">Altitude</p>
                        <p className="text-sm font-medium text-slate-700">{altitude != null ? `${altitude} m` : '—'}</p>
                      </div>
                      <div>
                        <p className="flex items-center gap-1 text-[11px]">
                          <Calendar className="w-3 h-3" /> Captured
                        </p>
                        <p className="text-sm font-medium text-slate-700">{captured ?? '—'}</p>
                      </div>
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-slate-400">No photo uploaded yet.</p>
                )}
              </SidePanel>

              <SidePanel title="Weather" icon={<Cloud className="w-4 h-4 text-sky-500" />}>
                {gis?.weather ? (
                  <>
                    <div className="flex items-center gap-3">
                      <Cloud className="w-8 h-8 text-sky-400" />
                      <div>
                        <p className="text-2xl font-bold text-slate-800">{gis.weather.temperature_c}°C</p>
                        <p className="text-xs text-slate-500">{gis.weather.source}</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3 mt-3 text-xs text-slate-500">
                      <div className="flex items-center gap-1.5">
                        <Droplet className="w-3.5 h-3.5 text-sky-400" />
                        <span className="text-sm font-medium text-slate-700">{gis.weather.humidity_percent}%</span> Humidity
                      </div>
                      <div className="flex items-center gap-1.5">
                        <Droplets className="w-3.5 h-3.5 text-sky-400" />
                        <span className="text-sm font-medium text-slate-700">{gis.weather.rainfall_mm} mm</span> Rainfall
                      </div>
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-slate-400">No weather data yet.</p>
                )}
              </SidePanel>

              <SidePanel title="Water Bodies" icon={<Waves className="w-4 h-4 text-sky-500" />}>
                {hasResult ? (
                  <>
                    <p className="text-2xl font-bold text-slate-800">{waterBodyCount}</p>
                    <p className="text-xs text-slate-500 mt-1">
                      {cv?.cv_model_status ?? 'No CV model status reported.'}
                    </p>
                    {cv?.water_bodies?.map((w, i) => (
                      <p key={i} className="text-xs text-slate-600 mt-2">
                        Body {i + 1}: {w.area_sqm} m² · {(w.confidence * 100).toFixed(0)}% confidence
                      </p>
                    ))}
                  </>
                ) : (
                  <p className="text-sm text-slate-400">No survey processed yet.</p>
                )}
              </SidePanel>
            </div>
          </div>

          {/* Detailed analysis */}
          <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            <div className="px-4 pt-3 pb-1 border-b border-slate-200">
              <p className="px-3 py-2 text-sm font-medium text-sky-600 border-b-2 border-sky-500 -mb-px inline-block">
                Detailed Analysis
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 p-6">
              <div className="flex items-center gap-4">
                <DonutGauge percent={vegPct ?? 0} />
                <div>
                  <p className="font-semibold text-slate-800">Vegetation Density</p>
                  <p className="text-xs text-slate-500 mt-1 max-w-[140px]">
                    {vegPct !== null
                      ? `${vegPct}% cover, estimated from field photo pixels (Excess Green Index).`
                      : 'Awaiting a processed survey.'}
                  </p>
                </div>
              </div>

              <div>
                <p className="text-sm font-medium text-slate-600 mb-2">NDVI (current reading)</p>
                {gis?.satellite?.ndvi_mean != null ? (
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-emerald-600">{gis.satellite.ndvi_mean.toFixed(2)}</span>
                    <span className="text-xs text-slate-500">from {gis.satellite.source}</span>
                  </div>
                ) : (
                  <p className="text-xs text-slate-400">
                    No NDVI value returned yet — historical trend isn&apos;t available from the current pipeline.
                  </p>
                )}
                {gis?.catchment_area_sqkm != null && (
                  <p className="text-xs text-slate-500 mt-3">
                    Catchment area (HydroBASINS): <span className="font-medium text-slate-700">{gis.catchment_area_sqkm} km²</span>
                  </p>
                )}
                {gis?.relief_ratio != null && (
                  <p className="text-xs text-slate-500 mt-1">
                    Relief ratio: <span className="font-medium text-slate-700">{gis.relief_ratio}</span>
                  </p>
                )}
              </div>

              <div>
                <p className="text-sm font-medium text-slate-600 mb-2">Satellite view</p>
                <div className="h-28 rounded-lg overflow-hidden relative bg-slate-100">
                  {gis?.satellite?.true_color_base64 ? (
                    <img
                      src={`data:image/png;base64,${gis.satellite.true_color_base64}`}
                      alt="Real satellite true-color view of the survey site"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-xs text-slate-400 text-center px-4">
                      No satellite image available for this location/date yet.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  iconBg,
  label,
  value,
  children,
}: {
  icon: React.ReactNode;
  iconBg: string;
  label: string;
  value: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <div className="flex items-center gap-3">
        <div className={`w-9 h-9 rounded-full flex items-center justify-center ${iconBg}`}>{icon}</div>
        <p className="text-sm text-slate-500">{label}</p>
      </div>
      <p className="text-2xl font-bold text-slate-800 mt-3">{value}</p>
      {children}
    </div>
  );
}

function SidePanel({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <div className="flex items-center gap-2 mb-3">
        {icon}
        <p className="text-sm font-semibold text-slate-700">{title}</p>
      </div>
      {children}
    </div>
  );
}

function LayerCheckbox({
  label,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  checked: boolean;
  onChange: () => void;
  disabled?: boolean;
}) {
  return (
    <label className={`flex items-center gap-2 py-1 ${disabled ? 'text-slate-300 cursor-not-allowed' : 'text-slate-600 cursor-pointer'}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        disabled={disabled}
        className="w-3.5 h-3.5 rounded accent-sky-500"
      />
      {label}
    </label>
  );
}

function DonutGauge({ percent }: { percent: number }) {
  const radius = 34;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percent / 100) * circumference;

  return (
    <div className="relative w-20 h-20 shrink-0">
      <svg viewBox="0 0 80 80" className="w-20 h-20 -rotate-90">
        <circle cx="40" cy="40" r={radius} fill="none" stroke="#e2e8f0" strokeWidth="8" />
        <circle
          cx="40"
          cy="40"
          r={radius}
          fill="none"
          stroke="#10b981"
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center text-sm font-bold text-slate-800">
        {percent}%
      </div>
    </div>
  );
}

function Starfield() {
  const stars = Array.from({ length: 140 }).map((_, i) => {
    const seed = (i * 9301 + 49297) % 233280;
    const rand = seed / 233280;
    return {
      left: `${(rand * 137.5) % 100}%`,
      top: `${((rand * 971) % 100).toFixed(2)}%`,
      size: 1 + (i % 3),
      delay: `${(i % 5) * 0.6}s`,
    };
  });

  return (
    <div className="fixed inset-0 -z-10 bg-slate-950 overflow-hidden pointer-events-none">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.08),transparent_45%),radial-gradient(circle_at_80%_70%,rgba(16,185,129,0.06),transparent_50%)]" />
      {stars.map((s, i) => (
        <span
          key={i}
          className="absolute rounded-full bg-white animate-pulse"
          style={{
            left: s.left,
            top: s.top,
            width: s.size,
            height: s.size,
            opacity: 0.5,
            animationDelay: s.delay,
            animationDuration: '3s',
          }}
        />
      ))}
    </div>
  );
}
