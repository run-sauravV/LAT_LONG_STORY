import os
import math
import logging
import numpy as np
import geopandas as gpd
from pysheds.grid import Grid
from shapely.geometry import Polygon, LineString

logger = logging.getLogger(__name__)

# PySheds default D8 dirmap: (N, NE, E, SE, S, SW, W, NW) -> (64,128,1,2,4,8,16,32)
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
DIR_LABELS = {64: "N", 128: "NE", 1: "E", 2: "SE", 4: "S", 8: "SW", 16: "W", 32: "NW"}

# Config — tune these here instead of hunting for magic numbers in the function body
STREAM_THRESHOLD = 200        # flow-accumulation cell count that defines a "stream"
SMOOTH_BUFFER_DEGREES = 0.0005  # ~50m, used to round jagged raster edges
SIMPLIFY_TOLERANCE_DEGREES = 0.0001


def _dominant_flow_direction(fdir, catchment_mask):
    """Majority D8 direction code within the catchment -> compass label."""
    values = fdir[catchment_mask]
    values = values[values != 0]
    if values.size == 0:
        return None
    codes, counts = np.unique(values, return_counts=True)
    dominant_code = int(codes[np.argmax(counts)])
    return DIR_LABELS.get(dominant_code, None)


def _cell_size_meters(grid, lat):
    """Approximate ground cell size (m) from affine transform, for lat/lon DEMs."""
    a = grid.affine
    deg_per_px_x = abs(a.a)
    deg_per_px_y = abs(a.e)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    cell_x_m = deg_per_px_x * m_per_deg_lon
    cell_y_m = deg_per_px_y * m_per_deg_lat
    return (cell_x_m + cell_y_m) / 2.0


def _slope_degrees(dem, cell_size_m):
    """
    Slope in degrees at every cell, computed from elevation gradient.
    NOTE: this is a simple proxy (slope steepness only) — it does NOT account for
    rainfall, soil type, or vegetation cover, so treat the result as a relative
    risk indicator, not a calibrated soil-loss estimate (that would need the
    full USLE model instead).
    """
    dy, dx = np.gradient(dem.astype(float), cell_size_m, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    return np.degrees(slope_rad)


def _erosion_risk_category(mean_slope_deg):
    """Bucket mean slope (degrees) into a qualitative erosion-risk label."""
    if mean_slope_deg is None:
        return None
    if mean_slope_deg < 5:
        return "Low"
    elif mean_slope_deg < 15:
        return "Moderate"
    elif mean_slope_deg < 25:
        return "High"
    else:
        return "Severe"


def process_hydrology(dem_path: str = "data/dem.tif", target_lat: float = 18.5204, target_lon: float = 73.8567):
    """
    Reads elevation data, calculates flow vectors, extracts smoothed catchment boundaries
    and stream networks, and saves output as clean GeoJSON + raster files.

    Returns a dict with everything compute_full_watershed() needs to assemble the
    final API response (see bottom of file).
    """
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM raster file not found at {dem_path}")

    os.makedirs("data", exist_ok=True)

    logger.info("Loading elevation grid into PySheds...")
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    # Fail fast with a clear message if the target point is outside the DEM's extent —
    # otherwise snap_to_mask/catchment can produce confusing downstream errors instead.
    left, bottom, right, top = grid.bbox
    if not (left <= target_lon <= right and bottom <= target_lat <= top):
        raise ValueError(
            f"Target coordinate ({target_lat}, {target_lon}) falls outside the DEM's "
            f"bounds ({bottom:.4f} to {top:.4f} lat, {left:.4f} to {right:.4f} lon). "
            f"Check the DEM was downloaded for the right location/buffer."
        )

    # 1. Condition the elevation terrain (fill depressions/pits)
    logger.info("Conditioning terrain (filling depressions)...")
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    conditioned_dem = grid.resolve_flats(flooded)

    # 2. Compute D8 Flow Direction
    logger.info("Computing D8 flow directions...")
    fdir = grid.flowdir(conditioned_dem, dirmap=DIRMAP)

    # 3. Calculate Flow Accumulation
    logger.info("Calculating flow accumulation...")
    acc = grid.accumulation(fdir, dirmap=DIRMAP)

    # 3b. Save flow direction + flow accumulation rasters (needed for API output)
    logger.info("Saving flow direction and flow accumulation rasters...")
    flowdir_path = "data/flow_direction.tif"
    flowacc_path = "data/flow_accumulation.tif"
    grid.to_raster(fdir, flowdir_path)
    grid.to_raster(acc, flowacc_path)
    logger.info("Flow direction saved to: %s", flowdir_path)
    logger.info("Flow accumulation saved to: %s", flowacc_path)

    # 4. Snap target coordinate to main stream channel
    logger.info("Snapping coordinates (%s, %s) to major flow path...", target_lat, target_lon)
    x_snap, y_snap = grid.snap_to_mask(acc > STREAM_THRESHOLD, (target_lon, target_lat))

    # 5. Extract Watershed Catchment Boundary
    logger.info("Extracting watershed catchment polygon...")
    catchment = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype="coordinate", dirmap=DIRMAP)

    # Keep an unclipped copy of catchment mask (same shape as original dem) for
    # elevation / flow-direction lookups before we clip the grid view below.
    catchment_mask_full = catchment.astype(bool)

    # 5b. Outlet elevation, relief ratio, dominant flow direction — MUST run before
    # grid.clip_to() below, since clip_to() resizes the grid's active view, and dem/fdir
    # here are still full-extent arrays. Running these after clip_to() causes a shape
    # mismatch between catchment_mask_full (full extent) and outputs like
    # distance_to_outlet() (which would use the now-clipped extent).
    logger.info("Computing outlet elevation...")
    row, col = grid.nearest_cell(x_snap, y_snap)
    outlet_elevation_m = float(dem[row, col])

    logger.info("Computing relief ratio...")
    catchment_elevations = dem[catchment_mask_full]
    max_elevation_m = float(np.max(catchment_elevations)) if catchment_elevations.size else outlet_elevation_m

    # Longest flow path: distance (in cell-steps) from the farthest cell to the outlet,
    # converted to meters using the DEM's approximate ground cell size.
    dist = grid.distance_to_outlet(x=x_snap, y=y_snap, fdir=fdir, xytype="coordinate", dirmap=DIRMAP)
    dist_in_catchment = dist[catchment_mask_full]
    dist_in_catchment = dist_in_catchment[np.isfinite(dist_in_catchment)]
    longest_path_cells = float(np.max(dist_in_catchment)) if dist_in_catchment.size else 0.0
    cell_size_m = _cell_size_meters(grid, target_lat)
    longest_path_m = longest_path_cells * cell_size_m

    relief_ratio = float((max_elevation_m - outlet_elevation_m) / longest_path_m) if longest_path_m > 0 else None

    # Dominant flow direction (compass label) across the catchment
    dominant_direction = _dominant_flow_direction(fdir, catchment_mask_full)

    # Slope-based erosion risk proxy (Option A: quick estimate, no external data).
    # Reuses the DEM and cell size already computed above — no new API calls.
    logger.info("Computing slope-based erosion risk...")
    slope_deg_grid = _slope_degrees(dem, cell_size_m)
    catchment_slopes = slope_deg_grid[catchment_mask_full]
    mean_slope_deg = float(np.mean(catchment_slopes)) if catchment_slopes.size else None
    erosion_risk_category = _erosion_risk_category(mean_slope_deg)

    # Clip view to catchment
    grid.clip_to(catchment)
    catchment_view = grid.view(catchment)

    # 6. Convert Catchment Raster to Polygons
    logger.info("Converting catchment raster to smoothed vector polygon...")
    catchment_int = catchment_view.astype(np.int32)
    shapes = grid.polygonize(catchment_int)

    polygons = [Polygon(shape["coordinates"][0]) for shape, val in shapes if val != 0]

    if not polygons:
        raise ValueError("No valid watershed polygon could be extracted.")

    gdf_boundary = gpd.GeoDataFrame({"geometry": polygons}, crs=grid.crs)

    # 7. Smooth Jagged Raster Edges (Chaikin / Morphological Buffer Smoothing)
    logger.info("Applying morphological smoothing to watershed polygon...")

    smoothed_geometries = []
    for geom in gdf_boundary["geometry"]:
        buffered = geom.buffer(SMOOTH_BUFFER_DEGREES, join_style=1).buffer(-SMOOTH_BUFFER_DEGREES, join_style=1)
        simplified = buffered.simplify(SIMPLIFY_TOLERANCE_DEGREES, preserve_topology=True)
        smoothed_geometries.append(simplified)

    gdf_boundary["geometry"] = smoothed_geometries

    # 8. Calculate Area in Square Kilometers
    gdf_projected = gdf_boundary.to_crs(epsg=3857)
    area_sq_km = float(gdf_projected.geometry.area.sum() / 1e6)

    # Export Watershed Boundary GeoJSON
    boundary_geojson = "data/watershed_boundary.geojson"
    gdf_boundary.to_file(boundary_geojson, driver="GeoJSON")
    logger.info("Smoothed Watershed Boundary saved to: %s", boundary_geojson)

    # 9. Extract Stream Network Vector Lines
    logger.info("Extracting stream network vectors...")
    branches = grid.extract_river_network(fdir, acc > STREAM_THRESHOLD, dirmap=DIRMAP)
    lines = [LineString(branch["geometry"]["coordinates"]) for branch in branches["features"]]

    streams_geojson = "data/stream_network.geojson"
    if lines:
        gdf_streams = gpd.GeoDataFrame({"geometry": lines}, crs=grid.crs)
        gdf_streams.to_file(streams_geojson, driver="GeoJSON")
        logger.info("Stream Network saved to: %s", streams_geojson)
    else:
        logger.warning("No stream lines extracted — streams_geojson will not be created.")

    logger.info("HYDROLOGY METRICS — area=%.2f sq km, outlet_elev=%.2f m, max_elev=%.2f m, "
                "relief_ratio=%s, dominant_dir=%s, mean_slope=%s deg, erosion_risk=%s",
                area_sq_km, outlet_elevation_m, max_elevation_m, relief_ratio,
                dominant_direction, mean_slope_deg, erosion_risk_category)

    return {
        "area_sq_km": area_sq_km,
        "snapped_coords": (y_snap, x_snap),
        "boundary_geojson": boundary_geojson,
        "streams_geojson": streams_geojson,
        "flow_direction_path": flowdir_path,
        "flow_accumulation_path": flowacc_path,
        "outlet_elevation_m": outlet_elevation_m,
        "max_elevation_m": max_elevation_m,
        "relief_ratio": relief_ratio,
        "dominant_flow_direction": dominant_direction,
        "mean_slope_deg": mean_slope_deg,
        "erosion_risk_category": erosion_risk_category,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output = process_hydrology("data/dem.tif", 18.5204, 73.8567)
    print(output)
