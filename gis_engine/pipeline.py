import os
import sys
import json
import logging

# Ensure the parent directory is added to sys.path so Python finds gis_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gis_engine.fetch_dem import download_dem_data, _validate_coordinates
from gis_engine.hydro_analysis import process_hydrology
try:
    from gis_engine.ndvi_gee import get_mean_ndvi_past_3_years
except ImportError:  # optional; compute_full_watershed() handles None
    get_mean_ndvi_past_3_years = None

logger = logging.getLogger(__name__)


def run_gis_pipeline(lat: float, lon: float, buffer: float = 0.02) -> dict:
    """
    Master function: Takes raw GPS coordinates from geotagged field photos
    and executes the end-to-end GIS workflow.

    Returns the raw process_hydrology() output — useful for local debugging,
    but Member 1 should call compute_full_watershed() below instead, since
    that returns the exact dict shape the backend/API contract expects.
    """
    _validate_coordinates(lat, lon)

    logger.info("STARTING GIS PIPELINE — target coordinates (%s, %s)", lat, lon)

    # 1. Download DEM raster for this location
    dem_path = download_dem_data(lat=lat, lon=lon, buffer=buffer, output_path="data/dem.tif")

    # 2. Extract watershed boundaries, stream vectors, and metrics
    results = process_hydrology(dem_path=dem_path, target_lat=lat, target_lon=lon)

    logger.info("PIPELINE COMPLETE")
    return results


def compute_full_watershed(lat: float, lon: float, buffer: float = 0.02) -> dict:
    """
    THIS is the function Member 1 plugs into app.py, replacing the current
    get_dem_from_earth_engine() flow output, per PROJECT_REPORT.md's spec:

        def compute_full_watershed(lat: float, lon: float) -> dict

    Internally: fetches the DEM (OpenTopography SRTM), runs the full hydrology
    analysis (watershed boundary, stream network, flow direction/accumulation,
    outlet elevation, relief ratio, slope-based erosion risk), fetches mean NDVI
    (past 3 years, via Google Earth Engine), then reshapes everything into the
    required response dict.
    """
    result = run_gis_pipeline(lat=lat, lon=lon, buffer=buffer)

    with open(result["boundary_geojson"], "r") as f:
        boundary_fc = json.load(f)
        boundary_feature = boundary_fc["features"][0] if boundary_fc.get("features") else boundary_fc

    # Mean NDVI (past 3 years) via Google Earth Engine. Wrapped in try/except so a
    # missing project ID, auth issue, or GEE outage doesn't crash the whole pipeline —
    # it just comes back as None, same pattern as the ndvi_timeseries placeholder below.
    try:
        if get_mean_ndvi_past_3_years is None:
            raise ImportError("gis_engine.ndvi_gee is not present")
        ndvi_result = get_mean_ndvi_past_3_years(lat=lat, lon=lon)
        mean_ndvi_3yr = ndvi_result["mean_ndvi"]
    except Exception as e:
        logger.warning("NDVI computation failed (%s). Returning None for mean_ndvi_3yr.", e)
        mean_ndvi_3yr = None

    return {
        "catchment_boundary": {
            "type": "Feature",
            "geometry": boundary_feature["geometry"],
            "properties": {"area_sqkm": result["area_sq_km"]},
        },
        "flow_accumulation_url": result["flow_accumulation_path"],
        "flow_direction_grid": result["flow_direction_path"],
        "outlet_elevation_m": result["outlet_elevation_m"],
        "relief_ratio": result["relief_ratio"],
        # NOTE: ndvi_timeseries (monthly, per report spec) still needs Sentinel Hub —
        # left empty until Member 1 fixes that auth. mean_ndvi_3yr below is a separate,
        # simpler addition computed via Google Earth Engine instead.
        "ndvi_timeseries": [],
        "mean_ndvi_3yr": mean_ndvi_3yr,
        # Slope-based erosion risk proxy (quick estimate from DEM, not full USLE model).
        "mean_slope_deg": result["mean_slope_deg"],
        "erosion_risk_category": result["erosion_risk_category"],
        # Extra field, also referenced in the report's AI-prompt spec for Member 4:
        "flow_direction_dominant": result["dominant_flow_direction"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Test run for local debugging
    test_lat, test_lon = 18.5204, 73.8567
    output = compute_full_watershed(test_lat, test_lon)
    print("Pipeline Output Data Summary:")
    print(json.dumps(output, indent=2, default=str))
