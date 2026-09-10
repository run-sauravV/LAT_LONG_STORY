import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import uuid
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import piexif
from dotenv import load_dotenv
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
EE_PROJECT_ID = os.getenv("EE_PROJECT_ID", "")
EE_SERVICE_KEY_JSON = os.getenv("EE_SERVICE_KEY_JSON", "")
OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY", "")

WATER_MODEL_PATH = os.getenv(
    "WATER_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "water_best.pt")
)
VEGETATION_MODEL_PATH = os.getenv(
    "VEGETATION_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "best.pt")
)

UPLOADS = {}
RESULTS = {}

_EE_READY = None

def _ee_init():
    """Initialize Earth Engine once: service-account key (servers) or user creds (laptop)."""
    global _EE_READY
    if _EE_READY is not None:
        return _EE_READY
    try:
        import ee
        if EE_SERVICE_KEY_JSON:
            import json
            from google.oauth2 import service_account
            info = json.loads(EE_SERVICE_KEY_JSON)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/earthengine"]
            )
            ee.Initialize(credentials=creds, project=EE_PROJECT_ID or info.get("project_id"))
            logger.info("Earth Engine: service-account auth OK")
        else:
            ee.Initialize(project=EE_PROJECT_ID)
            logger.info("Earth Engine: default auth OK")
        _EE_READY = True
    except Exception as e:
        logger.warning(f"Earth Engine init failed: {e}")
        _EE_READY = False
    return _EE_READY

class ExifData(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude: Optional[float] = None
    camera_pitch: Optional[float] = None
    camera_heading: Optional[float] = None
    timestamp: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    upload_id: str
    exif: ExifData
    message: str

class ResultsResponse(BaseModel):
    upload_id: str
    status: str
    cv_output: Optional[Dict[str, Any]] = None
    gis_output: Optional[Dict[str, Any]] = None
    processing_time_seconds: int
    error: Optional[str] = None

def extract_exif(file_bytes: bytes) -> Optional[dict]:
    try:
        exif_dict = piexif.load(file_bytes)
        gps = exif_dict.get("GPS", {})
        
        if piexif.GPSIFD.GPSLatitude in gps and piexif.GPSIFD.GPSLongitude in gps:
            lat_data = gps[piexif.GPSIFD.GPSLatitude]
            lat = float(lat_data[0][0]) / lat_data[0][1] + \
                  float(lat_data[1][0]) / (lat_data[1][1] * 60) + \
                  float(lat_data[2][0]) / (lat_data[2][1] * 3600)
            if gps.get(piexif.GPSIFD.GPSLatitudeRef) == b'S':
                lat = -lat
            
            lon_data = gps[piexif.GPSIFD.GPSLongitude]
            lon = float(lon_data[0][0]) / lon_data[0][1] + \
                  float(lon_data[1][0]) / (lon_data[1][1] * 60) + \
                  float(lon_data[2][0]) / (lon_data[2][1] * 3600)
            if gps.get(piexif.GPSIFD.GPSLongitudeRef) == b'W':
                lon = -lon
            
            altitude = None
            if piexif.GPSIFD.GPSAltitude in gps:
                alt_data = gps[piexif.GPSIFD.GPSAltitude]
                altitude = float(alt_data[0]) / alt_data[1]
            
            return {
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "altitude": round(altitude, 2) if altitude else None,
                "camera_pitch": None,
                "camera_heading": None,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.warning(f"EXIF extraction failed: {e}")
    
    return None

def get_dem_from_earth_engine(lat: float, lon: float) -> Dict[str, Any]:
    try:
        import ee
        import math
        import numpy as np

        if not _ee_init():
            raise RuntimeError("Earth Engine unavailable")

        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(5000).bounds()

        dem = ee.Image('USGS/SRTMGL1_003')
        terrain = ee.Terrain.products(dem)

        # Elevation, slope, aspect at the capture point
        center_stats = terrain.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(90),
            scale=30
        ).getInfo()

        center_elev = center_stats.get('elevation')
        dem_slope_deg = center_stats.get('slope')

        # Flow direction: sample 8 neighbors and find steepest downhill
        step = 0.0009
        offsets = {
            "N": (0,step), "NE": (step,step), "E": (step,0), "SE": (step,-step),
            "S": (0,-step), "SW": (-step,-step), "W": (-step,0), "NW": (-step,step)
        }

        drops = {}
        for direction, (dx, dy) in offsets.items():
            neighbor = ee.Geometry.Point([lon+dx, lat+dy])
            elev = dem.sample(neighbor, 30).first().get('elevation').getInfo()
            drops[direction] = (center_elev or 0) - (elev or 0)

        flow_direction = max(drops, key=drops.get)
        steepest_drop = drops[flow_direction]

        # Real watershed boundary from HydroBASINS (precomputed drainage basins)
        catchment_boundary = None
        catchment_area_sqkm = None
        catchment_geom = None
        try:
            for level in [9, 8, 7]:
                basins = ee.FeatureCollection(f'WWF/HydroSHEDS/v1/Basins/hybas_{level}')
                basin = basins.filterBounds(point).first()
                basin_info = basin.getInfo()
                if basin_info and basin_info.get('geometry'):
                    geom = basin.geometry()
                    area_sqkm = round(geom.area().getInfo() / 1e6, 2)
                    catchment_boundary = {
                        "type": "Feature",
                        "geometry": basin_info["geometry"],
                        "properties": {"area_sqkm": area_sqkm, "hydrobasins_level": level}
                    }
                    catchment_area_sqkm = area_sqkm
                    catchment_geom = geom
                    logger.info(f"Got real HydroBASINS polygon (level {level}), area={area_sqkm} sqkm")
                    break
        except Exception as e:
            logger.warning(f"HydroBASINS lookup failed: {e}")

        if catchment_area_sqkm is None:
            # Fallback estimate only if HydroBASINS lookup failed entirely
            avg_slope_pct = abs(steepest_drop) / 100.0
            catchment_area_sqkm = round(max(5.0, 100.0 / (1 + avg_slope_pct * 10)), 1)

        # Erosion risk from slope
        slope_val = dem_slope_deg or 0
        if slope_val < 5: erosion_risk = "Low"
        elif slope_val < 15: erosion_risk = "Moderate"
        elif slope_val < 25: erosion_risk = "High"
        else: erosion_risk = "Severe"

        # Relief ratio from regional stats
        region_stats = dem.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        ).getInfo()

        max_elev = region_stats.get('elevation_max', center_elev)
        min_elev = region_stats.get('elevation_min', center_elev)
        elev_range = (max_elev or 0) - (min_elev or 0)
        relief_ratio = round(elev_range / 10000, 4) if elev_range else None

        # Whole-watershed DEM stats clipped to the HydroBASINS polygon.
        # Coarse scale (150 m): level-7 basins can span thousands of sqkm,
        # so 30 m sampling would be slow and exceed pixel limits.
        catchment_elev_min = None
        catchment_elev_max = None
        catchment_elev_mean = None
        catchment_slope_mean = None
        catchment_relief_m = None
        catchment_relief_ratio = None
        if catchment_geom is not None:
            try:
                c_stats = dem.reduceRegion(
                    reducer=ee.Reducer.minMax().combine(
                        ee.Reducer.mean(), sharedInputs=True
                    ),
                    geometry=catchment_geom,
                    scale=150,
                    maxPixels=1e9
                ).getInfo()
                catchment_elev_min = c_stats.get('elevation_min')
                catchment_elev_max = c_stats.get('elevation_max')
                catchment_elev_mean = c_stats.get('elevation_mean')
                c_slope = terrain.select('slope').reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=catchment_geom,
                    scale=150,
                    maxPixels=1e9
                ).getInfo()
                catchment_slope_mean = c_slope.get('slope_mean')
                if catchment_elev_min is not None and catchment_elev_max is not None:
                    catchment_elev_min = round(float(catchment_elev_min), 1)
                    catchment_elev_max = round(float(catchment_elev_max), 1)
                    catchment_relief_m = round(catchment_elev_max - catchment_elev_min, 1)
                if catchment_elev_mean is not None:
                    catchment_elev_mean = round(float(catchment_elev_mean), 1)
                if catchment_slope_mean is not None:
                    catchment_slope_mean = round(float(catchment_slope_mean), 1)
                # Relief ratio = relief / characteristic basin length,
                # with sqrt(area) as the length proxy (documented estimate).
                if catchment_relief_m and catchment_area_sqkm:
                    basin_length_m = (catchment_area_sqkm * 1e6) ** 0.5
                    catchment_relief_ratio = round(catchment_relief_m / basin_length_m, 4)
                logger.info(
                    f"Catchment DEM: min={catchment_elev_min}m, max={catchment_elev_max}m, "
                    f"mean={catchment_elev_mean}m, relief={catchment_relief_m}m"
                )
            except Exception as e:
                logger.warning(f"Catchment DEM stats failed: {e}")

        # Real color-coded elevation tile layer for the map (genuine GEE raster, not decorative).
        # Clipped to the whole watershed polygon when available, else the local square.
        elevation_tile_url = None
        try:
            if catchment_elev_min is not None and catchment_elev_max is not None:
                vis_min, vis_max = catchment_elev_min, catchment_elev_max
            else:
                vis_min = min_elev if min_elev is not None else 0
                vis_max = max_elev if max_elev is not None else (vis_min + 500)
            clip_geom = catchment_geom if catchment_geom is not None else region
            dem_vis = dem.clip(clip_geom)
            map_id = dem_vis.getMapId({
                "min": vis_min,
                "max": vis_max,
                "palette": ["0000FF", "00FFFF", "00FF00", "FFFF00", "FF0000"]
            })
            elevation_tile_url = map_id["tile_fetcher"].url_format
        except Exception as e:
            logger.warning(f"Elevation tile layer generation failed: {e}")

        logger.info(f"GEE: elev={center_elev}m, slope={dem_slope_deg}°, flow={flow_direction}, risk={erosion_risk}")

        return {
            "elevation_m": round(float(center_elev), 1) if center_elev else None,
            "outlet_elevation_m": round(float(min_elev), 1) if min_elev else None,
            "max_elevation_m": round(float(max_elev), 1) if max_elev else None,
            "dem_source": "USGS SRTM GL1 (Google Earth Engine)",
            "resolution_m": 30,
            "flow_direction": flow_direction,
            "catchment_area_sqkm": catchment_area_sqkm,
            "catchment_boundary": catchment_boundary,
            "elevation_tile_url": elevation_tile_url,
            "steepest_drop_m": round(steepest_drop, 2),
            "dem_slope_deg": round(float(dem_slope_deg), 1) if dem_slope_deg else None,
            "erosion_risk_category": erosion_risk,
            "relief_ratio": relief_ratio,
            "catchment_elev_min_m": catchment_elev_min,
            "catchment_elev_max_m": catchment_elev_max,
            "catchment_elev_mean_m": catchment_elev_mean,
            "catchment_slope_mean_deg": catchment_slope_mean,
            "catchment_relief_m": catchment_relief_m,
            "catchment_relief_ratio": catchment_relief_ratio
        }

    except Exception as e:
        logger.warning(f"Earth Engine failed: {e}. Using mock data.")
        return {
            "elevation_m": 234.5,
            "outlet_elevation_m": None,
            "max_elevation_m": None,
            "dem_source": "Mock (Earth Engine unavailable)",
            "resolution_m": 30,
            "flow_direction": "S",
            "catchment_area_sqkm": 45.3,
            "catchment_boundary": None,
            "elevation_tile_url": None,
            "steepest_drop_m": None,
            "dem_slope_deg": None,
            "erosion_risk_category": None,
            "relief_ratio": None,
            "catchment_elev_min_m": None,
            "catchment_elev_max_m": None,
            "catchment_elev_mean_m": None,
            "catchment_slope_mean_deg": None,
            "catchment_relief_m": None,
            "catchment_relief_ratio": None
        }

def get_satellite_ndvi(lat: float, lon: float) -> Dict[str, Any]:
    try:
        import ee
        import numpy as np
        import base64
        from PIL import Image as PILImage

        if not _ee_init():
            raise RuntimeError("Earth Engine unavailable")

        point = ee.Geometry.Point([lon, lat])
        bbox = point.buffer(2000).bounds()

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(bbox)
            .filterDate("2026-01-01", "2026-09-07")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        image = collection.first()

        # Real NDVI at point
        ndvi_image = image.normalizedDifference(["B8", "B4"])
        ndvi_value = ndvi_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=bbox,
            scale=10
        ).get("nd").getInfo()

        # True color thumbnail (RGB)
        viz_params = {"min": 0, "max": 3000, "bands": ["B4", "B3", "B2"]}
        thumb_url = image.getThumbURL({
            "min": 0,
            "max": 3000,
            "bands": ["B4", "B3", "B2"],
            "region": bbox,
            "dimensions": 512,
            "format": "png"
        })

        thumb_resp = requests.get(thumb_url, timeout=30)
        true_color_base64 = None

        if thumb_resp.status_code == 200:
            pil_img = PILImage.open(BytesIO(thumb_resp.content))
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            true_color_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            logger.info(f"Got real satellite image from GEE for ({lat},{lon})")

        ndvi_mean = round(float(ndvi_value), 3) if ndvi_value is not None else None
        logger.info(f"Got real NDVI {ndvi_mean} from GEE for ({lat},{lon})")

        return {
            "ndvi_mean": ndvi_mean,
            "true_color_base64": true_color_base64,
            "source": "Sentinel-2 via Google Earth Engine",
            "date": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.warning(f"GEE satellite fetch failed: {e}. Using mock data.")
        return {
            "ndvi_mean": None,
            "true_color_base64": None,
            "source": "Sentinel-2 (mock fallback)",
            "date": datetime.utcnow().isoformat()
        }

def get_weather_data(lat: float, lon: float) -> Dict[str, Any]:
    if not OPENWEATHER_API_KEY:
        logger.info("OpenWeather API key missing, returning mock weather")
        return {
            "temperature_c": 25,
            "humidity_percent": 65,
            "rainfall_mm": 0,
            "source": "Mock"
        }
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        rainfall = data.get('rain', {}).get('1h', 0)
        
        logger.info(f"Got weather data for ({lat}, {lon})")
        return {
            "temperature_c": data['main']['temp'],
            "humidity_percent": data['main']['humidity'],
            "rainfall_mm": rainfall,
            "source": "OpenWeatherMap"
        }
    except Exception as e:
        logger.warning(f"OpenWeather failed: {e}. Using mock weather.")
        return {
            "temperature_c": 25,
            "humidity_percent": 65,
            "rainfall_mm": 0,
            "source": "Mock (OpenWeather failed)"
        }

def segment_photo(photo_bytes: bytes, exif: dict) -> Dict[str, Any]:
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(BytesIO(photo_bytes)).convert("RGB")
        img_array = np.array(img).astype(float)
        height, width = img_array.shape[:2]

        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        exg = 2 * g - r - b
        vegetation_mask = exg > 15
        vegetation_density = round(float(vegetation_mask.mean()), 3)

        gray = img_array.mean(axis=2)
        vertical_gradient = np.abs(np.diff(gray, axis=0))
        slope_estimate = round(float(vertical_gradient.mean()) * 2, 1)

        water_bodies = []
        erosion_zones = []
        vegetation_detections = []
        status_parts = []

        # --- Real trained model: vegetation vs bare/worked ground (detect task) ---
        if os.path.exists(VEGETATION_MODEL_PATH):
            try:
                from ultralytics import YOLO
                import cv2

                veg_model = YOLO(VEGETATION_MODEL_PATH)
                veg_class_names = veg_model.names  # expected: {0: 'vegetation', 1: 'works_bare'}

                bare_class_ids = {
                    cid for cid, name in veg_class_names.items()
                    if any(k in name.lower() for k in ["bare", "works", "erosion"])
                }
                veg_class_ids = {
                    cid for cid, name in veg_class_names.items()
                    if "veget" in name.lower()
                }

                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                veg_results = veg_model.predict(
                    source=img_bgr, conf=0.25, imgsz=640, device="cpu", verbose=False
                )
                veg_result = veg_results[0]

                if veg_result.boxes is not None:
                    boxes = veg_result.boxes
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    for i, cid in enumerate(class_ids):
                        x1, y1, x2, y2 = xyxy[i]
                        box_area_pixels = int((x2 - x1) * (y2 - y1))
                        # Rough estimate only — no camera calibration (focal length/distance)
                        # is available from a handheld photo, so this is not a precise ground area.
                        box_area_sqm_est = round(box_area_pixels * 0.09, 1)

                        if cid in bare_class_ids:
                            erosion_zones.append({
                                "label": veg_class_names[cid],
                                "confidence": round(float(confs[i]), 2),
                                "area_pixels": box_area_pixels,
                                "area_sqm_estimated": box_area_sqm_est,
                                "note": "Bare/worked ground detected by trained model — used as an erosion-susceptibility proxy, not a confirmed erosion classification."
                            })
                        elif cid in veg_class_ids:
                            vegetation_detections.append({
                                "confidence": round(float(confs[i]), 2),
                                "area_pixels": box_area_pixels,
                                "area_sqm_estimated": box_area_sqm_est
                            })

                status_parts.append(
                    f"Vegetation/bare-ground model: {len(vegetation_detections)} vegetation box(es), "
                    f"{len(erosion_zones)} bare-ground box(es) (task=detect, not segmentation)."
                )
                logger.info(status_parts[-1])

            except Exception as e:
                status_parts.append(f"Vegetation/bare-ground model failed: {e}")
                logger.warning(status_parts[-1])
        else:
            status_parts.append("No vegetation/bare-ground model found — using pixel-based (ExG) vegetation estimate only.")

        # --- Water detection model (separate file, expects a 'water' class) ---
        if os.path.exists(WATER_MODEL_PATH):
            try:
                from ultralytics import YOLO
                import cv2

                model = YOLO(WATER_MODEL_PATH)
                class_names = model.names  # {0: 'water', ...} for a properly trained model
                water_class_ids = {
                    cid for cid, name in class_names.items()
                    if "water" in name.lower()
                }

                if not water_class_ids:
                    # Model has no water class at all (e.g. generic COCO weights) —
                    # say so plainly instead of silently mislabeling other objects as water
                    msg = (
                        f"Model at {os.path.basename(WATER_MODEL_PATH)} has no 'water' class "
                        f"(classes: {list(class_names.values())[:6]}...). Not a water-detection model — "
                        f"skipping water body detection rather than reporting false positives."
                    )
                    status_parts.append(msg)
                    logger.warning(msg)
                else:
                    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    results = model.predict(
                        source=img_bgr, conf=0.30, imgsz=512, device="cpu", verbose=False
                    )
                    result = results[0]

                    if result.masks is not None and result.boxes is not None:
                        masks = result.masks.data.cpu().numpy()
                        boxes = result.boxes
                        class_ids = boxes.cls.cpu().numpy().astype(int)

                        for i, mask in enumerate(masks):
                            if class_ids[i] not in water_class_ids:
                                continue  # skip anything that isn't actually water
                            mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                            area_pixels = int((mask_resized > 0.5).sum())
                            area_sqm = round(area_pixels * 0.09, 1)
                            confidence = float(boxes.conf[i])
                            water_bodies.append({
                                "confidence": round(confidence, 2),
                                "area_sqm": area_sqm,
                                "area_pixels": area_pixels
                            })

                    msg = f"Water-trained model: {len(water_bodies)} water body(ies) found."
                    status_parts.append(msg)
                    logger.info(msg)

            except Exception as e:
                msg = f"Water model inference failed: {e}"
                status_parts.append(msg)
                logger.warning(msg)
        else:
            status_parts.append(
                f"No water-detection model found at {os.path.basename(WATER_MODEL_PATH)} — water body detection unavailable."
            )

        cv_model_status = " | ".join(status_parts)

        return {
            "erosion_zones": erosion_zones,
            "water_bodies": water_bodies,
            "vegetation_detections": vegetation_detections,
            "vegetation_density": vegetation_density,
            "slope_estimate": slope_estimate,
            "cv_model_status": cv_model_status
        }

    except Exception as e:
        logger.warning(f"Photo analysis failed: {e}. Using fallback values.")
        return {
            "erosion_zones": [],
            "water_bodies": [],
            "vegetation_detections": [],
            "vegetation_density": 0.5,
            "slope_estimate": 10.0,
            "cv_model_status": "analysis failed, fallback values used"
        }

def compute_runoff_coefficient(slope_deg: float, vegetation: float, rainfall: float) -> float:
    base_cn = 70

    if slope_deg > 20:
        base_cn += 10

    cn = base_cn + (1 - vegetation) * 5

    if rainfall > 50:
        cn += 5

    runoff = (cn - 30) / 70
    return round(min(max(runoff, 0), 1), 2)

def process_upload_background(upload_id: str, photo_bytes: bytes, exif_data: dict):
    try:
        start_time = datetime.utcnow()
        
        cv_result = segment_photo(photo_bytes, exif_data)
        gis_result = get_dem_from_earth_engine(exif_data["latitude"], exif_data["longitude"])
        weather = get_weather_data(exif_data["latitude"], exif_data["longitude"])
        satellite = get_satellite_ndvi(exif_data["latitude"], exif_data["longitude"])

        # Prefer real DEM slope; fall back to the photo-brightness proxy only if DEM failed
        slope_for_runoff = gis_result.get("dem_slope_deg")
        if slope_for_runoff is None:
            slope_for_runoff = cv_result.get("slope_estimate", 10.0)

        runoff = compute_runoff_coefficient(
            slope_for_runoff,
            cv_result.get("vegetation_density", 0.65),
            weather.get("rainfall_mm", 0)
        )
        
        gis_result["runoff_coefficient"] = runoff
        gis_result["weather"] = weather
        gis_result["satellite"] = satellite
        
        end_time = datetime.utcnow()
        processing_time = int((end_time - start_time).total_seconds())
        
        RESULTS[upload_id] = {
            "cv": cv_result,
            "gis": gis_result,
            "completed": end_time.isoformat(),
            "processing_time_seconds": processing_time,
            "error": None
        }
        
        logger.info(f"Upload {upload_id} completed in {processing_time}s")
    except Exception as e:
        logger.error(f"Background processing failed for {upload_id}: {e}")
        RESULTS[upload_id] = {
            "cv": None,
            "gis": None,
            "completed": datetime.utcnow().isoformat(),
            "processing_time_seconds": 0,
            "error": str(e)
        }

app = FastAPI(
    title="Watershed Backend",
    version="1.0.0",
    description="Field photo analysis - CV + GIS pipeline"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "uploads": len(UPLOADS),
        "processed": len(RESULTS)
    }

@app.post("/api/v1/upload", response_model=UploadResponse, status_code=201)
async def upload_photo(background_tasks: BackgroundTasks, photo: UploadFile = File(...)):
    file_bytes = await photo.read()
    
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    
    exif_data = extract_exif(file_bytes)
    
    if not exif_data:
        raise HTTPException(status_code=400, detail="No GPS data in photo EXIF")
    
    upload_id = str(uuid.uuid4())
    UPLOADS[upload_id] = {
        "exif": exif_data,
        "filename": photo.filename,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    RESULTS[upload_id] = {
        "cv": None,
        "gis": None,
        "completed": None,
        "processing_time_seconds": 0,
        "error": None
    }
    
    background_tasks.add_task(process_upload_background, upload_id, file_bytes, exif_data)
    
    logger.info(f"Upload {upload_id} accepted")
    
    return UploadResponse(
        success=True,
        upload_id=upload_id,
        exif=ExifData(**exif_data),
        message="Photo uploaded. Processing started."
    )

@app.get("/api/v1/results/{upload_id}", response_model=ResultsResponse)
async def get_results(upload_id: str):
    if upload_id not in UPLOADS:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    result = RESULTS.get(upload_id)
    
    if not result:
        return ResultsResponse(
            upload_id=upload_id,
            status="processing",
            cv_output=None,
            gis_output=None,
            processing_time_seconds=0
        )
    
    if result.get("error"):
        return ResultsResponse(
            upload_id=upload_id,
            status="failed",
            cv_output=None,
            gis_output=None,
            processing_time_seconds=result.get("processing_time_seconds", 0),
            error=result["error"]
        )
    
    if result.get("completed") is None:
        return ResultsResponse(
            upload_id=upload_id,
            status="processing",
            cv_output=None,
            gis_output=None,
            processing_time_seconds=0
        )
    
    return ResultsResponse(
        upload_id=upload_id,
        status="complete",
        cv_output=result["cv"],
        gis_output=result["gis"],
        processing_time_seconds=result["processing_time_seconds"]
    )

@app.get("/api/v1/uploads")
async def list_uploads():
    return {
        "count": len(UPLOADS),
        "uploads": [
            {
                "id": uid,
                "exif": UPLOADS[uid]["exif"],
                "status": "complete" if RESULTS[uid].get("completed") else "processing"
            }
            for uid in UPLOADS
        ]
    }

@app.delete("/api/v1/uploads/{upload_id}")
async def delete_upload(upload_id: str):
    if upload_id not in UPLOADS:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    UPLOADS.pop(upload_id, None)
    RESULTS.pop(upload_id, None)
    
    return {"success": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)