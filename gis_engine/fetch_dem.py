import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

OPENTOPO_API_KEY = os.environ.get("OPENTOPO_API_KEY")

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def _validate_coordinates(lat: float, lon: float) -> None:
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} is out of range (-90 to 90).")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude {lon} is out of range (-180 to 180).")


def download_dem_data(lat: float, lon: float, buffer: float = 0.02, output_path: str = "data/dem.tif") -> str:
    """
    Downloads SRTM 30m Digital Elevation Model (DEM) data using OpenTopography API.
    Retries on transient network/server errors before giving up.
    """
    if not OPENTOPO_API_KEY:
        raise EnvironmentError(
            "OPENTOPO_API_KEY is not set. Add it to your .env file and load it "
            "before calling download_dem_data()."
        )

    _validate_coordinates(lat, lon)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    south, north = lat - buffer, lat + buffer
    west, east = lon - buffer, lon + buffer

    url = (
        f"https://portal.opentopography.org/API/globaldem?"
        f"demtype=SRTMGL1&south={south}&north={north}&west={west}&east={east}"
        f"&outputFormat=GTiff&API_Key={OPENTOPO_API_KEY}"
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("Downloading DEM for (%s, %s) — attempt %d/%d", lat, lon, attempt, MAX_RETRIES)
        try:
            response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning("Network error on attempt %d: %s", attempt, e)
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            if "json" in content_type or "html" in content_type:
                # OpenTopography sometimes returns 200 with an error payload instead
                # of a real GeoTIFF (e.g. invalid bbox, quota exceeded).
                last_error = Exception(f"Unexpected response content-type '{content_type}': {response.text[:300]}")
                logger.warning(str(last_error))
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            logger.info("DEM successfully saved to %s", output_path)
            return output_path

        elif response.status_code in (429, 500, 502, 503, 504):
            # Rate-limited or transient server error — worth retrying
            last_error = Exception(f"API request failed with status {response.status_code}: {response.text[:300]}")
            logger.warning("Retryable error on attempt %d: %s", attempt, last_error)
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        else:
            # Non-retryable client error (bad request, auth failure, etc.)
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    raise Exception(f"DEM download failed after {MAX_RETRIES} attempts. Last error: {last_error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_lat, test_lon = 18.5204, 73.8567
    download_dem_data(test_lat, test_lon)
