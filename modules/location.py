from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_USER_AGENT    = os.environ.get("OSM_USER_AGENT",    "HazardRiskAssessmentSystem/1.0")
_NOMINATIM_URL = os.environ.get("OSM_NOMINATIM_URL", "https://nominatim.openstreetmap.org")
_OVERPASS_URL  = os.environ.get("OSM_OVERPASS_URL",  "https://overpass-api.de/api/interpreter")
_TIMEOUT       = int(os.environ.get("OSM_TIMEOUT",       "10"))
_RADIUS_M      = int(os.environ.get("OSM_SEARCH_RADIUS", "2000"))
_STRICT        = os.environ.get("OSM_STRICT", "0") == "1"
_COUNTRY_HINT  = os.environ.get("OSM_COUNTRY_HINT", "Philippines")

# Retry configuration for transient failures (timeouts, connection errors,
# 5xx responses).  429 rate-limit responses are handled separately with a
# fixed 5 s back-off and do not consume a retry attempt.
_MAX_RETRIES      = int(os.environ.get("OSM_MAX_RETRIES",       "2"))
_RETRY_BACKOFF_S  = float(os.environ.get("OSM_RETRY_BACKOFF_S", "1.5"))

# Headers sent on every request.  Content-Type is intentionally omitted here
# because it is only valid on POST requests (Overpass).  Setting it globally
# on GET requests (Nominatim) is technically incorrect and may confuse some
# proxies.  _get() adds it only when method="POST".
_BASE_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept":     "application/json",
}

# Nominatim requires ≥1 s between requests from the same IP.
_last_nominatim_call: float = 0.0
_MIN_NOMINATIM_INTERVAL = 1.1   # seconds


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GeoPoint:
    """Result of a geocoding or reverse-geocoding call."""
    lat:          float
    lon:          float
    display_name: str                  = ""
    osm_id:       Optional[int]        = None
    osm_type:     Optional[str]        = None
    place_rank:   Optional[int]        = None
    address:      Dict[str, str]       = field(default_factory=dict)
    is_fallback:  bool                 = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lat":          self.lat,
            "lon":          self.lon,
            "display_name": self.display_name,
            "osm_id":       self.osm_id,
            "osm_type":     self.osm_type,
            "place_rank":   self.place_rank,
            "address":      self.address,
            "is_fallback":  self.is_fallback,
        }


@dataclass
class HazardContext:
    """
    OSM-derived spatial context for a coordinate pair.
    All counts/distances are within OSM_SEARCH_RADIUS metres.
    """
    lat:                    float
    lon:                    float
    waterway_count:         int   = 0
    river_count:            int   = 0
    flood_zone:             bool  = False
    flood_zone_tags:        List[str]     = field(default_factory=list)
    river_distance_m:       float = math.inf
    landuse_tags:           List[str]     = field(default_factory=list)
    natural_tags:           List[str]     = field(default_factory=list)
    waterway_names:         List[str]     = field(default_factory=list)
    is_fallback:            bool  = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lat":               self.lat,
            "lon":               self.lon,
            "waterway_count":    self.waterway_count,
            "river_count":       self.river_count,
            "flood_zone":        self.flood_zone,
            "flood_zone_tags":   self.flood_zone_tags,
            "river_distance_m":  self.river_distance_m if not math.isinf(self.river_distance_m) else None,
            "landuse_tags":      self.landuse_tags,
            "natural_tags":      self.natural_tags,
            "waterway_names":    self.waterway_names,
            "is_fallback":       self.is_fallback,
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _rate_limit() -> None:
    """Enforce Nominatim's 1-request-per-second policy."""
    global _last_nominatim_call
    elapsed = time.monotonic() - _last_nominatim_call
    if elapsed < _MIN_NOMINATIM_INTERVAL:
        time.sleep(_MIN_NOMINATIM_INTERVAL - elapsed)
    _last_nominatim_call = time.monotonic()


def _get(url: str, params: Dict[str, Any] | None = None,
         data: str | None = None, method: str = "GET") -> Optional[Any]:
    """
    Thin HTTP wrapper with per-request error handling and retry logic.

    Content-Type is set only for POST requests (Overpass QL).  GET requests
    (Nominatim) do not include Content-Type — it has no meaning on a GET and
    setting it globally was incorrect in the original implementation.

    Retry behaviour:
      - Transient errors (timeout, connection error, 5xx) are retried up to
        _MAX_RETRIES times with exponential back-off starting at
        _RETRY_BACKOFF_S seconds.
      - 429 (rate-limited) gets a fixed 5 s sleep then returns None without
        consuming a retry — the caller decides how to handle it.
      - 4xx errors other than 429 are not retried (they indicate a bad
        request, not a transient failure).

    Returns parsed JSON or None on unrecoverable failure.
    """
    headers = dict(_BASE_HEADERS)
    if method == "POST":
        headers["Content-Type"] = "application/x-www-form-urlencoded"

    last_exc: Optional[Exception] = None

    for attempt in range(1 + _MAX_RETRIES):
        try:
            if method == "POST":
                resp = requests.post(url, data=data, headers=headers, timeout=_TIMEOUT)
            else:
                resp = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)

            if resp.status_code == 429:
                logger.warning("OSM rate-limited (429). Backing off 5 s …")
                time.sleep(5)
                return None

            if resp.status_code >= 500:
                logger.warning(
                    "OSM server error HTTP %d (attempt %d/%d): %s",
                    resp.status_code, attempt + 1, 1 + _MAX_RETRIES, url,
                )
                last_exc = None  # not an exception, but still retryable
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
                continue

            if resp.status_code >= 400:
                logger.warning("OSM client error HTTP %d for %s — not retrying.", resp.status_code, url)
                return None

            return resp.json()

        except requests.exceptions.Timeout as exc:
            logger.warning(
                "OSM request timed out after %ss (attempt %d/%d): %s",
                _TIMEOUT, attempt + 1, 1 + _MAX_RETRIES, url,
            )
            last_exc = exc

        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "OSM connection error (attempt %d/%d): %s",
                attempt + 1, 1 + _MAX_RETRIES, exc,
            )
            last_exc = exc

        except Exception as exc:
            logger.error("Unexpected OSM error: %s", exc)
            return None  # non-transient, don't retry

        if attempt < _MAX_RETRIES:
            sleep_s = _RETRY_BACKOFF_S * (2 ** attempt)
            logger.debug("Retrying in %.1f s …", sleep_s)
            time.sleep(sleep_s)

    logger.warning(
        "OSM request failed after %d attempt(s). Last error: %s",
        1 + _MAX_RETRIES, last_exc,
    )
    return None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in metres between two lat/lon points."""
    R = 6_371_000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _fallback_geo(name: str = "") -> GeoPoint:
    logger.warning("Returning fallback GeoPoint (OSM unavailable).")
    return GeoPoint(lat=0.0, lon=0.0, display_name=name, is_fallback=True)


def _fallback_hazard(lat: float = 0.0, lon: float = 0.0) -> HazardContext:
    logger.warning("Returning fallback HazardContext (OSM unavailable).")
    return HazardContext(lat=lat, lon=lon, is_fallback=True)


# ── Public: Geocoding ─────────────────────────────────────────────────────────

def geocode(location_name: str, country_hint: str = _COUNTRY_HINT) -> GeoPoint:
    """
    Forward-geocode a place name to coordinates using Nominatim.

    Args:
        location_name : e.g. "Barangay 1", "Batasan Hills, Quezon City"
        country_hint  : appended to the query to reduce ambiguity
                        (default: OSM_COUNTRY_HINT env var, fallback "Philippines")

    Returns:
        GeoPoint with lat/lon and address details.
        is_fallback=True if the API was unreachable after retries.

    Raises:
        RuntimeError if OSM_STRICT=1 and the API call fails.
    """
    query = f"{location_name}, {country_hint}" if country_hint else location_name
    logger.debug("Geocoding: '%s'", query)

    _rate_limit()
    result = _get(
        f"{_NOMINATIM_URL}/search",
        params={
            "q":               query,
            "format":          "jsonv2",
            "addressdetails":  1,
            "limit":           1,
            "accept-language": "en",
        },
    )

    if not result:
        if _STRICT:
            raise RuntimeError(f"Nominatim geocode failed for '{query}'")
        return _fallback_geo(location_name)

    hit = result[0]
    address = hit.get("address", {})

    gp = GeoPoint(
        lat          = float(hit["lat"]),
        lon          = float(hit["lon"]),
        display_name = hit.get("display_name", ""),
        osm_id       = hit.get("osm_id"),
        osm_type     = hit.get("osm_type"),
        place_rank   = hit.get("place_rank"),
        address      = address,
    )
    logger.info("Geocoded '%s' → (%.5f, %.5f) [%s]",
                location_name, gp.lat, gp.lon, gp.display_name)
    return gp


def reverse_geocode(lat: float, lon: float) -> GeoPoint:
    """
    Reverse-geocode coordinates to a place name using Nominatim.

    Args:
        lat, lon : WGS-84 decimal degrees

    Returns:
        GeoPoint with address details.
        is_fallback=True if the API was unreachable after retries.
    """
    logger.debug("Reverse geocoding: (%.5f, %.5f)", lat, lon)

    _rate_limit()
    result = _get(
        f"{_NOMINATIM_URL}/reverse",
        params={
            "lat":             lat,
            "lon":             lon,
            "format":          "jsonv2",
            "addressdetails":  1,
            "accept-language": "en",
        },
    )

    if not result or "error" in result:
        if _STRICT:
            raise RuntimeError(f"Nominatim reverse geocode failed for ({lat}, {lon})")
        return _fallback_geo()

    address = result.get("address", {})
    gp = GeoPoint(
        lat          = float(result.get("lat", lat)),
        lon          = float(result.get("lon", lon)),
        display_name = result.get("display_name", ""),
        osm_id       = result.get("osm_id"),
        osm_type     = result.get("osm_type"),
        place_rank   = result.get("place_rank"),
        address      = address,
    )
    logger.info("Reverse geocoded (%.5f, %.5f) → '%s'", lat, lon, gp.display_name)
    return gp


# ── Public: Hazard context via Overpass ───────────────────────────────────────

def get_hazard_context(lat: float, lon: float,
                       radius_m: int = _RADIUS_M) -> HazardContext:
    """
    Query OSM via Overpass for hazard-relevant features near (lat, lon).

    Fetches within `radius_m` metres:
      - Waterways (river, stream, canal, drain, ditch)
      - Natural water bodies (water, wetland)
      - Flood-tagged areas (flood_prone=yes, hazard=flood, etc.)
      - Landuse tags for context (farmland, residential, etc.)

    Args:
        lat, lon   : centre point (WGS-84)
        radius_m   : search radius in metres

    Returns:
        HazardContext dataclass.
        is_fallback=True if the API was unreachable after retries.
    """
    logger.debug("Overpass hazard query: (%.5f, %.5f) r=%dm", lat, lon, radius_m)

    query = f"""
[out:json][timeout:{_TIMEOUT}];
(
  way["waterway"~"river|stream|canal|drain|ditch"]
      (around:{radius_m},{lat},{lon});
  node["waterway"~"river|stream|canal|drain|ditch"]
      (around:{radius_m},{lat},{lon});
  way["natural"~"water|wetland|bay"]
      (around:{radius_m},{lat},{lon});
  way["flood_prone"="yes"]
      (around:{radius_m},{lat},{lon});
  way["hazard"="flood"]
      (around:{radius_m},{lat},{lon});
  way["landuse"]
      (around:{radius_m},{lat},{lon});
  node["natural"~"water|wetland"]
      (around:{radius_m},{lat},{lon});
);
out body center;
""".strip()

    result = _get(_OVERPASS_URL, data=query, method="POST")

    if not result:
        if _STRICT:
            raise RuntimeError(f"Overpass query failed for ({lat}, {lon})")
        return _fallback_hazard(lat, lon)

    elements: List[Dict] = result.get("elements", [])

    ctx = HazardContext(lat=lat, lon=lon)

    waterway_types  = {"river", "stream", "canal", "drain", "ditch"}
    flood_tag_keys  = {"flood_prone", "hazard", "flood_risk", "natural_hazard"}

    seen_waterway_names: set = set()

    for el in elements:
        tags: Dict[str, str] = el.get("tags", {})

        ww = tags.get("waterway", "").lower()
        if ww in waterway_types:
            ctx.waterway_count += 1
            if ww in {"river", "stream"}:
                ctx.river_count += 1

            name = tags.get("name") or tags.get("name:en")
            if name and name not in seen_waterway_names:
                ctx.waterway_names.append(name)
                seen_waterway_names.add(name)

            el_lat = el.get("lat") or el.get("center", {}).get("lat")
            el_lon = el.get("lon") or el.get("center", {}).get("lon")
            if el_lat and el_lon and ww in {"river", "stream"}:
                d = _haversine_m(lat, lon, float(el_lat), float(el_lon))
                if d < ctx.river_distance_m:
                    ctx.river_distance_m = d

        for key in flood_tag_keys:
            val = tags.get(key, "").lower()
            if val in {"yes", "flood", "high", "moderate"}:
                ctx.flood_zone = True
                label = f"{key}={val}"
                if label not in ctx.flood_zone_tags:
                    ctx.flood_zone_tags.append(label)

        nat = tags.get("natural", "").lower()
        if nat and nat not in ctx.natural_tags:
            ctx.natural_tags.append(nat)

        lu = tags.get("landuse", "").lower()
        if lu and lu not in ctx.landuse_tags:
            ctx.landuse_tags.append(lu)

    logger.info(
        "HazardContext (%.5f, %.5f): %d waterways (%d rivers), flood_zone=%s, dist=%.0fm",
        lat, lon,
        ctx.waterway_count, ctx.river_count,
        ctx.flood_zone,
        ctx.river_distance_m if not math.isinf(ctx.river_distance_m) else -1,
    )
    return ctx


# ── Public: Pipeline integration ─────────────────────────────────────────────

def enrich_report(HR: dict, E: dict,
                  country_hint: str = _COUNTRY_HINT) -> Tuple[GeoPoint, HazardContext]:
    """
    Enrich the hazard report and environmental indicators with OSM data.
    Mutates E in-place by adding OSM-derived keys.

    Args:
        HR           : hazard report dict (must contain "location")
        E            : environmental indicators dict (mutated in-place)
        country_hint : country appended to the location query

    Returns:
        (GeoPoint, HazardContext) tuple for further use downstream.

    New keys added to E:
        osm_lat               – barangay centroid latitude
        osm_lon               – barangay centroid longitude
        osm_display_name      – full OSM place name
        osm_waterway_count    – number of waterway features within search radius
        osm_river_count       – number of rivers/streams specifically
        osm_flood_zone        – bool: flood-tagged area detected
        osm_river_distance_m  – metres to nearest river (None if none found)
        osm_is_fallback       – bool: True if OSM was unreachable after retries
    """
    location_name = HR.get("location", "")
    if not location_name:
        logger.warning("HR missing 'location' key; skipping OSM enrichment.")
        geo = _fallback_geo()
        ctx = _fallback_hazard()
    else:
        geo = geocode(location_name, country_hint=country_hint)
        if geo.is_fallback:
            ctx = _fallback_hazard(geo.lat, geo.lon)
        else:
            ctx = get_hazard_context(geo.lat, geo.lon)

    E["osm_lat"]              = geo.lat
    E["osm_lon"]              = geo.lon
    E["osm_display_name"]     = geo.display_name
    E["osm_waterway_count"]   = ctx.waterway_count
    E["osm_river_count"]      = ctx.river_count
    E["osm_flood_zone"]       = ctx.flood_zone
    E["osm_river_distance_m"] = (
        ctx.river_distance_m if not math.isinf(ctx.river_distance_m) else None
    )
    E["osm_is_fallback"]      = geo.is_fallback or ctx.is_fallback

    logger.debug("E enriched with OSM data: %s", {k: v for k, v in E.items()
                                                    if k.startswith("osm_")})
    return geo, ctx


# ── Convenience: batch geocode a list of barangays ───────────────────────────

def batch_geocode(locations: List[str],
                  country_hint: str = _COUNTRY_HINT) -> List[GeoPoint]:
    """
    Geocode a list of location names, respecting Nominatim's rate limit.

    Args:
        locations    : list of location name strings
        country_hint : appended to each query

    Returns:
        List of GeoPoint objects (same order as input).
    """
    results = []
    for name in locations:
        results.append(geocode(name, country_hint=country_hint))
    return results