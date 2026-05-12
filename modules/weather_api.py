import logging
from datetime import datetime, timezone

import requests

from modules.config import get_openweather_key
from modules.database import engine

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# ── Per-barangay rainfall thresholds ─────────────────────────────────────────
# Matches normalization.py sensitivity bounds per barangay.
# HIGH barangays use lower thresholds → trigger alerts sooner.
# LOW barangays use higher thresholds → less sensitive.
_BARANGAY_RISK_THRESHOLDS = {
    7:  {"high": 15.0, "moderate": 4.0},   # Tayamaan    — HIGH
    15: {"high": 15.0, "moderate": 4.0},   # Poblacion 8 — HIGH
    4:  {"high": 25.0, "moderate": 8.0},   # San Luis    — LOW
    6:  {"high": 25.0, "moderate": 8.0},   # Tangkalan   — LOW
}
_DEFAULT_THRESHOLDS = {"high": 20.0, "moderate": 5.0}

# ── Per-barangay structural profiles for training data ────────────────────────
# soil = estimated saturation baseline, flood/storm_surge from hazard profile
_BARANGAY_PROFILES = {
    1:  {"soil": 2.07, "flood": 0.20, "storm_surge": 1.00},  # Balansay    MODERATE SSA3
    2:  {"soil": 2.03, "flood": 0.20, "storm_surge": 1.00},  # Fatima      MODERATE SSA3
    3:  {"soil": 2.10, "flood": 0.20, "storm_surge": 1.00},  # Payompon    MODERATE SSA3
    4:  {"soil": 2.22, "flood": 0.20, "storm_surge": 0.00},  # San Luis    LOW
    5:  {"soil": 2.20, "flood": 0.20, "storm_surge": 1.00},  # Talabaan    MODERATE SSA3
    6:  {"soil": 2.18, "flood": 0.60, "storm_surge": 0.00},  # Tangkalan   LOW medium flood
    7:  {"soil": 2.17, "flood": 0.60, "storm_surge": 1.00},  # Tayamaan    HIGH
    8:  {"soil": 2.14, "flood": 0.20, "storm_surge": 1.00},  # Poblacion 1 MODERATE SSA3
    9:  {"soil": 2.12, "flood": 0.20, "storm_surge": 1.00},  # Poblacion 2 MODERATE SSA3
    10: {"soil": 2.09, "flood": 0.20, "storm_surge": 1.00},  # Poblacion 3 MODERATE SSA3
    11: {"soil": 2.72, "flood": 0.20, "storm_surge": 1.00},  # Poblacion 4 MODERATE SSA3
    12: {"soil": 1.96, "flood": 0.20, "storm_surge": 1.00},  # Poblacion 5 MODERATE SSA3
    13: {"soil": 2.01, "flood": 0.20, "storm_surge": 1.00},  # Poblacion 6 MODERATE SSA3
    14: {"soil": 2.47, "flood": 0.20, "storm_surge": 1.00},  # Poblacion 7 MODERATE SSA3
    15: {"soil": 2.67, "flood": 0.60, "storm_surge": 1.00},  # Poblacion 8 HIGH
}


def _risk_label_for_barangay(rainfall: float, barangay_id: int) -> int:
    """
    Computes risk label (1/2/3) using per-barangay rainfall thresholds.
    Consistent with normalization bounds in normalization.py.
    """
    thresholds = _BARANGAY_RISK_THRESHOLDS.get(barangay_id, _DEFAULT_THRESHOLDS)
    if rainfall >= thresholds["high"]:
        return 3
    if rainfall >= thresholds["moderate"]:
        return 2
    return 1


def save_weather_data(weather: dict) -> None:
    """Saves current weather reading to weather_data table."""
    from sqlalchemy import text
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO weather_data (
                    city, temperature, pressure, humidity,
                    wind_speed, rainfall, timestamp
                ) VALUES (
                    :city, :temperature, :pressure, :humidity,
                    :wind_speed, :rainfall, :timestamp
                )
            """), {
                "city":        "Mamburao",
                "temperature": weather.get("temperature", 0.0),
                "pressure":    weather.get("pressure",    0.0),
                "humidity":    weather.get("humidity",    0.0),
                "wind_speed":  weather.get("wind_speed",  0.0),
                "rainfall":    weather.get("rainfall",    0.0),
                "timestamp":   datetime.now(timezone.utc),
            })
        logger.debug("Weather data saved to weather_data table.")
    except Exception as e:
        logger.error("save_weather_data error: %s", e)


def save_training_samples(weather: dict) -> None:
    """
    Saves one training record per barangay using current weather data.
    Each barangay gets a different risk_label based on its own
    rainfall sensitivity threshold — not a single shared label.

    This runs on every weather API call so barangay_training_data
    grows continuously with real observed weather data.
    """
    from sqlalchemy import text

    rainfall     = float(weather.get("rainfall", 0.0))
    humidity     = float(weather.get("humidity", 0.0))
    current_time = datetime.now(timezone.utc)

    try:
        with engine.begin() as conn:
            for barangay_id, profile in _BARANGAY_PROFILES.items():
                risk_label = _risk_label_for_barangay(rainfall, barangay_id)

                conn.execute(text("""
                    INSERT INTO barangay_training_data (
                        barangay_id, timestamp, rainfall, humidity,
                        soil, flood, storm_surge, risk_label
                    ) VALUES (
                        :barangay_id, :timestamp, :rainfall, :humidity,
                        :soil, :flood, :storm_surge, :risk_label
                    )
                """), {
                    "barangay_id": barangay_id,
                    "timestamp":   current_time,
                    "rainfall":    rainfall,
                    "humidity":    humidity,
                    "soil":        profile["soil"],
                    "flood":       profile["flood"],
                    "storm_surge": profile["storm_surge"],
                    "risk_label":  risk_label,
                })
        logger.info(
            "Training samples saved for all 15 barangays | "
            "rainfall=%.2f humidity=%.2f", rainfall, humidity
        )
    except Exception as e:
        logger.error("save_training_samples error: %s", e)


def get_weather(lat: float, lon: float) -> dict:
    """
    Fetches current weather from OpenWeatherMap API.
    Saves weather data and per-barangay training samples on every call.
    Returns zeros on failure — caller should use DB fallback.
    """
    _empty = {
        "temperature": 0.0,
        "humidity":    0.0,
        "pressure":    0.0,
        "wind_speed":  0.0,
        "rainfall":    0.0,
    }

    try:
        api_key = get_openweather_key()
    except RuntimeError as e:
        logger.error("Weather API key unavailable: %s", e)
        save_training_samples(_empty)
        return _empty

    url = f"{BASE_URL}?lat={lat}&lon={lon}&appid={api_key}&units=metric"

    try:
        response = requests.get(url, timeout=15)
        data     = response.json()

        # ── FIX: cod is only present on ERROR responses ───────────────────────
        # Successful responses have no "cod" field at all.
        # Old check: str(data.get("cod")) != "200" → always True → always zeros
        # New check: only fail if cod is explicitly present AND not 200
        cod = data.get("cod")
        if cod is not None and str(cod) != "200":
            logger.error("OpenWeatherMap API error (cod=%s): %s", cod, data)
            save_training_samples(_empty)
            return _empty

        main = data.get("main", {})
        wind = data.get("wind", {})
        rain = data.get("rain", {})

        features = {
            "temperature": float(main.get("temp",     0.0)),
            "humidity":    float(main.get("humidity", 0.0)),
            "pressure":    float(main.get("pressure", 0.0)),
            "wind_speed":  float(wind.get("speed",    0.0)),
            # rain["1h"] only appears when it is actually raining
            "rainfall":    float(rain.get("1h",       0.0)),
        }

        logger.info(
            "Weather fetched | lat=%.4f lon=%.4f | "
            "temp=%.1f humidity=%.1f rainfall=%.2f wind=%.1f",
            lat, lon,
            features["temperature"],
            features["humidity"],
            features["rainfall"],
            features["wind_speed"],
        )

        save_weather_data(features)
        save_training_samples(features)

        return features

    except requests.Timeout:
        logger.error("OpenWeatherMap API timed out for lat=%.4f lon=%.4f", lat, lon)
        save_training_samples(_empty)
        return _empty
    except Exception as e:
        logger.error("Weather API error: %s", e)
        save_training_samples(_empty)
        return _empty