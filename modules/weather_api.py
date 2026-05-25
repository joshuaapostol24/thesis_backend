import logging
from datetime import datetime, timezone

import requests

from modules.config import get_openweather_key
from modules.database import engine
from modules.normalization import BARANGAY_PROFILES
from modules.risk_adjustment import (
    ORANGE_RAINFALL_MM,
    YELLOW_RAINFALL_MM,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# ── Per-barangay rainfall thresholds ─────────────────────────────────────────
# Keep CNN-LSTM training labels aligned with PAGASA rainfall advisories.

# ── Per-barangay structural profiles for training data ────────────────────────
# soil = estimated saturation baseline; flood/storm_surge come from normalization.py
_BARANGAY_SOIL_BASELINES = {
    1:  2.07,
    2:  2.03,
    3:  2.10,
    4:  2.22,
    5:  2.20,
    6:  2.18,
    7:  2.17,
    8:  2.14,
    9:  2.12,
    10: 2.09,
    11: 2.72,
    12: 1.96,
    13: 2.01,
    14: 2.47,
    15: 2.67,
}


def _risk_label_for_barangay(rainfall: float, barangay_id: int) -> int:
    """
    Computes the CNN-LSTM training label using PAGASA rainfall thresholds.

    Labels remain 1/2/3 because the model predicts a 0-3 score; the final
    rule classification and rainfall guardrail separate HIGH from VERY HIGH.
    """
    if rainfall >= ORANGE_RAINFALL_MM:
        return 3
    if rainfall >= YELLOW_RAINFALL_MM:
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
    Each barangay gets the same PAGASA rainfall label, while structural
    flood/storm-surge features still differ by barangay.

    This runs on every weather API call so barangay_training_data
    grows continuously with real observed weather data.
    """
    from sqlalchemy import text

    rainfall     = float(weather.get("rainfall", 0.0))
    humidity     = float(weather.get("humidity", 0.0))
    current_time = datetime.now(timezone.utc)

    try:
        with engine.begin() as conn:
            for barangay_id, profile in BARANGAY_PROFILES.items():
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
                    "soil":        _BARANGAY_SOIL_BASELINES.get(barangay_id, 0.0),
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
