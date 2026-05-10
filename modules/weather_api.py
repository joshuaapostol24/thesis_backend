import logging
from datetime import datetime, timezone

import requests

from modules.config import get_openweather_key
from modules.database import engine
from modules.normalization import get_indicator_bounds, normalize

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Per-barangay rainfall thresholds that match normalization.py sensitivity.
# HIGH barangays (Tayamaan=7, Poblacion 8=15) use lower bounds → more sensitive.
# LOW barangays (San Luis=4, Tangkalan=6) use higher bounds → less sensitive.
_BARANGAY_RISK_THRESHOLDS = {
    7:  {"high": 15.0, "moderate": 4.0},   # Tayamaan — HIGH
    15: {"high": 15.0, "moderate": 4.0},   # Poblacion 8 — HIGH
    4:  {"high": 25.0, "moderate": 8.0},   # San Luis — LOW
    6:  {"high": 25.0, "moderate": 8.0},   # Tangkalan — LOW
}
_DEFAULT_THRESHOLDS = {"high": 20.0, "moderate": 5.0}


def _risk_label_for_barangay(rainfall: float, barangay_id: int) -> int:
    """
    Compute risk label (1/2/3) using per-barangay rainfall thresholds,
    consistent with the normalization bounds in normalization.py.
    """
    thresholds = _BARANGAY_RISK_THRESHOLDS.get(barangay_id, _DEFAULT_THRESHOLDS)
    if rainfall >= thresholds["high"]:
        return 3
    if rainfall >= thresholds["moderate"]:
        return 2
    return 1


def save_weather_data(weather: dict) -> None:
    from sqlalchemy import text
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO weather_data (
                    city, temperature, pressure, humidity,
                    wind_speed, rainfall, timestamp
                )
                VALUES (:city, :temperature, :pressure, :humidity,
                        :wind_speed, :rainfall, :timestamp)
            """), {
                "city":        "Mamburao",
                "temperature": weather.get("temperature"),
                "pressure":    weather.get("pressure"),
                "humidity":    weather.get("humidity"),
                "wind_speed":  weather.get("wind_speed"),
                "rainfall":    weather.get("rainfall"),
                "timestamp":   datetime.now(timezone.utc),
            })
        logger.info("Weather data saved.")
    except Exception as e:
        logger.error("save_weather_data error: %s", e)


def save_training_samples(weather: dict) -> None:
    from sqlalchemy import text

    barangay_profiles = {
        1:  {"soil": 2.07, "flood": 1.8, "storm_surge": 4.2},
        2:  {"soil": 2.03, "flood": 1.8, "storm_surge": 4.2},
        3:  {"soil": 2.10, "flood": 1.8, "storm_surge": 4.2},
        4:  {"soil": 2.22, "flood": 1.8, "storm_surge": 1.0},
        5:  {"soil": 2.20, "flood": 1.8, "storm_surge": 4.2},
        6:  {"soil": 2.18, "flood": 3.2, "storm_surge": 1.0},
        7:  {"soil": 2.17, "flood": 3.2, "storm_surge": 4.2},
        8:  {"soil": 2.14, "flood": 1.8, "storm_surge": 4.2},
        9:  {"soil": 2.12, "flood": 1.8, "storm_surge": 4.2},
        10: {"soil": 2.09, "flood": 1.8, "storm_surge": 4.2},
        11: {"soil": 2.72, "flood": 1.8, "storm_surge": 4.2},
        12: {"soil": 1.96, "flood": 1.8, "storm_surge": 1.0},
        13: {"soil": 2.01, "flood": 1.8, "storm_surge": 4.2},
        14: {"soil": 2.47, "flood": 3.2, "storm_surge": 1.0},
        15: {"soil": 2.67, "flood": 3.2, "storm_surge": 4.2},
    }

    rainfall     = weather.get("rainfall", 0)
    humidity     = weather.get("humidity", 0)
    current_time = datetime.now(timezone.utc)

    try:
        with engine.begin() as conn:
            for barangay_id, profile in barangay_profiles.items():
                # Use per-barangay thresholds so training labels match
                # the normalization sensitivity used during inference.
                risk_label = _risk_label_for_barangay(rainfall, barangay_id)

                conn.execute(text("""
                    INSERT INTO barangay_training_data (
                        barangay_id, timestamp, rainfall, humidity,
                        soil, flood, storm_surge, risk_label
                    )
                    VALUES (
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
        logger.info("Training samples saved.")
    except Exception as e:
        logger.error("save_training_samples error: %s", e)


def get_weather(lat: float, lon: float) -> dict:
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
        return _empty

    url = f"{BASE_URL}?lat={lat}&lon={lon}&appid={api_key}&units=metric"

    try:
        response = requests.get(url, timeout=15)
        data = response.json()

        if str(data.get("cod")) != "200":
            logger.error("OpenWeatherMap API error: %s", data)
            return _empty

        main = data.get("main", {})
        wind = data.get("wind", {})
        rain = data.get("rain", {})

        features = {
            "temperature": main.get("temp",     0.0),
            "humidity":    main.get("humidity", 0.0),
            "pressure":    main.get("pressure", 0.0),
            "wind_speed":  wind.get("speed",    0.0),
            "rainfall":    rain.get("1h",       0.0),
        }

        save_weather_data(features)
        save_training_samples(features)

        logger.info("Weather fetched: %s", features)
        return features

    except Exception as e:
        logger.error("Weather API error: %s", e)
        return _empty