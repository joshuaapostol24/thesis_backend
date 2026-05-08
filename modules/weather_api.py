import requests
import logging

from modules.database import get_connection
from datetime import datetime

logger = logging.getLogger(__name__)

# 🔑 YOUR API KEY
API_KEY = "cd2c4c876fb8dd39eaf19513133fd4d3"

# ✅ FREE WORKING ENDPOINT (NOT One Call 3.0)
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
def save_weather_data(weather: dict):

    conn = get_connection()
    cur = conn.cursor()

    try:

        cur.execute("""
            INSERT INTO weather_data (
                city,
                temperature,
                pressure,
                humidity,
                wind_speed,
                rainfall
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            "Mamburao",
            weather.get("temperature"),
            weather.get("pressure"),
            weather.get("humidity"),
            weather.get("wind_speed"),
            weather.get("rainfall")
        ))

        conn.commit()

        logger.info(
            "Weather data saved."
        )

    except Exception as e:

        logger.error(
            "save_weather_data error: %s",
            e
        )

    finally:

        cur.close()
        conn.close()


def save_training_samples(weather: dict):

    conn = get_connection()
    cur = conn.cursor()

    try:

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

        rainfall = weather.get(
            "rainfall",
            0
        )

        humidity = weather.get(
            "humidity",
            0
        )

        if rainfall >= 20:

            risk_label = 3

        elif rainfall >= 5:

            risk_label = 2

        else:

            risk_label = 1


        current_time = datetime.utcnow()
        
        for barangay_id, profile in barangay_profiles.items():

            cur.execute("""
                INSERT INTO barangay_training_data (
                    barangay_id,
                    timestamp,
                    rainfall,
                    humidity,
                    soil,
                    flood,
                    storm_surge,
                    risk_label
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                barangay_id,
                current_time,
                rainfall,
                humidity,
                profile["soil"],
                profile["flood"],
                profile["storm_surge"],
                risk_label
            ))

        conn.commit()

        logger.info(
            "Training samples saved."
        )

    except Exception as e:
        conn.rollback()
        logger.error(
            "save_training_samples error: %s",
            e
        )

    finally:

        cur.close()
        conn.close()

def get_weather(lat: float, lon: float) -> dict:
    url = (
        f"{BASE_URL}"
        f"?lat={lat}&lon={lon}"
        f"&appid={API_KEY}"
        f"&units=metric"
    )

    try:

        response = requests.get(
            url,
            timeout=15
        )

        data = response.json()

        print(
            "RAW API RESPONSE:",
            data
        )

        if str(data.get("cod")) != "200":

            logger.error(
                "API Error: %s",
                data
            )

            return {
                "temperature": 0.0,
                "humidity": 0.0,
                "pressure": 0.0,
                "wind_speed": 0.0,
                "rainfall": 0.0
            }

        main = data.get("main", {})
        wind = data.get("wind", {})
        rain = data.get("rain", {})

        features = {

            "temperature": main.get("temp", 0.0),

            "humidity": main.get(
                "humidity",
                0.0
            ),

            "pressure": main.get(
                "pressure",
                0.0
            ),

            "wind_speed": wind.get(
                "speed",
                0.0
            ),

            "rainfall": rain.get(
                "1h",
                0.0
            )
        }

        save_weather_data(features)

        save_training_samples(features)

        logger.info(
            "Weather fetched successfully: %s",
            features
        )

        return features

    except Exception as e:

        logger.error(
            "Weather API error: %s",
            e
        )

        return {
            "temperature": 0.0,
            "humidity": 0.0,
            "pressure": 0.0,
            "wind_speed": 0.0,
            "rainfall": 0.0
        }