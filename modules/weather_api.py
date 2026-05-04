import requests
import logging

logger = logging.getLogger(__name__)

# 🔑 YOUR API KEY
API_KEY = "cd2c4c876fb8dd39eaf19513133fd4d3"

# ✅ FREE WORKING ENDPOINT (NOT One Call 3.0)
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather(lat: float, lon: float) -> dict:
    """
    Fetch weather data for ML / LSTM (FREE OpenWeather API)
    Returns structured features for each barangay
    """

    url = (
        f"{BASE_URL}"
        f"?lat={lat}&lon={lon}"
        f"&appid={API_KEY}"
        f"&units=metric"
    )

    try:
        response = requests.get(url, timeout=15)
        data = response.json()

        # 🔍 DEBUG (IMPORTANT FOR THESIS TESTING)
        print("RAW API RESPONSE:", data)

        # ❌ Handle API errors
        if str(data.get("cod")) != "200":
            logger.error("API Error: %s", data)
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
            "humidity": main.get("humidity", 0.0),
            "pressure": main.get("pressure", 0.0),
            "wind_speed": wind.get("speed", 0.0),

            # ⚠️ rain is OPTIONAL (only appears if raining)
            "rainfall": rain.get("1h", 0.0)
        }

        logger.info("Weather fetched successfully: %s", features)

        return features

    except Exception as e:
        logger.error("Weather API error: %s", e)

        return {
            "temperature": 0.0,
            "humidity": 0.0,
            "pressure": 0.0,
            "wind_speed": 0.0,
            "rainfall": 0.0
        }