from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from ml_service_python.modules.context import load_context
from ml_service_python.modules.normalization import compute_weighted_scores
from ml_service_python.modules.rule_engine import compute_risk_score, apply_rules
from ml_service_python.modules.cnn_lstm import predict_risk, preload_all_models, BARANGAY_IDS
from ml_service_python.modules.fusion import fuse_risk
from ml_service_python.modules.feedback import store_data
from ml_service_python.modules.weather_api import get_weather
from ml_service_python.modules.database import (
    get_connection,
    get_barangay_features,
    get_barangay_centroid,
    get_barangay_name,
    get_barangay_hazard_profile,
    get_storm_surge_score,
    get_recent_weather,
    get_hazard_summary,
    list_barangay_profiles,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _env_enabled(name: str, default: str = "true") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "y", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _env_enabled("PRELOAD_MODELS_ON_STARTUP", "true"):
        preload_all_models()
    else:
        logger.info("Skipping model preload because PRELOAD_MODELS_ON_STARTUP=false.")
    yield


app = FastAPI(lifespan=lifespan)

cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    barangay_id: int
    hazard_type: str = "Flood"


@app.get("/")
def root():
    return {"message": "ML Hazard Prediction Service Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/barangays")
def list_barangays():
    """Returns all 15 barangays with coordinates and hazard profiles."""
    try:
        return list_barangay_profiles()
    except Exception as api_exc:
        logger.warning("list_barangays Supabase API fallback failed: %s", api_exc)

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                b.barangay_id, b.name, b.lat, b.lon,
                h.flood_hazard_level, h.flood_hazard_score,
                h.max_ssa_level, h.storm_surge_score, h.overall_hazard
            FROM barangay_list b
            LEFT JOIN barangay_hazard_profile h ON b.barangay_id = h.barangay_id
            ORDER BY b.barangay_id
        """)
        rows = cur.fetchall()
        return [
            {
                "barangay_id":        row[0],
                "name":               row[1],
                "lat":                float(row[2]),
                "lon":                float(row[3]),
                "flood_hazard_level": row[4],
                "flood_hazard_score": float(row[5]) if row[5] else 0.0,
                "max_ssa_level":      int(row[6]) if row[6] else 0,
                "storm_surge_score":  float(row[7]) if row[7] else 0.0,
                "overall_hazard":     row[8],
            }
            for row in rows
        ]
    except Exception as e:
        logger.error("list_barangays error: %s", e)
        return {"error": str(e)}
    finally:
        cur.close()
        conn.close()


@app.get("/barangays/{barangay_id}")
def get_barangay(barangay_id: int):
    """Returns full hazard profile for a specific barangay."""
    try:
        lat, lon       = get_barangay_centroid(barangay_id)
        name           = get_barangay_name(barangay_id)
        hazard         = get_barangay_hazard_profile(barangay_id)
        recent_weather = get_recent_weather(barangay_id, limit=5)
        return {
            "barangay_id":    barangay_id,
            "name":           name,
            "lat":            lat,
            "lon":            lon,
            **hazard,
            "recent_weather": recent_weather,
        }
    except Exception as e:
        logger.error("get_barangay error: %s", e)
        return {"error": str(e)}


@app.post("/predict-risk")
def predict(req: PredictRequest):
    try:
        # ── Step 1: Get coordinates and hazard profile (B parameter) ──────────
        lat, lon       = get_barangay_centroid(req.barangay_id)
        barangay_name  = get_barangay_name(req.barangay_id)
        hazard_profile = get_barangay_hazard_profile(req.barangay_id)

        # B = Barangay context data (Algorithm 3)
        # hazard_profile drives per-barangay weights and normalization bounds
        B = hazard_profile

        HR = {
            "type":        req.hazard_type,
            "location":    barangay_name,
            "barangay_id": req.barangay_id,
            "lat":         lat,
            "lon":         lon,
        }

        E = {
            "osm_lat": lat,
            "osm_lon": lon,
        }

        # ── Step 2: Live weather data ─────────────────────────────────────────
        weather = get_weather(lat, lon)
        E.update(weather)

        # ── Step 3: Flood + storm surge from hazard profile ───────────────────
        # Each barangay gets its own structural hazard values — not global
        E["flood"]       = hazard_profile["flood_hazard_score"]
        E["storm_surge"] = hazard_profile["storm_surge_score"]

        # ── Step 4: DB weather fallback if live API returns 0 ─────────────────
        db_features = get_barangay_features(req.barangay_id)
        if E.get("rainfall", 0) == 0 and db_features["rainfall"] > 0:
            E["rainfall"] = db_features["rainfall"]
            logger.info(
                "Barangay %d — using DB rainfall fallback: %.2f mm",
                req.barangay_id, E["rainfall"]
            )
        if E.get("humidity", 0) == 0 and db_features.get("humidity", 0) > 0:
            E["humidity"] = db_features["humidity"]
        E["soil"] = db_features.get("soil", E.get("soil", 0))
        if db_features.get("flood", 0) > 0:
            E["flood"] = db_features["flood"]
        if db_features.get("storm_surge", 0) > 0:
            E["storm_surge"] = db_features["storm_surge"]

        # ── Step 5: Load per-barangay context (weights from B) ────────────────
        # FIX: pass hazard_profile as B so each barangay gets correct weights
        hazard_type, barangay, indicators, weights, rules = load_context(HR, B)

        logger.info(
            "Barangay %d (%s) | overall=%s | weights=%s",
            req.barangay_id, barangay_name,
            hazard_profile["overall_hazard"], weights
        )

        # ── Step 6: Rule-based score with per-barangay normalization ──────────
        # FIX: pass barangay_id so normalization uses correct bounds per barangay
        rule_score       = compute_weighted_scores(HR, E, B, weights, req.barangay_id)
        rule_score_total = compute_risk_score(rule_score)

        # ── Step 7: CNN+LSTM prediction ───────────────────────────────────────
        recent_history = list(reversed(get_recent_weather(req.barangay_id, limit=10)))
        predicted  = predict_risk(req.barangay_id, E, {"history": recent_history})

        # ── Step 8: Fuse rule-based + ML scores ───────────────────────────────
        final_risk = fuse_risk(predicted, rule_score_total)
        risk_level = apply_rules(rules, final_risk)

        # ── Step 9: Store assessment for retraining ───────────────────────────
        HR["rainfall"] = E.get("rainfall")
        HR["humidity"] = E.get("humidity")
        HR["flood"]    = E.get("flood")
        HR["soil"]     = E.get("soil", 0)
        HR["storm_surge"] = E.get("storm_surge")
        store_data(HR, rule_score_total, predicted, final_risk, risk_level)

        return {
            "barangay_id":        req.barangay_id,
            "barangay_name":      barangay_name,
            "lat":                lat,
            "lon":                lon,
            # ── Weather ───────────────────────────────────────────────────────
            "rainfall":           E.get("rainfall"),
            "temperature":        E.get("temperature"),
            "humidity":           E.get("humidity"),
            "wind_speed":         E.get("wind_speed"),
            # ── Per-barangay hazard profile ───────────────────────────────────
            "flood_hazard_level": hazard_profile["flood_hazard_level"],
            "flood_hazard_score": hazard_profile["flood_hazard_score"],
            "storm_surge_score":  hazard_profile["storm_surge_score"],
            "max_ssa_level":      hazard_profile["max_ssa_level"],
            "overall_hazard":     hazard_profile["overall_hazard"],
            # ── Per-barangay weights used ─────────────────────────────────────
            "weights_used":       weights,
            # ── Risk scores ───────────────────────────────────────────────────
            "rule_score":         round(rule_score_total, 4),
            "predicted":          round(predicted, 4),
            "final_risk":         round(final_risk, 4),
            "risk_level":         risk_level,
        }

    except Exception as e:
        logger.exception("Prediction error for barangay_id=%d", req.barangay_id)
        return {"error": str(e)}


@app.get("/weather/recent")
def recent_weather():
    """Returns recent weather observations for Mamburao."""
    return get_recent_weather(1, limit=24)


@app.get("/hazard-summary")
def hazard_summary():
    """Returns hazard summary for all 15 barangays."""
    try:
        return get_hazard_summary()
    except Exception as api_exc:
        logger.warning("hazard_summary Supabase API fallback failed: %s", api_exc)

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                overall_hazard,
                COUNT(*) as count
            FROM barangay_hazard_profile
            GROUP BY overall_hazard
            ORDER BY count DESC
        """)
        rows = cur.fetchall()
        return {
            "summary":          [{"overall_hazard": r[0], "count": r[1]} for r in rows],
            "total_barangays":  15
        }
    except Exception as e:
        logger.error("hazard_summary error: %s", e)
        return {"error": str(e)}
    finally:
        cur.close()
        conn.close()
