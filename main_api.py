from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import logging

from ml_service_python.modules.location import enrich_report
from ml_service_python.modules.flood_hazard import enrich_E_with_shapefile
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
    get_storm_surge_score,
    get_barangay_centroid,
    get_barangay_name,
)

logging.basicConfig(level=logging.INFO)


# ── Startup: preload all 15 models in parallel ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_all_models()
    yield


app = FastAPI(lifespan=lifespan)


# ── Request model — only barangay_id needed ───────────────────────────────────
class PredictRequest(BaseModel):
    barangay_id: int
    hazard_type: str = "Flood"


@app.get("/")
def root():
    return {"message": "ML Hazard Prediction Service Running"}


# ── List all barangays with verified shapefile coordinates ────────────────────
@app.get("/barangays")
def list_barangays():
    """Returns all 15 barangays with shapefile-verified coordinates."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                row_number() OVER (ORDER BY "Var") AS barangay_id,
                "Var",
                ST_Y(ST_Centroid(geometry)) AS lat,
                ST_X(ST_Centroid(geometry)) AS lon
            FROM barangays
            ORDER BY "Var"
            LIMIT 15
        """)
        rows = cur.fetchall()
        return [
            {
                "barangay_id": int(row[0]),
                "var":         row[1],
                "lat":         round(float(row[2]), 6),
                "lon":         round(float(row[3]), 6),
            }
            for row in rows
        ]
    except Exception as e:
        return {"error": str(e)}
    finally:
        cur.close()
        conn.close()


# ── Main prediction endpoint ──────────────────────────────────────────────────
@app.post("/predict-risk")
def predict(req: PredictRequest):
    try:
        # ── Step 1: Get verified coordinates + name from DB ───────────────────
        lat, lon      = get_barangay_centroid(req.barangay_id)
        barangay_name = get_barangay_name(req.barangay_id)

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

        # ── Step 2: Weather using exact shapefile coordinates ─────────────────
        weather = get_weather(lat, lon)
        E.update(weather)

        # ── Step 3: OSM hazard context ────────────────────────────────────────
        enrich_report(HR, E)

        # ── Step 4: Flood hazard from DB ──────────────────────────────────────
        enrich_E_with_shapefile(HR, E)

        # ── Step 5: DB weather features as fallback ───────────────────────────
        db_features = get_barangay_features(req.barangay_id)
        if E.get("rainfall", 0) == 0 and db_features["rainfall"] > 0:
            E["rainfall"] = db_features["rainfall"]
        if E.get("flood", 0) == 0 and db_features["flood"] > 0:
            E["flood"] = db_features["flood"]

        # ── Step 6: Storm surge score ─────────────────────────────────────────
        storm_surge_score = get_storm_surge_score(lat, lon)
        E["storm_surge"]  = storm_surge_score

        # ── Step 7: Risk computation ──────────────────────────────────────────
        hazard_type, barangay, indicators, weights, rules = load_context(HR)

        rule_score       = compute_weighted_scores(HR, E, {}, weights)
        rule_score_total = compute_risk_score(rule_score)

        predicted  = predict_risk(req.barangay_id, E, {"history": []})
        final_risk = fuse_risk(rule_score_total, predicted)
        risk_level = apply_rules(rules, final_risk)

        # ── Step 8: Store assessment ──────────────────────────────────────────
        store_data(HR, rule_score_total, predicted, final_risk, risk_level)

        return {
            "barangay_id":       req.barangay_id,
            "barangay_name":     barangay_name,
            "lat":               lat,
            "lon":               lon,
            "rainfall":          E.get("rainfall"),
            "flood":             E.get("flood"),
            "storm_surge_score": storm_surge_score,
            "rule_score":        rule_score_total,
            "predicted":         predicted,
            "final_risk":        final_risk,
            "risk_level":        risk_level
        }

    except Exception as e:
        logging.exception("Prediction error")
        return {"error": str(e)}    