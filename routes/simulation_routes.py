"""
routes/simulation_routes.py
────────────────────────────
POST /simulate/
    • Runs all 15 barangays.
    • Saves results to Supabase (simulation_runs table).
    • Returns results + side-by-side comparison with the previous run
      (loaded from Supabase — survives server restarts).
    • Does NOT auto-post announcement — returns a prompt flag instead
      so the frontend can show a "Publish Announcement?" modal.

POST /simulate/barangay/{barangay_id}
    • Single-barangay detail view (unchanged).

GET /simulate/history
    • Returns the last N simulation runs from Supabase.

GET /simulate/last
    • Returns the most recent simulation run.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from modules.simulation_notifier import send_simulation_notification

from modules.simulation import run_simulation

router = APIRouter(prefix="/simulate", tags=["Simulation"])
logger = logging.getLogger(__name__)


# =========================================================
# SUPABASE CLIENT
# =========================================================

_supabase_client = None

def _get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
                "must be set in environment variables."
            )
        _supabase_client = create_client(url, key)
    return _supabase_client


# =========================================================
# DB HELPERS
# =========================================================

def _save_simulation(current_dict: dict) -> Optional[int]:
    """
    Insert a simulation run into Supabase simulation_runs table.
    Returns the new row ID, or None on failure.
    """
    try:
        supabase = _get_supabase()
        summary  = current_dict.get("summary", {})
        inputs   = current_dict.get("inputs", {})

        row = {
            "rainfall":        inputs.get("rainfall"),
            "humidity":        inputs.get("humidity"),
            "wind_speed":      inputs.get("wind_speed"),
            "temperature":     inputs.get("temperature"),
            "very_high_count": summary.get("very_high", 0),
            "high_count":      summary.get("high",      0),
            "moderate_count":  summary.get("moderate",  0),
            "low_count":       summary.get("low",       0),
            "very_low_count":  summary.get("very_low",  0),
            "barangays":       current_dict.get("barangays", []),
            "inputs":          inputs,
        }

        response = supabase.table("simulation_runs").insert(row).execute()
        new_id   = response.data[0]["id"] if response.data else None
        logger.info("Simulation saved to Supabase | id=%s", new_id)
        return new_id

    except Exception as exc:
        logger.error("Failed to save simulation to Supabase: %s", exc)
        return None


def _load_last_simulation() -> Optional[dict]:
    """
    Fetch the most recent simulation run from Supabase.
    Returns a dict shaped like SimulationResult.to_dict(), or None.
    """
    try:
        supabase  = _get_supabase()
        response  = (
            supabase.table("simulation_runs")
            .select("*")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if not response.data:
            return None

        row = response.data[0]

        return {
            "inputs":    row["inputs"],
            "barangays": row["barangays"],
            "summary": {
                "very_high": row["very_high_count"],
                "high":      row["high_count"],
                "moderate":  row["moderate_count"],
                "low":       row["low_count"],
                "very_low":  row["very_low_count"],
            },
            "_db_id":     row["id"],
            "_created_at": row["created_at"],
        }

    except Exception as exc:
        logger.error("Failed to load last simulation from Supabase: %s", exc)
        return None


def _load_simulation_history(limit: int = 10) -> list:
    """
    Fetch the last N simulation runs from Supabase (newest first).
    """
    try:
        supabase = _get_supabase()
        response = (
            supabase.table("simulation_runs")
            .select("id, created_at, rainfall, humidity, wind_speed, temperature, very_high_count, high_count, moderate_count, low_count, very_low_count")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []

    except Exception as exc:
        logger.error("Failed to load simulation history: %s", exc)
        return []

def _get_fcm_tokens(barangays: list[str] | None = None) -> list[str]:
    """
    Fetches FCM tokens for all approved users from Supabase.
    If barangays list is provided, only fetches tokens for users
    in those specific barangays — useful for targeted alerts.

    e.g. _get_fcm_tokens(["Barangay Balansay", "Barangay Talabaan"])
    """
    try:
        supabase = _get_supabase()
        query    = (
            supabase.table("users")
            .select("fcm_token, address")
            .eq("status", "approved")
            .not_.is_("fcm_token", "null")
        )

        rows   = query.execute().data or []
        tokens = []

        for row in rows:
            token = row.get("fcm_token", "").strip()
            if not token:
                continue
            # If barangay filter is given, only include matching users
            if barangays:
                if any(row.get("address", "") == brgy for brgy in barangays):
                    tokens.append(token)
            else:
                tokens.append(token)

        logger.info("FCM tokens fetched | count=%d", len(tokens))
        return tokens

    except Exception as exc:
        logger.error("Failed to fetch FCM tokens: %s", exc)
        return []
# =========================================================
# RISK HELPERS
# =========================================================

_RISK_ORDER = {
    "VERY HIGH": 0,
    "HIGH":      1,
    "MODERATE":  2,
    "LOW":       3,
    "VERY LOW":  4,
}

_RISK_PRIORITY = {
    "VERY HIGH": "High",
    "HIGH":      "High",
    "MODERATE":  "Moderate",
    "LOW":       "Low",
    "VERY LOW":  "Low",
}

_RISK_EMOJI = {
    "VERY HIGH": "🔴",
    "HIGH":      "🟠",
    "MODERATE":  "🟡",
    "LOW":       "🟢",
    "VERY LOW":  "⚪",
}


def _risk_emoji(level: str) -> str:
    return _RISK_EMOJI.get(level, "❓")


# =========================================================
# COMPARISON BUILDER
# =========================================================

def _build_comparison(current: dict, previous: Optional[dict]) -> list:
    prev_index: dict = {}
    if previous:
        for b in previous.get("barangays", []):
            prev_index[b["barangay_id"]] = b

    rows = []
    for b in current.get("barangays", []):
        bid = b["barangay_id"]
        current_entry = {
            "risk_level":  b["risk_level"],
            "final_score": b["final_score"],
            "rule_score":  b["rule_score"],
            "ml_score":    b["ml_score"],
        }
        row: dict = {
            "barangay_id":   bid,
            "barangay_name": b["barangay_name"],
            "current":       current_entry,
        }

        if bid in prev_index:
            pb       = prev_index[bid]
            cur_ord  = _RISK_ORDER.get(b["risk_level"],  99)
            prev_ord = _RISK_ORDER.get(pb["risk_level"], 99)
            direction = (
                "worsened"  if cur_ord < prev_ord else
                "improved"  if cur_ord > prev_ord else
                "unchanged"
            )
            row["previous"] = {
                "risk_level":  pb["risk_level"],
                "final_score": pb["final_score"],
                "rule_score":  pb["rule_score"],
                "ml_score":    pb["ml_score"],
            }
            row["delta"] = {
                "final_score":    round(b["final_score"] - pb["final_score"], 4),
                "rule_score":     round(b["rule_score"]  - pb["rule_score"],  4),
                "ml_score":       round(b["ml_score"]    - pb["ml_score"],    4),
                "risk_changed":   b["risk_level"] != pb["risk_level"],
                "risk_direction": direction,
            }

        rows.append(row)

    def sort_key(r):
        d = r.get("delta", {})
        return (
            {"worsened": 0, "unchanged": 1, "improved": 2}.get(
                d.get("risk_direction", "unchanged"), 1
            ),
            _RISK_ORDER.get(r["current"]["risk_level"], 99),
            -r["current"]["final_score"],
        )

    rows.sort(key=sort_key)
    return rows


# =========================================================
# ANNOUNCEMENT PAYLOAD BUILDER
# =========================================================

def _build_announcement_payload(
    simulation_dict: dict,
    inputs: dict,
) -> Optional[dict]:
    """
    If any barangay is HIGH or VERY HIGH, return a ready-to-post
    news payload that the frontend can send to POST /news/create
    after the user confirms.
    Returns None if threshold is not met.
    """
    barangays = simulation_dict.get("barangays", [])
    high_risk = [
        b for b in barangays
        if b["risk_level"] in ("VERY HIGH", "HIGH")
    ]

    if not high_risk:
        return None

    top          = sorted(high_risk, key=lambda b: _RISK_ORDER.get(b["risk_level"], 99))[0]
    top_level    = top["risk_level"]
    top_priority = _RISK_PRIORITY.get(top_level, "High")

    title = (
        f"{_risk_emoji(top_level)} Simulation Alert: {top_level} Flood Risk Detected "
        f"in {len(high_risk)} Barangay{'s' if len(high_risk) > 1 else ''}"
    )

    weather_line = (
        f"Simulated conditions — "
        f"Rainfall: {inputs.get('rainfall', 0):.1f} mm/h | "
        f"Wind: {inputs.get('wind_speed', 0):.1f} km/h | "
        f"Humidity: {inputs.get('humidity', 0):.1f}% | "
        f"Temp: {inputs.get('temperature', 0):.1f}°C"
    )

    barangay_lines = "\n".join(
        f"  {_risk_emoji(b['risk_level'])} {b['barangay_name']} "
        f"— {b['risk_level']} (score: {b['final_score']:.2f})"
        for b in high_risk
    )

    summary      = simulation_dict.get("summary", {})
    summary_line = (
        f"Summary: Very High: {summary.get('very_high', 0)} | "
        f"High: {summary.get('high', 0)} | "
        f"Moderate: {summary.get('moderate', 0)} | "
        f"Low: {summary.get('low', 0)}"
    )

    message = (
        f"{weather_line}\n\n"
        f"⚠️ High-risk barangays:\n{barangay_lines}\n\n"
        f"{summary_line}\n\n"
        f"This is an automated simulation alert. "
        f"Please verify with live sensor data before issuing evacuation orders."
    )

    return {
        "title":    title,
        "category": "Weather",
        "priority": top_priority,
        "date":     datetime.now(timezone.utc).isoformat(),
        "audience": "All Residents",
        "pinned":   "No",
        "message":  message,
    }


# =========================================================
# REQUEST MODEL
# =========================================================

class SimulationRequest(BaseModel):
    rainfall:    float = Field(..., ge=0,   le=500, description="Rainfall in mm/h")
    humidity:    float = Field(..., ge=0,   le=100, description="Humidity in %")
    wind_speed:  float = Field(..., ge=0,   le=300, description="Wind speed in km/h")
    temperature: float = Field(..., ge=-20, le=60,  description="Temperature in °C")

    class Config:
        schema_extra = {
            "example": {
                "rainfall":    25.0,
                "humidity":    85.0,
                "wind_speed":  40.0,
                "temperature": 30.0,
            }
        }


class PublishAnnouncementRequest(BaseModel):
    """
    Payload the frontend sends after the user confirms
    the 'Publish Announcement?' modal.
    Contains the announcement data + optional simulation context
    for the FCM notification body.
    """
    # Announcement fields (matches what _build_announcement_payload returns)
    title:    str
    message:  str
    category: str           = "Weather"
    priority: str           = "High"
    audience: str           = "All Residents"
    pinned:   str           = "No"
    date:     str           = ""

    # FCM targeting — if empty, notifies ALL approved users
    # Pass specific barangay names to target only affected residents
    # e.g. ["Barangay Balansay", "Barangay Talabaan"]
    target_barangays: List[str] = []

    # The full simulation dict is needed to build a rich FCM body.
    # Pass the same `barangays` + `summary` + `inputs` from the
    # POST /simulate/ response.
    simulation_snapshot: dict = {}

    class Config:
        schema_extra = {
            "example": {
                "title":    "🔴 Simulation Alert: VERY HIGH Flood Risk",
                "message":  "Rainfall: 80mm/h | ...",
                "category": "Weather",
                "priority": "High",
                "audience": "All Residents",
                "pinned":   "Yes",
                "target_barangays": [],
                "simulation_snapshot": {}
            }
        }

# =========================================================
# ROUTES
# =========================================================

@router.post("/")
def simulate_all(request: SimulationRequest):
    """
    Run a hazard simulation for ALL 15 barangays.

    • Saves results to Supabase (simulation_runs table).
    • Compares with the previous run loaded from Supabase.
    • If HIGH or VERY HIGH is detected, returns a suggested
      announcement payload under `suggest_announcement` so the
      frontend can prompt the user before publishing.
    """
    # Load previous run from Supabase for comparison
    previous = _load_last_simulation()

    try:
        result       = run_simulation(
            rainfall    = request.rainfall,
            humidity    = request.humidity,
            wind_speed  = request.wind_speed,
            temperature = request.temperature,
        )
        current_dict = result.to_dict()
        inputs       = current_dict["inputs"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Save to Supabase
    saved_id = _save_simulation(current_dict)

    # Build comparison
    comparison   = _build_comparison(current_dict, previous)
    has_previous = previous is not None
    prev_inputs  = previous.get("inputs") if previous else None

    # Build announcement suggestion (not posted yet — user must confirm)
    announcement_payload = _build_announcement_payload(current_dict, inputs)

    return {
        **current_dict,
        "simulation_id": saved_id,
        "comparison": {
            "has_previous":    has_previous,
            "previous_inputs": prev_inputs,
            "barangays":       comparison,
        },
        # Frontend shows a modal if this is not None
        "suggest_announcement": announcement_payload,
    }


@router.get("/last")
def get_last_simulation():
    """
    Return the most recent simulation run from Supabase.
    Useful for loading the last state on page refresh.
    """
    last = _load_last_simulation()
    if not last:
        raise HTTPException(status_code=404, detail="No simulation runs found.")
    return last


@router.get("/history")
def get_simulation_history(limit: int = 10):
    """
    Return the last N simulation runs (summary only, no full barangay data).
    Default: 10 most recent runs.
    """
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    history = _load_simulation_history(limit)
    return {"count": len(history), "runs": history}


@router.post("/publish-announcement")
def publish_announcement(request: PublishAnnouncementRequest):
    """
    Called by the frontend after the user confirms the
    'Publish Announcement?' modal.

    Does two things in sequence:
      1. POSTs the announcement to the news API so it appears
         in the Flutter NewsScreen (/api/news/all).
      2. Sends FCM push notifications to registered device tokens,
         optionally filtered by barangay.

    Returns a summary of what was saved and how many devices
    were notified.
    """
    import requests as http_requests

    announcement_payload = {
        "title":    request.title,
        "message":  request.message,
        "category": request.category,
        "priority": request.priority,
        "audience": request.audience,
        "pinned":   request.pinned,
        "date":     request.date or datetime.now(timezone.utc).isoformat(),
    }

    # ── 1. Save to news feed ──────────────────────────────────────────────
    news_saved    = False
    news_id       = None
    news_error    = None
    backend_url   = os.environ.get("BACKEND_URL", "https://resq-app-xsb98.ondigitalocean.app")
    news_endpoint = f"{backend_url}/api/news"

    try:
        resp = http_requests.post(
            news_endpoint,
            json=announcement_payload,
            timeout=15,
        )
        resp.raise_for_status()
        data     = resp.json()
        news_id  = str(data.get("id") or data.get("_id") or "")
        news_saved = True
        logger.info("Announcement posted to news feed | id=%s", news_id)

    except Exception as exc:
        news_error = str(exc)
        logger.error("Failed to post announcement: %s", exc)
        # Don't raise — still attempt FCM so users get notified

    # ── 2. Send FCM push notifications ───────────────────────────────────
    fcm_success = 0
    fcm_failure = 0
    failed_tokens: list[str] = []
    fcm_error   = None

    try:
        tokens = _get_fcm_tokens(
            barangays=request.target_barangays if request.target_barangays else None
        )

        if tokens and request.simulation_snapshot:
            fcm_result    = send_simulation_notification(
                simulation_dict=request.simulation_snapshot,
                tokens=tokens,
            )
            fcm_success   = fcm_result["success_count"]
            fcm_failure   = fcm_result["failure_count"]
            failed_tokens = fcm_result["failed_tokens"]

        elif tokens and not request.simulation_snapshot:
            # No snapshot provided — send a simple text-only notification
            from firebase_admin import messaging
            from modules.simulation_notifier import _init_firebase
            _init_firebase()

            simple_msg = messaging.MulticastMessage(
                tokens=tokens,
                notification=messaging.Notification(
                    title=request.title,
                    body=request.message[:200],   # FCM body limit safety trim
                ),
                data={"type": "news", "notification_type": "news"},
                android=messaging.AndroidConfig(
                    priority="high",
                    notification=messaging.AndroidNotificationDetails(
                        channel_id="resq_alerts",
                        sound="default",
                    ),
                ),
            )
            batch = messaging.send_each_for_multicast(simple_msg)
            fcm_success = batch.success_count
            fcm_failure = batch.failure_count

        else:
            logger.warning("No FCM tokens found — push notification skipped.")

    except Exception as exc:
        fcm_error = str(exc)
        logger.error("FCM notification failed: %s", exc)

    # ── Response ──────────────────────────────────────────────────────────
    # Raise only if BOTH saving and FCM completely failed
    if not news_saved and fcm_error:
        raise HTTPException(
            status_code=500,
            detail={
                "news_error": news_error,
                "fcm_error":  fcm_error,
            },
        )

    return {
        "news": {
            "saved": news_saved,
            "id":    news_id,
            "error": news_error,
        },
        "fcm": {
            "success_count": fcm_success,
            "failure_count": fcm_failure,
            "failed_tokens": failed_tokens,
            "error":         fcm_error,
        },
        "message": (
            f"Announcement {'saved' if news_saved else 'NOT saved'} to news feed. "
            f"Push notifications sent to {fcm_success} device(s)."
        ),
    }


@router.post("/barangay/{barangay_id}")
def simulate_one(barangay_id: int, request: SimulationRequest):
    """
    Run simulation for a single barangay.
    Useful for the detail view on the website.
    """
    if barangay_id not in range(1, 16):
        raise HTTPException(status_code=400, detail="barangay_id must be 1–15")
    try:
        result = run_simulation(
            rainfall     = request.rainfall,
            humidity     = request.humidity,
            wind_speed   = request.wind_speed,
            temperature  = request.temperature,
            barangay_ids = [barangay_id],
        )
        data = result.to_dict()
        if not data["barangays"]:
            raise HTTPException(status_code=404, detail="No result for this barangay")
        return {
            "inputs":   data["inputs"],
            "barangay": data["barangays"][0],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history/cleanup")
def cleanup_simulation_history(keep: int = 30):
    """
    Deletes old simulation runs, keeping only the most recent N.
    Default: keep=30. Use ?keep=50 to keep more.
    Prevents the simulation_runs table from growing forever.
    """
    if keep < 5 or keep > 200:
        raise HTTPException(
            status_code=400,
            detail="keep must be between 5 and 200"
        )
    try:
        supabase = _get_supabase()

        # Get IDs of the runs we want to KEEP
        keep_response = (
            supabase.table("simulation_runs")
            .select("id")
            .order("created_at", desc=True)
            .limit(keep)
            .execute()
        )

        if not keep_response.data:
            return {"deleted": 0, "kept": 0, "message": "No simulation runs found."}

        keep_ids = [row["id"] for row in keep_response.data]

        # Delete everything NOT in keep_ids
        # Supabase doesn't support NOT IN directly so we get all IDs first
        all_response = (
            supabase.table("simulation_runs")
            .select("id")
            .execute()
        )
        all_ids    = [row["id"] for row in (all_response.data or [])]
        delete_ids = [i for i in all_ids if i not in keep_ids]

        deleted_count = 0
        for old_id in delete_ids:
            supabase.table("simulation_runs").delete().eq("id", old_id).execute()
            deleted_count += 1

        logger.info(
            "Simulation cleanup | kept=%d deleted=%d",
            len(keep_ids), deleted_count
        )
        return {
            "deleted": deleted_count,
            "kept":    len(keep_ids),
            "message": f"Kept {len(keep_ids)} most recent runs. Deleted {deleted_count} old runs.",
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))