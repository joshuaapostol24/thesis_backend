"""
routes/simulation_routes.py
────────────────────────────
FastAPI routes for the simulation mode.

POST /simulate/
    • Runs all 15 barangays with the supplied weather conditions.
    • Returns the full results AND a side-by-side comparison with the
      previous simulation run (stored in-memory per server process;
      swap _last_simulation for a Redis/DB key for persistence).
    • Does NOT send a push notification automatically.

POST /simulate/notify
    • Takes the same weather inputs PLUS a list of FCM device tokens.
    • Runs the simulation, then fires an FCM push notification.
    • Use this when the user clicks "Send to Phone" on the website.

POST /simulate/barangay/{barangay_id}
    • Single-barangay detail view (unchanged).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from modules.simulation import run_simulation

router = APIRouter(prefix="/simulate", tags=["Simulation"])


# =========================================================
# IN-MEMORY COMPARISON STORE
# Replace with Redis / DB for multi-process deployments.
# =========================================================

_last_simulation: Optional[dict] = None   # stores the previous to_dict() result


# =========================================================
# REQUEST MODELS
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


class SimulationNotifyRequest(SimulationRequest):
    """Same weather inputs + FCM device tokens to push to."""
    fcm_tokens: List[str] = Field(
        ...,
        min_items=1,
        description="FCM registration tokens of the target devices.",
    )


# =========================================================
# HELPERS
# =========================================================

_RISK_ORDER = {
    "VERY HIGH": 0,
    "HIGH":      1,
    "MODERATE":  2,
    "LOW":       3,
    "VERY LOW":  4,
}


def _build_comparison(current: dict, previous: Optional[dict]) -> list:
    """
    Returns a list of per-barangay comparison rows, one entry per barangay.

    Shape of each row
    -----------------
    {
        "barangay_id":     int,
        "barangay_name":   str,

        "current": {
            "risk_level":  str,
            "final_score": float,
            "rule_score":  float,
            "ml_score":    float,
        },

        # Only present when a previous run exists:
        "previous": {
            "risk_level":  str,
            "final_score": float,
            "rule_score":  float,
            "ml_score":    float,
        },
        "delta": {
            "final_score":    float,   # positive = worsened
            "rule_score":     float,
            "ml_score":       float,
            "risk_changed":   bool,
            "risk_direction": str,     # "worsened" | "improved" | "unchanged"
        },
    }
    """
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
            pb = prev_index[bid]
            prev_entry = {
                "risk_level":  pb["risk_level"],
                "final_score": pb["final_score"],
                "rule_score":  pb["rule_score"],
                "ml_score":    pb["ml_score"],
            }

            d_final = round(b["final_score"] - pb["final_score"], 4)
            d_rule  = round(b["rule_score"]  - pb["rule_score"],  4)
            d_ml    = round(b["ml_score"]    - pb["ml_score"],    4)

            cur_ord  = _RISK_ORDER.get(b["risk_level"],  99)
            prev_ord = _RISK_ORDER.get(pb["risk_level"], 99)

            if cur_ord < prev_ord:
                direction = "worsened"
            elif cur_ord > prev_ord:
                direction = "improved"
            else:
                direction = "unchanged"

            row["previous"] = prev_entry
            row["delta"] = {
                "final_score":    d_final,
                "rule_score":     d_rule,
                "ml_score":       d_ml,
                "risk_changed":   b["risk_level"] != pb["risk_level"],
                "risk_direction": direction,
            }

        rows.append(row)

    # Sort: worsened first → then by current risk level → then by score desc
    def sort_key(r):
        direction_order = {"worsened": 0, "unchanged": 1, "improved": 2}
        delta = r.get("delta", {})
        return (
            direction_order.get(delta.get("risk_direction", "unchanged"), 1),
            _RISK_ORDER.get(r["current"]["risk_level"], 99),
            -r["current"]["final_score"],
        )

    rows.sort(key=sort_key)
    return rows


def _extract_inputs(sim: Optional[dict]) -> Optional[dict]:
    """Safely pull the inputs block from a stored simulation dict."""
    return sim.get("inputs") if sim else None


# =========================================================
# ROUTES
# =========================================================

@router.post("/")
def simulate_all(request: SimulationRequest):
    """
    Run a hazard simulation for ALL 15 barangays.
    Returns full results + a side-by-side comparison with the previous run.
    Read-only — no DB writes, no push notifications sent.
    The website should show a "Send to Phone" button that calls POST /simulate/notify.
    """
    global _last_simulation

    try:
        result       = run_simulation(
            rainfall    = request.rainfall,
            humidity    = request.humidity,
            wind_speed  = request.wind_speed,
            temperature = request.temperature,
        )
        current_dict = result.to_dict()
        comparison   = _build_comparison(current_dict, _last_simulation)
        has_previous = _last_simulation is not None
        prev_inputs  = _extract_inputs(_last_simulation)

        _last_simulation = current_dict          # save for next comparison

        return {
            **current_dict,
            "comparison": {
                "has_previous":    has_previous,
                "previous_inputs": prev_inputs,
                "barangays":       comparison,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notify")
def simulate_and_notify(request: SimulationNotifyRequest):
    """
    Run the simulation for ALL 15 barangays and send an FCM push
    notification to the supplied device tokens.

    Call this when the user clicks "Send to Phone" on the website.
    The response includes the full simulation results, comparison table,
    and the FCM delivery report.
    """
    global _last_simulation

    try:
        result       = run_simulation(
            rainfall    = request.rainfall,
            humidity    = request.humidity,
            wind_speed  = request.wind_speed,
            temperature = request.temperature,
        )
        current_dict = result.to_dict()
        comparison   = _build_comparison(current_dict, _last_simulation)
        has_previous = _last_simulation is not None
        prev_inputs  = _extract_inputs(_last_simulation)

        _last_simulation = current_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── FCM push ────────────────────────────────────────────────────
    try:
        from modules.simulation_notifier import send_simulation_notification
        fcm_report = send_simulation_notification(
            simulation_dict = current_dict,
            tokens          = request.fcm_tokens,
        )
    except Exception as e:
        # Don't fail the whole response if FCM errors
        fcm_report = {
            "success_count": 0,
            "failure_count": len(request.fcm_tokens),
            "failed_tokens": request.fcm_tokens,
            "error":         str(e),
        }

    return {
        **current_dict,
        "comparison": {
            "has_previous":    has_previous,
            "previous_inputs": prev_inputs,
            "barangays":       comparison,
        },
        "notification": fcm_report,
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