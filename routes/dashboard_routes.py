from fastapi import APIRouter, HTTPException
from supabase import create_client
import os

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

_RISK_ORDER = {
    "VERY HIGH": 0, "HIGH": 1,
    "MODERATE": 2, "LOW": 3, "VERY LOW": 4,
}


@router.get("/dashboard")
def get_dashboard_data():
    """
    Returns complete dashboard summary:
    - Total users and reports
    - Current highest risk level across all barangays
    - Count of HIGH/VERY HIGH barangays right now
    - Latest simulation run summary
    - Total news announcements count
    """

    # ── Total users ───────────────────────────────────────────────────────────
    users_response = (
        supabase.table("users")
        .select("*", count="exact")
        .execute()
    )
    total_users = users_response.count or 0

    # ── Total reports ─────────────────────────────────────────────────────────
    reports_response = (
        supabase.table("reports")
        .select("*", count="exact")
        .execute()
    )
    total_reports = reports_response.count or 0

    # ── Current live risk status ──────────────────────────────────────────────
    # Get latest risk assessment per barangay
    risk_response = (
        supabase.table("risk_assessments")
        .select("barangay_id, risk_level, final_risk, timestamp")
        .order("timestamp", desc=True)
        .limit(500)
        .execute()
    )

    seen            = set()
    latest_per_bgy  = []
    for row in (risk_response.data or []):
        bid = row["barangay_id"]
        if bid not in seen:
            seen.add(bid)
            latest_per_bgy.append(row)

    highest_risk  = "N/A"
    high_count    = 0
    risk_summary  = {"VERY HIGH": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0, "VERY LOW": 0}

    for row in latest_per_bgy:
        level = row.get("risk_level", "")
        if level in risk_summary:
            risk_summary[level] += 1
        if level in ("HIGH", "VERY HIGH"):
            high_count += 1
        if highest_risk == "N/A" or _RISK_ORDER.get(level, 99) < _RISK_ORDER.get(highest_risk, 99):
            highest_risk = level

    # ── Latest simulation run ─────────────────────────────────────────────────
    sim_response = (
        supabase.table("simulation_runs")
        .select(
            "id, created_at, rainfall, humidity, "
            "very_high_count, high_count, moderate_count, low_count, very_low_count"
        )
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    latest_simulation = None
    if sim_response.data:
        s = sim_response.data[0]
        latest_simulation = {
            "id":         s["id"],
            "created_at": s["created_at"],
            "rainfall":   s["rainfall"],
            "humidity":   s["humidity"],
            "summary": {
                "very_high": s["very_high_count"],
                "high":      s["high_count"],
                "moderate":  s["moderate_count"],
                "low":       s["low_count"],
                "very_low":  s["very_low_count"],
            }
        }

    # ── Total risk assessments ────────────────────────────────────────────────
    assessments_response = (
        supabase.table("risk_assessments")
        .select("*", count="exact")
        .execute()
    )
    total_assessments = assessments_response.count or 0

    return {
        "users":   {"total": total_users},
        "reports": {"total": total_reports},
        "risk": {
            "highest_level":    highest_risk,
            "high_count":       high_count,
            "barangays_at_risk": high_count,
            "summary":          risk_summary,
            "total_assessed":   len(latest_per_bgy),
        },
        "assessments":     {"total": total_assessments},
        "latest_simulation": latest_simulation,
    }


# ── Weather risk history (unchanged) ─────────────────────────────────────────

@router.get("/weather/history/{barangay_id}")
def get_weather_history(barangay_id: int):
    try:
        response = (
            supabase.table("risk_assessments")
            .select("*")
            .eq("barangay_id", barangay_id)
            .order("timestamp", desc=True)
            .limit(20)
            .execute()
        )
        return response.data
    except Exception as e:
        return {"error": str(e)}


@router.get("/weather/history")
def get_all_weather_history(limit: int = 50):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    try:
        response = (
            supabase.table("risk_assessments")
            .select(
                "barangay_id, timestamp, rainfall, humidity, "
                "temperature, wind_speed, final_risk, risk_level"
            )
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        return {"count": len(response.data), "records": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weather/history/summary/latest")
def get_latest_risk_per_barangay():
    try:
        response = (
            supabase.table("risk_assessments")
            .select(
                "barangay_id, timestamp, rainfall, humidity, "
                "temperature, wind_speed, final_risk, risk_level"
            )
            .order("timestamp", desc=True)
            .limit(500)
            .execute()
        )
        seen   = set()
        latest = []
        for row in response.data:
            bid = row["barangay_id"]
            if bid not in seen:
                seen.add(bid)
                latest.append(row)

        latest.sort(key=lambda r: (
            _RISK_ORDER.get(r.get("risk_level", ""), 99),
            -(r.get("final_risk") or 0),
        ))

        summary   = {"very_high": 0, "high": 0, "moderate": 0, "low": 0, "very_low": 0}
        level_map = {
            "VERY HIGH": "very_high", "HIGH": "high",
            "MODERATE": "moderate",  "LOW": "low", "VERY LOW": "very_low",
        }
        for row in latest:
            key = level_map.get(row.get("risk_level", ""))
            if key:
                summary[key] += 1

        return {"summary": summary, "barangays": latest}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Simulation history (unchanged) ───────────────────────────────────────────

@router.get("/simulation/history")
def get_simulation_history(limit: int = 10):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    try:
        response = (
            supabase.table("simulation_runs")
            .select(
                "id, created_at, rainfall, humidity, wind_speed, temperature, "
                "very_high_count, high_count, moderate_count, low_count, very_low_count"
            )
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"count": len(response.data), "runs": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation/history/{run_id}")
def get_simulation_run(run_id: int):
    try:
        response = (
            supabase.table("simulation_runs")
            .select("*")
            .eq("id", run_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Simulation run {run_id} not found.")
        row = response.data[0]
        return {
            "id": row["id"], "created_at": row["created_at"],
            "inputs": {
                "rainfall": row["rainfall"], "humidity": row["humidity"],
                "wind_speed": row["wind_speed"], "temperature": row["temperature"],
            },
            "summary": {
                "very_high": row["very_high_count"], "high": row["high_count"],
                "moderate": row["moderate_count"],   "low": row["low_count"],
                "very_low": row["very_low_count"],
            },
            "barangays": row["barangays"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation/history/latest")
def get_latest_simulation():
    try:
        response = (
            supabase.table("simulation_runs")
            .select("*")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail="No simulation runs found.")
        row = response.data[0]
        return {
            "id": row["id"], "created_at": row["created_at"],
            "inputs": {
                "rainfall": row["rainfall"], "humidity": row["humidity"],
                "wind_speed": row["wind_speed"], "temperature": row["temperature"],
            },
            "summary": {
                "very_high": row["very_high_count"], "high": row["high_count"],
                "moderate": row["moderate_count"],   "low": row["low_count"],
                "very_low": row["very_low_count"],
            },
            "barangays": row["barangays"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from fastapi import APIRouter, HTTPException
from supabase import create_client
import os

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

_RISK_ORDER = {
    "VERY HIGH": 0, "HIGH": 1,
    "MODERATE": 2, "LOW": 3, "VERY LOW": 4,
}


@router.get("/dashboard")
def get_dashboard_data():
    """
    Returns complete dashboard summary:
    - Total users and reports
    - Current highest risk level across all barangays
    - Count of HIGH/VERY HIGH barangays right now
    - Latest simulation run summary
    - Total news announcements count
    """

    # ── Total users ───────────────────────────────────────────────────────────
    users_response = (
        supabase.table("users")
        .select("*", count="exact")
        .execute()
    )
    total_users = users_response.count or 0

    # ── Total reports ─────────────────────────────────────────────────────────
    reports_response = (
        supabase.table("reports")
        .select("*", count="exact")
        .execute()
    )
    total_reports = reports_response.count or 0

    # ── Current live risk status ──────────────────────────────────────────────
    # Get latest risk assessment per barangay
    risk_response = (
        supabase.table("risk_assessments")
        .select("barangay_id, risk_level, final_risk, timestamp")
        .order("timestamp", desc=True)
        .limit(500)
        .execute()
    )

    seen            = set()
    latest_per_bgy  = []
    for row in (risk_response.data or []):
        bid = row["barangay_id"]
        if bid not in seen:
            seen.add(bid)
            latest_per_bgy.append(row)

    highest_risk  = "N/A"
    high_count    = 0
    risk_summary  = {"VERY HIGH": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0, "VERY LOW": 0}

    for row in latest_per_bgy:
        level = row.get("risk_level", "")
        if level in risk_summary:
            risk_summary[level] += 1
        if level in ("HIGH", "VERY HIGH"):
            high_count += 1
        if highest_risk == "N/A" or _RISK_ORDER.get(level, 99) < _RISK_ORDER.get(highest_risk, 99):
            highest_risk = level

    # ── Latest simulation run ─────────────────────────────────────────────────
    sim_response = (
        supabase.table("simulation_runs")
        .select(
            "id, created_at, rainfall, humidity, "
            "very_high_count, high_count, moderate_count, low_count, very_low_count"
        )
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    latest_simulation = None
    if sim_response.data:
        s = sim_response.data[0]
        latest_simulation = {
            "id":         s["id"],
            "created_at": s["created_at"],
            "rainfall":   s["rainfall"],
            "humidity":   s["humidity"],
            "summary": {
                "very_high": s["very_high_count"],
                "high":      s["high_count"],
                "moderate":  s["moderate_count"],
                "low":       s["low_count"],
                "very_low":  s["very_low_count"],
            }
        }

    # ── Total risk assessments ────────────────────────────────────────────────
    assessments_response = (
        supabase.table("risk_assessments")
        .select("*", count="exact")
        .execute()
    )
    total_assessments = assessments_response.count or 0

    return {
        "users":   {"total": total_users},
        "reports": {"total": total_reports},
        "risk": {
            "highest_level":    highest_risk,
            "high_count":       high_count,
            "barangays_at_risk": high_count,
            "summary":          risk_summary,
            "total_assessed":   len(latest_per_bgy),
        },
        "assessments":     {"total": total_assessments},
        "latest_simulation": latest_simulation,
    }


# ── Weather risk history (unchanged) ─────────────────────────────────────────

@router.get("/weather/history/{barangay_id}")
def get_weather_history(barangay_id: int):
    try:
        response = (
            supabase.table("risk_assessments")
            .select("*")
            .eq("barangay_id", barangay_id)
            .order("timestamp", desc=True)
            .limit(20)
            .execute()
        )
        return response.data
    except Exception as e:
        return {"error": str(e)}


@router.get("/weather/history")
def get_all_weather_history(limit: int = 50):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    try:
        response = (
            supabase.table("risk_assessments")
            .select(
                "barangay_id, timestamp, rainfall, humidity, "
                "temperature, wind_speed, final_risk, risk_level"
            )
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        return {"count": len(response.data), "records": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weather/history/summary/latest")
def get_latest_risk_per_barangay():
    try:
        response = (
            supabase.table("risk_assessments")
            .select(
                "barangay_id, timestamp, rainfall, humidity, "
                "temperature, wind_speed, final_risk, risk_level"
            )
            .order("timestamp", desc=True)
            .limit(500)
            .execute()
        )
        seen   = set()
        latest = []
        for row in response.data:
            bid = row["barangay_id"]
            if bid not in seen:
                seen.add(bid)
                latest.append(row)

        latest.sort(key=lambda r: (
            _RISK_ORDER.get(r.get("risk_level", ""), 99),
            -(r.get("final_risk") or 0),
        ))

        summary   = {"very_high": 0, "high": 0, "moderate": 0, "low": 0, "very_low": 0}
        level_map = {
            "VERY HIGH": "very_high", "HIGH": "high",
            "MODERATE": "moderate",  "LOW": "low", "VERY LOW": "very_low",
        }
        for row in latest:
            key = level_map.get(row.get("risk_level", ""))
            if key:
                summary[key] += 1

        return {"summary": summary, "barangays": latest}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Simulation history (unchanged) ───────────────────────────────────────────

@router.get("/simulation/history")
def get_simulation_history(limit: int = 10):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    try:
        response = (
            supabase.table("simulation_runs")
            .select(
                "id, created_at, rainfall, humidity, wind_speed, temperature, "
                "very_high_count, high_count, moderate_count, low_count, very_low_count"
            )
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"count": len(response.data), "runs": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation/history/{run_id}")
def get_simulation_run(run_id: int):
    try:
        response = (
            supabase.table("simulation_runs")
            .select("*")
            .eq("id", run_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Simulation run {run_id} not found.")
        row = response.data[0]
        return {
            "id": row["id"], "created_at": row["created_at"],
            "inputs": {
                "rainfall": row["rainfall"], "humidity": row["humidity"],
                "wind_speed": row["wind_speed"], "temperature": row["temperature"],
            },
            "summary": {
                "very_high": row["very_high_count"], "high": row["high_count"],
                "moderate": row["moderate_count"],   "low": row["low_count"],
                "very_low": row["very_low_count"],
            },
            "barangays": row["barangays"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation/history/latest")
def get_latest_simulation():
    try:
        response = (
            supabase.table("simulation_runs")
            .select("*")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=404, detail="No simulation runs found.")
        row = response.data[0]
        return {
            "id": row["id"], "created_at": row["created_at"],
            "inputs": {
                "rainfall": row["rainfall"], "humidity": row["humidity"],
                "wind_speed": row["wind_speed"], "temperature": row["temperature"],
            },
            "summary": {
                "very_high": row["very_high_count"], "high": row["high_count"],
                "moderate": row["moderate_count"],   "low": row["low_count"],
                "very_low": row["very_low_count"],
            },
            "barangays": row["barangays"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))