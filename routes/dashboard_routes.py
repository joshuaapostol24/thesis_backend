from fastapi import APIRouter, HTTPException
from supabase import create_client
from pymongo import MongoClient
import os

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

_RISK_ORDER = {
    "VERY HIGH": 0, "HIGH": 1,
    "MODERATE": 2, "LOW": 3, "VERY LOW": 4,
}

_mongo_client   = None
_news_collection = None

def _get_news_collection():
    global _mongo_client, _news_collection
    if _news_collection is None:
        mongo_url        = os.environ.get("MONGODB_URL", "mongodb://localhost:27017")
        _mongo_client    = MongoClient(mongo_url)
        db               = _mongo_client[os.environ.get("MONGODB_DB", "resq")]
        _news_collection = db["news"]
    return _news_collection


@router.get("/dashboard")
def get_dashboard_data():
    """
    Returns complete dashboard summary including:
    - Total users, reports, assessments
    - Pending reports count + latest pending report
    - Latest 5 news announcements
    - Latest simulation with top barangays ranked by risk
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

    # ── Pending reports ───────────────────────────────────────────────────────
    pending_response = (
        supabase.table("reports")
        .select("id, title, type, created_at")
        .eq("status", "pending")
        .order("created_at", desc=True)
        .execute()
    )
    pending_data    = pending_response.data or []
    pending_count   = len(pending_data)
    latest_pending  = None
    if pending_data:
        p = pending_data[0]
        latest_pending = {
            "title":      p.get("title") or p.get("type") or "Untitled Report",
            "created_at": p.get("created_at"),
        }

    # ── Total risk assessments ────────────────────────────────────────────────
    assessments_response = (
        supabase.table("risk_assessments")
        .select("*", count="exact")
        .execute()
    )
    total_assessments = assessments_response.count or 0

    # ── Latest simulation run with top barangays ──────────────────────────────
    sim_response = (
        supabase.table("simulation_runs")
        .select("*")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    latest_simulation = None
    if sim_response.data:
        s = sim_response.data[0]

        # Get barangay names from risk_assessments for this sim's timestamp
        # Find top barangays from the most recent risk assessments
        risk_response = (
            supabase.table("risk_assessments")
            .select("barangay_id, risk_level, final_risk, timestamp")
            .order("timestamp", desc=True)
            .limit(500)
            .execute()
        )
        seen           = set()
        latest_per_bgy = []
        for row in (risk_response.data or []):
            bid = row["barangay_id"]
            if bid not in seen:
                seen.add(bid)
                latest_per_bgy.append(row)

        latest_per_bgy.sort(key=lambda r: (
            _RISK_ORDER.get(r.get("risk_level", ""), 99),
            -(r.get("final_risk") or 0),
        ))

        # Try to get barangay names
        barangay_ids = [r["barangay_id"] for r in latest_per_bgy]
        barangay_names = {}
        if barangay_ids:
            try:
                bgy_response = (
                    supabase.table("barangays")
                    .select("id, name")
                    .in_("id", barangay_ids)
                    .execute()
                )
                for b in (bgy_response.data or []):
                    barangay_names[b["id"]] = b["name"]
            except Exception:
                pass

        top_barangays = []
        for rank, row in enumerate(latest_per_bgy, start=1):
            bid  = row["barangay_id"]
            name = barangay_names.get(bid, f"Barangay {bid}")
            final_risk = row.get("final_risk") or 0
            top_barangays.append({
                "rank":          rank,
                "barangay_id":   bid,
                "barangay_name": name,
                "risk_level":    row.get("risk_level", "N/A"),
                "final_risk":    round(float(final_risk) * 100, 1),
            })

        latest_simulation = {
            "id":           s["id"],
            "created_at":   s["created_at"],
            "rainfall":     s.get("rainfall"),
            "humidity":     s.get("humidity"),
            "top_barangays": top_barangays,
        }

    # ── Latest 5 news announcements ───────────────────────────────────────────
    latest_news = []
    try:
        collection = _get_news_collection()
        news_items = list(
            collection.find({}, {"_id": 0, "title": 1, "category": 1, "createdAt": 1, "date": 1})
            .sort("createdAt", -1)
            .limit(5)
        )
        for item in news_items:
            latest_news.append({
                "title":      item.get("title", ""),
                "category":   item.get("category", ""),
                "created_at": item.get("createdAt") or item.get("date"),
            })
    except Exception as e:
        print("News fetch error:", str(e))

    return {
        "users":       {"total": total_users},
        "reports":     {"total": total_reports},
        "assessments": {"total": total_assessments},
        "pending_reports": {
            "count":         pending_count,
            "latest_report": latest_pending,
        },
        "latest_news":       latest_news,
        "latest_simulation": latest_simulation,
    }


# ── Weather risk history ──────────────────────────────────────────────────────

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


# ── Simulation history ────────────────────────────────────────────────────────

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
            "barangays": row.get("barangays"),
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
            "barangays": row.get("barangays"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))