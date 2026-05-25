from fastapi import APIRouter
from supabase import create_client
import os

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

@router.get("/dashboard")
def get_dashboard_data():

    # TOTAL USERS
    users_response = (
        supabase
        .table("users")
        .select("*", count="exact")
        .execute()
    )

    total_users = users_response.count or 0

    # TOTAL REPORTS
    reports_response = (
        supabase
        .table("reports")
        .select("*", count="exact")
        .execute()
    )

    total_reports = reports_response.count or 0

    return {
        "users": {
            "total": total_users
        },
        "reports": {
            "total": total_reports
        }
    }


@router.get("/weather/history/{barangay_id}")
def get_weather_history(barangay_id: int):

    response = (
        supabase
        .table("weather_data")
        .select("*")
        .eq("barangay_id", barangay_id)
        .order("timestamp", desc=True)
        .limit(20)
        .execute()
    )

    return response.data