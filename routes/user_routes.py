import os

from fastapi import APIRouter

from dotenv import load_dotenv

from supabase import (
    create_client,
    Client
)

load_dotenv()

router = APIRouter(
    tags=["Users"]
)

supabase: Client = create_client(

    os.getenv("SUPABASE_URL"),

    os.getenv("SUPABASE_KEY")

)


@router.get("/users")
def get_users():

    result = supabase.table("user") \
        .select("*") \
        .eq("status", "verified") \
        .execute()

    return result.data


