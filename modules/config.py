from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()
else:
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

DEFAULT_SUPABASE_HOST = "db.jpovamcznyzoemcnjrgs.supabase.co"
DEFAULT_SUPABASE_DB = "postgres"
DEFAULT_SUPABASE_USER = "postgres"
DEFAULT_SUPABASE_PORT = "5432"


def get_database_url() -> str:
    direct_url = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")
    if direct_url:
        return _ensure_sslmode(direct_url)

    password = os.environ.get("SUPABASE_DB_PASSWORD")
    if not password:
        raise RuntimeError(
            "Missing database config. Set DATABASE_URL, SUPABASE_DB_URL, or "
            "SUPABASE_DB_PASSWORD in your .env file."
        )

    host = os.environ.get("SUPABASE_DB_HOST", DEFAULT_SUPABASE_HOST)
    db = os.environ.get("SUPABASE_DB_NAME", DEFAULT_SUPABASE_DB)
    user = os.environ.get("SUPABASE_DB_USER", DEFAULT_SUPABASE_USER)
    port = os.environ.get("SUPABASE_DB_PORT", DEFAULT_SUPABASE_PORT)
    return _ensure_sslmode(f"postgresql://{user}:{password}@{host}:{port}/{db}")


def _ensure_sslmode(url: str) -> str:
    if "sslmode=" in url:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}sslmode=require"


def get_supabase_url() -> str:
    value = os.environ.get("SUPABASE_URL") or "https://jpovamcznyzoemcnjrgs.supabase.co"
    if not value:
        raise RuntimeError("Missing SUPABASE_URL in your .env file.")
    return value


def get_supabase_key() -> str:
    value = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not value:
        raise RuntimeError("Missing SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY in your .env file.")
    return value
