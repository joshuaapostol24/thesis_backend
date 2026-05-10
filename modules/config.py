from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    # Manual .env parsing fallback (no python-dotenv installed)
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(
                    key.strip(),
                    value.strip().strip("\"'")
                )


# =========================================================
# DATABASE URL
# =========================================================

def get_database_url() -> str:
    """
    Build the PostgreSQL connection URL from environment variables.
    Raises RuntimeError immediately if required vars are missing —
    no hardcoded fallback credentials.
    """
    direct_url = (
        os.environ.get("DATABASE_URL")
        or os.environ.get("SUPABASE_DB_URL")
    )
    if direct_url:
        return _ensure_sslmode(direct_url)

    # All of these must be set in .env or the environment.
    # Missing any one of them is a configuration error, not a runtime condition.
    required = {
        "SUPABASE_DB_PASSWORD": os.environ.get("SUPABASE_DB_PASSWORD"),
        "SUPABASE_DB_HOST":     os.environ.get("SUPABASE_DB_HOST"),
        "SUPABASE_DB_USER":     os.environ.get("SUPABASE_DB_USER"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in your .env file or environment."
        )

    host     = required["SUPABASE_DB_HOST"]
    password = required["SUPABASE_DB_PASSWORD"]
    user     = required["SUPABASE_DB_USER"]
    db       = os.environ.get("SUPABASE_DB_NAME", "postgres")
    port     = os.environ.get("SUPABASE_DB_PORT", "5432")

    return _ensure_sslmode(
        f"postgresql://{user}:{password}@{host}:{port}/{db}"
    )


# =========================================================
# SSL MODE
# =========================================================

def _ensure_sslmode(url: str) -> str:
    if "sslmode=" in url:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}sslmode=require"


# =========================================================
# SUPABASE URL
# =========================================================

def get_supabase_url() -> str:
    value = os.environ.get("SUPABASE_URL")
    if not value:
        raise RuntimeError(
            "Missing SUPABASE_URL in your .env file."
        )
    return value


# =========================================================
# SUPABASE KEY
# =========================================================

def get_supabase_key() -> str:
    value = (
        os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    )
    if not value:
        raise RuntimeError(
            "Missing SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY "
            "in your .env file."
        )
    return value


# =========================================================
# OPENWEATHERMAP KEY
# =========================================================

def get_openweather_key() -> str:
    value = os.environ.get("OPENWEATHER_API_KEY")
    if not value:
        raise RuntimeError(
            "Missing OPENWEATHER_API_KEY in your .env file."
        )
    return value


# =========================================================
# JWT SECRET
# =========================================================

def get_secret_key() -> str:
    value = os.environ.get("SECRET_KEY")
    if not value:
        raise RuntimeError(
            "Missing SECRET_KEY in your .env file. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return value