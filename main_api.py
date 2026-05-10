import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from modules.cnn_lstm import preload_all_models
from routes.auth_routes import router as auth_router
from routes.barangay_routes import router as barangay_router
from routes.prediction_routes import router as prediction_router
from routes.sms_routes import router as sms_router
from routes.simulation_routes import router as simulation_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# ENVIRONMENT HELPERS
# =========================================================

def _env_enabled(name: str, default: str = "true") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "y", "on"}


# =========================================================
# APP STARTUP
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load hazard profiles cache for fast lookups
    from modules.database import load_hazard_cache
    try:
        load_hazard_cache()
        logger.info("Hazard profile cache loaded successfully.")
    except Exception as e:
        logger.warning("Failed to preload hazard cache: %s (will use per-request DB queries)", e)

    # Optionally preload models
    if _env_enabled("PRELOAD_MODELS_ON_STARTUP", "false"):
        import threading
        threading.Thread(target=preload_all_models, daemon=True).start()
        logger.info("Background model preload started.")
    else:
        logger.info("Skipping model preload (PRELOAD_MODELS_ON_STARTUP not set).")
    yield

    logger.info("Application shutting down.")


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Disaster Management API",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

cors_origins_raw = os.environ.get("CORS_ORIGINS", "")
cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
allow_credentials = _env_enabled("CORS_ALLOW_CREDENTIALS", "false")

if "*" in cors_origins and allow_credentials:
    logger.warning(
        "CORS_ORIGINS=* combined with CORS_ALLOW_CREDENTIALS=true is invalid. "
        "Disabling credentials for wildcard origin."
    )
    allow_credentials = False

if not cors_origins:
    logger.warning(
        "CORS_ORIGINS not set. Defaulting to no CORS (same-origin only). "
        "Set CORS_ORIGINS in your .env to allow cross-origin requests."
    )
    cors_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {"message": "Disaster Management API Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# ROUTES
# =========================================================

app.include_router(auth_router)
app.include_router(barangay_router)
app.include_router(prediction_router)
app.include_router(sms_router)
app.include_router(simulation_router)