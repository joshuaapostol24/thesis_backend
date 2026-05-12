import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.news_routes import router as news_router
from modules.cnn_lstm import preload_all_models
from routes.barangay_routes import router as barangay_router
from routes.prediction_routes import router as prediction_router
from routes.simulation_routes import router as simulation_router
from routes.sms_routes import router as sms_router

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
    from modules.database import load_hazard_cache
    try:
        load_hazard_cache()
        logger.info("Hazard profile cache loaded successfully.")
    except Exception as e:
        logger.warning("Failed to preload hazard cache: %s (will use per-request DB queries)", e)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

app.include_router(barangay_router)
app.include_router(prediction_router)
app.include_router(simulation_router)
app.include_router(news_router)
app.include_router(sms_router)

# =========================================================
# LOGIN
# =========================================================

from pydantic import BaseModel

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/login")
def login(data: LoginRequest):
    if (
        data.email == "admin@resq.com"
        and
        data.password == "admin123"
    ):
        return {
            "success": True,
            "name": "Administrator",
            "email": "admin@resq.com",
        }

    return {
        "success": False,
        "message": "Invalid email or password"
    }
