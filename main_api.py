from contextlib import asynccontextmanager
import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from modules.cnn_lstm import preload_all_models

from routes.auth_routes import (
    router as auth_router
)

from routes.barangay_routes import (
    router as barangay_router
)

from routes.prediction_routes import (
    router as prediction_router
)

from routes.sms_routes import (
    router as sms_router
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# ENVIRONMENT
# =========================================================

def _env_enabled(
    name: str,
    default: str = "true"
) -> bool:

    return os.environ.get(
        name,
        default
    ).lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on"
    }


# =========================================================
# APP STARTUP
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    if _env_enabled(
        "PRELOAD_MODELS_ON_STARTUP",
        "false"
    ):

        import threading

        threading.Thread(
            target=preload_all_models,
            daemon=True
        ).start()

        logger.info(
            "Background model preload started."
        )

    else:

        logger.info(
            "Skipping model preload."
        )

    yield


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Disaster Management API",
    version="1.0.0",
    lifespan=lifespan
)


# =========================================================
# CORS
# =========================================================

cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "*"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Disa    ster Management API Running"
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =========================================================
# ROUTES
# =========================================================

app.include_router(auth_router)
app.include_router(barangay_router)
app.include_router(prediction_router)
app.include_router(sms_router)
