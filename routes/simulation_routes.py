from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional

from modules.simulation import run_simulation

router = APIRouter(prefix="/simulate", tags=["Simulation"])


# =========================================================
# REQUEST MODEL
# =========================================================

class SimulationRequest(BaseModel):
    rainfall:     float = Field(..., ge=0,   le=500, description="Rainfall in mm/h")
    humidity:     float = Field(..., ge=0,   le=100, description="Humidity in %")
    wind_speed:   float = Field(..., ge=0,   le=300, description="Wind speed in km/h")
    temperature:  float = Field(..., ge=-20, le=60,  description="Temperature in °C")
    barangay_ids: Optional[List[int]] = Field(
        default=None,
        description="Barangay IDs to simulate (1–15). Omit for all 15.",
    )

    @validator("barangay_ids", each_item=True, pre=True)
    def validate_barangay_id(cls, v):
        if v not in range(1, 16):
            raise ValueError(f"barangay_id must be between 1 and 15, got {v}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "rainfall":     25.0,
                "humidity":     85.0,
                "wind_speed":   40.0,
                "temperature":  30.0,
                "barangay_ids": None,
            }
        }


# =========================================================
# ROUTES
# =========================================================

@router.post("/")
def simulate_all(request: SimulationRequest):
    """
    Run a hazard simulation for all 15 barangays (or a subset).
    Read-only — nothing is written to the DB and no alerts are sent.
    Results are sorted HIGH → MODERATE → LOW.
    """
    try:
        result = run_simulation(
            rainfall     = request.rainfall,
            humidity     = request.humidity,
            wind_speed   = request.wind_speed,
            temperature  = request.temperature,
            barangay_ids = request.barangay_ids,
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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