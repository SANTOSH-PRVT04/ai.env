# server.py — QTrack OpenEnv REST API Server
# Implements /reset, /step, /state endpoints as required by OpenEnv spec

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import uvicorn

from env import QTrackEnv

app = FastAPI(
    title="QTrack Hospital Queue Environment",
    description="OpenEnv-compatible REST API for QTrack",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance
_env: Optional[QTrackEnv] = None


# ── Request / Response Models ─────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None   # optional — no body required


class StepRequest(BaseModel):
    action_type: str = Field(default="noop", description="reschedule | alert_staff | open_walkin | noop")
    dept:        str = Field(default="",     description="Target department name")
    delay_mins:  int = Field(default=15,     description="Minutes to delay (reschedule only)")


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment. Returns first observation."""
    global _env
    _env = QTrackEnv(seed=request.seed)
    obs  = _env.reset()
    return {
        "observation": {
            "departments":      obs.departments,
            "overloaded_depts": obs.overloaded_depts,
            "idle_depts":       obs.idle_depts,
            "total_patients":   obs.total_patients,
            "avg_wait_minutes": obs.avg_wait_minutes,
            "critical_count":   obs.critical_count,
            "step_number":      obs.step_number,
            "timestamp":        obs.timestamp,
        },
        "status": "reset_ok",
    }


@app.post("/step")
def step(request: StepRequest = StepRequest()):
    """Execute one action. Returns observation, reward, done, info."""
    global _env
    if _env is None:
        _env = QTrackEnv()
        _env.reset()

    action = {
        "action_type": request.action_type,
        "dept":        request.dept,
        "delay_mins":  request.delay_mins,
    }
    result = _env.step(action)
    obs    = result.observation

    return {
        "observation": {
            "departments":      obs.departments,
            "overloaded_depts": obs.overloaded_depts,
            "idle_depts":       obs.idle_depts,
            "total_patients":   obs.total_patients,
            "avg_wait_minutes": obs.avg_wait_minutes,
            "critical_count":   obs.critical_count,
            "step_number":      obs.step_number,
            "timestamp":        obs.timestamp,
        },
        "reward": result.reward,
        "done":   result.done,
        "info":   result.info,
    }


@app.get("/state")
@app.post("/state")
def state():
    """Return full serialisable environment state."""
    global _env
    if _env is None:
        _env = QTrackEnv()
        _env.reset()
    return _env.state()


@app.get("/health")
def health():
    return {"status": "ok", "env": "QTrack-HospitalQueue-v1"}


@app.get("/")
def root():
    return {
        "name":    "QTrack Hospital Queue Environment",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
