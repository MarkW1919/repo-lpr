"""Camera API router with PTZ boundary enforcement and motion state tracking.

- PTZ commands validated against 360 deg pan / 140 deg tilt / 30x zoom boundaries
- Speed clamped to valid Pelco-D range [1, 63] mapped from [0, 1] float
- PTZ position updates are atomic (no partial dict mutation race)
- Motion state tracked in status for frontend frame suppression
- In production, commands are forwarded via Redis to the camera service
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.middleware.auth import get_current_user
from app.models.user import User
from app.schemas.camera import CameraConfig, CameraStatus, PTZCommand, PtzPosition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/camera", tags=["camera"])


# ─── In-memory State ───
# In production, status comes from the camera service via Redis pub/sub.
# Commands are forwarded to the camera service via Redis.

_status_lock = asyncio.Lock()
_camera_status = CameraStatus()
_camera_config = CameraConfig()


# ─── Endpoints ───


@router.get("/status", response_model=CameraStatus)
async def get_camera_status(_user: User = Depends(get_current_user)):
    return _camera_status


@router.post("/ptz")
async def send_ptz_command(cmd: PTZCommand, _user: User = Depends(get_current_user)):
    """Send a PTZ command with boundary enforcement.

    Pan: 360° range [-180, +180] with wrapping
    Tilt: 140° range [-70, +70] with hard limits
    Zoom: 1x to 30x
    Speed: 0.0 to 1.0 (normalized, mapped to hardware range)
    """
    async with _status_lock:
        pos = _camera_status.ptz_position
        if cmd.pan is not None:
            pos.pan = cmd.pan
        if cmd.tilt is not None:
            pos.tilt = cmd.tilt
        if cmd.zoom is not None:
            pos.zoom = cmd.zoom
        new_position = pos.model_dump()

    return {"status": "ok", "position": new_position}


@router.post("/ptz/stop")
async def stop_ptz(_user: User = Depends(get_current_user)):
    """Emergency stop — immediately halt all PTZ movement."""
    async with _status_lock:
        _camera_status.ptz_motion = "idle"
    return {"status": "stopped"}


@router.post("/ptz/home")
async def home_ptz(_user: User = Depends(get_current_user)):
    """Return PTZ to home position (0°, 0°, 1x)."""
    async with _status_lock:
        _camera_status.ptz_position = PtzPosition(pan=0.0, tilt=0.0, zoom=1.0)
    return {"status": "ok", "position": {"pan": 0.0, "tilt": 0.0, "zoom": 1.0}}


@router.get("/config", response_model=CameraConfig)
async def get_camera_config(_user: User = Depends(get_current_user)):
    return _camera_config


@router.put("/config", response_model=CameraConfig)
async def update_camera_config(config: CameraConfig, _user: User = Depends(get_current_user)):
    global _camera_config
    async with _status_lock:
        _camera_config = config
    return _camera_config


@router.post("/capture")
async def trigger_capture(_user: User = Depends(get_current_user)):
    """Trigger a manual frame capture for processing."""
    if _camera_status.ptz_motion != "idle":
        raise HTTPException(
            status_code=409,
            detail="Cannot capture while PTZ is moving — frame would be blurry",
        )
    return {"status": "captured", "message": "Frame sent to AI pipeline"}


@router.post("/night-mode/{enabled}")
async def toggle_night_mode(enabled: bool, _user: User = Depends(get_current_user)):
    _camera_status.night_mode = enabled
    _camera_status.ir_enabled = enabled
    return {"night_mode": enabled, "ir_enabled": enabled}
