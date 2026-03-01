"""Main camera controller that orchestrates all camera subsystems.

Performance optimizations:
- Base64 frame encoding (33% smaller than hex, reduces Redis bandwidth)
- Cached PTZ position (avoid dict copy per frame)
- Pre-allocated JPEG encode params
"""

import asyncio
import base64
import json
import logging
import os
import time

import cv2
import numpy as np
import yaml

from camera.calibration import CameraCalibration
from camera.capture import VideoCapture
from camera.night_vision import NightVisionController
from camera.preprocessor import ImagePreprocessor
from camera.ptz import PTZController
from camera.rtsp_server import RTSPServer

logger = logging.getLogger(__name__)


class CameraController:
    """Main controller that integrates all camera subsystems."""

    def __init__(self, config_path: str = "config.yaml"):
        try:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("Camera config not found: %s", config_path)
            raise
        except yaml.YAMLError as e:
            logger.error("Invalid camera config YAML: %s", e)
            raise

        self.capture = VideoCapture(self.config)
        self.ptz = PTZController(self.config.get("ptz", {}))
        self.night_vision = NightVisionController(self.config.get("night_vision", {}))
        self.preprocessor = ImagePreprocessor(self.config)
        self.rtsp = RTSPServer(self.config)
        self.calibration = CameraCalibration()

        self._running = False

    def start(self):
        """Start all camera subsystems."""
        logger.info("Starting camera controller...")

        # Start video capture
        if not self.capture.start():
            logger.error("Failed to start video capture")
            return False

        # Start RTSP streaming
        self.rtsp.start()

        # Start night vision auto-monitoring
        self.night_vision.start_auto_monitoring(self.capture.read)

        self._running = True
        logger.info("Camera controller started")
        return True

    def get_frame(self, preprocess: bool = True) -> np.ndarray | None:
        """Get the latest preprocessed frame.

        Args:
            preprocess: Apply CLAHE/denoise preprocessing

        Returns:
            BGR frame or None
        """
        frame = self.capture.read()
        if frame is None:
            return None

        # Apply distortion correction
        if self.calibration.is_calibrated:
            frame = self.calibration.undistort(frame)

        # Apply preprocessing
        if preprocess:
            frame = self.preprocessor.process(frame)

        return frame

    def get_frame_timestamped(self, preprocess: bool = True):
        """Get the latest preprocessed frame with timestamp metadata.

        Returns:
            TimestampedFrame with processed image, or None
        """
        ts_frame = self.capture.read_timestamped()
        if ts_frame is None or ts_frame.frame is None:
            return None

        frame = ts_frame.frame

        # Apply distortion correction
        if self.calibration.is_calibrated:
            frame = self.calibration.undistort(frame)

        # Apply preprocessing
        if preprocess:
            frame = self.preprocessor.process(frame)

        # Return updated timestamped frame with processed image
        ts_frame.frame = frame
        return ts_frame

    def is_ptz_moving(self) -> bool:
        """Check if PTZ is mid-motion (frames may be blurry)."""
        return self.ptz.is_moving

    def get_status(self) -> dict:
        """Get current camera status including PTZ motion state."""
        return {
            "is_connected": self.capture.is_running,
            "resolution": f"{self.config['camera']['width']}x{self.config['camera']['height']}",
            "fps": self.capture.actual_fps,
            "night_mode": self.night_vision.is_night_mode,
            "ir_enabled": self.night_vision.is_ir_enabled,
            "ptz_position": self.ptz.position,
            "ptz_motion": self.ptz.motion_state.value,
            "ptz_connected": self.ptz.is_connected,
            "streaming": self.rtsp.is_running,
            "rtsp_url": self.rtsp.url if self.rtsp.is_running else None,
        }

    def stop(self):
        """Stop all camera subsystems."""
        self._running = False
        self.capture.stop()
        self.rtsp.stop()
        self.night_vision.stop()
        self.ptz.close()
        logger.info("Camera controller stopped")


async def main():
    """Main entry point for the camera service.

    Captures frames and publishes them to Redis for the AI pipeline.
    Uses adaptive timing to maintain target FPS and timestamped frames
    for pipeline synchronization.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config_path = os.environ.get("CAMERA_CONFIG", "config.yaml")
    controller = CameraController(config_path)

    if not controller.start():
        logger.error("Failed to start camera controller")
        return

    # Connect to Redis for frame publishing
    import redis.asyncio as aioredis

    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    redis_client = aioredis.from_url(redis_url)

    target_fps = controller.config["camera"].get("fps", 30)
    frame_interval = 1.0 / target_fps
    status_interval = 5.0
    last_status = 0
    last_seq = -1
    frames_sent = 0
    fps_log_interval = 10.0
    last_fps_log = time.monotonic()

    # Pre-allocated JPEG encode params (avoid list alloc per frame)
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 80]

    try:
        while True:
            loop_start = time.monotonic()

            # Skip frame acquisition when PTZ is moving — frames captured
            # during motor motion are blurry and useless for LPR/inference.
            # RTSP stream continues (viewers see the pan) but we don't push
            # to the inference pipeline.
            if controller.is_ptz_moving():
                await asyncio.sleep(frame_interval)
                continue

            ts_frame = controller.get_frame_timestamped(preprocess=True)

            if ts_frame is not None and ts_frame.sequence != last_seq:
                last_seq = ts_frame.sequence
                frame = ts_frame.frame

                # Encode frame for Redis transport (base64 = 33% smaller than hex)
                success, buffer = cv2.imencode(".jpg", frame, jpeg_params)
                if not success or buffer is None:
                    logger.warning("JPEG encoding failed, skipping frame")
                    continue
                frame_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")

                frame_data = json.dumps(
                    {
                        "frame": frame_b64,
                        "timestamp": ts_frame.wall_time,
                        "sequence": ts_frame.sequence,
                        "fps_actual": round(ts_frame.fps_actual, 1),
                        "ptz_position": controller.ptz.position,
                    }
                )

                try:
                    await redis_client.publish("reposcan:frames", frame_data)
                except Exception as e:
                    logger.warning("Failed to publish frame: %s", e)
                frames_sent += 1

            # Publish status periodically
            now = time.time()
            if now - last_status > status_interval:
                status = controller.get_status()
                try:
                    await redis_client.publish(
                        "reposcan:alerts",
                        json.dumps({"type": "system:camera", "data": status}),
                    )
                except Exception as e:
                    logger.warning("Failed to publish status: %s", e)
                last_status = now

            # Log throughput periodically
            mono_now = time.monotonic()
            if mono_now - last_fps_log >= fps_log_interval:
                actual = frames_sent / (mono_now - last_fps_log)
                logger.info(
                    "Camera publish rate: %.1f FPS (%d frames in %.0fs)",
                    actual,
                    frames_sent,
                    mono_now - last_fps_log,
                )
                frames_sent = 0
                last_fps_log = mono_now

            # Adaptive sleep: account for processing time to hit target FPS
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        pass
    finally:
        controller.stop()
        try:
            await asyncio.wait_for(redis_client.close(), timeout=2)
        except (TimeoutError, Exception):
            logger.warning("Redis close timed out or failed")


if __name__ == "__main__":
    asyncio.run(main())
