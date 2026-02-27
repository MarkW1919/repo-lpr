"""Complete AI inference pipeline for RepoScan Pro.

Orchestrates vehicle detection, plate detection, OCR, classification,
and multi-object tracking into a single end-to-end pipeline.

Optimized for sustained 25-30 FPS on Jetson Orin Nano Super:
- Non-blocking async API dispatch (fire-and-forget)
- Frame-budget-aware processing with skip logic
- Thermal monitoring with automatic throttling
- Bounded detection history to prevent memory growth
- Reduced per-frame object churn (cached day_dir, JPEG params)
- Base64 frame transport (33% smaller than hex encoding)
"""

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import cv2
import httpx
import numpy as np
import yaml

from ai.plate_detector import PlateDetector
from ai.plate_ocr import PlateOCR
from ai.tracker import MultiObjectTracker
from ai.vehicle_classifier import VehicleClassifier
from ai.vehicle_detector import VehicleDetector

logger = logging.getLogger(__name__)


class ThermalMonitor:
    """Monitor Jetson GPU/CPU temperature with multi-zone throttling.

    Three thermal zones provide graduated response:
    - Normal (< 70C): full speed, all features enabled
    - Warm (70-80C): skip every other frame, reduce GPU load
    - Hot (>= 80C): skip 2 of 3 frames, aggressive power saving

    Reads max temperature across all available Jetson thermal zones.
    Hysteresis gap (resume at 65C) prevents rapid oscillation.
    """

    WARM_TEMP = 70.0
    THROTTLE_TEMP = 80.0
    RESUME_TEMP = 65.0
    THERMAL_ZONES: list[str] = [  # noqa: RUF012
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone1/temp",
    ]

    def __init__(self):
        self._throttle_level = 0  # 0=normal, 1=warm, 2=hot
        self._last_check = 0.0
        self._check_interval = 2.0
        self._last_temp = 0.0

    @property
    def throttle_level(self) -> int:
        """0=normal, 1=warm (skip some frames), 2=hot (heavy skip)."""
        now = time.monotonic()
        if now - self._last_check < self._check_interval:
            return self._throttle_level
        self._last_check = now
        temp = self._read_temp()
        if temp is None:
            return self._throttle_level
        self._last_temp = temp

        old_level = self._throttle_level
        if temp >= self.THROTTLE_TEMP:
            self._throttle_level = 2
        elif temp >= self.WARM_TEMP:
            self._throttle_level = 1
        elif temp < self.RESUME_TEMP:
            self._throttle_level = 0

        if self._throttle_level != old_level:
            labels = ["normal", "warm", "hot"]
            logger.warning(f"Thermal zone: {labels[self._throttle_level]} ({temp:.1f}C)")
        return self._throttle_level

    @property
    def is_throttled(self) -> bool:
        return self.throttle_level > 0

    @property
    def temperature(self) -> float | None:
        return self._read_temp()

    def _read_temp(self) -> float | None:
        """Read max temperature across all available thermal zones."""
        max_temp = None
        for zone in self.THERMAL_ZONES:
            try:
                with open(zone) as f:
                    temp = int(f.read().strip()) / 1000.0
                    if max_temp is None or temp > max_temp:
                        max_temp = temp
            except (FileNotFoundError, ValueError, PermissionError):
                continue
        return max_temp


class InferencePipeline:
    """End-to-end AI pipeline for vehicle and plate detection."""

    def __init__(self, config_path: str = "config.yaml"):
        try:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {config_path}: {e}")
            raise

        _precision = self.config.get("precision", "fp16")

        pipeline_cfg = self.config.get("pipeline", {})
        self.max_fps = pipeline_cfg.get("max_fps", 30)
        self.skip_frames = pipeline_cfg.get("skip_frames", 0)
        self.save_captures = pipeline_cfg.get("save_captures", True)
        self.capture_dir = Path(
            os.environ.get("CAPTURE_DIR", pipeline_cfg.get("capture_dir", "/data/captures"))
        )
        self.thumbnail_size = tuple(pipeline_cfg.get("thumbnail_size", [320, 240]))
        self.api_endpoint = pipeline_cfg.get("api_endpoint", "http://backend:8000/api/detections/")

        self.capture_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        logger.info("Initializing AI pipeline...")
        self.vehicle_detector = VehicleDetector(self.config["vehicle_detector"])
        self.plate_detector = PlateDetector(self.config["plate_detector"])
        self.plate_ocr = PlateOCR(self.config["plate_ocr"])
        self.vehicle_classifier = VehicleClassifier(self.config["vehicle_classifier"])
        self.tracker = MultiObjectTracker(self.config.get("tracker", {}))
        self.thermal = ThermalMonitor()

        self._frame_count = 0
        self._api_token: str | None = None
        self._http_client = httpx.AsyncClient(
            timeout=5.0,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

        # Async detection dispatch queue (non-blocking sends)
        self._dispatch_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=64)
        self._dispatcher_task: asyncio.Task | None = None

        # Performance counters
        self._perf_sum_ms = 0.0
        self._perf_count = 0

        # Pre-allocated JPEG encode params (avoids list alloc per imencode call)
        self._jpeg_full_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        self._jpeg_thumb_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
        self._jpeg_plate_params = [cv2.IMWRITE_JPEG_QUALITY, 90]

        # Cached day directory (refreshed once per day, not per frame)
        self._cached_day_str = ""
        self._cached_day_dir: Path | None = None
        self._cached_plate_dir: Path | None = None

        logger.info("AI pipeline initialized")

    async def start_dispatcher(self):
        """Start the background API dispatch task."""
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())

    async def _dispatch_loop(self):
        """Background task that sends detections to API without blocking pipeline."""
        while True:
            try:
                result = await self._dispatch_queue.get()
                await self._send_detection(result)
                self._dispatch_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Dispatch error: {e}")

    async def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        heading: float | None = None,
        speed: float | None = None,
        is_night: bool = False,
    ) -> list[dict]:
        """Process a single frame through the complete pipeline.

        Args:
            frame: BGR image from camera
            timestamp: Frame capture wall-clock time (epoch seconds)
            latitude: GPS latitude
            longitude: GPS longitude
            heading: GPS heading in degrees
            speed: Speed in mph
            is_night: Whether the camera is in night/IR mode

        Returns:
            List of detection results (duplicates filtered)
        """
        self._frame_count += 1

        # Graduated thermal-aware frame skipping
        # Level 0: no skip (or configured skip_frames)
        # Level 1 (warm): skip every other frame
        # Level 2 (hot): skip 2 of 3 frames
        thermal_level = self.thermal.throttle_level
        effective_skip = self.skip_frames
        if thermal_level >= 2:
            effective_skip = max(effective_skip, 2)
        elif thermal_level >= 1:
            effective_skip = max(effective_skip, 1)

        if effective_skip > 0 and self._frame_count % (effective_skip + 1) != 0:
            return []

        start_time = time.monotonic()
        results = []

        # 1. Detect vehicles
        vehicle_detections = self.vehicle_detector.detect(frame)

        # 2. Update tracker
        confirmed_tracks = self.tracker.update(vehicle_detections)

        # Build track ID lookup for classification caching
        track_by_bbox: dict[tuple, int] = {}
        for track in confirmed_tracks:
            tb = tuple(track.bbox)
            track_by_bbox[tb] = track.track_id

        # 3. For each detected vehicle, detect plates and classify
        for det in vehicle_detections:
            bbox = det["bbox"]
            vehicle_img = self._crop(frame, bbox)

            if vehicle_img.size == 0:
                continue

            frame_ts = timestamp or time.time()

            # Find matching track ID for this detection (by closest bbox)
            det_track_id = self._find_track_id(bbox, track_by_bbox)

            result = {
                "id": str(uuid.uuid4()),
                "vehicle_type": det.get("class"),
                "vehicle_confidence": det["confidence"],
                "bbox": bbox,
                "latitude": latitude,
                "longitude": longitude,
                "heading": heading,
                "speed": speed,
                "plate_reads": [],
                "timestamp": datetime.fromtimestamp(frame_ts, tz=UTC).isoformat(),
            }

            # 3a. Detect plates first (higher priority than classification)
            plate_detections = self.plate_detector.detect(vehicle_img)

            for plate_det in plate_detections:
                plate_bbox = plate_det["bbox"]
                plate_img = self.plate_detector.extract_plate_image(
                    vehicle_img, plate_bbox, is_night=is_night
                )

                # Compute full-frame plate bbox (accounting for padding offset)
                pad_frac = 0.1  # matches extract_plate_image default
                pad_x = int(plate_bbox[2] * pad_frac)
                pad_y = int(plate_bbox[3] * pad_frac)
                full_plate_bbox = [
                    bbox[0] + max(0, plate_bbox[0] - pad_x),
                    bbox[1] + max(0, plate_bbox[1] - pad_y),
                    plate_bbox[2] + 2 * pad_x,
                    plate_bbox[3] + 2 * pad_y,
                ]

                # 3b. OCR the plate (with state, bbox for merging)
                ocr_result = self.plate_ocr.read(
                    plate_img,
                    plate_bbox=full_plate_bbox,
                )
                if ocr_result:
                    # Skip duplicates (same plate seen within TTL)
                    if ocr_result.get("is_duplicate"):
                        continue

                    plate_read = {
                        "plate_text": ocr_result["plate_text"],
                        "confidence": ocr_result["confidence"],
                        "raw_ocr": ocr_result["raw_ocr"],
                        "plate_bbox": full_plate_bbox,
                    }

                    if self.save_captures:
                        plate_path = self._save_plate_image(plate_img)
                        plate_read["plate_image_path"] = plate_path

                    result["plate_reads"].append(plate_read)

            # 3c. Classify vehicle (after plates — lower priority, uses track cache)
            classification = self.vehicle_classifier.classify(vehicle_img, track_id=det_track_id)
            if classification:
                result["vehicle_make"] = classification.get("make")
                result["vehicle_model"] = classification.get("model")
                result["vehicle_year"] = classification.get("year")
                result["vehicle_color"] = classification.get("color")

            if self.save_captures and (result["plate_reads"] or det["confidence"] > 0.8):
                img_path, thumb_path = self._save_capture(frame, bbox)
                result["image_path"] = img_path
                result["thumbnail_path"] = thumb_path

            results.append(result)

            # Non-blocking dispatch to API (only if we have plate reads or high-conf vehicle)
            if result["plate_reads"] or det["confidence"] > 0.8:
                try:
                    self._dispatch_queue.put_nowait(result)
                except asyncio.QueueFull:
                    logger.debug("Dispatch queue full, dropping oldest detection")
                    try:
                        self._dispatch_queue.get_nowait()
                        self._dispatch_queue.put_nowait(result)
                    except asyncio.QueueEmpty:
                        pass

        # Performance tracking
        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._perf_sum_ms += elapsed_ms
        self._perf_count += 1

        if self._frame_count % 30 == 0:
            avg_ms = self._perf_sum_ms / max(self._perf_count, 1)
            temp = self.thermal.temperature
            temp_str = f", {temp:.0f}C" if temp is not None else ""
            logger.info(
                f"Pipeline: {len(vehicle_detections)} vehicles, "
                f"{sum(len(r['plate_reads']) for r in results)} plates, "
                f"{elapsed_ms:.1f}ms (avg {avg_ms:.1f}ms){temp_str}"
            )
            self._perf_sum_ms = 0.0
            self._perf_count = 0

        return results

    async def _send_detection(self, result: dict):
        """Send detection to backend API."""
        try:
            payload = {
                "vehicle_type": result.get("vehicle_type"),
                "vehicle_color": result.get("vehicle_color"),
                "vehicle_make": result.get("vehicle_make"),
                "vehicle_model": result.get("vehicle_model"),
                "vehicle_year": result.get("vehicle_year"),
                "vehicle_confidence": result.get("vehicle_confidence"),
                "bbox_x": result["bbox"][0],
                "bbox_y": result["bbox"][1],
                "bbox_w": result["bbox"][2],
                "bbox_h": result["bbox"][3],
                "image_path": result.get("image_path"),
                "thumbnail_path": result.get("thumbnail_path"),
                "latitude": result.get("latitude"),
                "longitude": result.get("longitude"),
                "heading": result.get("heading"),
                "speed": result.get("speed"),
                "plate_reads": [
                    {
                        "plate_text": pr["plate_text"],
                        "confidence": pr["confidence"],
                        "raw_ocr": pr.get("raw_ocr"),
                        "plate_image_path": pr.get("plate_image_path"),
                        "plate_bbox_x": pr["plate_bbox"][0] if pr.get("plate_bbox") and len(pr["plate_bbox"]) > 0 else None,
                        "plate_bbox_y": pr["plate_bbox"][1] if pr.get("plate_bbox") and len(pr["plate_bbox"]) > 1 else None,
                        "plate_bbox_w": pr["plate_bbox"][2] if pr.get("plate_bbox") and len(pr["plate_bbox"]) > 2 else None,
                        "plate_bbox_h": pr["plate_bbox"][3] if pr.get("plate_bbox") and len(pr["plate_bbox"]) > 3 else None,
                    }
                    for pr in result.get("plate_reads", [])
                ],
            }

            headers = {}
            if self._api_token:
                headers["Authorization"] = f"Bearer {self._api_token}"

            await self._http_client.post(self.api_endpoint, json=payload, headers=headers)
        except httpx.RequestError as e:
            logger.warning(f"Failed to send detection to API: {e}")
        except Exception as e:
            logger.warning(f"Unexpected dispatch error: {e}")

    @staticmethod
    def _find_track_id(det_bbox: list[int], track_by_bbox: dict[tuple, int]) -> int | None:
        """Find the track ID whose bbox best matches a detection bbox.

        Uses IoU matching to associate detections with confirmed tracks.
        Returns None if no track overlaps sufficiently (IoU > 0.3).
        """
        if not track_by_bbox:
            return None

        best_id = None
        best_iou = 0.3  # minimum IoU threshold

        dx, dy, dw, dh = det_bbox
        da = dw * dh
        if da <= 0:
            return None

        for (tx, ty, tw, th), tid in track_by_bbox.items():
            ix1 = max(dx, tx)
            iy1 = max(dy, ty)
            ix2 = min(dx + dw, tx + tw)
            iy2 = min(dy + dh, ty + th)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = da + tw * th - inter
            iou = inter / max(union, 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_id = tid

        return best_id

    def _crop(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        h, w = frame.shape[:2]
        x, y, bw, bh = bbox
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        return frame[y1:y2, x1:x2]

    def _get_day_dir(self) -> Path:
        """Get today's capture directory, cached to avoid per-frame strftime + mkdir."""
        day_str = datetime.now().strftime("%Y-%m-%d")
        if day_str != self._cached_day_str:
            self._cached_day_str = day_str
            self._cached_day_dir = self.capture_dir / day_str
            self._cached_day_dir.mkdir(parents=True, exist_ok=True)
            self._cached_plate_dir = self._cached_day_dir / "plates"
            self._cached_plate_dir.mkdir(parents=True, exist_ok=True)
        return self._cached_day_dir

    def _save_capture(self, frame: np.ndarray, bbox: list[int]) -> tuple[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        day_dir = self._get_day_dir()

        annotated = frame.copy()
        x, y, w, h = bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        img_path = str(day_dir / f"{ts}_full.jpg")
        cv2.imwrite(img_path, annotated, self._jpeg_full_params)

        thumb = cv2.resize(annotated, self.thumbnail_size)
        thumb_path = str(day_dir / f"{ts}_thumb.jpg")
        cv2.imwrite(thumb_path, thumb, self._jpeg_thumb_params)

        return img_path, thumb_path

    def _save_plate_image(self, plate_img: np.ndarray) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._get_day_dir()  # ensure plate dir exists

        path = str(self._cached_plate_dir / f"{ts}_plate.jpg")
        cv2.imwrite(path, plate_img, self._jpeg_plate_params)
        return path

    def set_api_token(self, token: str):
        self._api_token = token

    def reset(self):
        self.tracker.reset()
        self._frame_count = 0

    async def cleanup(self):
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatcher_task
        await self._http_client.aclose()

        # Release GPU resources
        self.vehicle_detector.engine.close()
        self.plate_detector.engine.close()
        self.plate_ocr.engine.close()
        self.vehicle_classifier.engine.close()


async def main():
    """Main entry point for the AI pipeline service."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config_path = os.environ.get("AI_CONFIG", "config.yaml")
    pipeline = InferencePipeline(config_path)

    await pipeline.start_dispatcher()

    import redis.asyncio as aioredis

    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    redis_client = aioredis.from_url(redis_url)

    logger.info("AI pipeline waiting for frames...")

    pubsub = redis_client.pubsub()
    await pubsub.subscribe("reposcan:frames")

    frame_min_interval = 1.0 / pipeline.max_fps
    last_process_time = 0.0

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            # Backpressure: skip frame if still within budget
            now = time.monotonic()
            if now - last_process_time < frame_min_interval:
                continue

            try:
                frame_data = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Invalid JSON in frame message: %s", e)
                continue
            raw = frame_data.get("frame")
            if not raw:
                continue
            # Support both base64 (new) and hex (legacy) encoding
            try:
                frame_bytes = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
            except Exception:
                try:
                    frame_bytes = np.frombuffer(bytes.fromhex(raw), dtype=np.uint8)
                except Exception as e:
                    logger.warning("Failed to decode frame data: %s", e)
                    continue
            frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

            if frame is not None:
                last_process_time = time.monotonic()
                await pipeline.process_frame(
                    frame,
                    timestamp=frame_data.get("timestamp"),
                    latitude=frame_data.get("latitude"),
                    longitude=frame_data.get("longitude"),
                    heading=frame_data.get("heading"),
                    speed=frame_data.get("speed"),
                )
    except asyncio.CancelledError:
        pass
    finally:
        await pipeline.cleanup()
        await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
