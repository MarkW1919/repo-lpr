"""RTSP streaming server using GStreamer."""

import logging
import subprocess

logger = logging.getLogger(__name__)


class RTSPServer:
    """RTSP server for live camera feed streaming.

    Streams H.264/H.265 encoded video via GStreamer RTSP server.
    """

    def __init__(self, config: dict):
        rtsp_cfg = config.get("rtsp", {})
        self.enabled = rtsp_cfg.get("enabled", True)
        self.port = rtsp_cfg.get("port", 8554)
        self.path = rtsp_cfg.get("path", "/stream")
        self.codec = rtsp_cfg.get("codec", "h264")
        self.bitrate = rtsp_cfg.get("bitrate", 4000000)
        self.keyframe_interval = rtsp_cfg.get("keyframe_interval", 30)

        cam_cfg = config.get("camera", {})
        self.width = cam_cfg.get("width", 1920)
        self.height = cam_cfg.get("height", 1080)
        self.fps = cam_cfg.get("fps", 30)

        self._process: subprocess.Popen | None = None
        self._running = False

    def _build_pipeline(self) -> str:
        """Build GStreamer RTSP server pipeline."""
        encoder = "nvv4l2h264enc" if self.codec == "h264" else "nvv4l2h265enc"
        payloader = "rtph264pay" if self.codec == "h264" else "rtph265pay"

        pipeline = (
            f"( nvarguscamerasrc sensor-id=0 "
            f"! video/x-raw(memory:NVMM),width={self.width},height={self.height},"
            f"framerate={self.fps}/1 "
            f"! {encoder} bitrate={self.bitrate} "
            f"iframeinterval={self.keyframe_interval} "
            f"preset-level=1 control-rate=1 "
            f"! {payloader} name=pay0 pt=96 )"
        )
        return pipeline

    def start(self):
        """Start RTSP server using gst-rtsp-launch or test-launch."""
        if not self.enabled:
            logger.info("RTSP server disabled")
            return

        pipeline = self._build_pipeline()

        # Use GStreamer RTSP test server
        cmd = [
            "gst-rtsp-launch",
            f"--port={self.port}",
            f"--mapping={self.path}",
            pipeline,
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._running = True
            logger.info(f"RTSP server started at rtsp://0.0.0.0:{self.port}{self.path}")
        except FileNotFoundError:
            logger.warning(
                "gst-rtsp-launch not found. Install gstreamer1.0-rtsp or use GstRtspServer API."
            )

    @property
    def url(self) -> str:
        return f"rtsp://0.0.0.0:{self.port}{self.path}"

    @property
    def is_running(self) -> bool:
        if self._process:
            return self._process.poll() is None
        return False

    def stop(self):
        """Stop RTSP server."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("RTSP server did not terminate, killing")
                self._process.kill()
                try:
                    self._process.wait(timeout=2)
                except (subprocess.TimeoutExpired, OSError):
                    logger.debug("RTSP process already reaped")
            self._process = None
        self._running = False
        logger.info("RTSP server stopped")
