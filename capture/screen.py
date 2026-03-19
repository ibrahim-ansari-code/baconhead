"""
capture/screen.py — Screen capture and OCR for the Roblox obby agent.

Public interface:
    Capturer.current_stage : int   — most recently read stage number
    Capturer.death_event   : bool  — True for exactly one frame when death detected

All other state is internal.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import mss
import mss.tools
import numpy as np
import yaml
from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)["capture"]


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------

class _EasyOCRBackend:
    def __init__(self, gpu: bool) -> None:
        import easyocr  # deferred — slow to import
        self._reader = easyocr.Reader(["en"], gpu=gpu)
        log.info("EasyOCR backend initialised (gpu=%s)", gpu)

    def read(self, image: np.ndarray) -> str:
        results = self._reader.readtext(image, detail=0)
        return " ".join(results).strip()


class _TesseractBackend:
    def __init__(self) -> None:
        import pytesseract  # deferred
        self._pytesseract = pytesseract
        log.info("Pytesseract backend initialised")

    def read(self, image: np.ndarray) -> str:
        pil_img = Image.fromarray(image)
        # PSM 7 = treat as single line of text; whitelist digits and slash
        cfg = "--psm 7"
        return self._pytesseract.image_to_string(pil_img, config=cfg).strip()


# ---------------------------------------------------------------------------
# Stage number parser
# ---------------------------------------------------------------------------

def _parse_stage(raw: str) -> Optional[int]:
    """
    Extract the stage number from raw OCR text like 'Stage 12 (5%)' or 'Stage 12/100'.
    Falls back to the first integer found if the pattern isn't matched.
    Returns None if nothing parseable is found.
    """
    import re
    # Primary: match "Stage <number>" explicitly
    m = re.search(r"[Ss]tage\s+(\d+)", raw)
    if m:
        return int(m.group(1))
    # Fallback: first number in string
    numbers = re.findall(r"\d+", raw)
    if numbers:
        return int(numbers[0])
    return None


# ---------------------------------------------------------------------------
# Main Capturer class
# ---------------------------------------------------------------------------

class Capturer:
    """
    Handles screen capture, OCR, and death detection.

    Usage:
        cap = Capturer()
        while True:
            cap.tick()
            stage = cap.current_stage
            died  = cap.death_event
    """

    def __init__(self) -> None:
        cfg = _load_config()

        self._game_region: Optional[dict] = cfg.get("game_region")  # None = auto
        self._hud_roi: dict = cfg["hud_roi"]
        self._fps: float = cfg.get("fps", 4)
        self._death_debounce: int = cfg.get("death_debounce_frames", 2)

        # Void HSV bounds for death detection (Phase 2)
        self._void_hsv_lower = np.array(cfg.get("void_hsv_lower", [0, 0, 0]), dtype=np.uint8)
        self._void_hsv_upper = np.array(cfg.get("void_hsv_upper", [179, 255, 50]), dtype=np.uint8)
        self._death_ratio_threshold: float = cfg.get("death_void_ratio_threshold", 0.85)

        # OCR backend
        backend_name = cfg.get("ocr_backend", "easyocr").lower()
        gpu = cfg.get("ocr_gpu", False)
        if backend_name == "easyocr":
            self._ocr = _EasyOCRBackend(gpu=gpu)
        elif backend_name == "pytesseract":
            self._ocr = _TesseractBackend()
        else:
            raise ValueError(f"Unknown ocr_backend: {backend_name!r}")

        # State
        self.current_stage: int = 1
        self.death_event: bool = False
        self.last_raw_ocr: str = ""  # raw OCR output from last tick
        self.last_frame: Optional[np.ndarray] = None  # BGR frame from last tick

        self._void_counter: int = 0  # consecutive void-fall frames
        self._frame_interval: float = 1.0 / self._fps
        self._last_tick: float = 0.0

        log.info(
            "Capturer ready — game_region=%s hud_roi=%s fps=%s",
            self._game_region,
            self._hud_roi,
            self._fps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Capture one frame, run OCR and death detection, update public state."""
        now = time.monotonic()
        elapsed = now - self._last_tick
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)
        self._last_tick = time.monotonic()

        with mss.mss() as sct:
            region = self._resolve_region(sct)
            raw_frame = np.array(sct.grab(region))  # BGRA uint8

        # Convert to BGR (drop alpha)
        frame_bgr = raw_frame[:, :, :3]
        self.last_frame = frame_bgr

        # --- OCR ---
        stage = self._run_ocr(frame_bgr)
        if stage is not None:
            self.current_stage = stage

        # --- Death detection ---
        self.death_event = self._detect_death(frame_bgr)

    def tick_fast(self) -> None:
        """Capture frame and run death detection only (no OCR). For 20fps loops."""
        with mss.mss() as sct:
            region = self._resolve_region(sct)
            raw_frame = np.array(sct.grab(region))  # BGRA uint8

        frame_bgr = raw_frame[:, :, :3]
        self.last_frame = frame_bgr

        self.death_event = self._detect_death(frame_bgr)

    def run_loop(self) -> None:
        """Blocking capture loop. Runs until KeyboardInterrupt."""
        log.info("Capture loop starting at %.1f fps", self._fps)
        try:
            while True:
                self.tick()
        except KeyboardInterrupt:
            log.info("Capture loop stopped by user")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_region(self, sct: mss.base.MSSBase) -> dict:
        """Return the mss grab region for the game window."""
        if self._game_region is not None:
            return self._game_region
        # Fall back to primary monitor
        monitor = sct.monitors[1]  # index 1 = primary monitor
        return {
            "top": monitor["top"],
            "left": monitor["left"],
            "width": monitor["width"],
            "height": monitor["height"],
        }

    def _crop_hud(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Crop the HUD stage counter region from a full game frame."""
        roi = self._hud_roi
        t = roi["top"]
        l = roi["left"]
        h = roi["height"]
        w = roi["width"]
        return frame_bgr[t : t + h, l : l + w]

    def _run_ocr(self, frame_bgr: np.ndarray) -> Optional[int]:
        """Run OCR on the HUD crop and return the parsed stage number."""
        hud_crop = self._crop_hud(frame_bgr)
        raw = self._ocr.read(hud_crop)
        self.last_raw_ocr = raw
        parsed = _parse_stage(raw)
        log.debug("OCR raw=%r parsed=%s", raw, parsed)
        return parsed

    def _detect_death(self, frame_bgr: np.ndarray) -> bool:
        """
        Detect death by void fall using HSV void ratio.
        Returns True on the first frame of a confirmed death (after debounce).
        """
        from vision.perception import compute_scene_state

        state = compute_scene_state(frame_bgr, self._void_hsv_lower, self._void_hsv_upper)
        void_ratio = state["void_ratio"]

        log.debug("void_ratio=%.3f", void_ratio)

        if void_ratio > self._death_ratio_threshold:
            self._void_counter += 1
        else:
            self._void_counter = 0

        if self._void_counter == self._death_debounce:
            log.info("Death detected (void_ratio=%.3f)", void_ratio)
            return True

        return False
