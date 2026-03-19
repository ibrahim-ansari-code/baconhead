"""
agent/twotier.py — Two-tier agent: Gemini planner + CNN controller.

Gemini provides high-level instructions every 1.5s.
The CNN produces actions at ~20fps from stacked frames.
Planner instructions bias the CNN logits before argmax.

Public interface:
    TwoTierAgent.run(duration_seconds, dry_run) -> None
"""

from __future__ import annotations

import atexit
import logging
import signal
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from agent.planner import GeminiPlanner
from capture.screen import Capturer
from control.actions import (
    ACTION_NAMES,
    begin_action,
    end_action,
    release_all,
    set_camera_angle,
)
from vision.model import ObbyCNN
from vision.preprocess import preprocess_frame
from vision.stacker import FrameStacker

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_twotier_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)["twotier"]


class TwoTierAgent:
    """
    Two-tier agent combining a Gemini high-level planner with a CNN controller.

    The CNN runs at ~20fps producing action logits. The planner's cached
    instruction biases those logits before argmax.
    """

    def __init__(
        self,
        capturer: Capturer,
        checkpoint_path: str | None = None,
        bias_scale: float | None = None,
        use_gemini: bool = True,
    ) -> None:
        cfg = _load_twotier_config()

        self._capturer = capturer
        self._cfg = cfg
        self._fps: int = cfg.get("cnn_fps", 20)
        self._frame_interval: float = 1.0 / self._fps
        self._ocr_every_n: int = cfg.get("ocr_every_n_frames", 10)
        self._respawn_wait: float = cfg.get("respawn_wait", 3.0)
        self._respawn_timeout: float = cfg.get("respawn_timeout", 10.0)
        self._respawn_void_threshold: float = cfg.get("respawn_void_threshold", 0.3)
        self._bias_scale: float = bias_scale if bias_scale is not None else cfg.get("bias_scale", 2.0)

        # CNN — default path is same file that scripts/train_bc.py saves (bc_best.pt)
        _default_cp = cfg.get("checkpoint", "checkpoints/bc_best.pt")
        cp = checkpoint_path or _default_cp
        if not Path(cp).is_absolute():
            cp = str(Path(__file__).resolve().parent.parent / cp)
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._model = ObbyCNN(n_actions=6)
        state = torch.load(cp, map_location=self._device, weights_only=False)
        if "model_state_dict" in state:
            self._model.load_state_dict(state["model_state_dict"])
        else:
            self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()
        log.info("CNN loaded from %s on %s", cp, self._device)

        # Frame stacker
        self._stacker = FrameStacker(stack_size=4)

        # Planner
        self._planner: GeminiPlanner | None = None
        self._planner_result: dict = {"instruction": "idle", "confidence": 0.0, "reason": "init"}
        self._planner_lock = threading.Lock()
        if use_gemini:
            self._planner = GeminiPlanner()
            t = threading.Thread(target=self._planner_loop, daemon=True)
            t.start()

        # Build instruction → action index map
        self._instruction_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(ACTION_NAMES)
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, duration_seconds: int = 300, dry_run: bool = False) -> None:
        """Run the two-tier agent for *duration_seconds*."""
        # Safety: release all keys on exit
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        log.info(
            "Starting two-tier agent — duration=%ds fps=%d dry_run=%s gemini=%s",
            duration_seconds, self._fps, dry_run, self._planner is not None,
        )

        # One-time camera setup
        if not dry_run:
            set_camera_angle()
            time.sleep(0.5)

        # Initial capture and stacker reset
        self._capturer.tick()
        frame = self._capturer.last_frame
        processed = preprocess_frame(frame)
        self._stacker.reset(processed)

        start = time.monotonic()
        frame_count = 0
        deaths = 0

        try:
            while time.monotonic() - start < duration_seconds:
                tick_start = time.monotonic()

                # 1. Capture — tick_fast most frames, full tick every N for OCR
                if frame_count % self._ocr_every_n == 0:
                    self._capturer.tick()
                else:
                    self._capturer.tick_fast()

                # 2. Death check
                if self._capturer.death_event:
                    deaths += 1
                    log.info("Death #%d at frame %d", deaths, frame_count)
                    if not dry_run:
                        self._handle_death()
                    frame_count += 1
                    continue

                # 3. Preprocess and stack
                frame = self._capturer.last_frame
                processed = preprocess_frame(frame)
                stacked = self._stacker.push(processed)

                # 4. CNN inference
                logits = self._cnn_inference(stacked)

                # 5. Planner bias (non-blocking — reads from background thread)
                planner_result = None
                if self._planner is not None:
                    with self._planner_lock:
                        planner_result = self._planner_result

                if planner_result is not None:
                    logits = self._bias_logits(logits, planner_result)

                # 6. Select and execute action
                action = int(np.argmax(logits))

                # Idle suppression: if CNN picks idle, fall back to best non-idle action
                if action == 5:
                    non_idle_logits = logits.copy()
                    non_idle_logits[5] = -np.inf
                    action = int(np.argmax(non_idle_logits))
                    if frame_count % 20 == 0:
                        log.info("Frame %d: idle suppressed → %s", frame_count, ACTION_NAMES[action])

                if frame_count % 20 == 0:  # log once per second at 20fps
                    log.info(
                        "Frame %d: action=%s logits=%s",
                        frame_count, ACTION_NAMES[action],
                        np.round(logits, 2).tolist(),
                    )

                if not dry_run:
                    begin_action(action)

                frame_count += 1

                # 7. Frame rate sleep
                elapsed = time.monotonic() - tick_start
                remaining = self._frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        finally:
            self._cleanup()

        elapsed_total = time.monotonic() - start
        log.info(
            "Agent stopped — %d frames in %.1fs (%.1f fps), %d deaths",
            frame_count, elapsed_total,
            frame_count / max(elapsed_total, 0.001), deaths,
        )

    # ------------------------------------------------------------------
    # Background planner thread
    # ------------------------------------------------------------------

    def _planner_loop(self) -> None:
        """Daemon thread: calls Gemini every INTERVAL seconds without blocking the main loop."""
        while True:
            frame = self._capturer.last_frame
            if frame is not None:
                result = self._planner._call_gemini(frame)
                with self._planner_lock:
                    self._planner_result = result
            time.sleep(self._planner.INTERVAL)

    # ------------------------------------------------------------------
    # Death / respawn
    # ------------------------------------------------------------------

    def _handle_death(self) -> None:
        """Handle death: release inputs, wait for respawn, reset camera and stacker."""
        end_action()
        release_all()

        # Wait for respawn animation
        log.info("Waiting %.1fs for respawn animation...", self._respawn_wait)
        time.sleep(self._respawn_wait)

        # Poll until void_ratio drops (character back on platform)
        deadline = time.monotonic() + self._respawn_timeout
        while time.monotonic() < deadline:
            self._capturer.tick_fast()
            frame = self._capturer.last_frame
            from vision.perception import compute_scene_state
            state = compute_scene_state(
                frame,
                self._capturer._void_hsv_lower,
                self._capturer._void_hsv_upper,
            )
            if state["void_ratio"] < self._respawn_void_threshold:
                log.info("Respawn detected (void_ratio=%.3f)", state["void_ratio"])
                break
            time.sleep(0.1)
        else:
            log.warning("Respawn timeout — continuing anyway")

        # Re-set camera and stabilize
        set_camera_angle()
        time.sleep(0.5)

        # Reset stacker with fresh frame
        self._capturer.tick_fast()
        processed = preprocess_frame(self._capturer.last_frame)
        self._stacker.reset(processed)

    # ------------------------------------------------------------------
    # CNN + bias
    # ------------------------------------------------------------------

    def _cnn_inference(self, stacked: np.ndarray) -> np.ndarray:
        """Run CNN forward pass. Returns logits as numpy array of shape (6,)."""
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
        return logits.squeeze(0).cpu().numpy()

    def _bias_logits(self, logits: np.ndarray, planner_result: dict) -> np.ndarray:
        """Add planner bias to the CNN logits."""
        instruction = planner_result.get("instruction", "idle")
        confidence = planner_result.get("confidence", 0.0)

        idx = self._instruction_to_idx.get(instruction)
        if idx is not None:
            logits = logits.copy()
            logits[idx] += self._bias_scale * confidence

        return logits

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        end_action()
        release_all()

    def _signal_handler(self, signum, frame) -> None:
        log.info("Signal %d received — cleaning up", signum)
        self._cleanup()
        raise SystemExit(0)
