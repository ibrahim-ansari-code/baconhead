"""
agent/twotier.py — CNN-only agent controller.

The CNN produces actions at ~20fps from stacked frames.

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
    CNN-only agent controller.

    The CNN runs at ~20fps producing action logits.
    """

    def __init__(
        self,
        capturer: Capturer,
        checkpoint_path: str | None = None,
        enable_self_improvement: bool = False,
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
        self._model_lock = threading.RLock()
        log.info("CNN loaded from %s on %s", cp, self._device)

        # Frame stacker
        self._stacker = FrameStacker(stack_size=4)

        # Death tracking
        self._deaths: int = 0

        # Self-improvement components
        self._self_improvement = enable_self_improvement
        self._recorder = None
        self._retrainer = None
        self._reward_calc = None
        self._episode_reward: float = 0.0
        self._progress_estimator = None
        self._latest_progress: float | None = None
        self._progress_frame_count: int = 0
        self._progress_every_n: int = cfg.get("progress_every_n_frames", 40)
        if enable_self_improvement:
            from training.reward import RewardCalculator
            self._reward_calc = RewardCalculator()
            from agent.progress_estimator import GeminiProgressEstimator
            self._progress_estimator = GeminiProgressEstimator()
        if enable_self_improvement:
            from agent.experience_recorder import ExperienceRecorder
            from agent.background_retrainer import BackgroundRetrainer
            self._recorder = ExperienceRecorder(demos_dir="demos")
            self._retrainer = BackgroundRetrainer(demos_dir="demos", checkpoint_dir="checkpoints")
            self._retrainer.start()
            t = threading.Thread(target=self._hotswap_loop, daemon=True)
            t.start()
            log.info("Self-improvement enabled: recorder + retrainer + hot-swap")

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
            "Starting CNN agent — duration=%ds fps=%d dry_run=%s",
            duration_seconds, self._fps, dry_run,
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

        if self._reward_calc is not None:
            pass  # RewardCalculator is stateless; nothing to reset
        self._episode_reward = 0.0

        start = time.monotonic()
        frame_count = 0
        prev_stage: int | None = None

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
                    self._deaths += 1
                    if self._reward_calc is not None:
                        step_reward = self._reward_calc.compute(
                            progress=None, death_event=True, stuck=False
                        )
                        self._episode_reward += step_reward
                        self._latest_progress = None
                        log.info(
                            "Death #%d at frame %d — step_reward=%.4f episode_reward=%.4f",
                            self._deaths, frame_count, step_reward, self._episode_reward,
                        )
                    else:
                        log.info("Death #%d at frame %d", self._deaths, frame_count)
                    if self._recorder is not None:
                        self._recorder.on_death()
                    if not dry_run:
                        self._handle_death()
                    self._episode_reward = 0.0
                    frame_count += 1
                    continue

                # 3. Preprocess and stack
                frame = self._capturer.last_frame
                processed = preprocess_frame(frame)
                stacked = self._stacker.push(processed)

                # 4. CNN inference
                logits = self._cnn_inference(stacked)

                # 5. Select action
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

                # Progress estimation (rate-limited Gemini call)
                if self._progress_estimator is not None:
                    self._progress_frame_count += 1
                    if self._progress_frame_count % self._progress_every_n == 0:
                        prev_progress = self._latest_progress
                        self._latest_progress = self._progress_estimator.estimate(frame)
                        log.info(
                            "Frame %d: Gemini progress=%.3f (prev=%.3f)",
                            frame_count,
                            self._latest_progress if self._latest_progress is not None else -1.0,
                            prev_progress if prev_progress is not None else -1.0,
                        )

                # Reward tracking
                if self._reward_calc is not None:
                    step_reward = self._reward_calc.compute(
                        progress=self._latest_progress, death_event=False, stuck=False
                    )
                    self._episode_reward += step_reward
                    if frame_count % 20 == 0:
                        log.info(
                            "Frame %d: step_reward=%.4f episode_reward=%.4f",
                            frame_count, step_reward, self._episode_reward,
                        )

                # Self-improvement: record frame and detect stage advances
                if self._recorder is not None:
                    curr_stage = self._capturer.current_stage
                    self._recorder.record(self._capturer.last_frame, action, curr_stage)
                    if prev_stage is not None and curr_stage > prev_stage:
                        self._recorder.on_stage_advance(curr_stage)
                    prev_stage = curr_stage

                frame_count += 1

                # 7. Frame rate sleep
                elapsed = time.monotonic() - tick_start
                remaining = self._frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        finally:
            # Flush self-demo data if quality threshold met
            if self._recorder is not None:
                run_path = self._recorder.flush_run()
                if run_path is not None and self._retrainer is not None:
                    self._retrainer.notify_new_run(run_path)
            if self._retrainer is not None:
                self._retrainer.stop()
            self._cleanup()

        elapsed_total = time.monotonic() - start
        log.info(
            "Agent stopped — %d frames in %.1fs (%.1f fps), %d deaths",
            frame_count, elapsed_total,
            frame_count / max(elapsed_total, 0.001), self._deaths,
        )
        if self._reward_calc is not None:
            log.info("Final cumulative episode reward: %.4f", self._episode_reward)

    # ------------------------------------------------------------------
    # Hot-swap loop (self-improvement)
    # ------------------------------------------------------------------

    def _hotswap_loop(self) -> None:
        """Daemon thread: polls for bc_pending.flag and hot-swaps model weights."""
        flag = Path("checkpoints/bc_pending.flag")
        while True:
            time.sleep(30)
            if flag.exists():
                try:
                    new_cp = flag.read_text().strip() or "checkpoints/bc_best.pt"
                    state = torch.load(new_cp, map_location=self._device, weights_only=False)
                    sd = state.get("model_state_dict", state)
                    with self._model_lock:
                        self._model.load_state_dict(sd)
                        self._model.eval()
                    flag.unlink()
                    log.info("Hot-swapped model from %s", new_cp)
                except Exception:
                    log.exception("Hot-swap failed")

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
    # CNN inference
    # ------------------------------------------------------------------

    def _cnn_inference(self, stacked: np.ndarray) -> np.ndarray:
        """Run CNN forward pass. Returns logits as numpy array of shape (6,)."""
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self._device)
        with self._model_lock:
            with torch.no_grad():
                logits = self._model(tensor)
        return logits.squeeze(0).cpu().numpy()

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
