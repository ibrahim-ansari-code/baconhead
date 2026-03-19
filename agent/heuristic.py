"""
agent/heuristic.py — Phase 3 heuristic + LLM hybrid agent.

DEPRECATED (2026-03-17): The original heuristic approach (motion detection, forward-by-default)
has been invalidated. The target obby uses static platforms and gaps — frame differencing detects
no moving obstacles, and forward-by-default does not navigate gaps or turns. The motion detection
and forward-by-default selection logic are commented out below pending a new Phase 3 design.

Still active:
    - LLM integration (called every 5 seconds)
    - Session logger (per-frame CSV)
    - Death/respawn idle handling
    - Stuck detection timer (infrastructure only — strafe fallback disabled)

Implements the action_fn(stage, death) -> int interface expected by control/loop.run_loop().
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from agent.logger import SessionLogger
from control.actions import ACTION_NAMES

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action indices (mirror control/actions.py)
# ---------------------------------------------------------------------------
FORWARD = 0
LEFT = 1
RIGHT = 2
JUMP = 3
FORWARD_JUMP = 4
IDLE = 5

# ---------------------------------------------------------------------------
# LLM decision → action mapping
# ---------------------------------------------------------------------------
_LLM_ACTION_MAP = {
    "continue": FORWARD,
    "strafe_left": LEFT,
    "strafe_right": RIGHT,
    "wait": IDLE,
}

# Motion mask thresholds — DISABLED: static obby, no moving obstacles
# _MOTION_THRESHOLD = 25
# _MOTION_ACTIVITY_THRESHOLD = 0.05

# Stuck detection (timer infrastructure kept; strafe fallback disabled)
_STUCK_SECONDS = 8.0
_LLM_STUCK_SECONDS = 10.0  # tell the LLM agent is stuck after this long

# LLM call interval
_LLM_INTERVAL = 5.0  # seconds between LLM calls


class HeuristicAgent:
    """
    Callable heuristic agent: agent(stage, death) -> action_index.

    Args:
        capturer: capture.screen.Capturer instance (for last_frame access).
        llm_provider: Optional callable(prompt: str) -> str for LLM calls.
                      If None, LLM integration is disabled.
    """

    def __init__(self, capturer, llm_provider=None) -> None:
        self._capturer = capturer
        self._llm_provider = llm_provider
        self._logger = SessionLogger()

        # Motion mask state
        self._prev_gray: Optional[np.ndarray] = None

        # Stage tracking for stuck detection
        self._last_stage: int = 1
        self._last_stage_change: float = time.time()

        # Strafe state — alternates direction when stuck
        self._strafe_direction: int = RIGHT
        self._strafe_until: float = 0.0
        self._forward_until: float = 0.0  # forced forward after strafe burst

        # LLM state
        self._last_llm_call: float = 0.0
        self._llm_decision: Optional[str] = None
        self._llm_override_until: float = 0.0

        # Action history for LLM context
        self._action_history: deque[str] = deque(maxlen=5)

        # Death state — wait for death_event to clear
        self._was_dead: bool = False

        log.info(
            "HeuristicAgent ready (llm=%s)",
            "enabled" if llm_provider else "disabled",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, stage: int, death: bool) -> int:
        """Select an action given current stage and death event."""
        now = time.time()
        motion_mean = 0.0

        # DISABLED: motion mask computation — static obby has no moving obstacles
        # motion_mask = self._compute_motion_mask()
        # if motion_mask is not None:
        #     motion_mean = float(np.mean(motion_mask) / 255.0)
        motion_mask = None  # placeholder so logging still runs

        # Track stage changes for stuck detection
        if stage != self._last_stage:
            self._last_stage = stage
            self._last_stage_change = now
            self._strafe_until = 0.0  # reset strafe on progress
            self._forward_until = 0.0

        stuck_duration = now - self._last_stage_change

        # --- LLM call (every 5 seconds) ---
        llm_decision_this_frame: Optional[str] = None
        if self._llm_provider and (now - self._last_llm_call >= _LLM_INTERVAL):
            llm_decision_this_frame = self._call_llm(
                stage, stuck_duration >= _LLM_STUCK_SECONDS
            )
            self._last_llm_call = now

        # --- Action selection ---
        action = self._select_action(
            death, motion_mask, motion_mean, stuck_duration, now
        )

        action_name = ACTION_NAMES[action]
        self._action_history.append(action_name)

        # --- Log frame ---
        self._logger.log_frame(
            current_stage=stage,
            death_event=death,
            action_taken=action_name,
            llm_decision=llm_decision_this_frame,
            motion_mask_mean=motion_mean,
        )

        return action

    def close(self) -> None:
        self._logger.close()

    # ------------------------------------------------------------------
    # Action selection logic
    # ------------------------------------------------------------------

    def _select_action(
        self,
        death: bool,
        motion_mask: Optional[np.ndarray],
        motion_mean: float,
        stuck_duration: float,
        now: float,
    ) -> int:
        # Rule 1: Idle when dead — wait for respawn
        if death:
            self._was_dead = True
            return IDLE

        # Brief pause after respawn before resuming
        if self._was_dead:
            self._was_dead = False
            self._last_stage_change = now  # reset stuck timer on respawn
            return IDLE

        # Rule 2: LLM override (camera turns or strategic direction)
        if now < self._llm_override_until and self._llm_decision is not None:
            action = _LLM_ACTION_MAP.get(self._llm_decision)
            if action is not None:
                return action

        # DISABLED: strafe/forward-burst stuck recovery — static platform obby
        # made this logic counterproductive (no moving obstacles, gaps require
        # precise timing not random strafing). Pending new Phase 3 design.
        # if now < self._strafe_until:
        #     return self._strafe_direction
        # if now < self._forward_until:
        #     return FORWARD_JUMP
        # if stuck_duration >= _STUCK_SECONDS:
        #     self._strafe_until = now + 0.8
        #     self._forward_until = now + 0.8 + 2.0
        #     self._strafe_direction = LEFT if self._strafe_direction == RIGHT else RIGHT
        #     return self._strafe_direction

        # DISABLED: motion-triggered jump — no moving obstacles in target obby
        # if motion_mask is not None and self._check_lower_center_motion(motion_mask):
        #     return FORWARD_JUMP

        # DISABLED: forward-by-default — doesn't navigate static gaps or turns
        # return FORWARD

        # Fallback: idle until a new Phase 3 strategy is implemented
        return IDLE

    # DISABLED: static obby has no moving obstacles — motion detection not useful
    # def _check_lower_center_motion(self, motion_mask: np.ndarray) -> bool:
    #     h, w = motion_mask.shape[:2]
    #     roi = motion_mask[h * 2 // 3 :, w // 4 : w * 3 // 4]
    #     mean_activation = np.mean(roi) / 255.0
    #     return mean_activation > _MOTION_ACTIVITY_THRESHOLD

    # ------------------------------------------------------------------
    # Motion mask — DISABLED (static obby, no moving obstacles)
    # ------------------------------------------------------------------

    # def _compute_motion_mask(self) -> Optional[np.ndarray]:
    #     """Compute binary motion mask via frame differencing."""
    #     frame = self._capturer.last_frame
    #     if frame is None:
    #         return None
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.resize(gray, (84, 84))
    #     prev = self._prev_gray
    #     self._prev_gray = gray
    #     if prev is None:
    #         return None
    #     diff = cv2.absdiff(prev, gray)
    #     _, binary = cv2.threshold(diff, _MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    #     return binary

    # ------------------------------------------------------------------
    # LLM integration
    # ------------------------------------------------------------------

    def _call_llm(self, stage: int, is_stuck: bool) -> Optional[str]:
        """Call the LLM for a strategic decision. Returns the decision string."""
        recent_actions = list(self._action_history)
        prompt = (
            "You are controlling a Roblox obby character. "
            "Respond with exactly one word: continue, strafe_left, strafe_right, "
            "or wait.\n\n"
            f"Current stage: {stage}\n"
            f"Last 5 actions: {recent_actions}\n"
            f"Stuck (no progress in 10s): {is_stuck}\n\n"
            "What should the agent do?"
        )

        try:
            raw = self._llm_provider(prompt).strip().lower()
        except Exception:
            log.exception("LLM call failed")
            return None

        # Validate response
        valid = {
            "continue",
            "strafe_left",
            "strafe_right",
            "wait",
        }
        if raw not in valid:
            log.warning("LLM returned invalid decision: %r — ignoring", raw)
            return None

        log.info("LLM decision: %s (stage=%d stuck=%s)", raw, stage, is_stuck)
        self._llm_decision = raw

        # Apply LLM override for ~3 seconds
        self._llm_override_until = time.time() + 3.0

        return raw


# ---------------------------------------------------------------------------
# LLM provider factory
# ---------------------------------------------------------------------------

def create_llm_provider(
    backend: str = "anthropic",
    model: str | None = None,
) -> callable:
    """
    Create a callable(prompt) -> str LLM provider.

    Supports 'anthropic' (Claude) and 'openai' (GPT-4o) backends.
    API key is read from ANTHROPIC_API_KEY or OPENAI_API_KEY env vars.
    """
    if backend == "anthropic":
        model = model or "claude-sonnet-4-20250514"
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.warning("ANTHROPIC_API_KEY not set — LLM disabled")
            return None

        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        def _call_anthropic(prompt: str) -> str:
            resp = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        return _call_anthropic

    elif backend == "openai":
        model = model or "gpt-4o"
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.warning("OPENAI_API_KEY not set — LLM disabled")
            return None

        import openai

        client = openai.OpenAI(api_key=api_key)

        def _call_openai(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content

        return _call_openai

    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}")
