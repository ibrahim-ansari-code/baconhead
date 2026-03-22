"""
agent/planner.py — Gemini high-level planner for the two-tier agent.

Calls Gemini every INTERVAL seconds with a screenshot and returns a
structured instruction dict. Between API calls, returns the cached result.
The API call runs in a background thread so it never blocks the main loop.

Public interface:
    GeminiPlanner.tick(frame: np.ndarray) -> dict
        Returns {"instruction": str, "confidence": float, "reason": str}
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are the high-level planner for an autonomous Roblox obby agent.\n"
    "Camera: third-person, fixed forward angle, pitched 25° below horizontal. The camera does NOT rotate — it is locked.\n"
    "The character is navigating circular platforms over a void.\n\n"
    "Your job: look at the screenshot, identify the next visible platform or checkpoint ahead, and return a single JSON object with:\n"
    '  "instruction": one of ["forward", "left", "right", "jump", "forward_jump", "idle"]\n'
    '  "confidence": float 0.0-1.0\n'
    '  "reason": one sentence explaining your choice\n\n'
    "Decision rules:\n"
    "- Default to 'forward' or 'forward_jump' to make progress toward the next platform.\n"
    "- Use 'left' or 'right' only when the path clearly curves or the next platform is off to the side.\n"
    "- Use 'jump' or 'forward_jump' when there is a gap to cross.\n"
    "- Use 'idle' only if no safe move is apparent (e.g. character is mid-air with no platform visible).\n"
    "- Prefer action with confidence >= 0.65 so the agent can act decisively.\n\n"
    "Respond with only valid JSON. No markdown, no explanation outside the JSON."
)

# Regex to extract JSON from responses that may include markdown fences
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


class GeminiPlanner:
    """
    High-level planner using Gemini Vision.

    Calls the Gemini API at most once every INTERVAL seconds.
    Between calls, returns the cached last result.
    API calls run in a background thread to avoid blocking the main loop.
    """

    VALID_INSTRUCTIONS = {"forward", "left", "right", "jump", "forward_jump", "idle"}
    INTERVAL = 1.5  # seconds between API calls

    def __init__(self) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            log.warning("GEMINI_API_KEY not set — GeminiPlanner will return idle")

        from google import genai
        self._client = genai.Client(api_key=api_key) if api_key else None

        self._last_call: float = 0.0
        self._cached: dict = {
            "instruction": "idle",
            "confidence": 0.0,
            "reason": "init",
        }
        self._pending: bool = False
        self._lock = threading.Lock()

        log.info("GeminiPlanner ready (interval=%.1fs)", self.INTERVAL)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def tick(
        self,
        frame: np.ndarray,
        *,
        stage: Optional[int] = None,
        just_died: bool = False,
        death_count: int = 0,
    ) -> dict:
        """
        Return a planning decision for the given BGR frame.

        Uses cached result if called within INTERVAL seconds of the last
        API call. Otherwise fires a background Gemini call and returns
        cached until it completes.
        """
        now = time.time()
        if now - self._last_call >= self.INTERVAL and not self._pending:
            self._last_call = now
            self._pending = True
            t = threading.Thread(
                target=self._bg_call,
                args=(frame.copy(), stage, just_died, death_count),
                daemon=True,
            )
            t.start()

        return self._cached

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _bg_call(
        self,
        frame: np.ndarray,
        stage: Optional[int],
        just_died: bool,
        death_count: int,
    ) -> None:
        """Background thread: call Gemini and update cache."""
        try:
            result = self._call_gemini(
                frame, stage=stage, just_died=just_died, death_count=death_count
            )
            with self._lock:
                self._cached = result
        finally:
            self._pending = False

    def _call_gemini(
        self,
        frame: np.ndarray,
        stage: Optional[int] = None,
        just_died: bool = False,
        death_count: int = 0,
    ) -> dict:
        """Call Gemini Vision with the frame. Returns result dict."""
        if self._client is None:
            log.warning("No Gemini client (API key missing) — returning cached")
            return self._cached

        try:
            from google.genai import types

            # BGR → RGB → JPEG bytes
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            jpeg_bytes = buf.getvalue()

            image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

            # Build context prefix
            context_line = ""
            if stage is not None:
                context_line += f"Current stage: {stage}. "
            if just_died:
                context_line += (
                    f"The agent just respawned (total deaths this session: {death_count}). "
                    "Be cautious — look before moving. "
                )

            prompt = context_line + "What action should the agent take next?" if context_line else "What action should the agent take next?"

            response = self._client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=300,
                    response_mime_type="application/json",
                ),
            )

            raw_text = response.text if response.text else ""
            raw_text = raw_text.strip()
            if not raw_text:
                log.warning("Gemini returned empty response — returning cached")
                return self._cached

            # Try direct parse first, then regex extract
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                match = _JSON_RE.search(raw_text)
                if match:
                    parsed = json.loads(match.group())
                else:
                    log.warning("Could not extract JSON from Gemini response: %r", raw_text[:200])
                    return self._cached

            instruction = str(parsed.get("instruction", "idle")).lower()
            if instruction not in self.VALID_INSTRUCTIONS:
                log.warning(
                    "Gemini returned invalid instruction %r — using idle", instruction
                )
                instruction = "idle"

            confidence = float(parsed.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            reason = str(parsed.get("reason", "")).strip()

            result = {
                "instruction": instruction,
                "confidence": confidence,
                "reason": reason,
            }
            log.info(
                "Gemini: instruction=%s confidence=%.2f reason=%s",
                instruction,
                confidence,
                reason,
            )
            return result

        except Exception:
            log.warning("Gemini call failed — returning cached", exc_info=True)
            return self._cached
