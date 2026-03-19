"""
agent/progress_estimator.py — Gemini-based progress estimator for the obby agent.

Stateless w.r.t. timing — the caller is responsible for rate-limiting.

Public interface:
    GeminiProgressEstimator.estimate(frame_bgr: np.ndarray) -> float | None
"""

from __future__ import annotations

import json
import logging
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a progress monitor for a Roblox obstacle course (obby) agent.\n"
    "Camera: third-person, fixed forward angle pitched slightly downward. The character is visible on screen.\n\n"
    "COURSE STRUCTURE — this is critical for accurate estimation:\n"
    "The obby is divided into segments separated by checkpoint platforms. A checkpoint platform is a large white/grey square tile with a bold red upward-pointing arrow printed on it.\n"
    "- progress=0.0 means the character is standing ON or immediately beside the START checkpoint platform (the one they just came from, behind/below them).\n"
    "- progress=1.0 means the character has reached the NEXT checkpoint platform (the red arrow tile visible ahead/above).\n"
    "- Between the two checkpoint platforms is a series of colored floating blocks the character must jump across. Your job is to estimate how far through THAT gap the character currently is.\n\n"
    "How to estimate progress:\n"
    "- If the large red-arrow checkpoint platform is prominently visible BELOW or BEHIND the character, progress is near 0.0.\n"
    "- If the character is in the middle of the colored jumping blocks with both checkpoints roughly equidistant, progress is ~0.5.\n"
    "- If the next red-arrow checkpoint platform is clearly visible AHEAD and close, progress is near 1.0.\n"
    "- If the character is standing ON the next checkpoint platform, progress = 1.0.\n"
    "- Use relative position between the two red-arrow platforms as your primary signal, not platform color or height alone.\n\n"
    "progress: float between 0.0 and 1.0\n"
    "reason: one sentence referencing the checkpoint platforms and colored blocks visible\n\n"
    'Respond with only valid JSON: {"progress": <float>, "reason": "<string>"}. No markdown, no extra text.'
)


class GeminiProgressEstimator:
    """
    Stateless Gemini-based progress estimator.

    Calls Gemini with a screenshot and returns a progress score in [0.0, 1.0].
    Rate-limiting is the caller's responsibility.
    """

    def __init__(self) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            log.warning("GEMINI_API_KEY not set — GeminiProgressEstimator will return None")

        from google import genai
        self._client = genai.Client(api_key=api_key) if api_key else None

        log.info("GeminiProgressEstimator ready")

    def estimate(self, frame_bgr: np.ndarray) -> float | None:
        """
        Estimate progress for the given BGR frame.

        Returns a float in [0.0, 1.0] or None on any failure.
        """
        if self._client is None:
            return None

        try:
            from google.genai import types

            # BGR → RGB → JPEG bytes
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            jpeg_bytes = buf.getvalue()

            image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

            response = self._client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image_part, "Estimate the character's progress through the obstacle course."],
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=500,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            raw_text = response.text.strip()
            # Extract first complete JSON object regardless of surrounding prose or fences
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw_text = raw_text[start : end + 1]
            try:
                parsed = json.loads(raw_text)
            except Exception:
                log.warning("GeminiProgressEstimator: failed to parse JSON: %r", raw_text)
                return None

            progress = float(parsed.get("progress", 0.0))
            progress = max(0.0, min(1.0, progress))
            reason = str(parsed.get("reason", "")).strip()

            log.debug("GeminiProgressEstimator: progress=%.3f reason=%s", progress, reason)
            return progress

        except Exception:
            log.warning("GeminiProgressEstimator: Gemini call failed", exc_info=True)
            return None
