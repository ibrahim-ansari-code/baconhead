"""
Claude Sonnet — game-agnostic planning for the Roblox bot.

Sends screenshots to Claude and lets it figure out the game, objectives,
and actions. No hardcoded game knowledge — works on any Roblox game.

API:
  plan_with_goal(frame, context_text, api_key, last_goal, last_failure)
      → (goal: str, steps: list[tuple[str, int]])

  verify_goal(plan_frame, current_frame, goal, api_key)
      → ("achieved"|"in_progress"|"failed", reason)

  survey_pick_best(frames_4, api_key)
      → int (0-3): index of best direction to face
"""

import base64
import io
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np

from llm_agent.physics import degrees_to_ms, MAX_MOVEMENT_MS

SCOUT_MODEL = "claude-sonnet-4-6"
PLAN_ACTIONS = (
    "W", "A", "S", "D", "W+space", "none", "look_left", "look_right"
)

# ── Image helpers ─────────────────────────────────────────────────────────────


def _frame_to_base64(frame: np.ndarray, max_size: int = 512) -> str:
    from PIL import Image

    pil = Image.fromarray(frame.astype(np.uint8))
    w, h = pil.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil = pil.resize(
            (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
        )
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Movement reference (Roblox engine constants, not game-specific) ───────────

_MOVEMENT_REFERENCE = (
    "MOVEMENT REFERENCE (Roblox default walk speed):\n"
    "  W 150ms  = ~2 studs  (nudge — fine position near a door)\n"
    "  W 300ms  = ~5 studs  (short step)\n"
    "  W 500ms  = ~8 studs  (medium walk)\n"
    "  W 800ms  = ~13 studs (long walk)\n"
    "  W 1200ms = ~19 studs (cross a courtyard)\n"
    "  W 2000ms = ~32 studs (cross a large open area)\n"
    "  W+space  = jump forward ~6 studs\n"
    "  A/D 200ms = sidestep ~3 studs\n"
    "  look_right/left: specify degrees "
    "(10° fine, 45° slight, 90° quarter, 180° U-turn)\n"
)


# ── survey_pick_best ──────────────────────────────────────────────────────────


def survey_pick_best(
    frames: List[np.ndarray],
    api_key: Optional[str] = None,
    model: str = SCOUT_MODEL,
) -> int:
    """
    Given 4 frames (0°, 90°, 180°, 270°), send all to Claude and ask
    which direction has the best accessible destination.
    Returns index 0-3. Falls back to 0 on error.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or len(frames) != 4:
        return 0

    prompt = (
        "You control a Roblox character. These 4 images show 4 directions "
        "the character can face (0°, 90° right, 180° behind, 270° left).\n\n"
        "Which direction has the best path or destination to move toward?\n\n"
        "Consider:\n"
        "  - Is there a clear walkable path (not blocked by a wall)?\n"
        "  - Is there a destination visible (building, platform, objective)?\n"
        "  - How close and accessible is it?\n"
        "  - Avoid directions that lead to water, void, or map edges.\n\n"
        "Reply with ONLY valid JSON:\n"
        '{"best": 0, "reason": "why this direction"}\n\n'
        "where best is 0, 1, 2, or 3."
    )

    content = []
    for i, f in enumerate(frames):
        b64 = _frame_to_base64(f)
        content.append({"type": "text", "text": f"Direction {i} ({i * 90}°):"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })
    content.append({"type": "text", "text": prompt})

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        print("[scout] Survey: picking best of 4 directions...", flush=True)
        resp = client.messages.create(
            model=model,
            max_tokens=100,
            temperature=0.1,
            messages=[{"role": "user", "content": content}],
        )
        text = (resp.content[0].text if resp.content else "").strip()
        print(f"[scout] survey: {text!r}", flush=True)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(text[start:end])
            best = int(obj.get("best", 0))
            return max(0, min(3, best))
    except Exception as e:
        print(f"[scout] survey ERROR: {e}", flush=True)
    return 0


# ── plan_with_goal ────────────────────────────────────────────────────────────


def plan_with_goal(
    frame: np.ndarray,
    context_text: str,
    api_key: Optional[str] = None,
    last_goal: Optional[str] = None,
    last_failure: Optional[str] = None,
    model: str = SCOUT_MODEL,
) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Send screenshot + context to Claude, get back a goal and action plan.
    Claude figures out what game this is and what to do — no hardcoded hints.
    Returns (goal, [(action, ms), ...]).
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[scout] No API key — fallback plan", flush=True)
        return _fallback_plan()

    b64 = _frame_to_base64(frame)

    failure_block = ""
    if last_failure:
        failure_block = (
            f"\n!! PREVIOUS PLAN FAILED: \"{last_failure}\"\n"
            "You MUST try a COMPLETELY different approach.\n"
            "If you walked forward and got stuck, TURN first (90° or more).\n"
            "If you turned right last time, try turning LEFT.\n"
            "Do NOT repeat the failed approach.\n"
        )

    prompt = (
        "You control a Roblox character in third-person view. The character "
        "is the figure in the center of the screen, facing AWAY from you.\n\n"
        "Look at the screenshot VERY carefully and determine:\n"
        "  1. What game is this? What are the objectives?\n"
        "  2. What is directly in front / left / right of the character?\n"
        "  3. Can the character walk forward without hitting something?\n"
        "  4. What should the character do to make progress?\n\n"
        "--- SITUATION ---\n"
        + context_text
        + "\n--- END SITUATION ---\n\n"
        + _MOVEMENT_REFERENCE + "\n"
        + failure_block
        + "\nBased on what you see, create a SPECIFIC plan.\n"
        "Reply with ONLY valid JSON — no explanation, no markdown:\n"
        "{\n"
        '  "observation": "what you see in the screenshot",\n'
        '  "goal": "specific destination and how to reach it",\n'
        '  "steps": [\n'
        '    {"action": "look_right", "degrees": 90},\n'
        '    {"action": "W", "ms": 600},\n'
        '    {"action": "W", "ms": 500}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "  - 3 to 6 steps\n"
        "  - look actions: use 'degrees' key. movement: use 'ms' key\n"
        "  - valid: W, A, S, D, W+space, look_left, look_right, none\n"
        "  - choose ms DYNAMICALLY: far = 1000-2000, close = 150-300\n"
        "  - if a WALL is ahead → first step MUST be a turn\n"
        "  - if WATER or VOID is visible → turn AWAY from it immediately\n"
        "  - NEVER plan 3+ W steps without a look step in between"
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        print("[scout] Planning...", flush=True)
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = (resp.content[0].text if resp.content else "").strip()
        print(f"[scout] plan: {text!r}", flush=True)
    except Exception as e:
        print(f"[scout] plan ERROR: {e}", flush=True)
        return _fallback_plan()

    result = _parse_plan(text)
    if result is None:
        print("[scout] Parse failed — fallback", flush=True)
        return _fallback_plan()

    goal, steps = result
    print(f"[scout] Goal: {goal!r}  Steps: {steps}", flush=True)
    return goal, steps


# ── verify_goal ───────────────────────────────────────────────────────────────


def verify_goal(
    plan_frame: np.ndarray,
    current_frame: np.ndarray,
    goal: str,
    api_key: Optional[str] = None,
    model: str = SCOUT_MODEL,
) -> Tuple[str, str]:
    """
    Compare before / after screenshots to judge goal progress.
    Returns (status, reason) where status is achieved / in_progress / failed.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "in_progress", ""

    b64_before = _frame_to_base64(plan_frame)
    b64_after = _frame_to_base64(current_frame)

    prompt = (
        "You control a Roblox character.\n"
        "Image 1 = BEFORE executing the plan. Image 2 = AFTER.\n"
        f'Goal: "{goal}"\n\n'
        "Compare the two images:\n"
        "  1. Has the character moved? (different background / angle?)\n"
        "  2. Is the character closer to the goal?\n"
        "  3. If images look NEARLY IDENTICAL → character is STUCK.\n\n"
        "Reply with ONLY valid JSON:\n"
        '{"status": "achieved", "reason": "reached destination"}\n'
        '{"status": "in_progress", "reason": "moved closer"}\n'
        '{"status": "failed", "reason": "what went wrong"}\n\n'
        "If the two images look almost the same, respond \"failed\"."
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        print("[scout] Verifying goal...", flush=True)
        resp = client.messages.create(
            model=model,
            max_tokens=100,
            temperature=0.1,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_before,
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_after,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = (resp.content[0].text if resp.content else "").strip()
        print(f"[scout] verify: {text!r}", flush=True)
    except Exception as e:
        print(f"[scout] verify ERROR: {e}", flush=True)
        return "in_progress", ""

    return _parse_verify(text)


# ── Parsing helpers ───────────────────────────────────────────────────────────


def _strip_markdown_json(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def _parse_plan(text: str) -> Optional[Tuple[str, List[Tuple[str, int]]]]:
    try:
        text = _strip_markdown_json(text)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        obj = json.loads(text[start:end])
        goal = str(obj.get("goal", "")).strip()
        raw_steps = obj.get("steps", [])
        if not goal or not isinstance(raw_steps, list) or not raw_steps:
            return None

        steps: List[Tuple[str, int]] = []
        for s in raw_steps[:6]:
            action = str(s.get("action", "")).strip()
            if action.lower() == "space":
                action = "W+space"
            if action not in PLAN_ACTIONS:
                continue
            is_look = action.lower() in ("look_left", "look_right")
            if is_look:
                degrees = s.get("degrees")
                if degrees is not None:
                    ms = degrees_to_ms(float(degrees))
                else:
                    ms = max(100, min(2000, int(s.get("ms", 364))))
            else:
                ms = max(100, min(MAX_MOVEMENT_MS, int(s.get("ms", 400))))
            steps.append((action, ms))
        if not steps:
            return None
        return goal, steps
    except Exception:
        return None


def _parse_verify(text: str) -> Tuple[str, str]:
    try:
        text = _strip_markdown_json(text)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            return "in_progress", ""
        obj = json.loads(text[start:end])
        status = str(obj.get("status", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()
        if status in ("achieved", "in_progress", "failed"):
            return status, reason
    except Exception:
        pass
    return "in_progress", ""


def _fallback_plan() -> Tuple[str, List[Tuple[str, int]]]:
    """Safe default plan when Claude is unavailable."""
    import random

    options = [
        (
            "look around and walk forward",
            [("look_right", 364), ("W", 500), ("W", 400)],
        ),
        (
            "walk forward and explore",
            [("W", 500), ("W", 400), ("W", 300)],
        ),
        (
            "turn and walk forward",
            [("look_left", 364), ("W", 500), ("W", 400)],
        ),
    ]
    return random.choice(options)
