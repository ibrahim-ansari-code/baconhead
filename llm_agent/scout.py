"""
Claude Sonnet (Anthropic vision): goal-oriented planning for the Roblox bot.

Primary API:
  plan_with_goal(frame, context_text, api_key, game, last_goal, last_failure)
      → (goal: str, steps: list[tuple[str, int]])

  verify_goal(plan_frame, current_frame, goal, api_key)
      → ("achieved"|"in_progress"|"failed", reason)

  survey_pick_best(frames_4, game, api_key)
      → int (0-3): index of the best direction to face
"""

import base64
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np

from llm_agent.physics import degrees_to_ms, LOOK_PX_PER_DEGREE, MAX_MOVEMENT_MS

SCOUT_MODEL  = "claude-sonnet-4-6"
PLAN_ACTIONS = ("W", "A", "S", "D", "W+space", "none", "look_left", "look_right")


# ── Image helpers ──────────────────────────────────────────────────────────────

def _frame_to_base64(frame: np.ndarray, max_size: int = 512) -> str:
    from PIL import Image
    import io
    pil = Image.fromarray(frame.astype(np.uint8))
    w, h = pil.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Game context ───────────────────────────────────────────────────────────────

GAME_CONTEXT = {
    "nds": (
        "GAME: Natural Disaster Survival.\n"
        "OBJECTIVE: Survive the disaster by getting INSIDE a building BEFORE the disaster hits.\n"
        "\n"
        "MAP LAYOUT: The map is a square island surrounded by WATER on all sides.\n"
        "  Buildings are in the CENTER of the island. The EDGES have green grass ending in water.\n"
        "  If you see water/ocean taking up a large part of the screen → you are at the EDGE.\n"
        "  IMMEDIATELY turn 180° toward the center of the island where the buildings are.\n"
        "\n"
        "PHASE 1 — FIND A BUILDING:\n"
        "  → Walk toward the CENTER of the island (away from edges).\n"
        "  → Use long W steps (800-2000ms) when crossing open ground.\n"
        "  → After every long walk, do a look step to check your bearings.\n"
        "\n"
        "PHASE 2 — APPROACH THE BUILDING:\n"
        "  → Look for DOORS — dark rectangles at ground level on the building walls.\n"
        "  → If you see a building WALL with no door → turn 90° to find the door side.\n"
        "  → Walk toward the door. As you get closer, SHORTEN your steps (300-500ms).\n"
        "\n"
        "PHASE 3 — ENTER THE BUILDING:\n"
        "  → When the door fills most of the screen, you're very close.\n"
        "  → Use fine-adjustment turns (10-20°) to center the door in your view.\n"
        "  → Use short W steps (150-300ms) to walk through the doorway.\n"
        "  → If there's a step/lip at the entrance → W+space to jump over it.\n"
        "  → Use A or D (200ms) to sidestep if you're slightly misaligned with the door.\n"
        "  → Once inside, STOP — you're safe.\n"
        "\n"
        "AFTER disaster warning (red text at top):\n"
        "  - Flood/Tsunami → climb stairs or to high ground INSIDE a building (W+space to go up stairs).\n"
        "  - Fire → get inside a building NOW.\n"
        "  - Tornado/Blizzard → get inside a solid building NOW.\n"
        "  - Earthquake → move to open ground.\n"
        "\n"
        "ABSOLUTE RULES (NEVER violate these):\n"
        "  - If water/ocean is visible ahead → TURN 180° IMMEDIATELY. You will die if you walk into water.\n"
        "  - If you see green grass with brown edge and then blue → that is the MAP EDGE. TURN AROUND.\n"
        "  - Buildings are ALWAYS toward the center, NEVER at the edges.\n"
        "  - NEVER walk toward the edge of the island.\n"
        "  - NEVER jump in place — W+space only when entering doors or climbing stairs."
    ),
    "obby": (
        "GAME: Obby (obstacle course).\n"
        "OBJECTIVE: Reach the next checkpoint by crossing platforms.\n"
        "\n"
        "RULES:\n"
        "  - Always move toward the next visible platform or checkpoint.\n"
        "  - W+space to jump over gaps — move forward before jumping.\n"
        "  - On narrow platforms, use shorter W durations (200-300ms).\n"
        "  - NEVER jump in place."
    ),
    "brookhaven": (
        "GAME: Brookhaven RP — open-world city.\n"
        "OBJECTIVE: Roam naturally. Explore roads, buildings, parks.\n"
        "\n"
        "RULES:\n"
        "  - Walk along roads and paths. Turn at intersections.\n"
        "  - If blocked by a wall, turn and find a new direction.\n"
        "  - NEVER spin in circles."
    ),
}

_MOVEMENT_REFERENCE = (
    "MOVEMENT REFERENCE (choose ms dynamically based on distance):\n"
    "  W 150ms  = ~2 studs  (nudge — fine-tune position near a door)\n"
    "  W 300ms  = ~5 studs  (short step — approaching a close object)\n"
    "  W 500ms  = ~8 studs  (medium walk)\n"
    "  W 800ms  = ~13 studs (long walk)\n"
    "  W 1200ms = ~19 studs (cross a courtyard)\n"
    "  W 2000ms = ~32 studs (cross a large open area)\n"
    "  W+space  = jump forward ~6 studs (use to cross thresholds, climb steps, get over small obstacles)\n"
    "  A/D 200ms = sidestep ~3 studs (align with a doorway)\n"
    "  look_right/left: specify degrees (10° fine adjustment, 45° slight turn, 90° quarter turn, 180° U-turn)\n"
    "\n"
    "ENTERING BUILDINGS:\n"
    "  1. Look for the DOOR — a dark rectangle at ground level on one side of the building.\n"
    "  2. Turn to face the door DIRECTLY (use small 10-30° adjustments if needed).\n"
    "  3. Walk toward it with W. If close, use SHORT steps (150-300ms) to avoid overshooting.\n"
    "  4. If there's a step/threshold at the entrance, use W+space to jump over it.\n"
    "  5. Once through the door, you're inside — stop and reassess."
)


# ── survey_pick_best ──────────────────────────────────────────────────────────

def survey_pick_best(
    frames: List[np.ndarray],
    game: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = SCOUT_MODEL,
) -> int:
    """
    Given exactly 4 frames (0°, 90°, 180°, 270°), send all to Claude in one
    call and ask which direction has the best accessible destination.

    Returns the index (0-3) of the best direction.
    Falls back to 0 on error.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or len(frames) != 4:
        return 0

    if (game or "").lower() == "nds":
        what = "a building entrance (door/opening at ground level) that you can walk to and enter"
    elif (game or "").lower() == "obby":
        what = "the next platform or checkpoint you can reach"
    else:
        what = "a clear path, road, building entrance, or interesting destination to walk toward"

    prompt = (
        "You control a Roblox character. These 4 images show 4 directions the character can face "
        "(0°, 90° right, 180° behind, 270° left).\n\n"
        f"Which direction has the best {what}?\n\n"
        "Consider:\n"
        "  - Is there a clear path (not blocked by a wall right in front)?\n"
        "  - Is there an actual destination visible (building, door, platform)?\n"
        "  - How close/accessible is it?\n\n"
        "Reply with ONLY valid JSON:\n"
        '{"best": 0, "reason": "why this direction is best"}\n\n'
        "where best is 0, 1, 2, or 3 (the image number, 0-indexed)."
    )

    content = []
    for i, f in enumerate(frames):
        b64 = _frame_to_base64(f)
        content.append({
            "type": "text",
            "text": f"Direction {i} ({i * 90}°):"
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
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
        print(f"[scout] survey response: {text!r}", flush=True)
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(text[start:end])
            best = int(obj.get("best", 0))
            reason = obj.get("reason", "")
            print(f"[scout] Survey picked direction {best}: {reason}", flush=True)
            return max(0, min(3, best))
    except Exception as e:
        print(f"[scout] survey ERROR: {e}", flush=True)
    return 0


# ── plan_with_goal ─────────────────────────────────────────────────────────────

def plan_with_goal(
    frame: np.ndarray,
    context_text: str,
    api_key: Optional[str] = None,
    game: Optional[str] = None,
    last_goal: Optional[str] = None,
    last_failure: Optional[str] = None,
    model: str = SCOUT_MODEL,
) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Ask Claude for a goal + ordered action sequence given the current screenshot.

    Returns (goal: str, steps: list of (action, ms) tuples).
    Falls back to a safe default plan on any error.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[scout] No API key — using fallback plan", flush=True)
        return _fallback_plan()

    b64       = _frame_to_base64(frame)
    game_hint = GAME_CONTEXT.get((game or "").lower(), "")

    failure_block = ""
    if last_failure:
        failure_block = (
            f"\n!! PREVIOUS PLAN FAILED: \"{last_failure}\"\n"
            f"You MUST try a COMPLETELY different approach.\n"
            f"If you walked forward and got stuck, TURN first (90° or more).\n"
            f"If you turned right last time, try turning LEFT.\n"
            f"Do NOT repeat the failed approach.\n"
        )

    prompt = (
        "You control a Roblox character in third-person view. The character is the figure "
        "in the center of the screen, facing AWAY from you (into the screen).\n\n"
        "Look at the screenshot VERY carefully and answer these questions silently:\n"
        "  1. What is directly in FRONT of the character? (wall, door, open space, building side?)\n"
        "  2. Can the character walk forward without hitting something?\n"
        "  3. If there's a building, WHERE is the door/entrance? (left side, right side, other direction?)\n\n"
        "--- SITUATION ---\n"
        + context_text
        + "\n--- END SITUATION ---\n\n"
        + (game_hint + "\n\n" if game_hint else "")
        + _MOVEMENT_REFERENCE + "\n\n"
        + failure_block
        + "\nBased on what you see, create a SPECIFIC plan.\n"
        "Reply with ONLY valid JSON — no explanation, no markdown:\n"
        '{\n'
        '  "observation": "what is in front, left, and right of the character",\n'
        '  "goal": "specific destination and how to reach it",\n'
        '  "steps": [\n'
        '    {"action": "look_right", "degrees": 90},\n'
        '    {"action": "W", "ms": 600},\n'
        '    {"action": "W", "ms": 500}\n'
        '  ]\n'
        '}\n\n'
        "Rules:\n"
        "  - 3 to 6 steps\n"
        "  - look actions: use 'degrees' key (any value, e.g. 15, 45, 90, 180). Movement: use 'ms' key (any value 100-2000)\n"
        "  - valid: W, A, S, D, W+space, look_left, look_right, none\n"
        "  - choose ms DYNAMICALLY: far away = large ms (1000-2000), close = small ms (150-300)\n"
        "  - if a WALL or building SIDE is directly ahead → first step MUST be a turn\n"
        "  - if you see a building DOOR → turn to face it precisely (small degree adjustments), then walk toward it\n"
        "  - if CLOSE to a door/entrance → use short W steps (150-300ms) and W+space to step over thresholds\n"
        "  - use A/D (200ms) to sidestep and align with doorways\n"
        "  - NEVER plan 3+ W steps in a row without a look step — check your surroundings"
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        print("[scout] Planning goal + steps...", flush=True)
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = (resp.content[0].text if resp.content else "").strip()
        print(f"[scout] plan response: {text!r}", flush=True)
    except Exception as e:
        print(f"[scout] plan ERROR: {e}", flush=True)
        return _fallback_plan()

    result = _parse_plan(text)
    if result is None:
        print("[scout] Plan parse failed — using fallback", flush=True)
        return _fallback_plan()

    goal, steps = result
    print(f"[scout] Goal: {goal!r}  Steps: {steps}", flush=True)
    return goal, steps


def _strip_markdown_json(text: str) -> str:
    """Strip ```json ... ``` wrappers that Claude sometimes adds."""
    import re
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        return m.group(1)
    return text

def _parse_plan(text: str) -> Optional[Tuple[str, List[Tuple[str, int]]]]:
    """Parse Claude's plan JSON into (goal, [(action, ms), ...])."""
    try:
        text = _strip_markdown_json(text)
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        obj   = json.loads(text[start:end])
        goal  = str(obj.get("goal", "")).strip()
        raw_steps = obj.get("steps", [])
        if not goal or not isinstance(raw_steps, list) or len(raw_steps) == 0:
            return None
        steps: List[Tuple[str, int]] = []
        for s in raw_steps[:5]:
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


def _fallback_plan() -> Tuple[str, List[Tuple[str, int]]]:
    """Safe default plan when Claude is unavailable or parse fails."""
    import random
    options = [
        ("look around and walk forward", [("look_right", 364), ("W", 500), ("W", 400)]),
        ("walk forward and explore", [("W", 500), ("W", 400), ("W", 300)]),
        ("turn and walk forward", [("look_left", 364), ("W", 500), ("W", 400)]),
    ]
    return random.choice(options)


# ── verify_goal ────────────────────────────────────────────────────────────────

def verify_goal(
    plan_frame: np.ndarray,
    current_frame: np.ndarray,
    goal: str,
    api_key: Optional[str] = None,
    model: str = SCOUT_MODEL,
) -> Tuple[str, str]:
    """
    Compare before/after screenshots to judge goal progress.

    Returns (status, reason) where status is "achieved", "in_progress", or "failed".
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "in_progress", ""

    b64_before = _frame_to_base64(plan_frame)
    b64_after  = _frame_to_base64(current_frame)

    prompt = (
        f'You control a Roblox character.\n'
        f'Image 1 = BEFORE executing the plan. Image 2 = AFTER.\n'
        f'Goal: "{goal}"\n\n'
        "Compare the two images carefully:\n"
        "  1. Has the character's position changed? (different background, different angle?)\n"
        "  2. Is the character closer to the goal?\n"
        "  3. If the images look NEARLY IDENTICAL, the character is STUCK and did not move.\n\n"
        "Reply with ONLY valid JSON:\n"
        '{"status": "achieved", "reason": "reached the destination"}\n'
        '{"status": "in_progress", "reason": "moved closer, keep going"}\n'
        '{"status": "failed", "reason": "describe what went wrong"}\n\n'
        "IMPORTANT: If the two images look almost the same (same walls, same ground, same angle), "
        "that means the character is STUCK and you MUST respond with \"failed\". "
        "Only say \"in_progress\" if you can see clear evidence of movement."
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
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_before},
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_after},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = (resp.content[0].text if resp.content else "").strip()
        print(f"[scout] verify response: {text!r}", flush=True)
    except Exception as e:
        print(f"[scout] verify ERROR: {e}", flush=True)
        return "in_progress", ""

    return _parse_verify(text)


def _parse_verify(text: str) -> Tuple[str, str]:
    try:
        text = _strip_markdown_json(text)
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end <= start:
            return "in_progress", ""
        obj    = json.loads(text[start:end])
        status = str(obj.get("status", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()
        if status in ("achieved", "in_progress", "failed"):
            return status, reason
    except Exception:
        pass
    return "in_progress", ""


# ── Legacy (kept for backward compat with tests) ──────────────────────────────

def _parse_single_action(text: str) -> Optional[Tuple[str, int]]:
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        obj = json.loads(text[start:end])
        action = str(obj.get("action", "")).strip()
        if action.lower() == "space":
            action = "W+space"
        if action not in PLAN_ACTIONS:
            return None
        is_look = action.lower() in ("look_left", "look_right")
        if is_look:
            degrees = obj.get("degrees")
            if degrees is not None:
                ms = degrees_to_ms(float(degrees))
            else:
                ms = max(100, min(2000, int(obj.get("ms", 364))))
        else:
            ms = max(100, min(MAX_MOVEMENT_MS, int(obj.get("ms", 400))))
        return (action, ms)
    except Exception:
        return None


def plan_one_action(
    frame: np.ndarray,
    context_text: str,
    api_key: Optional[str] = None,
    game: Optional[str] = None,
    model: str = SCOUT_MODEL,
) -> Tuple[str, int]:
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ("W", 400)

    b64       = _frame_to_base64(frame)
    game_hint = GAME_CONTEXT.get((game or "").lower(), "")

    prompt = (
        "You control a Roblox character. Look at the screenshot and the situation summary below.\n"
        "Choose ONE action to take right now.\n\n"
        "--- SITUATION ---\n"
        + context_text
        + "\n--- END SITUATION ---\n\n"
        + (game_hint + "\n\n" if game_hint else "")
        + _MOVEMENT_REFERENCE + "\n\n"
        "ACTIONS (pick exactly one):\n"
        "  W          = run forward\n"
        "  A          = strafe left\n"
        "  D          = strafe right\n"
        "  S          = back up\n"
        "  W+space    = run forward AND jump\n"
        "  look_left  = rotate camera left\n"
        "  look_right = rotate camera right\n"
        "  none       = pause/wait\n\n"
        "Reply with ONLY valid JSON:\n"
        '{"action": "W", "ms": 400}\n'
        "or for camera rotation:\n"
        '{"action": "look_right", "degrees": 90}'
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=80,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = (resp.content[0].text if resp.content else "").strip()
    except Exception as e:
        print(f"[scout] ERROR: {e}", flush=True)
        return ("W", 400)

    result = _parse_single_action(text)
    return result if result is not None else ("W", 400)


def _default_plan_10s() -> List[Tuple[str, int]]:
    import random
    plans = [
        [("W", 500), ("none", 200)],
        [("look_right", 364), ("W", 500)],
        [("look_left", 364), ("W", 500)],
        [("W", 500), ("W+space", 400)],
        [("W", 400), ("A", 200), ("W", 400)],
    ]
    return random.choice(plans)


def plan_next_10s(
    frame: np.ndarray,
    api_key: Optional[str] = None,
    model: str = SCOUT_MODEL,
    current_plan_remaining: Optional[List[Tuple[str, int]]] = None,
    executed_recent: Optional[List[str]] = None,
    user_pattern: Optional[str] = None,
    last_objective: Optional[str] = None,
    look_streak: int = 0,
    edge_distances: Optional[np.ndarray] = None,
    flow_mean: Optional[float] = None,
    game: Optional[str] = None,
) -> Tuple[List[Tuple[str, int]], str, bool]:
    lines = []
    if last_objective:
        lines.append(f"Current goal: {last_objective}")
    if executed_recent:
        lines.append(f"Last actions: {', '.join(executed_recent[-6:])}")
    if look_streak >= 2:
        lines.append(f"STUCK LOOKING: {look_streak} consecutive look actions — move forward instead")
    if flow_mean is not None:
        if flow_mean < 0.3:
            lines.append("Motion: stationary")
        elif flow_mean < 1.5:
            lines.append(f"Motion: slow (flow={flow_mean:.1f})")
        else:
            lines.append(f"Motion: moving (flow={flow_mean:.1f})")
    context_text = "\n".join(lines) if lines else "No additional context."
    action, ms = plan_one_action(frame, context_text, api_key=api_key, game=game, model=model)
    return [(action, ms)], "", False
