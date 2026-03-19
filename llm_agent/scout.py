"""
Claude Sonnet 4.6 (Anthropic vision): decide the next single action for the bot.

Public API:
  plan_one_action(frame, context_text, api_key, game) -> (action, ms)
  plan_next_10s(...) kept for backward compatibility with tests.
"""

import base64
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np

SCOUT_MODEL  = "claude-sonnet-4-6"
PLAN_ACTIONS = ("W", "A", "S", "D", "W+space", "none", "look_left", "look_right")

# Calibrated: 220px drag ≈ 400ms ≈ ~45° in Roblox (0.55 px/ms → 11.25°/100ms)
DEGREES_PER_100MS = 11.25


def _degrees_to_ms(degrees: float) -> int:
    ms = int(abs(degrees) / DEGREES_PER_100MS * 100)
    return max(100, min(2000, ms))


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


def _load_physics(physics_path: str = "episode_data/physics.json") -> Optional[dict]:
    if os.path.isfile(physics_path):
        try:
            with open(physics_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


GAME_CONTEXT = {
    "nds": (
        "GAME: Natural Disaster Survival.\n"
        "GOAL: Survive the disaster. Always read the WARNING text at the top of the screen first.\n"
        "\n"
        "PHASE 1 — No warning visible yet:\n"
        "  → Immediately find and move toward the nearest building, house, or indoor structure.\n"
        "  → Look around to spot buildings — rooftops, windows, doors. Walk toward the closest one.\n"
        "  → Enter through a door if possible. Being inside = safe from most disasters.\n"
        "  → If already inside a building, move toward the center or upper floors.\n"
        "\n"
        "PHASE 2 — Warning is visible at top of screen:\n"
        "  - Flood/Tsunami → find high ground immediately (stairs, hills, tall buildings). Use W+space to climb.\n"
        "  - Tornado → get inside a solid building immediately.\n"
        "  - Blizzard/Acid Rain → go indoors or under cover NOW.\n"
        "  - Earthquake → move to open ground away from falling debris.\n"
        "  - Meteor → find any building or overhang for cover.\n"
        "\n"
        "GENERAL:\n"
        "  - NEVER jump in place — W+space only when moving forward.\n"
        "  - If other players are running somewhere, follow them.\n"
        "  - Stay away from water edges and map boundaries.\n"
        "  - If you see a LOBBY (grey island with water on all sides, no buildings), stand still (none)."
    ),
    "obby": (
        "GAME: Obby (obstacle course).\n"
        "GOAL: Reach the next checkpoint by crossing platforms without falling.\n"
        "RULES:\n"
        "  - Always move toward the next visible platform.\n"
        "  - W+space to jump over gaps — time it so you're already moving forward.\n"
        "  - Never jump in place.\n"
        "  - If a platform is narrow, slow down (shorter W durations)."
    ),
    "brookhaven": (
        "GAME: Brookhaven RP — open-world roleplay city.\n"
        "GOAL: Roam the city naturally. Explore roads, buildings, parks. Look like a real player.\n"
        "RULES:\n"
        "  - The world is flat (roads, sidewalks). No falling hazards. Don't jump randomly.\n"
        "  - Walk along roads and paths. Turn at intersections. Enter buildings through doors.\n"
        "  - If blocked by a wall or obstacle, turn and find a new direction.\n"
        "  - Look around (look_left/look_right) at intersections to decide which way.\n"
        "  - NEVER spin in circles. ONE look then walk."
    ),
}


def _parse_single_action(text: str) -> Optional[Tuple[str, int]]:
    """Parse a single JSON object {action, ms} or {action, degrees} from Claude response."""
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
                ms = _degrees_to_ms(float(degrees))
            else:
                ms = obj.get("ms", 400)
                ms = max(100, min(2000, int(ms)))
        else:
            ms = obj.get("ms", 400)
            ms = max(100, min(800, int(ms)))
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
    """
    Ask Claude Sonnet 4.6 for a single action given the current screenshot + situation context.
    Returns (action, ms). Falls back to ("W", 400) on any error.

    context_text: rich situation summary built by _build_context() in run_takeover.py.
                  Includes phase, water/void %, motion, edge distances, stuck count,
                  disaster warning status, and recent actions.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[scout] No API key — using fallback W action", flush=True)
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
        + "CAMERA: look_left/look_right use degrees. Small correction=15-30°, corner=60-90°, U-turn=150-180°.\n\n"
        "ACTIONS (pick exactly one):\n"
        "  W          = run forward (ms: 300-600)\n"
        "  A          = strafe left (ms: 200-400)\n"
        "  D          = strafe right (ms: 200-400)\n"
        "  S          = back up (ms: 200-400)\n"
        "  W+space    = run forward AND jump (ms: 300-500)\n"
        "  look_left  = rotate camera left\n"
        "  look_right = rotate camera right\n"
        "  none       = pause/wait (ms: 200-500)\n\n"
        "Reply with ONLY valid JSON — no explanation, no markdown:\n"
        '{"action": "W", "ms": 400}\n'
        "or for camera rotation:\n"
        '{"action": "look_right", "degrees": 60}'
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        print("[scout] Asking Claude for next action...", flush=True)
        resp = client.messages.create(
            model=model,
            max_tokens=80,
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
        print(f"[scout] response: {text!r}", flush=True)
    except Exception as e:
        print(f"[scout] ERROR: {e}", flush=True)
        return ("W", 400)

    result = _parse_single_action(text)
    if result is None:
        print("[scout] Parse failed — using fallback W", flush=True)
        return ("W", 400)

    print(f"[scout] Action: {result[0]!r} {result[1]}ms", flush=True)
    return result


# ── backward-compatible batch planner (kept for tests) ─────────────────────────

def _default_plan_10s() -> List[Tuple[str, int]]:
    import random
    plans = [
        [("W", 500), ("none", 200)],
        [("look_right", 400), ("W", 500)],
        [("look_left", 400), ("W", 500)],
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
    """
    Backward-compatible wrapper: builds a context string from the legacy parameters
    and delegates to plan_one_action, returning a single-element plan list.
    """
    lines = []
    if last_objective:
        lines.append(f"Current goal: {last_objective}")
    if executed_recent:
        lines.append(f"Last actions: {', '.join(executed_recent[-6:])}")
    if look_streak >= 2:
        lines.append(f"STUCK LOOKING: {look_streak} consecutive look actions — move forward instead")
    if edge_distances is not None and len(edge_distances) == 4:
        labels = ["top", "right", "bottom", "left"]
        parts = []
        for label, dist in zip(labels, edge_distances):
            pct = int(dist * 100)
            tag = " (VERY CLOSE)" if pct < 10 else " (close)" if pct < 25 else ""
            parts.append(f"{label}={pct}%{tag}")
        lines.append("Platform edges: " + ", ".join(parts))
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
