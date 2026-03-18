"""
Claude Sonnet 4.6 (Anthropic vision): plan the next ~10 seconds of obby actions.

Only one public function is exposed: plan_next_10s().
"""

import base64
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np

SCOUT_MODEL    = "claude-sonnet-4-6"
PLAN_ACTIONS   = ("W", "A", "S", "D", "W+space", "none", "look_left", "look_right")
DEFAULT_MOVE_MS = 700
DEFAULT_LOOK_MS = 300
PLAN_TARGET_MS  = 3_000

# Calibrated: 220px drag ≈ 400ms ≈ ~45° in Roblox (0.55 px/ms, ~9°/100ms)
DEGREES_PER_100MS = 9.0

def _degrees_to_ms(degrees: float) -> int:
    """Convert desired rotation degrees to look duration ms."""
    ms = int(abs(degrees) / DEGREES_PER_100MS * 100)
    return max(100, min(500, ms))


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


def _physics_hint(physics: Optional[dict]) -> str:
    if not physics:
        return ""
    w   = physics.get("w_px_per_ms", 0)
    sp  = physics.get("space_px_per_ms", 0)
    lk  = physics.get("look_px_per_ms", 0)
    if w <= 0:
        return ""
    lines = [
        f"Movement physics (px/ms from calibration): W={w:.3f}  space={sp:.3f}  look={lk:.3f}.",
        f"  → W for 500ms ≈ {w * 500:.0f}px forward.  space for 280ms ≈ {sp * 280:.0f}px arc.",
        "Use these to estimate whether a gap can be cleared before choosing durations.",
    ]
    return "\n".join(lines)


def _parse_plan_json(text: str) -> List[Tuple[str, int]]:
    """Extract a JSON array of {action, ms} objects from the LLM response."""
    start = text.find("[")
    if start == -1:
        return []
    depth, end = 0, -1
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return []
    try:
        raw = json.loads(text[start:end])
    except json.JSONDecodeError:
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        action = item.get("action") or item.get("action_name")
        if action is None:
            continue
        action = str(action).strip()

        # For look actions: accept degrees field and convert to ms
        is_look = action.lower() in ("look_left", "look_right")
        degrees = item.get("degrees")
        ms      = item.get("ms") or item.get("duration_ms") or item.get("duration")

        if is_look and degrees is not None:
            try:
                ms = _degrees_to_ms(float(degrees))
            except (TypeError, ValueError):
                pass

        if ms is None:
            continue

        # Normalise standalone "space" to "W+space" — jumping in place is useless
        if action.lower() == "space":
            action = "W+space"
        if action not in PLAN_ACTIONS:
            continue
        try:
            ms = int(ms)
        except (TypeError, ValueError):
            continue
        ms = max(100, min(2000, ms))
        out.append((action, ms))
    return out


def _default_plan_10s() -> List[Tuple[str, int]]:
    """Randomised fallback plan covering ~3s with varied actions."""
    import random
    plans = [
        [("W", 700), ("space", 280), ("W", 600), ("none", 150)],
        [("look_right", 300), ("W", 700), ("space", 260), ("W", 500)],
        [("look_left", 300), ("W", 700), ("space", 260), ("W", 500)],
        [("W", 800), ("space", 300), ("W", 700), ("none", 150)],
        [("W", 600), ("A", 200), ("W", 600), ("space", 280)],
    ]
    return random.choice(plans)


def _spatial_hint(
    edge_distances: Optional[np.ndarray],
    flow_mean: Optional[float],
) -> str:
    """Build a plain-English spatial context string from edge distances and flow."""
    parts = []
    if edge_distances is not None and len(edge_distances) == 4:
        labels = ["top", "right", "bottom", "left"]
        edge_strs = []
        for label, dist in zip(labels, edge_distances):
            pct = int(dist * 100)
            if pct < 10:
                edge_strs.append(f"{label}={pct}% (VERY CLOSE TO EDGE)")
            elif pct < 25:
                edge_strs.append(f"{label}={pct}% (close)")
            else:
                edge_strs.append(f"{label}={pct}%")
        parts.append("Platform edge distances (% of screen to edge): " + ", ".join(edge_strs) + ".")
        # warn about dangerous edges
        dangers = [l for l, d in zip(labels, edge_distances) if d < 0.12]
        if dangers:
            parts.append(f"⚠ DANGER: very close to {'/'.join(dangers)} edge — avoid moving that direction.")
    if flow_mean is not None:
        if flow_mean < 0.5:
            parts.append("Motion: nearly stationary (character barely moving).")
        elif flow_mean < 3.0:
            parts.append(f"Motion: slow movement (flow={flow_mean:.1f}).")
        else:
            parts.append(f"Motion: actively moving (flow={flow_mean:.1f}).")
    return "\n".join(parts)


GAME_CONTEXT = {
    "nds": (
        "GAME: Natural Disaster Survival.\n"
        "GOAL: Survive the disaster. Read the warning text at the top of the screen.\n"
        "RULES:\n"
        "  - Flood/Tsunami warning → find high ground immediately (stairs, hills, tall buildings). Use W+space to climb.\n"
        "  - Tornado warning → find a solid building and go inside.\n"
        "  - Blizzard/Acid Rain → go indoors or find shelter.\n"
        "  - Earthquake → move away from falling debris, get to open ground.\n"
        "  - NEVER jump in place — W+space only when moving toward safety.\n"
        "  - If you see other players running somewhere, follow them (they know the map).\n"
        "  - Prioritise elevation and indoor cover over staying still."
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
}


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
    Ask Claude Sonnet 4.6 (vision) to plan the next ~10 seconds.
    Returns (plan: list of (action, ms), objectives: str, popup_detected: bool).

    look_streak: number of consecutive look_left/look_right actions already executed.
                 When >= 2, the prompt tells the model to use W or W+space instead of more turns.
    edge_distances: (4,) array [top, right, bottom, left] normalised to [0, 1].
    flow_mean: mean optical flow magnitude over the last frame pair.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    b64        = _frame_to_base64(frame)
    physics    = _load_physics()
    phys_hint  = _physics_hint(physics)
    spat_hint  = _spatial_hint(edge_distances, flow_mean)
    game_hint  = GAME_CONTEXT.get((game or "").lower(), "")

    context = ""
    if last_objective:
        context += f"Current goal: {last_objective}\n\n"
    if executed_recent:
        context += f"Recently executed actions (in order): {', '.join(executed_recent[-12:])}\n\n"
    if user_pattern and user_pattern.strip():
        context += f"User play style: {user_pattern.strip()[:200]}\n\n"
    if current_plan_remaining:
        rem = ", ".join(f"{a}({ms}ms)" for a, ms in current_plan_remaining[:10])
        context += (
            "Already queued actions (DO NOT repeat these in your plan):\n"
            f"  {rem}\n\n"
            "Output ONLY actions that come AFTER the above.\n\n"
        )

    look_note = ""
    if look_streak >= 2:
        look_note = (
            f"\n⚠ The bot has already done {look_streak} consecutive look actions with no progress. "
            "DO NOT start with another look_left or look_right. "
            "Start with W, W+space, or space to actually move toward the goal.\n"
        )

    prompt = (
        "You are controlling a Roblox character. Look at the screenshot and plan the next 3 seconds of movement.\n\n"
        + (game_hint + "\n\n" if game_hint else "")
        + context
        + (phys_hint + "\n\n" if phys_hint else "")
        + (spat_hint + "\n\n" if spat_hint else "")
        + look_note
        + "\nCAMERA ROTATION:\n"
        "  look_left / look_right use a 'degrees' field (not ms) to specify exact rotation.\n"
        "  Small correction: 10-20°  |  Turn a corner: 45-90°  |  Full U-turn: 150-180°\n"
        "  ONE look action max per plan. After looking you MUST move (W, A, D, or W+space).\n\n"
        "ACTIONS:\n"
        "  W          = run forward\n"
        "  A / D      = strafe left/right (small nudge, no camera change)\n"
        "  S          = back up\n"
        "  W+space    = run forward AND jump (only way to jump — never jump in place)\n"
        "  look_left  = rotate camera left  — use {\"action\":\"look_left\",\"degrees\":45}\n"
        "  look_right = rotate camera right — use {\"action\":\"look_right\",\"degrees\":30}\n"
        "  none       = pause 100-200ms (after landing)\n\n"
        "RULES:\n"
        "  1. Study the screenshot — find the path/platform/safe zone.\n"
        "  2. Path straight ahead → W immediately.\n"
        "  3. Need to turn → ONE look action with exact degrees, then W.\n"
        "  4. Gap → W+space (run and jump together).\n"
        "  5. NEVER use 2 look actions. NEVER look without moving after.\n"
        "  6. Near an edge → S first.\n\n"
        "Step 1 — Popup: POPUP=1 only if menu FULLY blocks screen.\n"
        "Step 2 — Objective: OBJECTIVES=one sentence.\n"
        "Step 3 — Plan: JSON array, ~3 seconds total (2000-3500ms of W/W+space, excluding look).\n"
        "  Look format:  {\"action\": \"look_right\", \"degrees\": 45}\n"
        "  Move format:  {\"action\": \"W\", \"ms\": 700}\n"
        "  Example: [{\"action\":\"look_right\",\"degrees\":60},{\"action\":\"W\",\"ms\":700},{\"action\":\"W+space\",\"ms\":400},{\"action\":\"W\",\"ms\":600}]\n\n"
        "Reply format: POPUP=0/1, OBJECTIVES=..., then ONLY the JSON array."
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        print("[scout] Planning next 10s with Claude Sonnet 4.6...", flush=True)
        resp = client.messages.create(
            model=model,
            max_tokens=900,
            temperature=0.5,
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
        text = resp.content[0].text if resp.content else ""
        print(f"[scout] response ({len(text)} chars)", flush=True)
    except Exception as e:
        print(f"[scout] ERROR: {e}", flush=True)
        return _default_plan_10s(), "", False

    popup = bool(re.search(r"POPUP\s*=\s*1", text, re.IGNORECASE))
    objectives = ""
    m = re.search(r"OBJECTIVES\s*=\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        objectives = m.group(1).strip()[:120]

    plan = _parse_plan_json(text)
    if not plan:
        print("[scout] Plan parse failed — using default.", flush=True)
        return _default_plan_10s(), objectives, popup

    # ── hard enforcement: max 1 look action, look ms capped at 500 ──────────
    look_count = 0
    cleaned = []
    for action, ms in plan:
        is_look = action.lower() in ("look_left", "look_right")
        if is_look:
            if look_count >= 1:
                print(f"[scout] Dropping extra look '{action}' (max 1 per plan)", flush=True)
                continue
            ms = min(ms, 500)  # hard cap ~50°
            look_count += 1
        cleaned.append((action, ms))
    plan = cleaned

    # ── enforce: look can't be followed by another look ──────────────────────
    final = []
    for i, (action, ms) in enumerate(plan):
        is_look = action.lower() in ("look_left", "look_right")
        prev_is_look = final and final[-1][0].lower() in ("look_left", "look_right")
        if is_look and prev_is_look:
            print(f"[scout] Replacing consecutive look with W", flush=True)
            final.append(("W", 600))
        else:
            final.append((action, ms))
    plan = final

    total_ms = sum(ms for _, ms in plan)
    if total_ms < 1000:
        print(f"[scout] Plan too short ({total_ms}ms) — using default.", flush=True)
        return _default_plan_10s(), objectives, popup

    print(f"[scout] Plan: {len(plan)} steps, {total_ms}ms, obj={objectives!r} popup={popup}", flush=True)
    return plan, objectives, popup
