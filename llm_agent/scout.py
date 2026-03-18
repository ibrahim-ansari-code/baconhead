"""
Llama 4 Scout (Groq vision): plan the next ~10 seconds of obby actions.

Only one public function is exposed: plan_next_10s().
The old score_actions_with_scout / REWARD1-10 multi-output scoring has been removed —
it produced noisy, repetitive results. We now generate one structured plan per call.
"""

import base64
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np

SCOUT_MODEL    = "meta-llama/llama-4-maverick-17b-128e-instruct"
PLAN_ACTIONS   = ("W", "A", "S", "D", "space", "none", "look_left", "look_right")
DEFAULT_MOVE_MS = 700
DEFAULT_LOOK_MS = 350
PLAN_TARGET_MS  = 10_000


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
        ms     = item.get("ms") or item.get("duration_ms") or item.get("duration")
        if action is None or ms is None:
            continue
        action = str(action).strip()
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
    """Randomised fallback plan covering ~10s with varied actions."""
    import random
    plans = [
        [("look_right", 350), ("W", 700), ("space", 280), ("W", 700), ("none", 200),
         ("W", 800), ("look_left", 300), ("W", 700), ("space", 280), ("none", 200)],
        [("W", 600), ("space", 250), ("W", 700), ("look_right", 350), ("W", 800),
         ("space", 280), ("none", 200), ("W", 700), ("look_left", 300), ("W", 600)],
        [("look_left", 350), ("W", 700), ("space", 280), ("W", 600), ("none", 200),
         ("look_right", 300), ("W", 800), ("space", 260), ("W", 700), ("none", 150)],
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
) -> Tuple[List[Tuple[str, int]], str, bool]:
    """
    Ask Llama 4 Scout (vision) to plan the next ~10 seconds.
    Returns (plan: list of (action, ms), objectives: str, popup_detected: bool).

    look_streak: number of consecutive look_left/look_right actions already executed.
                 When >= 2, the prompt tells the model to use W or W+space instead of more turns.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    b64      = _frame_to_base64(frame)
    physics  = _load_physics()
    phys_hint = _physics_hint(physics)

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

    # Anti-look-repeat instruction
    look_note = ""
    if look_streak >= 2:
        look_note = (
            f"\n⚠ The bot has already done {look_streak} consecutive look actions with no progress. "
            "DO NOT start with another look_left or look_right. "
            "Start with W, W+space, or space to actually move toward the goal.\n"
        )

    prompt = (
        "You are controlling a Roblox obby character. Plan the next 10 seconds of actions.\n\n"
        + context
        + (phys_hint + "\n\n" if phys_hint else "")
        + look_note
        + "\nACTIONS:\n"
        "  W = run forward (only if path is straight ahead)\n"
        "  A/D = strafe left/right (small adjustments)\n"
        "  S = back up\n"
        "  space = jump (200-350ms tap for precision; 500ms for wide gaps)\n"
        "  look_left / look_right = rotate camera (ALWAYS before W if the path bends that way)\n"
        "  none = pause 150-300ms (after landing or when in danger)\n\n"
        "STRATEGY:\n"
        "  1. If the next platform is to the LEFT → look_left first, then W.\n"
        "  2. If the next platform is to the RIGHT → look_right first, then W.\n"
        "  3. To cross a gap: W (run) then space (jump).\n"
        "  4. Do NOT repeat look_left/look_right more than 2 times — if you already turned, move.\n"
        "  5. After landing from a jump, add a short none (150ms) before the next move.\n\n"
        "Step 1 — Popup: ONLY return POPUP=1 if a dialog/menu is CLEARLY and FULLY blocking the game view. "
        "If the game is visible and playable, POPUP=0.\n"
        "Step 2 — Objective: OBJECTIVES=one short sentence.\n"
        "Step 3 — Plan: JSON array. REQUIREMENTS:\n"
        "  - MUST have AT LEAST 8 steps\n"
        "  - MUST total AT LEAST 8000ms\n"
        "  - Each step: {\"action\": \"W\", \"ms\": 700}\n"
        "  - Duration guide: look 300-450ms | jump 250-350ms | run 500-900ms | pause 150-250ms\n"
        "  Example (8 steps ~8s): [{\"action\":\"look_right\",\"ms\":350},{\"action\":\"W\",\"ms\":700},"
        "{\"action\":\"space\",\"ms\":280},{\"action\":\"W\",\"ms\":700},{\"action\":\"none\",\"ms\":200},"
        "{\"action\":\"W\",\"ms\":700},{\"action\":\"space\",\"ms\":300},{\"action\":\"W\",\"ms\":800}]\n\n"
        "Reply format: POPUP=0/1, OBJECTIVES=..., then ONLY the JSON array."
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        print("[scout] Planning next 10s...", flush=True)
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=900,
            temperature=0.5,
        )
        text = resp.choices[0].message.content or ""
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

    total_ms = sum(ms for _, ms in plan)

    # If Scout returned a suspiciously short plan, extend it with sensible moves
    if total_ms < 3000:
        print(f"[scout] Plan too short ({total_ms}ms, {len(plan)} steps) — extending.", flush=True)
        filler = [("W", 700), ("space", 280), ("W", 700), ("none", 200),
                  ("W", 700), ("space", 280), ("W", 700), ("none", 200)]
        plan = plan + filler
        total_ms = sum(ms for _, ms in plan)

    print(f"[scout] Plan: {len(plan)} steps, {total_ms}ms, obj={objectives!r} popup={popup}", flush=True)
    return plan, objectives, popup
