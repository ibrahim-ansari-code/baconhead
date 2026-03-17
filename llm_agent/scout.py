"""
Llama 4 Scout (Groq Vision): send screen image, get scores for 10 actions; or plan next 10s of actions.
"""

import base64
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np

SCOUT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_ACTIONS = ["W", "A", "S", "D", "space", "none", "look_left", "look_right", "look_left", "look_right"]

# Actions allowed in 10s plans; typical durations (ms) for prompting
PLAN_ACTIONS = ("W", "A", "S", "D", "space", "none", "look_left", "look_right")
DEFAULT_MOVE_MS = 900
DEFAULT_LOOK_MS = 400
PLAN_TARGET_MS = 10_000


def _frame_to_base64(frame: np.ndarray, max_size: int = 512) -> str:
    """Encode frame as JPEG base64; resize to max_size on long edge to stay under 4MB."""
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


def score_actions_with_scout(
    frame: np.ndarray,
    actions: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model: str = SCOUT_MODEL,
    last_actions: Optional[List[str]] = None,
    last_objective: Optional[str] = None,
    user_pattern: Optional[str] = None,
) -> Tuple[List[float], float, str, str, bool]:
    """
    Send frame to Llama 4 Scout; ask it to rate each of 10 actions 0-1, AVOID=0/1, POPUP=0/1.
    Returns (list of 10 scores, avoid_penalty 0 or 1, raw response text, objectives_string, popup_detected).
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    actions = actions or DEFAULT_ACTIONS
    if len(actions) != 10:
        actions = (actions * (10 // len(actions) + 1))[:10]
    b64 = _frame_to_base64(frame)
    context = ""
    if last_objective:
        context += f"Current goal (carry forward): {last_objective}\n\n"
    if last_actions:
        context += f"Recent actions you just took (in order): {', '.join(last_actions[-8:])}\n"
        context += "Do NOT keep repeating the same action if it didn't make progress (e.g. jumping in place many times → give space a LOW score like 0.1). Pick the action that best moves toward the goal.\n\n"
    if user_pattern and user_pattern.strip():
        context += f"User activity pattern (when they were playing): {user_pattern.strip()[:200]}\nPrefer actions that continue this play style when reasonable.\n\n"
    prompt = (
        "You are watching a game (obby/platformer). Decide what the player should do next.\n\n"
        + context +
        "Step 1 - Objectives: In one short line, what is the main goal RIGHT NOW? (e.g. reach the next flag, get to the platform, move forward, avoid the gap)\n\n"
        "Step 2 - Danger: If the screen shows death, game over, falling, or losing health, reply AVOID=1 else AVOID=0.\n\n"
        "Step 3 - Popup: Is there a popup, dialog, or menu covering the game (e.g. with an X or No button)? Reply POPUP=1 if yes, POPUP=0 if no.\n\n"
        "Step 4 - Score each action 0-1. CRITICAL: If the path or next platform is to the LEFT, give look_left 0.8-1.0 and W 0.1-0.2 (going W will run straight off). "
        "If the path is to the RIGHT, give look_right 0.8-1.0 and W 0.1-0.2. Only give W 0.7+ when the path is clearly straight ahead. "
        "When unsure or when the character might fall if they go straight, prefer look_left or look_right (0.6+) to turn the camera first. "
        "Give look_left/look_right 0.1-0.2 only when going straight is clearly correct. Jumping in place = space 0.1-0.2.\n\n"
        "Reply format: AVOID=0 or 1, POPUP=0 or 1, OBJECTIVES=your goal, then exactly 10 scores in order.\n"
        "REWARD1=score for W, REWARD2=score for A, REWARD3=S, REWARD4=D, REWARD5=space, REWARD6=none, REWARD7=look_left, REWARD8=look_right, REWARD9=look_left, REWARD10=look_right.\n"
        "Write ONLY the number after each equals sign (e.g. REWARD1=0.8 REWARD2=0.2 REWARD3=0.1 ... REWARD10=0.5). Do not put action names after the number.\n"
        "Example: REWARD1=0.2 REWARD2=0.1 REWARD3=0.1 REWARD4=0.1 REWARD5=0.1 REWARD6=0.1 REWARD7=0.9 REWARD8=0.1 REWARD9=0.1 REWARD10=0.1"
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        print("[scout] Calling Groq Llama 4 Scout (vision)...", flush=True)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=256,
            temperature=0.3,
        )
        text = resp.choices[0].message.content or ""
        print(f"[scout] response ({len(text)} chars): {text[:200]!r}...", flush=True)
    except Exception as e:
        print(f"[scout] ERROR: {e}", flush=True)
        return [0.0] * 10, 0.0, str(e), "", False
    avoid = 1.0 if re.search(r"AVOID\s*=\s*1", text, re.IGNORECASE) else 0.0
    popup = bool(re.search(r"POPUP\s*=\s*1", text, re.IGNORECASE))
    objectives = ""
    m = re.search(r"OBJECTIVES\s*=\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        objectives = m.group(1).strip()[:120]
    scores = _parse_rewards(text, n=10, actions=actions)
    if objectives:
        print(f"[scout] objectives: {objectives}", flush=True)
    if popup:
        print("[scout] POPUP=1 detected — will click close (X / No)", flush=True)
    print(f"[scout] parsed AVOID={avoid:.0f} POPUP={int(popup)} REWARD1..10={[f'{s:.2f}' for s in scores]}", flush=True)
    return scores, avoid, text, objectives, popup


def _parse_rewards(text: str, n: int = 10, actions: Optional[List[str]] = None) -> List[float]:
    """Extract REWARD1=... REWARD2=... from response. If model writes REWARD1=0.9 look_left, assign 0.9 to look_left index."""
    default_actions = ["W", "A", "S", "D", "space", "none", "look_left", "look_right", "look_left", "look_right"]
    actions = actions or default_actions
    if len(actions) < n:
        actions = (actions * (n // len(actions) + 1))[:n]
    scores = [0.0] * n
    # Indices already set by name (e.g. REWARD1=0.9 look_left) — don't overwrite with position later
    name_set: set = set()
    valid_action_names = {a.lower() for a in actions} | {"w", "a", "s", "d"}
    # Match REWARDi=number optionally followed by action name (so we fix misattribution)
    for i in range(1, n + 1):
        m = re.search(rf"REWARD{i}\s*=\s*([\d.]+)\s*([A-Za-z_]+)?", text, re.IGNORECASE)
        if m:
            try:
                value = float(m.group(1))
            except ValueError:
                continue
            action_after = (m.group(2) or "").strip()
            # Only treat as action name if it's a known action (ignore "REWARD" from next token)
            if action_after and action_after.lower() in valid_action_names and not action_after.upper().startswith("REWARD"):
                action_lower = action_after.lower()
                idx = next(
                    (j for j, a in enumerate(actions) if a.lower() == action_lower or a == action_after),
                    i - 1,
                )
                scores[idx] = value
                name_set.add(idx)
            else:
                if (i - 1) not in name_set:
                    scores[i - 1] = value
    return scores[:n]


def _parse_plan_json(text: str) -> List[Tuple[str, int]]:
    """Extract a JSON array of {action, ms} from response. Returns list of (action, ms)."""
    # Find a [...] array in the text
    start = text.find("[")
    if start == -1:
        return []
    depth = 0
    end = -1
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
        ms = item.get("ms") or item.get("duration_ms") or item.get("duration")
        if action is None or ms is None:
            continue
        action = str(action).strip()
        if action not in PLAN_ACTIONS:
            continue
        try:
            ms = int(ms)
        except (TypeError, ValueError):
            continue
        ms = max(100, min(2000, ms))  # clamp 100–2000 ms per step
        out.append((action, ms))
    return out


def _default_plan_10s() -> List[Tuple[str, int]]:
    """Safe fallback plan: a few moves and looks to fill ~10s."""
    return [
        ("W", DEFAULT_MOVE_MS),
        ("look_right", DEFAULT_LOOK_MS),
        ("W", DEFAULT_MOVE_MS),
        ("none", 400),
        ("W", DEFAULT_MOVE_MS),
        ("look_left", DEFAULT_LOOK_MS),
        ("W", DEFAULT_MOVE_MS),
        ("W", DEFAULT_MOVE_MS),
        ("none", 400),
    ]


def plan_next_10s(
    frame: np.ndarray,
    api_key: Optional[str] = None,
    model: str = SCOUT_MODEL,
    current_plan_remaining: Optional[List[Tuple[str, int]]] = None,
    executed_recent: Optional[List[str]] = None,
    user_pattern: Optional[str] = None,
    last_objective: Optional[str] = None,
) -> Tuple[List[Tuple[str, int]], str, bool]:
    """
    Ask the model to plan the next ~10 seconds of actions (no forcing).
    If current_plan_remaining is set, the model plans what to do AFTER that plan; it must not re-output those actions.
    Returns (list of (action, duration_ms), objectives_string, popup_detected).
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    b64 = _frame_to_base64(frame)
    context = ""
    if last_objective:
        context += f"Current goal: {last_objective}\n\n"
    if executed_recent:
        context += f"Recently executed (in order): {', '.join(executed_recent[-10:])}\n\n"
    if user_pattern and user_pattern.strip():
        context += f"User play style (when they were playing): {user_pattern.strip()[:200]}\n\n"
    if current_plan_remaining:
        remaining_str = ", ".join(f"{a}({ms}ms)" for a, ms in current_plan_remaining[:12])
        context += (
            "The bot is ALREADY executing (or about to execute) these actions — do NOT include them in your plan:\n"
            f"  {remaining_str}\n\n"
            "Output ONLY the next 10 seconds of actions that happen AFTER the above.\n\n"
        )
    prompt = (
        "You are controlling a character in an obby/platformer game. Plan the next 10 seconds of actions.\n\n"
        + context +
        "What each action does:\n"
        "- W = move forward, A = strafe left, S = move back, D = strafe right\n"
        "- space = jump\n"
        "- look_left = turn camera left, look_right = turn camera right\n"
        "- none = do nothing for a short moment (e.g. 200–400ms)\n\n"
        "Goals: reach the next platform, don't fall, complete the obby. "
        "Only turn (look_left/look_right) when the path is not straight ahead. "
        "Only jump when needed. Don't repeat the same action many times if it doesn't help.\n\n"
        "Step 1 - Popup: Is there a popup/dialog covering the game? Reply POPUP=1 or POPUP=0.\n"
        "Step 2 - Objective: One short line: OBJECTIVES=your goal for this 10s.\n"
        "Step 3 - Plan: Reply with a JSON array only. Each element: {\"action\": \"W\", \"ms\": 900}. "
        "Use ms: 300–500 for look_left/look_right, 600–1000 for W/A/S/D/space, 0–400 for none. "
        "Total duration about 10000ms. Example: [{\"action\":\"W\",\"ms\":900},{\"action\":\"look_right\",\"ms\":400},{\"action\":\"W\",\"ms\":900}]\n\n"
        "Reply: POPUP=0 or 1, OBJECTIVES=..., then the JSON array."
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        print("[scout] Planning next 10s (Groq vision)...", flush=True)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.3,
        )
        text = resp.choices[0].message.content or ""
        print(f"[scout] plan response ({len(text)} chars)", flush=True)
    except Exception as e:
        print(f"[scout] plan ERROR: {e}", flush=True)
        return _default_plan_10s(), "", False
    popup = bool(re.search(r"POPUP\s*=\s*1", text, re.IGNORECASE))
    objectives = ""
    m = re.search(r"OBJECTIVES\s*=\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        objectives = m.group(1).strip()[:120]
    plan = _parse_plan_json(text)
    if not plan:
        print("[scout] plan parse failed, using default 10s", flush=True)
        return _default_plan_10s(), objectives, popup
    total_ms = sum(ms for _, ms in plan)
    print(f"[scout] plan: {len(plan)} steps, total {total_ms}ms, objectives={objectives!r} popup={popup}", flush=True)
    return plan, objectives, popup


def get_best_action(
    frame: np.ndarray,
    actions: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    last_actions: Optional[List[str]] = None,
    last_objective: Optional[str] = None,
) -> Tuple[str, List[float], float, str]:
    """Score 10 actions with Scout, return (best_action_name, list_of_scores, avoid_penalty, objectives)."""
    actions = actions or DEFAULT_ACTIONS
    if len(actions) != 10:
        actions = (actions * (10 // len(actions) + 1))[:10]
    scores, avoid, _, objectives, _ = score_actions_with_scout(
        frame, actions=actions, api_key=api_key,
        last_actions=last_actions, last_objective=last_objective,
    )
    idx = max(range(len(scores)), key=lambda i: scores[i])
    return actions[idx], scores, avoid, objectives
