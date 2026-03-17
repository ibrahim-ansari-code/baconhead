"""
CEM: 10 noisy action options, score with reward model + LLM (Scout), apply avoid penalty, pick best.
Millisecond-accurate timing for execution.
"""

import time
from typing import Optional, List, Tuple

import numpy as np

from reward.avoids import get_avoid_penalty


def run_cem(
    frame: np.ndarray,
    actions: Optional[List[str]] = None,
    reward_model=None,
    device=None,
    scout_api_key: Optional[str] = None,
    use_reward_model: bool = False,
    use_scout: bool = True,
    avoid_weight: float = 1.0,
    last_actions: Optional[List[str]] = None,
    last_objective: Optional[str] = None,
    user_pattern: Optional[str] = None,
    mock_scout_result: Optional[Tuple[List[float], float, str]] = None,
) -> Tuple[str, List[float], float, str, bool]:
    """
    Run CEM: 10 options, score each with Scout (context-aware), subtract avoid penalty, pick best.
    Returns (best_action, list_of_scores, combined_reward_for_best, objectives_string, popup_detected).
    mock_scout_result: optional (scores[10], avoid_pen, objectives_str) or (..., popup) for testing without API.
    """
    from llm_agent.scout import score_actions_with_scout

    default_actions = ["W", "A", "S", "D", "space", "none", "look_left", "look_right", "look_left", "look_right"]
    actions = actions or default_actions
    if len(actions) != 10:
        actions = (actions * (10 // len(actions) + 1))[:10]

    # 1) Scout scores + avoid + objectives + popup (or mock for tests)
    objectives = ""
    popup = False
    if mock_scout_result is not None:
        scout_scores, avoid_pen = mock_scout_result[0], mock_scout_result[1]
        objectives = mock_scout_result[2] if len(mock_scout_result) > 2 else ""
        popup = bool(mock_scout_result[3]) if len(mock_scout_result) > 3 else False
        scout_scores = (scout_scores + [0.5] * 10)[:10]
    elif use_scout and scout_api_key is not None:
        scout_scores, avoid_pen, raw, objectives, popup = score_actions_with_scout(
            frame, actions=actions, api_key=scout_api_key,
            last_actions=last_actions, last_objective=last_objective,
            user_pattern=user_pattern,
        )
        print(f"[cem] Scout: avoid_pen={avoid_pen:.0f} scores={[f'{s:.2f}' for s in scout_scores]}", flush=True)
    else:
        scout_scores = [0.5] * 10
        avoid_pen = get_avoid_penalty(frame, caption=None, use_vision=True)
        print(f"[cem] No Scout; avoid_pen(BLIP)={avoid_pen:.0f}", flush=True)

    # 2) Scout scores minus avoid; optional reward model: low r(s) -> boost "none"
    scores = [scout_scores[i] - avoid_weight * avoid_pen for i in range(10)]
    if use_reward_model and reward_model is not None and device is not None:
        import torch
        from reward.combined import frame_to_tensor
        with torch.no_grad():
            x = frame_to_tensor(frame, height=84, width=84).to(device)
            logit = reward_model(x).item()
            r_state = max(0.0, min(1.0, torch.sigmoid(torch.tensor(logit)).item()))
        none_idx = next((i for i, a in enumerate(actions) if a == "none"), None)
        if none_idx is not None and r_state < 0.35:
            scores[none_idx] += 0.4
            print(f"[cem] reward r(s)={r_state:.2f} low -> boost none", flush=True)

    # 3) Repeat penalty: if same action 3+ times in a row (or space 2+), downweight it so we get variety
    if last_actions and len(last_actions) >= 2:
        repeated = last_actions[-1]
        n_repeat = 1
        for a in reversed(last_actions[:-1]):
            if a == repeated:
                n_repeat += 1
            else:
                break
        if n_repeat >= 2 and repeated == "space":
            space_idx = next((i for i, a in enumerate(actions) if a == "space"), None)
            if space_idx is not None:
                scores[space_idx] = max(0.0, scores[space_idx] - 0.5)
                print(f"[cem] anti-repeat: downweighted space (jump in place)", flush=True)
        elif n_repeat >= 3:
            for i, a in enumerate(actions):
                if a == repeated:
                    scores[i] = max(0.0, scores[i] - 0.4)
            print(f"[cem] anti-repeat: downweighted {repeated!r} (same action {n_repeat}x)", flush=True)
    # Downweight look_left/look_right after 2x same (avoid spinning)
    if last_actions and len(last_actions) >= 2 and last_actions[-1] == last_actions[-2]:
        repeated = last_actions[-1]
        if repeated in ("look_left", "look_right"):
            for i, a in enumerate(actions):
                if a == repeated:
                    scores[i] = max(0.0, scores[i] - 0.5)
            print(f"[cem] anti-repeat: downweighted {repeated!r} (avoid spin)", flush=True)

    # No forced alternation: pick best action from Scout/reward; only repeat penalties and avoid apply
    idx = max(range(10), key=lambda i: scores[i])
    best_action = actions[idx]
    print(f"[cem] best=#{idx} {best_action!r} combined_r={scores[idx]:.3f}", flush=True)
    return best_action, scores, scores[idx], objectives, popup


# Mouse delta in pixels for look actions (Roblox needs a big move to turn noticeably)
LOOK_DX = 220
LOOK_DY = 120


def execute_action_ms(action: str, duration_ms: int = 5000) -> None:
    """Execute a single action (key name or look_left/look_right) for duration_ms milliseconds."""
    if action is None or action.lower() == "none":
        time.sleep(duration_ms / 1000.0)
        return
    action_lower = action.lower() if isinstance(action, str) else ""
    if action_lower in ("look_left", "look_right", "look_up", "look_down"):
        try:
            import pyautogui
            look_ms = min(duration_ms, 350)
            steps = max(6, look_ms // 45)
            dx = (LOOK_DX if action_lower == "look_right" else -LOOK_DX if action_lower == "look_left" else 0) // steps
            dy = (LOOK_DY if action_lower == "look_down" else -LOOK_DY if action_lower == "look_up" else 0) // steps
            step_ms = look_ms / steps
            for _ in range(steps):
                pyautogui.move(dx, dy)
                time.sleep(step_ms / 1000.0)
            time.sleep(max(0, (duration_ms - look_ms) / 1000.0))
        except Exception:
            time.sleep(duration_ms / 1000.0)
        return
    key_map = {
        "W": "w", "A": "a", "S": "s", "D": "d",
        "space": "space", "Space": "space",
    }
    k = key_map.get(action, action.lower() if isinstance(action, str) else None)
    if k is None:
        time.sleep(duration_ms / 1000.0)
        return
    try:
        import pyautogui
        key = "space" if k == "space" else k
        pyautogui.keyDown(key)
        try:
            time.sleep(duration_ms / 1000.0)
        finally:
            pyautogui.keyUp(key)
    except Exception:
        time.sleep(duration_ms / 1000.0)
