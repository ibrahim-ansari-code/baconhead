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
) -> Tuple[str, List[float], float, str, bool, Optional[int]]:
    """
    Run CEM: 10 options, score each with Scout (context-aware), subtract avoid penalty, pick best.
    Returns (best_action, list_of_scores, combined_reward_for_best, objectives_string, popup_detected, duration_ms_override).
    mock_scout_result: optional (scores[10], avoid_pen, objectives_str) or (..., popup) for testing without API.
    """
    from llm_agent.scout import score_actions_with_scout

    default_actions = ["W", "A", "S", "D", "space", "none", "look_left", "look_right", "look_left", "look_right"]
    actions = actions or default_actions
    if len(actions) != 10:
        actions = (actions * (10 // len(actions) + 1))[:10]

    # 1) Scout scores + avoid + objectives + popup + optional BEST/DURATION_MS (or mock for tests)
    objectives = ""
    popup = False
    best_override = None
    duration_override = None
    if mock_scout_result is not None:
        scout_scores, avoid_pen = mock_scout_result[0], mock_scout_result[1]
        objectives = mock_scout_result[2] if len(mock_scout_result) > 2 else ""
        popup = bool(mock_scout_result[3]) if len(mock_scout_result) > 3 else False
        scout_scores = (scout_scores + [0.5] * 10)[:10]
    elif use_scout and scout_api_key is not None:
        scout_scores, avoid_pen, raw, objectives, popup, best_override, duration_override = score_actions_with_scout(
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
        from PIL import Image as _Image
        import numpy as _np
        def _frame_to_tensor(frame, height=84, width=84):
            pil = _Image.fromarray(frame.astype(_np.uint8)).resize((width, height), _Image.Resampling.LANCZOS)
            x = _np.array(pil).astype(_np.float32) / 255.0
            return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            x = _frame_to_tensor(frame, height=84, width=84).to(device)
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
    best_action = best_override if best_override else actions[idx]
    print(f"[cem] best=#{idx} {best_action!r} combined_r={scores[idx]:.3f}", flush=True)
    return best_action, scores, scores[idx], objectives, popup, duration_override


# Base look speed: pixels per ms. Roblox camera scales linearly with drag distance.
# 220px / 400ms = 0.55 px/ms — a 400ms look rotates ~45 degrees.
LOOK_PX_PER_MS = 0.55
LOOK_DY = 0

KEY_MAP = {
    "w": "w", "a": "a", "s": "s", "d": "d",
    "space": "space",
}


def _normalize_key(part: str):
    p = part.strip().lower()
    return KEY_MAP.get(p, p) if p in KEY_MAP else None


def execute_action_ms(action: str, duration_ms: int = 5000) -> None:
    """Execute action for duration_ms (positive only). Supports combos: W+space, look_left+W.
    Look = right-click + mouse move. Keys can be held together."""
    duration_ms = max(1, int(duration_ms))
    if action is None:
        time.sleep(duration_ms / 1000.0)
        return
    action_str = action.strip()
    if not action_str or action_str.lower() == "none":
        time.sleep(duration_ms / 1000.0)
        return
    parts = [p.strip() for p in action_str.split("+") if p.strip()]
    look_part = None
    key_parts = []
    for p in parts:
        pl = p.lower()
        if pl in ("look_left", "look_right", "look_up", "look_down"):
            look_part = pl
        else:
            k = _normalize_key(p) or (pl if pl in KEY_MAP else None)
            if k:
                key_parts.append(k)
    import pyautogui
    from capture.screen import look_camera, get_roblox_region
    try:
        if look_part and key_parts:
            # look + movement simultaneously: hold keys in main thread, look via Quartz
            look_ms     = min(duration_ms, 3000)
            look_px     = int(LOOK_PX_PER_MS * look_ms)
            look_dx_val = (look_px if look_part == "look_right" else
                           -look_px if look_part == "look_left" else 0)
            import threading
            done = threading.Event()
            def _hold():
                for key in key_parts:
                    pyautogui.keyDown(key)
                done.wait(timeout=duration_ms / 1000.0 + 0.5)
                for key in key_parts:
                    pyautogui.keyUp(key)
            t = threading.Thread(target=_hold, daemon=True)
            t.start()
            look_camera(look_dx_val, look_ms)
            time.sleep(max(0, (duration_ms - look_ms) / 1000.0))
            done.set()
            t.join(timeout=0.5)
        elif look_part:
            look_ms     = min(duration_ms, 3000)
            look_px     = int(LOOK_PX_PER_MS * look_ms)
            look_dx_val = (look_px if look_part == "look_right" else
                           -look_px if look_part == "look_left" else 0)
            look_camera(look_dx_val, look_ms)
            time.sleep(max(0, (duration_ms - look_ms) / 1000.0))
        elif key_parts:
            for key in key_parts:
                pyautogui.keyDown(key)
            try:
                time.sleep(duration_ms / 1000.0)
            finally:
                for key in key_parts:
                    pyautogui.keyUp(key)
        else:
            time.sleep(duration_ms / 1000.0)
    except Exception:
        try:
            pyautogui.mouseUp(button="right")
            for key in ("w", "a", "s", "d", "space"):
                try:
                    pyautogui.keyUp(key)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(duration_ms / 1000.0)
