#!/usr/bin/env python3
"""
Idle takeover — goal-oriented planning loop with persistent memory.

Watches for user idle, then uses Claude vision + GameSense to play the game.
No hardcoded game knowledge — works on any Roblox game.

Usage:
    python run_takeover.py --idle 7 --monitor bot_log.tsv
    python run_takeover.py --idle 10 --model game_sense.pt
"""

import argparse
import os
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

import mss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from capture.screen import capture_region, get_roblox_region, focus_roblox_and_click
from reward.input_state import start_listener, is_active, set_bot_pressing
from llm_agent.actions import execute_action_ms
from llm_agent.scout import plan_with_goal, verify_goal, survey_pick_best
from llm_agent.physics import degrees_to_ms, MAX_MOVEMENT_MS
from vision.game_sense import heuristic_state


# ── Frame comparison ──────────────────────────────────────────────────────────


def _frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity [0, 1]. 1.0 = identical. Downscaled grayscale MAD."""
    from PIL import Image as _Image

    sz = (112, 112)
    a_sm = np.array(
        _Image.fromarray(a.astype(np.uint8)).resize(sz, _Image.Resampling.LANCZOS)
    ).astype(np.float32)
    b_sm = np.array(
        _Image.fromarray(b.astype(np.uint8)).resize(sz, _Image.Resampling.LANCZOS)
    ).astype(np.float32)
    diff = np.abs(a_sm.mean(axis=2) - b_sm.mean(axis=2)).mean()
    return max(0.0, 1.0 - diff / 60.0)


# ── Survey ────────────────────────────────────────────────────────────────────


def run_survey(region, sct, api_key, log_fn) -> int:
    """Capture 4 compass frames, pick the best direction via Claude."""
    log_fn("[survey] Scanning 4 directions...")
    turn_ms = degrees_to_ms(90)
    frames = []

    for i in range(4):
        frame = capture_region(region=region, sct=sct)
        frames.append(frame)
        log_fn(f"[survey] Captured angle {i * 90}°")
        if i < 3:
            set_bot_pressing(True)
            execute_action_ms("look_right", turn_ms)
            set_bot_pressing(False)
            time.sleep(0.15)

    # Complete 360° back to original facing
    set_bot_pressing(True)
    execute_action_ms("look_right", turn_ms)
    set_bot_pressing(False)
    time.sleep(0.15)

    if not api_key:
        return -1

    best = survey_pick_best(frames, api_key=api_key)
    log_fn(f"[survey] Best direction: angle {best * 90}°")

    if best > 0:
        set_bot_pressing(True)
        execute_action_ms("look_right", degrees_to_ms(best * 90))
        set_bot_pressing(False)
        time.sleep(0.15)

    return best


# ── Persistent memory ─────────────────────────────────────────────────────────


class BotMemory:
    """Survives user-takeover events. Tracks what the bot has tried."""

    def __init__(self):
        self.action_history: deque = deque(maxlen=50)
        self.goal_history: deque = deque(maxlen=10)
        self.failure_reasons: deque = deque(maxlen=10)
        self.consecutive_failures = 0
        self.total_plans = 0
        self.has_surveyed = False
        self.last_survey_direction = -1

    def record_action(self, action: str, ms: int):
        self.action_history.append((action, ms))

    def record_goal(self, goal: str, status: str, reason: str = ""):
        self.goal_history.append((goal, status, reason))
        if status == "failed":
            self.consecutive_failures += 1
            if reason:
                self.failure_reasons.append(reason)
        elif status == "achieved":
            self.consecutive_failures = 0
        self.total_plans += 1

    def needs_survey(self) -> bool:
        if not self.has_surveyed:
            return True
        return self.consecutive_failures >= 3

    def mark_surveyed(self, direction: int):
        self.has_surveyed = True
        self.last_survey_direction = direction
        self.consecutive_failures = 0

    def get_history_summary(self) -> str:
        parts = []
        if self.action_history:
            recent = list(self.action_history)[-8:]
            action_strs = [f"{a}({ms}ms)" for a, ms in recent]
            parts.append(f"Recent actions: {', '.join(action_strs)}")

        if self.goal_history:
            recent_goals = list(self.goal_history)[-3:]
            for goal, status, reason in recent_goals:
                entry = f"  - '{goal}' → {status}"
                if reason:
                    entry += f" ({reason})"
                parts.append(entry)
            if parts:
                parts.insert(len(parts) - len(recent_goals), "Recent goals:")

        if self.failure_reasons:
            recent_failures = list(self.failure_reasons)[-3:]
            parts.append(
                f"Failed approaches to AVOID: {'; '.join(recent_failures)}"
            )

        return "\n".join(parts) if parts else ""

    def get_last_failure(self) -> Optional[str]:
        if self.failure_reasons:
            return self.failure_reasons[-1]
        return None


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Idle takeover — goal-oriented Claude loop"
    )
    parser.add_argument("--idle", type=float, default=3.0)
    parser.add_argument("--full-screen", action="store_true")
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--no-scout", action="store_true")
    parser.add_argument("--monitor", type=str, default=None)
    parser.add_argument("--max-step-ms", type=int, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained GameSense model (game_sense.pt)",
    )
    args = parser.parse_args()

    # ── Region ────────────────────────────────────────────────────────────────
    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) != 4:
            raise ValueError("--region must be left,top,width,height")
        region = {
            "left": parts[0],
            "top": parts[1],
            "width": parts[2],
            "height": parts[3],
        }
    elif args.full_screen:
        with mss.mss() as m:
            mon = m.monitors[0]
        region = {
            "left": mon["left"],
            "top": mon["top"],
            "width": mon["width"],
            "height": mon["height"],
        }
    else:
        for attempt in range(30):
            region = get_roblox_region()
            if region is not None:
                break
            if attempt == 0:
                print("Waiting for Roblox window...", flush=True)
            time.sleep(3)
        if region is None:
            print("Error: Roblox window not found.", file=sys.stderr)
            sys.exit(1)
    print(f"Region: {region}", flush=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not args.no_scout and not api_key:
        print("Warning: ANTHROPIC_API_KEY not set.", file=sys.stderr)

    # ── Load GameSense model if available ─────────────────────────────────────
    game_sense = None
    model_path = args.model
    if model_path and os.path.isfile(model_path):
        try:
            from vision.game_sense import load_game_sense

            game_sense = load_game_sense(model_path)
            print(f"GameSense model loaded: {model_path}", flush=True)
        except Exception as e:
            print(f"GameSense load failed: {e} — using heuristics", flush=True)
    elif model_path:
        print(f"GameSense model not found: {model_path} — using heuristics", flush=True)

    def detect_state(frame: np.ndarray) -> tuple:
        """Returns (state_label, confidence) via model or heuristic."""
        if game_sense is not None:
            return game_sense.predict(frame)
        return heuristic_state(frame)

    start_listener()
    sct = mss.mss()
    idle_sec = args.idle
    max_step_ms = args.max_step_ms or MAX_MOVEMENT_MS
    monitor_path = args.monitor
    use_scout = not args.no_scout and bool(api_key)

    bot_active = [False]
    focused_this_session = [False]
    last_bot_event_time = [0.0]

    # Plan state (cleared on user takeover)
    current_goal: Optional[str] = None
    action_queue: deque = deque()
    plan_frame: Optional[np.ndarray] = None
    no_progress_streak = [0]

    NO_PROGRESS_THRESHOLD = 0.88
    NO_PROGRESS_ABORT = 2

    memory = BotMemory()

    def log(msg: str):
        print(msg, flush=True)

    def monitor_log(action: str, ms: int):
        if not monitor_path:
            return
        try:
            with open(monitor_path, "a") as f:
                f.write(f"{time.time():.3f}\tscout\t{action}\t{ms}\n")
        except Exception:
            pass

    def _clear_plan(
        reason: str = "", failed: bool = False, failure_reason: str = ""
    ):
        nonlocal current_goal, plan_frame
        action_queue.clear()
        if failed and current_goal:
            memory.record_goal(current_goal, "failed", failure_reason or reason)
        current_goal = None
        plan_frame = None
        no_progress_streak[0] = 0
        if reason:
            log(f"[takeover] Plan cleared: {reason}")

    log(
        f"Watching. Takeover after {idle_sec}s idle. "
        f"Scout={'on' if use_scout else 'off'}. "
        f"GameSense={'model' if game_sense else 'heuristic'}."
    )
    log("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.15)
            frame = capture_region(region=region, sct=sct)

            # ── User active? ──────────────────────────────────────────────────
            bot_cooldown = (
                time.perf_counter() - last_bot_event_time[0]
            ) < (idle_sec + 1.0)
            if not bot_cooldown and is_active(idle_sec):
                if bot_active[0]:
                    log("[takeover] User active — pausing bot.")
                    bot_active[0] = False
                    focused_this_session[0] = False
                    _clear_plan("user took over")
                continue

            # ── Bot activates ─────────────────────────────────────────────────
            if not bot_active[0]:
                bot_active[0] = True
                log("[takeover] Idle — bot active.")

            if not focused_this_session[0]:
                set_bot_pressing(True)
                focus_roblox_and_click()
                set_bot_pressing(False)
                focused_this_session[0] = True
                last_bot_event_time[0] = time.perf_counter()
                time.sleep(0.3)
                continue

            # ── GameSense state detection ─────────────────────────────────────
            state, confidence = detect_state(frame)

            if state == "dead":
                log(f"[takeover] State: DEAD ({confidence:.2f}) — waiting 3s")
                _clear_plan("dead")
                memory.has_surveyed = False
                set_bot_pressing(True)
                execute_action_ms("none", 3000)
                set_bot_pressing(False)
                last_bot_event_time[0] = time.perf_counter()
                continue

            if state == "danger":
                log(
                    f"[takeover] State: DANGER ({confidence:.2f}) — turning 180°"
                )
                _clear_plan(
                    "danger detected",
                    failed=True,
                    failure_reason="walked toward danger — must go opposite direction",
                )
                set_bot_pressing(True)
                execute_action_ms("look_right", degrees_to_ms(180))
                set_bot_pressing(False)
                last_bot_event_time[0] = time.perf_counter()
                continue

            if state == "menu":
                log(f"[takeover] State: MENU ({confidence:.2f}) — waiting 2s")
                set_bot_pressing(True)
                execute_action_ms("none", 2000)
                set_bot_pressing(False)
                last_bot_event_time[0] = time.perf_counter()
                continue

            # ── Need a new plan? ──────────────────────────────────────────────
            if not action_queue:
                # Verify previous goal
                if current_goal and plan_frame is not None:
                    sim = _frame_similarity(plan_frame, frame)
                    if sim > NO_PROGRESS_THRESHOLD:
                        log(
                            f"[takeover] Frame diff {sim:.2f} — stuck, plan failed"
                        )
                        _clear_plan(
                            "no visual change",
                            failed=True,
                            failure_reason="character stuck — did not move",
                        )
                    elif use_scout:
                        status, fail_reason = verify_goal(
                            plan_frame, frame, current_goal, api_key
                        )
                        log(f"[takeover] Verify '{current_goal}': {status}")
                        if status == "achieved":
                            memory.record_goal(current_goal, "achieved")
                            _clear_plan("goal achieved")
                        elif status == "failed":
                            _clear_plan(
                                "goal failed",
                                failed=True,
                                failure_reason=fail_reason,
                            )
                        else:
                            memory.record_goal(current_goal, "in_progress")
                            _clear_plan("extending plan")
                    else:
                        _clear_plan("no scout")

                # ── Survey (only when needed) ─────────────────────────────────
                if memory.needs_survey() and use_scout:
                    run_survey(region, sct, api_key, log)
                    last_bot_event_time[0] = time.perf_counter()
                    memory.mark_surveyed(0)
                    frame = capture_region(region=region, sct=sct)

                # ── Build context with memory + GameSense ─────────────────────
                context_lines = []
                h = frame.shape[0]
                top_strip = frame[: max(1, int(h * 0.08)), :]
                red_pct = float(
                    (
                        (top_strip[..., 0].astype(int) > 150)
                        & (top_strip[..., 1].astype(int) < 80)
                        & (top_strip[..., 2].astype(int) < 80)
                    ).mean()
                )
                if red_pct > 0.04:
                    context_lines.append(
                        "DISASTER WARNING visible — act urgently!"
                    )

                # Include GameSense state in context for Claude
                context_lines.append(f"GameSense state: {state} ({confidence:.2f})")

                history = memory.get_history_summary()
                if history:
                    context_lines.append(history)

                context_text = (
                    "\n".join(context_lines)
                    if context_lines
                    else "First action — no prior context."
                )

                # ── Plan ──────────────────────────────────────────────────────
                if use_scout:
                    goal, steps = plan_with_goal(
                        frame,
                        context_text=context_text,
                        api_key=api_key,
                        last_goal=current_goal,
                        last_failure=memory.get_last_failure(),
                    )
                else:
                    import random

                    goal = "walk forward and explore"
                    steps = random.choice([
                        [("W", 500), ("W", 400)],
                        [("look_right", degrees_to_ms(90)), ("W", 500)],
                        [("look_left", degrees_to_ms(90)), ("W", 500)],
                    ])

                current_goal = goal
                plan_frame = frame
                no_progress_streak[0] = 0
                action_queue.clear()
                for step in steps:
                    action_queue.append(step)
                log(f"[plan] '{goal}' → {steps}")

            # ── Execute next step ─────────────────────────────────────────────
            if not action_queue:
                continue

            action, action_ms = action_queue.popleft()
            is_look = "look_" in action.lower()
            action_ms = max(
                100, min(action_ms, max_step_ms if not is_look else 2000)
            )

            pre_frame = capture_region(region=region, sct=sct)

            log(f"  → {action} {action_ms}ms (steps_left={len(action_queue)})")
            monitor_log(action, action_ms)
            set_bot_pressing(True)
            execute_action_ms(action, duration_ms=action_ms)
            set_bot_pressing(False)
            last_bot_event_time[0] = time.perf_counter()
            memory.record_action(action, action_ms)

            time.sleep(0.05)
            post_frame = capture_region(region=region, sct=sct)

            # Per-step danger check
            if not is_look and action.lower() != "none":
                post_state, post_conf = detect_state(post_frame)
                if post_state == "danger":
                    log("[takeover] Post-step DANGER — aborting, turning 180°")
                    _clear_plan(
                        "walked to danger",
                        failed=True,
                        failure_reason="walked toward danger — must go opposite",
                    )
                    set_bot_pressing(True)
                    execute_action_ms("look_right", degrees_to_ms(180))
                    set_bot_pressing(False)
                    last_bot_event_time[0] = time.perf_counter()
                    continue

            # Per-step stuck check (movement actions only)
            if not is_look and action.lower() != "none":
                sim = _frame_similarity(pre_frame, post_frame)
                if sim > NO_PROGRESS_THRESHOLD:
                    no_progress_streak[0] += 1
                    log(
                        f"  ⚠ No progress (sim={sim:.2f}, "
                        f"streak={no_progress_streak[0]})"
                    )
                    if no_progress_streak[0] >= NO_PROGRESS_ABORT:
                        log("[takeover] STUCK: aborting plan, turning away")
                        _clear_plan(
                            "stuck",
                            failed=True,
                            failure_reason="walked into wall — no movement for 2 steps",
                        )
                        import random

                        deg = random.choice([90, -90, 135, -135])
                        turn = "look_right" if deg > 0 else "look_left"
                        set_bot_pressing(True)
                        execute_action_ms(turn, degrees_to_ms(abs(deg)))
                        set_bot_pressing(False)
                        last_bot_event_time[0] = time.perf_counter()
                        memory.record_action(turn, degrees_to_ms(abs(deg)))
                        continue
                else:
                    no_progress_streak[0] = 0

    except KeyboardInterrupt:
        pass
    finally:
        sct.close()
    print("Done.")


if __name__ == "__main__":
    main()
