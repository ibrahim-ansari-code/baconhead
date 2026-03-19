#!/usr/bin/env python3
"""
Idle takeover — single-action-per-Claude-call loop.

When the user goes idle, the bot:
  1. Captures a fresh screenshot every action cycle.
  2. Computes situation signals: phase, water/void ahead, optical flow,
     edge distances, stuck count, disaster warning, recent actions.
  3. Passes all signals as rich text context to Claude alongside the screenshot.
  4. Claude returns ONE action. Execute it (~300-600ms). Repeat.
  5. Stops the moment the user presses any key.

Usage:
    python run_takeover.py --idle 3 --game nds
    python run_takeover.py --idle 3 --game nds --monitor bot_log.tsv
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
from reward.input_state import start_listener, is_active, get_recent_activity_summary
from llm_agent.cem import execute_action_ms
from llm_agent.scout import plan_one_action, _default_plan_10s


# ── window focus check ─────────────────────────────────────────────────────────

def _is_roblox_focused() -> bool:
    """Return True if Roblox is currently the frontmost application."""
    try:
        import subprocess
        out = subprocess.run(
            ["osascript", "-e", "name of (info for (path to frontmost application))"],
            capture_output=True, text=True, timeout=1,
        )
        return "roblox" in (out.stdout or "").lower()
    except Exception:
        return False


# ── context builder ────────────────────────────────────────────────────────────

def _build_context(
    frame: np.ndarray,
    prev_frame: Optional[np.ndarray],
    last_actions: List[str],
    stuck_count: int,
) -> Tuple[str, float]:
    """
    Compute all situation signals from the current frame and return:
      (context_text: str, flow_mean: float)

    context_text is injected into the Claude prompt alongside the screenshot.
    flow_mean is returned so the caller can track stuck state over time.
    """
    lines = []
    h, w = frame.shape[:2]

    # ── 1. Phase detection ────────────────────────────────────────────────────
    top_strip = frame[:max(1, int(h * 0.08)), :]
    red_pct   = float(
        ((top_strip[..., 0].astype(int) > 150) &
         (top_strip[..., 1].astype(int) < 80)  &
         (top_strip[..., 2].astype(int) < 80)).mean()
    )
    mean_bright = float(frame.mean())
    screen_std  = float(frame.std())

    if red_pct > 0.04:
        lines.append("PHASE: active_round")
        lines.append("DISASTER WARNING IS ACTIVE — survival is the immediate priority")
    elif mean_bright > 210 and screen_std < 25:
        lines.append("PHASE: dead (bright white respawn screen — wait for the game to reload)")
    elif mean_bright < 15:
        lines.append("PHASE: dead (black screen — loading or just died)")
    else:
        lines.append("PHASE: active_round")

    # ── 2. Water / void ahead ─────────────────────────────────────────────────
    # Sample a horizontal band in the center of the screen at eye-level (35-65% height)
    band = frame[int(h * 0.35):int(h * 0.65), int(w * 0.2):int(w * 0.8)]
    r, g, b = band[..., 0].astype(int), band[..., 1].astype(int), band[..., 2].astype(int)
    water_mask = (b - r > 50) & (b > 140)
    void_mask  = (r + g + b) < 55
    danger_pct = float((water_mask | void_mask).mean())

    if danger_pct > 0.30:
        lines.append(
            f"DANGER: {int(danger_pct*100)}% of the area directly ahead is water or void. "
            "Do NOT walk forward — turn first using look_left or look_right."
        )
    elif danger_pct > 0.10:
        lines.append(f"Caution: {int(danger_pct*100)}% water/void visible in forward area — be careful.")
    else:
        lines.append(f"Forward area: {int(danger_pct*100)}% water/void (path looks clear).")

    # ── 3. Optical flow / motion ──────────────────────────────────────────────
    flow_mean = 0.0
    try:
        import cv2
        from PIL import Image as _Image
        sz = (224, 224)
        frame_sm = np.array(_Image.fromarray(frame.astype(np.uint8)).resize(sz, _Image.Resampling.LANCZOS))
        if prev_frame is not None:
            prev_sm = np.array(_Image.fromarray(prev_frame.astype(np.uint8)).resize(sz, _Image.Resampling.LANCZOS))
            g1   = cv2.cvtColor(prev_sm, cv2.COLOR_RGB2GRAY)
            g2   = cv2.cvtColor(frame_sm, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag  = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_mean = float(mag.mean())
    except Exception:
        pass

    if flow_mean < 0.3:
        motion_str = "stationary — character is not moving"
    elif flow_mean < 1.5:
        motion_str = f"slow (flow={flow_mean:.1f})"
    else:
        motion_str = f"moving well (flow={flow_mean:.1f})"
    lines.append(f"Motion: {motion_str}")

    # ── 4. Stuck counter ──────────────────────────────────────────────────────
    if stuck_count >= 3:
        lines.append(
            f"STUCK: character has not moved for {stuck_count} consecutive cycles. "
            "You MUST change direction — use look_right or look_left, then W."
        )

    # ── 5. Platform edge distances ────────────────────────────────────────────
    try:
        from reward.collect_episodes import compute_edge_distances
        from PIL import Image as _Image
        frame_sm2 = np.array(_Image.fromarray(frame.astype(np.uint8)).resize((224, 224), _Image.Resampling.LANCZOS))
        edge_dists = compute_edge_distances(frame_sm2)
        labels     = ["top", "right", "bottom", "left"]
        parts      = []
        for label, dist in zip(labels, edge_dists):
            pct = int(dist * 100)
            tag = " (VERY CLOSE)" if pct < 10 else " (close)" if pct < 25 else ""
            parts.append(f"{label}={pct}%{tag}")
        lines.append("Platform edges: " + ", ".join(parts))
        dangers = [l for l, d in zip(labels, edge_dists) if d < 0.12]
        if dangers:
            lines.append(f"EDGE DANGER: very close to {'/'.join(dangers)} edge — do not move that direction")
    except Exception:
        pass

    # ── 6. Recent actions ─────────────────────────────────────────────────────
    if last_actions:
        lines.append(f"Last {min(6, len(last_actions))} actions: {', '.join(list(last_actions)[-6:])}")

    return "\n".join(lines), flow_mean


# ── misc helpers ───────────────────────────────────────────────────────────────

def _default_action() -> Tuple[str, int]:
    import random
    return random.choice([("W", 400), ("look_right", 400), ("W", 500)])


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Idle takeover — single-action Claude loop")
    parser.add_argument("--idle",        type=float, default=3.0,  help="Idle seconds before takeover")
    parser.add_argument("--full-screen", action="store_true",       help="Capture full monitor")
    parser.add_argument("--region",      type=str,   default=None,  help="left,top,width,height")
    parser.add_argument("--no-scout",    action="store_true",       help="Skip Claude (use random fallback)")
    parser.add_argument("--monitor",     type=str,   default=None,  help="Log decisions to this TSV file")
    parser.add_argument("--max-step-ms", type=int,   default=800,   help="Cap any single action to this many ms")
    parser.add_argument("--game",        type=str,   default=None,  help="Game context: nds | obby | brookhaven")
    args = parser.parse_args()

    # ── region ────────────────────────────────────────────────────────────────
    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) != 4:
            raise ValueError("--region must be left,top,width,height")
        region = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
        print("Using region from --region:", region, flush=True)
    elif args.full_screen:
        with mss.mss() as m:
            mon = m.monitors[0]
        region = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
        print("Using full primary monitor:", region, flush=True)
    else:
        for attempt in range(30):
            region = get_roblox_region()
            if region is not None:
                break
            if attempt == 0:
                print("Waiting for Roblox window... (Ctrl+C to abort)", flush=True)
            time.sleep(3)
        if region is None:
            print("Error: Roblox window not found. Use --region or --full-screen.", file=sys.stderr)
            sys.exit(1)
        print("Using Roblox window:", region, flush=True)

    # ── api key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not args.no_scout and not api_key:
        print("Warning: ANTHROPIC_API_KEY not set; using random fallback actions.", file=sys.stderr)

    # ── setup ─────────────────────────────────────────────────────────────────
    start_listener()
    sct          = mss.mss()
    idle_sec     = args.idle
    max_step_ms  = max(100, args.max_step_ms)
    monitor_path = args.monitor
    game_name    = (args.game or "").strip().lower()
    use_scout    = not args.no_scout and bool(api_key)

    BOT_COOLDOWN  = 3.0   # seconds to ignore user-active after bot action (pyautogui trips pynput)
    last_bot_time = [0.0]
    bot_active    = [False]

    last_actions: deque = deque(maxlen=20)
    prev_frame:   Optional[np.ndarray] = None

    # Rolling flow_mean window for stuck detection
    flow_window:  deque = deque(maxlen=5)
    stuck_count   = [0]

    def log(msg: str):
        print(msg, flush=True)

    def monitor_log(action: str, ms: int):
        if not monitor_path:
            return
        try:
            with open(monitor_path, "a") as f:
                f.write(f"{time.time():.3f}\t{action}\t{ms}\n")
        except Exception:
            pass

    log(f"Watching. Takeover after {idle_sec}s idle. Scout={'on' if use_scout else 'off (random)'}.")
    log("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.15)   # ~6 Hz polling — responsive but not CPU-intensive
            frame = capture_region(region=region, sct=sct)
            now   = time.perf_counter()

            # ── user active check ─────────────────────────────────────────────
            in_cooldown = (now - last_bot_time[0]) < BOT_COOLDOWN
            if not in_cooldown and is_active(idle_sec):
                if bot_active[0]:
                    log("[takeover] User active — stopping bot.")
                    bot_active[0]   = False
                    stuck_count[0]  = 0
                    flow_window.clear()
                prev_frame = frame
                continue

            if not bot_active[0]:
                bot_active[0] = True
                log("[takeover] Idle — starting takeover.")

            # ── build situation context ───────────────────────────────────────
            context_text, flow_mean = _build_context(
                frame, prev_frame, list(last_actions), stuck_count[0]
            )
            prev_frame = frame

            # ── update stuck counter ──────────────────────────────────────────
            flow_window.append(flow_mean)
            if len(flow_window) >= 4 and (sum(flow_window) / len(flow_window)) < 0.35:
                stuck_count[0] += 1
            else:
                stuck_count[0] = 0

            # ── get one action from Claude ────────────────────────────────────
            if use_scout:
                action, action_ms = plan_one_action(
                    frame,
                    context_text=context_text,
                    api_key=api_key,
                    game=game_name,
                )
            else:
                action, action_ms = _default_action()

            # Look actions have their own internal cap (degrees → ms via _degrees_to_ms),
            # so don't apply max_step_ms to them — it would truncate large turns.
            is_look = "look_" in action.lower()
            if is_look:
                action_ms = max(100, action_ms)
            else:
                action_ms = min(max(100, action_ms), max_step_ms)

            # ── ensure Roblox is focused before every action ──────────────────
            if not _is_roblox_focused():
                log("[takeover] Roblox not focused — focusing...")
                focus_roblox_and_click()
                time.sleep(0.2)

            # ── execute ───────────────────────────────────────────────────────
            first_context_line = context_text.splitlines()[0] if context_text else ""
            log(f"[takeover] {action!r} {action_ms}ms | {first_context_line}")
            monitor_log(action, action_ms)
            execute_action_ms(action, duration_ms=action_ms)
            last_bot_time[0] = time.perf_counter()
            last_actions.append(action)

    except KeyboardInterrupt:
        pass
    finally:
        sct.close()
    print("Done.")


if __name__ == "__main__":
    main()
