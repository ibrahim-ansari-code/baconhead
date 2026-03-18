"""
Episode collector for the outcome model.

Records 10-second windows of real gameplay as:
  - frame:          screenshot at t=0 (224x224 RGB)
  - flow_mag:       mean optical-flow magnitude per time bucket (9 buckets)
  - flow_dir:       mean optical-flow direction per time bucket (9 buckets, radians)
  - edge_distances: pixel distance from frame centre to nearest Canny edge [up,down,left,right]
  - key_events:     all key press/release events in the window (key_idx, t_down_ms, t_up_ms)
  - survived:       1 = no death detected, 0 = death/fall detected

Death detection: bright-white flash (mean pixel > 200) OR large frame diff (>50 MAD).

Usage:
    python -m reward.collect_episodes --seconds 120 --out-dir episode_data
"""

import argparse
import json
import os
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import mss
import numpy as np

from capture.screen import get_roblox_region, capture_region
from reward.input_state import start_listener

# ── key vocabulary ─────────────────────────────────────────────────────────────
KEY_VOCAB = ["w", "a", "s", "d", "space", "shift", "q", "e"]
KEY_TO_IDX = {k: i for i, k in enumerate(KEY_VOCAB)}
MAX_KEY_EVENTS = 64          # max events stored per episode
EPISODE_SECS   = 10.0        # window length
N_BUCKETS      = 9           # optical-flow time buckets
FLOW_FPS       = 5.0         # frames/s for optical flow capture
FRAME_SIZE     = 224


# ── per-episode key-event recorder ─────────────────────────────────────────────

class _EpisodeKeyRecorder:
    """Collects raw key events during a window, thread-safe via the global pynput listener."""

    def __init__(self):
        self._events: List[Tuple[int, float, float]] = []  # (key_idx, t_down_ms, t_up_ms)
        self._in_progress: dict = {}   # key_idx -> t_down (perf_counter)
        self._lock = threading.Lock()
        self._t0: float = time.perf_counter()

    def reset(self, t0: float):
        with self._lock:
            self._events = []
            self._in_progress = {}
            self._t0 = t0

    def on_press(self, key_name: str):
        idx = KEY_TO_IDX.get(key_name.lower())
        if idx is None:
            return
        now = time.perf_counter()
        with self._lock:
            if idx not in self._in_progress:
                self._in_progress[idx] = now

    def on_release(self, key_name: str):
        idx = KEY_TO_IDX.get(key_name.lower())
        if idx is None:
            return
        now = time.perf_counter()
        with self._lock:
            t_down = self._in_progress.pop(idx, now)
            t_down_ms = (t_down - self._t0) * 1000.0
            t_up_ms   = (now   - self._t0) * 1000.0
            if len(self._events) < MAX_KEY_EVENTS:
                self._events.append((idx, t_down_ms, t_up_ms))

    def get_events(self) -> List[Tuple[int, float, float]]:
        """Return a copy of collected events; close any still-held keys at window end."""
        now = time.perf_counter()
        with self._lock:
            evts = list(self._events)
            for idx, t_down in self._in_progress.items():
                t_down_ms = (t_down - self._t0) * 1000.0
                t_up_ms   = (now    - self._t0) * 1000.0
                evts.append((idx, t_down_ms, t_up_ms))
            return evts


# ── optical flow helpers ────────────────────────────────────────────────────────

def _to_gray(frame: np.ndarray) -> np.ndarray:
    """RGB (H,W,3) → uint8 gray (H,W)."""
    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def _optical_flow_mag_dir(prev_gray: np.ndarray, next_gray: np.ndarray) -> Tuple[float, float]:
    """Dense optical flow between two gray frames; returns (mean_magnitude, mean_direction_rad)."""
    try:
        import cv2
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(mag.mean()), float(ang.mean())
    except Exception:
        return 0.0, 0.0


def compute_flow_features(
    frames_sequence: List[np.ndarray],
    n_buckets: int = N_BUCKETS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a sequence of RGB frames captured during the episode, compute per-bucket
    mean optical-flow magnitude and direction.
    Returns flow_mag (n_buckets,) and flow_dir (n_buckets,) float32 arrays.
    """
    if len(frames_sequence) < 2:
        return np.zeros(n_buckets, np.float32), np.zeros(n_buckets, np.float32)

    grays = [_to_gray(f) for f in frames_sequence]
    # Compute pairwise flow
    pair_mags, pair_dirs = [], []
    for i in range(len(grays) - 1):
        m, d = _optical_flow_mag_dir(grays[i], grays[i + 1])
        pair_mags.append(m)
        pair_dirs.append(d)

    # Distribute pairs into buckets
    n_pairs = len(pair_mags)
    flow_mag = np.zeros(n_buckets, np.float32)
    flow_dir = np.zeros(n_buckets, np.float32)
    for bucket in range(n_buckets):
        lo = int(bucket * n_pairs / n_buckets)
        hi = int((bucket + 1) * n_pairs / n_buckets)
        if hi <= lo:
            hi = lo + 1
        hi = min(hi, n_pairs)
        if lo < hi:
            flow_mag[bucket] = float(np.mean(pair_mags[lo:hi]))
            flow_dir[bucket] = float(np.mean(pair_dirs[lo:hi]))
    return flow_mag, flow_dir


# ── edge distance helpers ───────────────────────────────────────────────────────

def compute_edge_distances(frame: np.ndarray) -> np.ndarray:
    """
    Run Canny edge detection on the frame; from the frame center compute pixel distance
    to the nearest strong edge in each of [up, down, left, right].
    Returns float32 array of shape (4,), values in [0, 1] normalised by frame dimension.
    """
    try:
        import cv2
        gray = _to_gray(frame)
        h, w = gray.shape
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)  # (H, W) bool-ish
        cy, cx = h // 2, w // 2

        def _dist_to_edge(arr_1d: np.ndarray) -> float:
            idxs = np.where(arr_1d > 0)[0]
            if len(idxs) == 0:
                return 1.0
            return float(np.min(np.abs(idxs - len(arr_1d) // 2))) / len(arr_1d)

        up_col    = edges[:cy, cx][::-1]   # from centre upward
        down_col  = edges[cy:, cx]          # from centre downward
        left_row  = edges[cy, :cx][::-1]    # from centre leftward
        right_row = edges[cy, cx:]          # from centre rightward

        return np.array([
            _dist_to_edge(up_col),
            _dist_to_edge(down_col),
            _dist_to_edge(left_row),
            _dist_to_edge(right_row),
        ], dtype=np.float32)
    except Exception:
        return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)


# ── death detection ─────────────────────────────────────────────────────────────

def _is_death_frame(frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> bool:
    """
    Heuristic death detection:
    1. Bright-white flash: mean pixel > 200 across all channels.
    2. Large sudden frame diff compared to previous: mean absolute diff > 50.
    """
    mean_bright = float(frame.mean())
    if mean_bright > 200:
        return True
    if prev_frame is not None:
        diff = float(np.mean(np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32))))
        if diff > 50:
            return True
    return False


# ── episode key-event array packing ────────────────────────────────────────────

def pack_key_events(events: List[Tuple[int, float, float]]) -> np.ndarray:
    """
    Pack list of (key_idx, t_down_ms, t_up_ms) into fixed (MAX_KEY_EVENTS, 3) float32 array.
    Unused slots are zeros.
    """
    arr = np.zeros((MAX_KEY_EVENTS, 3), dtype=np.float32)
    for i, (kidx, tdown, tup) in enumerate(events[:MAX_KEY_EVENTS]):
        arr[i, 0] = kidx
        arr[i, 1] = tdown
        arr[i, 2] = tup
    return arr


# ── main collection logic ───────────────────────────────────────────────────────

def run_collect(
    out_dir: str,
    total_seconds: float = 120.0,
    episode_secs: float = EPISODE_SECS,
    region: Optional[dict] = None,
    stop_event: Optional[threading.Event] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    # ── pynput listener wrapping the episode recorder ──
    recorder = _EpisodeKeyRecorder()

    def _hook_press(key):
        try:
            k = key.char if hasattr(key, "char") and key.char else key.name
        except Exception:
            k = getattr(key, "name", str(key))
        recorder.on_press(k or "")

    def _hook_release(key):
        try:
            k = key.char if hasattr(key, "char") and key.char else key.name
        except Exception:
            k = getattr(key, "name", str(key))
        recorder.on_release(k or "")

    from pynput import keyboard as _kb
    listener = _kb.Listener(on_press=_hook_press, on_release=_hook_release)
    listener.daemon = True
    listener.start()

    sct = mss.mss()

    # ── storage lists ──
    all_frames:         List[np.ndarray] = []
    all_flow_mag:       List[np.ndarray] = []
    all_flow_dir:       List[np.ndarray] = []
    all_edge_distances: List[np.ndarray] = []
    all_key_events:     List[np.ndarray] = []
    all_survived:       List[int]        = []

    flow_interval = 1.0 / FLOW_FPS
    t_start_global = time.perf_counter()
    ep_count = 0

    print(f"[collect_episodes] Starting. Play Roblox for {total_seconds:.0f}s. Ctrl+C to stop early.", flush=True)

    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            if time.perf_counter() - t_start_global >= total_seconds:
                break

            # ── start of episode ──────────────────────────────────────────────
            t_ep_start = time.perf_counter()
            recorder.reset(t_ep_start)

            initial_frame_full = capture_region(region=region, sct=sct)
            from PIL import Image as _Image
            initial_frame = np.array(
                _Image.fromarray(initial_frame_full.astype(np.uint8))
                .resize((FRAME_SIZE, FRAME_SIZE), _Image.Resampling.LANCZOS)
            )
            edge_dists = compute_edge_distances(initial_frame)

            # ── collect frames for optical flow during episode ──────────────
            flow_frames: List[np.ndarray] = []
            death_detected = False
            prev_flow_frame = None
            t_next_flow = t_ep_start
            t_ep_end = t_ep_start + episode_secs

            while time.perf_counter() < t_ep_end:
                if stop_event and stop_event.is_set():
                    break
                now = time.perf_counter()
                if now >= t_next_flow:
                    raw = capture_region(region=region, sct=sct)
                    small = np.array(
                        _Image.fromarray(raw.astype(np.uint8))
                        .resize((FRAME_SIZE, FRAME_SIZE), _Image.Resampling.LANCZOS)
                    )
                    if not death_detected and _is_death_frame(small, prev_flow_frame):
                        death_detected = True
                    flow_frames.append(small)
                    prev_flow_frame = small
                    t_next_flow += flow_interval
                else:
                    time.sleep(min(0.05, t_next_flow - now))

            # ── finalise episode ──────────────────────────────────────────────
            key_events = recorder.get_events()
            flow_mag, flow_dir = compute_flow_features(flow_frames, N_BUCKETS)
            survived = 0 if death_detected else 1

            all_frames.append(initial_frame)
            all_flow_mag.append(flow_mag)
            all_flow_dir.append(flow_dir)
            all_edge_distances.append(edge_dists)
            all_key_events.append(pack_key_events(key_events))
            all_survived.append(survived)
            ep_count += 1

            n_died = sum(1 for s in all_survived if s == 0)
            print(
                f"[collect_episodes] ep {ep_count}: survived={survived} "
                f"keys={len(key_events)} flow_mean={flow_mag.mean():.2f} "
                f"edges={edge_dists.round(2)} | total={ep_count} died={n_died}",
                flush=True,
            )

    except KeyboardInterrupt:
        pass
    finally:
        sct.close()
        listener.stop()

    if not all_frames:
        print("[collect_episodes] No episodes collected.", flush=True)
        return

    # ── save ──────────────────────────────────────────────────────────────────
    npz_path = os.path.join(out_dir, "data.npz")
    # Load and append to existing data if present
    existing: dict = {}
    if os.path.isfile(npz_path):
        try:
            existing = dict(np.load(npz_path, allow_pickle=False))
            print(f"[collect_episodes] Appending to existing {len(existing.get('survived', []))} episodes.", flush=True)
        except Exception:
            existing = {}

    def _cat(key, new_arr):
        if key in existing:
            return np.concatenate([existing[key], new_arr], axis=0)
        return new_arr

    frames_arr      = _cat("frames",         np.stack(all_frames, 0))
    flow_mag_arr    = _cat("flow_mag",        np.stack(all_flow_mag, 0))
    flow_dir_arr    = _cat("flow_dir",        np.stack(all_flow_dir, 0))
    edge_dist_arr   = _cat("edge_distances",  np.stack(all_edge_distances, 0))
    key_events_arr  = _cat("key_events",      np.stack(all_key_events, 0))
    survived_arr    = _cat("survived",        np.array(all_survived, dtype=np.int8))

    np.savez(
        npz_path,
        frames=frames_arr,
        flow_mag=flow_mag_arr,
        flow_dir=flow_dir_arr,
        edge_distances=edge_dist_arr,
        key_events=key_events_arr,
        survived=survived_arr,
    )
    total_eps = len(survived_arr)
    n_survived = int(survived_arr.sum())
    config = {
        "total_episodes": total_eps,
        "n_survived": n_survived,
        "n_fell": total_eps - n_survived,
        "episode_secs": episode_secs,
        "n_buckets": N_BUCKETS,
        "flow_fps": FLOW_FPS,
        "frame_size": FRAME_SIZE,
        "max_key_events": MAX_KEY_EVENTS,
        "key_vocab": KEY_VOCAB,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(
        f"[collect_episodes] Saved {total_eps} episodes "
        f"({n_survived} survived, {total_eps - n_survived} fell) to {out_dir}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Collect episode data for the outcome model")
    parser.add_argument("--out-dir",  type=str,   default="episode_data")
    parser.add_argument("--seconds",  type=float, default=120.0, help="Total recording time")
    parser.add_argument("--episode-secs", type=float, default=EPISODE_SECS, help="Length of each episode window")
    parser.add_argument("--region",   type=str,   default=None, help="left,top,width,height")
    parser.add_argument("--full-screen", action="store_true")
    args = parser.parse_args()

    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        region = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    elif args.full_screen:
        import mss as _mss
        with _mss.mss() as m:
            mon = m.monitors[0]
        region = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
    else:
        region = get_roblox_region()
        if region:
            print(f"Using Roblox window: {region}", flush=True)
        else:
            print("Roblox window not found; using primary monitor.", flush=True)

    run_collect(out_dir=args.out_dir, total_seconds=args.seconds, episode_secs=args.episode_secs, region=region)


if __name__ == "__main__":
    main()
