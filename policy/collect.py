"""
Collect (frame, action_id) data for the policy model.
You run Roblox; we capture screenshots and label each with an oracle (scout, avoid_only, etc.).
"""

import argparse
import json
import os
import threading
import time
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from capture.screen import capture_region, get_roblox_region, capture_loop
from policy.oracles import ACTION_NAMES, get_oracle


def run_collect(
    out_dir: str,
    oracle_name: str = "avoid_only",
    region: Optional[dict] = None,
    fps: float = 10.0,
    sample_every: int = 3,
    frame_height: int = 224,
    frame_width: int = 224,
    max_samples: int = 0,
    stop_event: Optional[threading.Event] = None,
    reward_model_path: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Capture frames and label each with the chosen oracle. Save to out_dir/data.npz and config.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = None
    reward_model = None
    if reward_model_path and os.path.isfile(reward_model_path) and oracle_name in ("scout", "cem_no_scout"):
        import torch
        from reward.model import load_reward_model
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        reward_model = load_reward_model(reward_model_path, device=device)
    oracle = get_oracle(oracle_name, api_key=api_key, reward_model=reward_model, device=device)

    frames_list = []
    actions_list = []
    n = [0]
    log_every = 30

    def on_frame(frame: np.ndarray, timestamp: float):
        n[0] += 1
        if n[0] % sample_every != 0:
            return
        if max_samples > 0 and len(frames_list) >= max_samples:
            if stop_event:
                stop_event.set()
            return
        small = _resize_frame(frame, frame_height, frame_width)
        try:
            action_id = oracle(frame)
        except Exception as e:
            print(f"[policy/collect] oracle error: {e}", flush=True)
            action_id = 0
        frames_list.append(small)
        actions_list.append(action_id)
        num = len(frames_list)
        if num % log_every == 0:
            from collections import Counter
            c = Counter(actions_list)
            print(f"  [policy/collect] {num} samples | oracle={oracle_name} | actions: {dict(c)}", flush=True)

    try:
        capture_loop(region=region, fps=fps, callback=on_frame, stop_event=stop_event)
    except KeyboardInterrupt:
        pass

    if not frames_list:
        print("No samples collected.")
        return

    frames = np.stack(frames_list, axis=0)
    actions = np.array(actions_list, dtype=np.int64)
    np.savez(
        os.path.join(out_dir, "data.npz"),
        frames=frames,
        actions=actions,
    )
    config = {
        "oracle": oracle_name,
        "fps": fps,
        "sample_every": sample_every,
        "frame_height": frame_height,
        "frame_width": frame_width,
        "n_samples": len(frames_list),
        "action_names": ACTION_NAMES,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved {len(frames_list)} samples to {out_dir} (oracle={oracle_name})", flush=True)


def _resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray(frame.astype(np.uint8))
    pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(pil)


def main():
    parser = argparse.ArgumentParser(description="Collect policy data: capture frames, label with oracle")
    parser.add_argument("--out-dir", type=str, default="policy_data", help="Output directory")
    parser.add_argument("--oracle", type=str, default="avoid_only",
                        choices=["scout", "avoid_only", "cem_no_scout", "random", "forward"],
                        help="Which oracle labels each frame")
    parser.add_argument("--region", type=str, default=None, help="left,top,width,height")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--frame-size", type=int, default=224, help="Height and width of saved frames")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples then exit (0 = unlimited)")
    parser.add_argument("--reward-model", type=str, default=None, help="Path to reward model for scout/cem_no_scout")
    parser.add_argument("--seconds", type=float, default=None, help="Run for N seconds then exit")
    args = parser.parse_args()

    region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        if len(parts) != 4:
            raise ValueError("--region must be left,top,width,height")
        region = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    else:
        region = get_roblox_region()
        if region:
            print("Using Roblox window:", region, flush=True)
        else:
            print("Roblox window not found; using primary monitor.", flush=True)

    stop = threading.Event()
    if args.seconds is not None:
        def stop_after():
            time.sleep(args.seconds)
            stop.set()
        threading.Thread(target=stop_after, daemon=True).start()

    run_collect(
        out_dir=args.out_dir,
        oracle_name=args.oracle,
        region=region,
        fps=args.fps,
        sample_every=args.sample_every,
        frame_height=args.frame_size,
        frame_width=args.frame_size,
        max_samples=args.max_samples or 0,
        stop_event=stop,
        reward_model_path=args.reward_model,
        api_key=os.environ.get("GROQ_API_KEY"),
    )


if __name__ == "__main__":
    main()
