"""
run_twotier.py — Launch the Phase 4 two-tier agent.

Usage:
    python run_twotier.py                  # full 5-min run
    python run_twotier.py --duration 600   # 10 minutes
    python run_twotier.py --dry            # no inputs sent
    python run_twotier.py --no-gemini      # CNN only, no planner bias
    python run_twotier.py --checkpoint path/to/model.pt
"""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

from agent.twotier import TwoTierAgent
from capture.screen import Capturer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase 4 two-tier agent")
    parser.add_argument("--duration", type=int, default=300, help="Run duration in seconds (default: 300)")
    parser.add_argument("--dry", action="store_true", help="Capture only, no inputs sent")
    parser.add_argument("--no-gemini", action="store_true", help="CNN only, no planner bias")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to CNN checkpoint")
    parser.add_argument("--bias-scale", type=float, default=None, help="Planner bias scale (default: from config)")
    args = parser.parse_args()

    print(f"Switch to Roblox now — starting in 3... ", end="", flush=True)
    import time
    for i in range(3, 0, -1):
        print(f"{i}... ", end="", flush=True)
        time.sleep(1)
    print("GO!")

    capturer = Capturer()
    agent = TwoTierAgent(
        capturer=capturer,
        checkpoint_path=args.checkpoint,
        bias_scale=args.bias_scale,
        use_gemini=not args.no_gemini,
    )
    agent.run(duration_seconds=args.duration, dry_run=args.dry)


if __name__ == "__main__":
    main()
