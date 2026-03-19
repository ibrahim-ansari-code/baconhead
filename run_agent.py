"""
run_agent.py — Launch the Phase 3 heuristic hybrid agent.

Usage:
    python run_agent.py              # full run with LLM
    python run_agent.py --no-llm     # heuristic only, no LLM calls
    python run_agent.py --dry        # capture only, no inputs sent
    python run_agent.py --frames 500 # stop after 500 frames
"""

from __future__ import annotations

import argparse
import atexit
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from agent.heuristic import HeuristicAgent, create_llm_provider
from capture.screen import Capturer
from control.loop import run_loop

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase 3 heuristic hybrid agent")
    parser.add_argument("--dry", action="store_true", help="Capture only, no inputs")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM calls")
    parser.add_argument("--frames", type=int, default=0, help="Max frames (0=forever)")
    args = parser.parse_args()

    # Load config
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f).get("agent", {})

    # Set up LLM provider
    llm_provider = None
    if not args.no_llm:
        llm_provider = create_llm_provider(
            backend=cfg.get("llm_backend", "anthropic"),
            model=cfg.get("llm_model"),
        )
        if llm_provider is None:
            logging.getLogger(__name__).warning(
                "LLM provider unavailable — running heuristic only"
            )

    # Create capturer and agent
    capturer = Capturer()
    agent = HeuristicAgent(capturer=capturer, llm_provider=llm_provider)
    atexit.register(agent.close)

    # Pass the same capturer to the loop so the agent can read last_frame
    run_loop(
        action_fn=agent,
        dry_run=args.dry,
        max_frames=args.frames,
        capturer=capturer,
    )


if __name__ == "__main__":
    main()
