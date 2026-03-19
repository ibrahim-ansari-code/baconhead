"""
scripts/run_self_improve.py — Full self-improvement cycle.

Runs the two-tier agent with self-demo collection, background BC retraining,
live hot-swap, and optional PPO fine-tuning at the end.

Usage:
    python scripts/run_self_improve.py                      # 5 min agent run
    python scripts/run_self_improve.py --duration 600       # 10 min agent run
    python scripts/run_self_improve.py --duration 300 --ppo # agent run + PPO after
    python scripts/run_self_improve.py --ppo-only           # skip agent, just PPO
    python scripts/run_self_improve.py --cycles 3           # repeat agent+PPO 3 times
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
)
log = logging.getLogger("self_improve")


def count_demo_runs(demos_dir: Path) -> int:
    """Count BCDataset-compatible demo runs."""
    return len([
        d for d in demos_dir.iterdir()
        if d.is_dir() and (d / "frames.npz").exists() and (d / "actions.npy").exists()
    ]) if demos_dir.exists() else 0


def count_self_runs(demos_dir: Path) -> int:
    """Count self-play demo runs specifically."""
    return len([
        d for d in demos_dir.iterdir()
        if d.is_dir() and d.name.endswith("_self")
        and (d / "frames.npz").exists() and (d / "actions.npy").exists()
    ]) if demos_dir.exists() else 0


def run_agent(duration: int, dry_run: bool) -> None:
    """Run TwoTierAgent with self-improvement enabled."""
    from capture.screen import Capturer
    from agent.twotier import TwoTierAgent

    if not dry_run:
        print(f"\nSwitch to Roblox now — starting in 5... ", end="", flush=True)
        for i in range(5, 0, -1):
            print(f"{i}... ", end="", flush=True)
            time.sleep(1)
        print("GO!\n")

    cap = Capturer()
    agent = TwoTierAgent(
        cap,
        enable_self_improvement=True,
    )
    agent.run(duration_seconds=duration, dry_run=dry_run)


def run_ppo(timesteps: int) -> bool:
    """Run PPO fine-tuning with BC warm-start."""
    train_script = Path(__file__).resolve().parent / "run_training.py"
    log.info("Starting PPO fine-tuning for %d timesteps...", timesteps)

    result = subprocess.run(
        [sys.executable, str(train_script), "--timesteps", str(timesteps)],
        timeout=3600,
    )

    if result.returncode == 0:
        log.info("PPO fine-tuning completed successfully")
        return True
    else:
        log.error("PPO fine-tuning failed (rc=%d)", result.returncode)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Full self-improvement cycle")
    parser.add_argument(
        "--duration", type=int, default=300,
        help="Agent run duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--ppo", action="store_true",
        help="Run PPO fine-tuning after agent collection",
    )
    parser.add_argument(
        "--ppo-only", action="store_true",
        help="Skip agent run, only do PPO fine-tuning",
    )
    parser.add_argument(
        "--ppo-timesteps", type=int, default=10000,
        help="PPO training timesteps per cycle (default: 10000)",
    )
    parser.add_argument(
        "--cycles", type=int, default=1,
        help="Number of collect→train cycles to run (default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run agent in dry-run mode (no keyboard output)",
    )
    args = parser.parse_args()

    demos_dir = Path("demos")
    checkpoint = Path("checkpoints/bc_best.pt")

    print("=" * 60)
    print("Self-Improvement Cycle")
    print("=" * 60)
    print(f"  Checkpoint:    {checkpoint} ({'exists' if checkpoint.exists() else 'MISSING'})")
    print(f"  Total demos:   {count_demo_runs(demos_dir)}")
    print(f"  Self demos:    {count_self_runs(demos_dir)}")
    print(f"  Cycles:        {args.cycles}")
    print(f"  Agent duration:{args.duration}s per cycle")
    print(f"  PPO after:     {args.ppo or args.ppo_only}")
    print("=" * 60)

    if not checkpoint.exists() and not args.ppo_only:
        log.warning("No bc_best.pt found — agent will fail. Train BC first with: python scripts/train_bc.py")
        sys.exit(1)

    try:
        for cycle in range(1, args.cycles + 1):
            print(f"\n{'='*60}")
            print(f"Cycle {cycle}/{args.cycles}")
            print(f"{'='*60}")

            demos_before = count_self_runs(demos_dir)

            # Phase 1: Run agent and collect self-demos
            if not args.ppo_only:
                log.info("Phase 1: Running agent for %ds with self-demo collection...", args.duration)
                run_agent(args.duration, dry_run=args.dry_run)

                demos_after = count_self_runs(demos_dir)
                new_demos = demos_after - demos_before
                log.info("Collected %d new self-demos (total: %d)", new_demos, demos_after)

            # Phase 2: PPO fine-tuning (if requested)
            if args.ppo or args.ppo_only:
                if not checkpoint.exists():
                    log.warning("Skipping PPO — no bc_best.pt checkpoint")
                else:
                    log.info("Phase 2: PPO fine-tuning (%d timesteps)...", args.ppo_timesteps)
                    run_ppo(args.ppo_timesteps)

    except KeyboardInterrupt:
        log.info("Ctrl+C — stopping. Buffered self-demos were flushed to disk.")

    # Final summary
    print(f"\n{'='*60}")
    print("Self-Improvement Stopped" if False else "Self-Improvement Complete")
    print(f"{'='*60}")
    print(f"  Total demos:   {count_demo_runs(demos_dir)}")
    print(f"  Self demos:    {count_self_runs(demos_dir)}")
    print(f"  Checkpoint:    {checkpoint} ({'exists' if checkpoint.exists() else 'MISSING'})")


if __name__ == "__main__":
    main()
