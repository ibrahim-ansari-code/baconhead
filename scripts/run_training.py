"""
scripts/run_training.py — Launch PPO training on the Roblox obby.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --timesteps 500000
    python scripts/run_training.py --resume checkpoints/obby_ppo.zip
"""

from __future__ import annotations

import argparse
import atexit
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from agent.env import ObbyEnv
from training.callbacks import EpisodeLogger
from training.curriculum import CurriculumTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def load_training_config() -> dict:
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("training", {})
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Roblox obby")
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training timesteps (default: from config or 1000000)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to saved model .zip to resume training",
    )
    args = parser.parse_args()

    cfg = load_training_config()

    total_timesteps = args.timesteps or cfg.get("total_timesteps", 1_000_000)
    stuck_timeout = cfg.get("stuck_timeout", 30.0)

    # Create environment
    env = ObbyEnv(stuck_timeout=stuck_timeout)
    env = Monitor(env)
    atexit.register(env.close)

    # Curriculum and logging
    curriculum = CurriculumTracker()
    callback = EpisodeLogger(curriculum, log_path="logs/training.csv")

    # PPO model
    if args.resume:
        log.info("Resuming from %s", args.resume)
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=cfg.get("learning_rate", 2.5e-4),
            n_steps=cfg.get("n_steps", 128),
            batch_size=cfg.get("batch_size", 32),
            n_epochs=cfg.get("n_epochs", 4),
            gamma=cfg.get("gamma", 0.99),
            clip_range=cfg.get("clip_range", 0.1),
            verbose=1,
            tensorboard_log="logs/tensorboard/",
        )

    log.info("Starting PPO training for %d timesteps", total_timesteps)

    # Save checkpoints directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Auto-save every 512 timesteps (~40 episodes)
    checkpoint_cb = CheckpointCallback(
        save_freq=512,
        save_path=str(checkpoint_dir),
        name_prefix="obby_ppo",
    )

    # Save on Ctrl+C
    def _save_on_interrupt(sig, frame):
        save_path = checkpoint_dir / "obby_ppo_interrupt.zip"
        log.info("Ctrl+C caught — saving model to %s", save_path)
        model.save(str(save_path))
        env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _save_on_interrupt)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[callback, checkpoint_cb],
    )

    save_path = checkpoint_dir / "obby_ppo.zip"
    model.save(str(save_path))
    log.info("Model saved to %s", save_path)


if __name__ == "__main__":
    main()
