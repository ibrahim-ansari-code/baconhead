# Roblox auto-play (capture → vision → reward → CEM planning)

Capture Roblox gameplay, learn from the user, take over when idle, and execute CEM-style plans.

## Todo list

- [x] **Capture** – Get gameplay from Roblox (screen capture, window or region, configurable FPS)
- [x] **Reward model** – Learn r(s) from user play (active vs idle frames)
- [x] **Preset avoids** – Losing health, falling off map, death screen (penalty in reward)
- [x] **Combined reward** – r_total = r_cnn(s) - avoid_penalty(s)
- [x] **Llama 4 Scout** – Groq Vision, score 10 actions per frame
- [x] **CEM** – 10 options, score with Scout + reward model + avoids, pick best, ms timing
- [x] **Idle takeover** – When idle N s, run CEM every 5s, execute best action for 5000 ms

## Setup

```bash
cd "sf tripo"
pip install -r requirements.txt
```

## Capture (current)

```bash
# Capture at 10 FPS; auto-detect Roblox window on Mac, else primary monitor
python run_capture.py

# Full primary monitor, no window detection
python run_capture.py --no-window-detect

# Custom region (left, top, width, height)
python run_capture.py --region 100,100,1280,720

# Different FPS
python run_capture.py --fps 15

# Report what we see (BLIP caption every N frames)
python run_capture.py --report --report-every 15 --seconds 30
```

Press Ctrl+C to stop. With `--report`, a vision model describes each Nth frame so you can confirm we're interpreting the screen.

## Reward model

Learns r(s) from your play: frames where you're pressing keys = high reward, idle = low. Used later for CEM planning.

**1. Collect data** (play the game; we record frames + key state):

```bash
# Default: save to reward_data/, sample every 3 frames, 84x84, active = key in last 2s
python -m reward.collect --out-dir reward_data

# Limit time or number of samples
python -m reward.collect --out-dir reward_data --seconds 120
python -m reward.collect --out-dir reward_data --max-samples 2000
```

On macOS, add **Terminal** (or **Python**) to **Accessibility** (System Settings → Privacy & Security → Accessibility), then quit and reopen Terminal. If key logging doesn’t work when the game is focused.

**2. Train**

```bash
python -m reward.train_reward --data reward_data --out reward_model.pt --epochs 20
```

**3. Use in code**

```python
from reward.model import load_reward_model
import torch
model = load_reward_model("reward_model.pt")
# frame: (1, 3, 84, 84) tensor in [0,1]; or (B, 3, 84, 84)
r = model(frame)
```

## Full pipeline: notice play, reward model, then take over with CEM

1. **Collect** (notice how the player plays): `python -m reward.collect --out-dir reward_data --seconds 120`
2. **Train** reward model: `python -m reward.train_reward --data reward_data --out reward_model.pt --epochs 20`
3. **Take over when idle:** `python run_takeover.py` (Roblox window only; use `--full-screen` for whole screen, `--idle 3` to change idle seconds)

When you don't press any key for `--idle` seconds (default 3), we run **CEM** every **5s**: 10 action options (W, A, S, D, space, none, …), score each with **Llama 4 Scout** (vision) and the **reward model**, subtract **preset avoid** penalty (losing health, falling off map, death screen), pick best, then execute that key for **5000 ms** (millisecond-accurate). Repeat until you press a key again.

- **Preset avoids** ([reward/avoids.py](reward/avoids.py)): Keywords in frame caption (BLIP) trigger penalty: death, game over, falling, void, respawn, low health, etc. Combined with learned r(s) in [reward/combined.py](reward/combined.py).
- **CEM** ([llm_agent/cem.py](llm_agent/cem.py)): `run_cem(frame, reward_model=..., scout_api_key=...)` returns best action and scores. `execute_action_ms(action, duration_ms=5000)` holds the key for the given milliseconds.

## Config

[config.yaml](config.yaml): `capture.*`, `agent.idle_seconds`, `agent.interval_seconds`, `agent.duration_ms`, `avoids.*` (preset phrases for penalty).

## Tests (offline, no Roblox)

Run CEM + action-diversity tests with mock Scout (no API key needed):

```bash
python tests/test_cem_offline.py
```

Tests: Scout reply parsing, mock CEM prefers W when scored high, anti-repeat space after 2x jump, anti-repeat any action after 3x same, loop gets at least 2 distinct actions, and mock CEM runtime. With `GROQ_API_KEY` set, one real Scout call is timed for tuning intervals.
