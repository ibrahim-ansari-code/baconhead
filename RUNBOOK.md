# Runbook

## Quick start: full pipeline from zero to self-improving agent

Follow these steps in order. Each step depends on the previous one.

### Step 1 — Record human demos (do this first, only needed once)

You need at least 10 demo runs before anything else works.

1. Open Roblox to the obby
2. Run this, then immediately switch to the Roblox window:
   ```bash
   /opt/homebrew/bin/python3.14 scripts/record_demo.py
   ```
3. Play normally with **W / A / D / Space**. Ctrl+C when done.
4. Repeat until you have 10+ runs in `demos/`. Mix up your inputs — don't just hold W.

### Step 2 — Train the base CNN

```bash
/opt/homebrew/bin/python3.14 scripts/train_bc.py
```

Wait for it to finish. It prints val accuracy at the end. You need **≥ 70%** to continue. If it's below 60%, go back to step 1 and record more demos.

### Step 3 — Validate

```bash
/opt/homebrew/bin/python3.14 scripts/validate_bc.py
```

All checks should say PASS. If not, go back to step 2.

### Step 4 — Run the self-improving agent

1. Open Roblox to the obby, make sure the character is alive and on a platform
2. Run:
   ```bash
   /opt/homebrew/bin/python3.14 scripts/run_self_improve.py --cycles 3 --ppo
   ```
3. Switch to Roblox during the 5-second countdown
4. Watch it play. Each cycle:
   - Agent plays for 5 min — this is the agent playing live, you can watch it
   - It collects self-demos from successful runs (2+ stage advances)
   - BC retrains in the background when 3+ self-demos accumulate
   - Model hot-swaps live (no restart)
   - After the 5 min, PPO fine-tunes from the BC weights against live Roblox (also live play, keep Roblox focused)
   - PPO exports improved weights back to `bc_best.pt`
   - Next cycle starts with the improved model
5. When all 3 cycles finish, validate the result:
   ```bash
   /opt/homebrew/bin/python3.14 scripts/validate_bc.py
   ```

**Ctrl+C:** Stops the whole script immediately. Buffered self-demos still get flushed to disk before exit — you don't lose data.

**Regression detection:** If a retrained model is worse than the previous one, you'll see this in the log output:
```
WARNING — Regression detected: new val_accuracy=0.XXX < previous=0.XXX — skipping hot-swap
```
This means the retrainer threw away the bad model and kept the old one active. The agent keeps running on the previous good weights. No action needed from you.

---
---

## Reference: individual scripts

## 1. Record a demo run

Switch to Roblox before running. The script gives you a 5-second countdown.

```bash
/opt/homebrew/bin/python3.14 scripts/record_demo.py
```

- Play normally using **W / A / D / Space**
- Stop with **Ctrl+C** when done
- Saves to `demos/run_NNN/` automatically (increments each time)
- Prints an action distribution summary at the end — check that no action is at 0%

Aim for at least 10 runs before training. More is better. Try to cover different parts of the obby and include varied inputs (don't just hold W the whole time).

---

## 2. Train the CNN

```bash
/opt/homebrew/bin/python3.14 scripts/train_bc.py
```

- Trains for up to 50 epochs with early stopping (stops if no improvement for 10 epochs)
- Saves the best checkpoint to `checkpoints/bc_best.pt` automatically
- Prints per-action accuracy each epoch so you can see if any action is being ignored
- At the end it tells you if you passed the gate (≥70% val accuracy)

If val accuracy is below 60%: collect more demos before re-training. Don't keep training on bad data.

---

## 3. Validate the trained model

```bash
/opt/homebrew/bin/python3.14 scripts/validate_bc.py
```

Checks:
- Dataset loads correctly
- Checkpoint loads into the CNN
- Inference runs in under 100ms per frame
- Val accuracy ≥ 70%
- CNN predicts at least 4 distinct actions (not collapsed to one)
- Runs 10 live screen captures and prints what action it would pick

---

## 4. Run the agent

```bash
# Default (uses checkpoints/bc_best.pt automatically)
/opt/homebrew/bin/python3.14 run_twotier.py --duration 60

# CNN only, no Gemini bias
/opt/homebrew/bin/python3.14 run_twotier.py --duration 60 --no-gemini

# Specific checkpoint
/opt/homebrew/bin/python3.14 run_twotier.py --checkpoint checkpoints/bc_best.pt
```

---

## 5. Self-improvement (agent learns from its own play)

Runs the agent with live self-demo collection, background BC retraining, and model hot-swap. The agent gets better while it plays.

```bash
# Basic: 5 min agent run with self-demo collection + background retrain
/opt/homebrew/bin/python3.14 scripts/run_self_improve.py

# Longer session
/opt/homebrew/bin/python3.14 scripts/run_self_improve.py --duration 600

# Agent run + PPO fine-tuning after
/opt/homebrew/bin/python3.14 scripts/run_self_improve.py --ppo

# Full loop: 3 cycles of collect → PPO → collect → PPO → collect → PPO
/opt/homebrew/bin/python3.14 scripts/run_self_improve.py --cycles 3 --ppo

# Just PPO fine-tuning from current bc_best.pt (no agent run)
/opt/homebrew/bin/python3.14 scripts/run_self_improve.py --ppo-only

# Dry run (no keyboard output, for testing the pipeline)
/opt/homebrew/bin/python3.14 scripts/run_self_improve.py --dry-run
```

What happens under the hood:
1. Agent plays Roblox with self-demo collection enabled
2. Every time the agent advances 2+ stages in a run, those frames get saved as a new demo (`demos/run_NNN_self/`)
3. After 3 self-demos accumulate, BC retrains in the background
4. When retraining finishes, the new model is hot-swapped into the live agent (no restart needed)
5. If `--ppo` is set, PPO fine-tuning runs after using BC weights as a warm-start
6. PPO weights get exported back to `bc_best.pt` so the next cycle starts from the improved model

---

## Typical iteration loops

**Manual (human demos):**
1. Record 3–5 new demo runs
2. Run `train_bc.py` — check if val accuracy improved
3. Run `validate_bc.py` — confirm gate passed
4. Run `run_twotier.py --duration 60 --no-gemini` — watch the character

**Automated (self-improvement):**
1. Run `run_self_improve.py --ppo --cycles 3` — let it collect and improve
2. Run `validate_bc.py` — confirm model hasn't regressed
3. Run `run_twotier.py --duration 60` — watch the improved agent
