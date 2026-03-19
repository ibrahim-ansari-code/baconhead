# Recording Demos & Training the CNN

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

## 4. Run the agent with the new checkpoint

```bash
# Default (uses checkpoints/bc_best.pt automatically)
/opt/homebrew/bin/python3.14 run_twotier.py --duration 60

# CNN only, no Gemini bias
/opt/homebrew/bin/python3.14 run_twotier.py --duration 60 --no-gemini

# Specific checkpoint
/opt/homebrew/bin/python3.14 run_twotier.py --checkpoint checkpoints/bc_best.pt
```

---

## Typical iteration loop

1. Record 3–5 new demo runs
2. Run `train_bc.py` — check if val accuracy improved
3. Run `validate_bc.py` — confirm gate passed
4. Run `run_twotier.py --duration 60 --no-gemini` — watch the character
