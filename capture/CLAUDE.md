# capture/CLAUDE.md — Screen capture and OCR

## Responsibility
This module handles all pixel acquisition and stage counter reading. Nothing else. It does not make decisions, send inputs, or touch the CNN.

---

## Screen capture
Use `mss` for screen capture. It is faster than pyautogui for repeated frame grabs.

- Capture only the Roblox game window, not the full desktop
- Define the game viewport ROI at startup by detecting the window bounds
- Target capture rate: 2-5 fps to match the main loop frame limiter

```python
import mss
with mss.mss() as sct:
    frame = sct.grab(game_region)  # game_region = {"top": y, "left": x, "width": w, "height": h}
```

---

## HUD region of interest (ROI)
The stage counter is displayed in a fixed position on the HUD. Do not run OCR on the full frame — crop tightly to the counter region only.

- Identify the stage counter position on the target obby before running
- Define the HUD crop coordinates as a constant in a config file, not hardcoded in logic
- The crop should be small — typically under 200×50px
- If the counter position varies between obbies, implement a one-time calibration step at startup

---

## OCR implementation
Use `easyocr` as the default. Fall back to `pytesseract` if easyocr accuracy is insufficient.

```python
import easyocr
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if available

def read_stage(hud_crop):
    results = reader.readtext(hud_crop, allowlist='0123456789/')
    # parse the first number found
    ...
```

- Use `allowlist='0123456789/'` to restrict OCR to digits and slash only — reduces false reads
- Parse the result to extract the current stage number as an integer
- If no number is detected, return the last known stage number (do not return 0)
- Log every raw OCR result alongside the parsed value for debugging

---

## Death detection
Detect death by measuring the fraction of the full frame that matches the void's color.

- The void has a specific, consistent color throughout the obby — identify its HSV range empirically before deploying (see config.yaml void_hsv_lower / void_hsv_upper)
- Compute the void ratio: fraction of full-frame pixels that fall within the void HSV range
- If void ratio exceeds the configured threshold (default 0.85) for 2+ consecutive frames, flag as death event
- Also watch for the respawn UI — it has distinctive colors distinct from normal gameplay
- Debounce: require the signal to persist for 2+ consecutive frames before triggering
- Do NOT use brightness delta — the void is a specific color, not a generic fade to black or white

**Third-person camera caveat — TBD:** The default void ratio threshold (0.85) was calibrated for first-person, where death causes the void to flood the entire screen. In third-person, the character falls through space with the void filling much less of the frame (character, platform remnants, and HUD all remain visible longer). The 0.85 threshold will likely stop firing reliably. **This threshold must be re-calibrated for third-person as part of the Phase 2 gate — do not mark Phase 2 complete until death detection is re-validated against a deliberate death in third-person mode.**

---

## OCR validation requirement
Per PHASES.md, OCR must achieve >95% accuracy before any other module is built.

Validation process:
1. Run the capture loop on a live obby session
2. Log every OCR read alongside the ground truth (manually verified)
3. Calculate accuracy over at least 20 distinct stage numbers
4. Do not proceed to Phase 2 until this threshold is met

If accuracy cannot reach 95% with easyocr or pytesseract, replace with a small custom digit classifier (e.g. a 5-layer CNN trained on ~500 screenshots of the specific font).

---

## Output interface
This module exposes two values to the rest of the system:

- `current_stage: int` — the most recently read stage number
- `death_event: bool` — True for one frame when a death is detected

No other data leaves this module.
