# PHASES.md — Implementation order and phase gates

## Overview
Implementation is strictly sequential. Each phase has a gate condition that must be met before the next phase begins. Do not jump ahead.

---

## Phase 1 — Capture pipeline validation
**Goal:** Reliable screen capture of the game viewport.

**Status:** Complete. OCR gate passed.

Tasks:
- Set up mss screen capture of the game viewport
- Validate the capture is fast (<5ms per frame), consistently cropped to the game window, and produces frames in the expected format
- Validate a live preview shows clean output with no taskbar or other apps bleeding in

Gate condition: Capture pipeline producing clean, correctly cropped frames at target resolution. **PASSED.**

---

## Phase 2 — Perception layer
**Goal:** Structured scene understanding from raw screenshots — void detection, edge detection, death detection.

**Camera note:** This branch uses third-person camera. A first-person comparison branch exists for reference. Third-person is the active camera mode on this branch.

Tasks:
- Implement void detection using the void's consistent color as an anchor signal
- Implement edge proximity detection using the bottom-center strip void ratio
- Implement left/right platform mass comparison for directional awareness
- Implement death detection via respawn UI color check or full-screen void ratio threshold (>80–90%)
- Validate on a live frame: walk to a platform edge and confirm edge detection fires; die deliberately and confirm death detection fires

Gate condition: All three signals (edge proximity, directional bias, death) produce correct output on live frames before any data collection begins.

---

## Phase 3 — Demonstration recording and behavioral cloning training
**Goal:** A trained CNN that takes an 84×84 screenshot and outputs the correct action.

Tasks:
- Record 25–30 clean, focused third-person runs on the scoped obby section (same start position, same camera angle set via `set_camera_angle()`, same graphics settings every run)
- Include deliberate edge-recovery demonstrations in each run so the model sees near-fall correction behavior
- Store each run as a sequence of (screenshot, action) pairs in `demos/`
- Resize frames to 84×84 for training
- Train a CNN with convolutional layers → flatten → fully connected → 4-class softmax (forward, left, right, jump)
- Split data 85% train / 15% validation
- Train for ~30 epochs, save the checkpoint with the lowest validation loss

Gate condition: Validation accuracy above 70%. If below 60%, collect more demonstration data — do not continue training past this point without addressing the data quality.

---

## Phase 4 — Two-tier agent deployment
**Goal:** Full running agent combining Gemini high-level planner and CNN low-level controller, with a robust death/respawn cycle.

Tasks:
- Implement Gemini planner loop: call Gemini every 1.5–2 seconds with the current full screenshot, receive a high-level intent string
- Implement CNN controller loop: run inference every ~50ms (20fps) using the latest 84×84 screenshot
- Optionally bias CNN output using Gemini's current intent (e.g. weight jump action higher when Gemini signals a jump is needed)
- Implement death handling: detect death → release all held keys → wait for respawn animation → confirm respawn → wait for character to fully load → resume main loop
- Run for 5 minutes and confirm the death/respawn cycle completes cleanly on every death

Gate condition: Agent runs continuously for 5 minutes with clean death/respawn handling on every death. Section completed successfully on 50%+ of attempts.
