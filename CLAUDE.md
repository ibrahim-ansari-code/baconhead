# CLAUDE.md — Roblox Obby RL Agent

## Project goal
Build an autonomous agent that plays Roblox obstacle courses (obbies) using only screen pixels and OS-level input emulation. No game memory hooks, no internal APIs. The agent must generalize across any obby that displays a visible stage counter on the HUD.

## Approach
Behavioral cloning first:
1. Capture human demonstration data (`screenshot` -> `action`) from a single scoped obby section.
2. Train a CNN to imitate those actions (behavioral cloning).
3. Deploy a two-tier agent:
   - Gemini as the high-level planner, updating the intent every 1.5–2 seconds.
   - The CNN as the low-level controller, producing actions at ~20fps (about every 50ms) from the latest screenshot.

Do not skip to phase 2 before phase 1 is complete and validated. See PHASES.md for phase gates.

## Tech stack
- Python 3.10+
- `mss` — screen capture
- `OpenCV` + `PIL` — vision preprocessing
- `easyocr` or `pytesseract` — stage counter OCR
- `PyTorch` — CNN
- `pynput` — OS-level input emulation
- Gemini API — high-level planning for the two-tier agent

## Folder structure
```
/
├── CLAUDE.md           ← this file
├── PHASES.md           ← implementation order and phase gates
├── RISKS.md            ← known risks and hard constraints
├── scripts/            ← standalone validation scripts (one per module)
│   ├── validate_ocr.py ← Phase 1: OCR accuracy validation
│   └── test_integration.py ← Phase 3: ObbyEnv integration test
├── demos/              ← recorded demonstration data for behavioral cloning
├── capture/
│   ├── CLAUDE.md       ← screen capture and OCR spec
│   └── ...
├── vision/
│   ├── CLAUDE.md       ← CNN architecture spec
│   └── ...
├── agent/
│   ├── CLAUDE.md       ← heuristic + LLM hybrid agent spec
│   └── ...
├── training/
│   ├── CLAUDE.md       ← reward function and RL training spec
│   └── ...
└── control/
    ├── CLAUDE.md       ← input emulation and action space spec
    └── ...
```

## Camera
Third-person camera. Fixed forward-facing angle, pitched 20–30° below horizontal. Set once at episode start via `set_camera_angle()` (see control/CLAUDE.md) and never touched again during a run. Mouse movement is not part of the action space — camera angle is fixed throughout each episode.

## Key constraints
- Pixel-only observation. Never access game memory, APIs, or internal state.
- OS-level input only. Use pynput for all keyboard and mouse emulation.
- Camera is third-person, fixed at episode start — no camera rotation actions exist.
- OCR must be validated on the target obby's font before training begins.
- Do not proceed past phase gates without explicit confirmation. See PHASES.md.

## Testing standards
Every module must have a corresponding validation script in `scripts/` before it is considered complete. Each script must:
- Be standalone and runnable with a single command from the project root
- Complete in under 2 minutes
- Produce clear PASS/FAIL output for every check
- Validate the module against the real Roblox environment — no mocked data
- If the script captures the screen at any point, print a countdown (e.g. "Switch to Roblox now — starting in 3... 2... 1...") before the first capture so the tester has time to focus the Roblox window

## Current focus
- Agent cannot yet complete a single checkpoint reliably.
- Priority is improving action quality via Gemini planner integration into the two-tier agent.
- Stage-based reward improvements are deferred until the agent can actually advance stages.

## Sub-module docs
Each subfolder has its own CLAUDE.md. Always read the relevant subfolder CLAUDE.md before working in that module. When in doubt about scope, refer back to this file.
