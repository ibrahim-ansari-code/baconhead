# control/CLAUDE.md — Input emulation and action space

## Responsibility
This module sends keyboard and mouse inputs to Roblox via OS-level emulation. It receives an action index and executes it. Nothing else happens here — no decisions, no perception.

---

## Input library
Use `pynput` for all input emulation. Do not use `pyautogui` for key hold actions — pynput gives finer control over keydown and keyup separately, which is required for held movement keys.

```python
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

keyboard = KeyboardController()
mouse = MouseController()
```

---

## Action space
6 discrete actions. This is the full action space for both the heuristic agent and the RL policy.

| Index | Action | Keys |
|---|---|---|
| 0 | Move forward | W held |
| 1 | Move left | A held |
| 2 | Move right | D held |
| 3 | Jump | Space tap |
| 4 | Move forward + jump | W held + Space tap |
| 5 | Idle | No input |

Key hold duration: 200ms per step at 5fps. Adjust proportionally if fps changes.

Do not add more actions without updating the gymnasium observation space in agent/CLAUDE.md and revalidating the CNN output layer size in vision/CLAUDE.md.

---

## Camera control
Camera is third-person, fixed at episode start. Camera rotation is **not** part of the action space and is never triggered during a run — not by the CNN, not by the LLM planner.

Call `set_camera_angle()` exactly once at episode start, before the main loop begins. Do not call it again.

`set_camera_angle()` does the following:
1. Moves the mouse to the center of the game window
2. Holds right-click (Roblox uses right-click-drag to rotate the camera)
3. Moves the mouse downward to pitch the camera 20–30° below horizontal, forward-facing
4. Releases right-click

This produces a stable third-person view where the character is visible from behind and 2–3 platforms ahead are in frame.

```python
def set_camera_angle():
    game_center = (window_x + window_w // 2, window_y + window_h // 2)
    mouse.position = game_center
    mouse.press(Button.right)
    mouse.move(0, -60)  # pitch down ~25° below horizontal
    mouse.release(Button.right)
```

---

## Frame rate limiter
Cap the main loop at 5fps. This matches human-like input rates and prevents Roblox from flagging unusual input frequency.

```python
import time

FRAME_DURATION = 1.0 / 5  # 200ms per frame

loop_start = time.time()
# ... execute action ...
elapsed = time.time() - loop_start
if elapsed < FRAME_DURATION:
    time.sleep(FRAME_DURATION - elapsed)
```

---

## Input safety
- Always release all keys at episode reset — do not leave keys held across episodes
- On program exit, release all keys and mouse buttons immediately
- Never send more than one keydown without a corresponding keyup

```python
def release_all():
    for key in ['w', 'a', 'd']:
        keyboard.release(key)
    keyboard.release(Key.space)
    mouse.release(Button.right)
```

Call `release_all()` at every episode reset and on program exit.
