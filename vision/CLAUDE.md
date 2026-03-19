# vision/CLAUDE.md — CNN architecture and frame processing

## Responsibility
This module handles all visual preprocessing and the CNN model definition. It takes raw frames in and produces a feature vector out. It does not make action decisions or compute rewards.

---

## Frame preprocessing pipeline
Every frame goes through this pipeline before entering the CNN:

1. Crop to game viewport (remove OS chrome, taskbar, etc.)
2. Grayscale vs. color input — **TBD, revisit after Phase 2 perception validation.** With the third-person camera, the character sprite is now visible in frame and color may aid sprite identification. For now, the pipeline converts to grayscale, but this decision should be re-evaluated before behavioral cloning data collection begins.
3. Resize to 84×84 pixels
4. Normalize pixel values to range 0.0→1.0

```python
import cv2
import numpy as np

def preprocess_frame(frame_bgr, viewport_region):
    cropped = frame_bgr[viewport_region]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized.astype(np.float32) / 255.0
    return normalized
```

---

## Frame stacking
Stack the last 4 preprocessed frames into a single tensor. This gives the CNN implicit motion information without optical flow.

- Maintain a deque of length 4
- At startup, fill the deque with copies of the first frame
- After each new frame, pop the oldest and append the new one
- Stack along axis 0 to produce shape (4, 84, 84)

```python
from collections import deque
frame_buffer = deque(maxlen=4)
```

---

## Frame differencing
Run frame differencing in parallel with the stacked frame pipeline to detect moving obstacles.

- Subtract the previous frame from the current frame (absolute difference)
- Threshold the result to produce a binary motion mask
- This is a separate output — it does not replace the stacked frames fed to the CNN

```python
motion_mask = cv2.absdiff(prev_frame, curr_frame)
_, motion_binary = cv2.threshold(motion_mask, 25, 255, cv2.THRESH_BINARY)
```

---

## Third-person perception notes
With the third-person camera:
- The character sprite is visible in frame and can be used as a position reference (e.g. distance from platform edge, lateral alignment).
- The wider FOV means 2–3 platforms are typically visible simultaneously, compared to the narrow single-platform view in first-person.
- Frame stacking still provides motion information; the character sprite's movement across frames is an additional implicit motion signal.

---

## CNN architecture
This is the Nature DQN architecture, slightly adapted. Use Stable Baselines3's `CnnPolicy` — it implements this architecture by default and does not need to be defined manually unless customization is required.

| Layer | Config | Output shape |
|---|---|---|
| Input | 4 stacked grayscale frames | (4, 84, 84) |
| Conv1 | 32 filters, 8×8 kernel, stride 4, ReLU | (32, 20, 20) |
| Conv2 | 64 filters, 4×4 kernel, stride 2, ReLU | (64, 9, 9) |
| Conv3 | 64 filters, 3×3 kernel, stride 1, ReLU | (64, 7, 7) |
| Flatten | — | 3136 |
| Linear | 3136 → 512, ReLU | 512 |
| Output | 512 → N (number of actions) | N |

If implementing manually in PyTorch:

```python
import torch.nn as nn

class ObbyCNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.fc(self.conv(x))
```

---

## Inference speed requirement
Per PHASES.md Phase 4 gate: CNN forward pass must complete in under 100ms on target hardware to be compatible with the 2-5 fps main loop. Validate this before integrating into the loop.

---

## Depth estimation layer

Depth Anything V2 Small runs alongside the existing OpenCV perception layer on every frame. The two signals are complementary — OpenCV void detection is fast and color-based; depth estimation is geometry-based and color-agnostic. Both feed into the final scene state.

**Model:** Depth Anything V2 Small specifically. Medium and Large are too slow for real-time use at 20fps on a gaming PC without a dedicated inference budget. Do not substitute.

### Responsibilities

- Generate a depth map of the full screenshot on every frame
- Use the **bottom-center strip** of the depth map to determine ground state — this is a second signal alongside the OpenCV void ratio, not a replacement
- Compare **left vs. right** depth averages across the depth map to determine which direction the next platform is
- Estimate **forward depth** (center strip) to know how far ahead the next platform is

### Integration approach

OpenCV and depth signals are computed **independently** and merged into a single scene state dictionary. Neither replaces the other.

```
scene_state = {
    "void_ratio": float,          # from OpenCV HSV mask
    "depth_ground_state": str,    # "floor" | "void" | "uncertain"
    "depth_left_avg": float,      # mean depth of left half
    "depth_right_avg": float,     # mean depth of right half
    "depth_forward": float,       # mean depth of center-forward strip
    "signal_conflict": bool,      # True if OpenCV and depth disagree on ground state
}
```

If the two signals contradict each other (e.g. void ratio says death but depth map shows solid ground ahead), set `signal_conflict: True` in the scene state and surface it to the high-level planner. Do not silently resolve the conflict in this layer.

### Latency constraint

Depth Anything V2 Small must be benchmarked on target hardware before integrating into the live loop. If it cannot complete inference in **under 30ms per frame**, it must be moved to the Gemini tier (1.5–2 second intervals) rather than the CNN tier. See RISKS.md Risk 7.

---

## Output interface
This module exposes:

- `features: torch.Tensor` shape (512,) — the feature vector from the FC layer, fed to the policy
- `motion_mask: np.ndarray` shape (84, 84) — binary motion mask, available to the agent as auxiliary input
- `scene_state: dict` — merged OpenCV + depth perception signals (see Depth estimation layer above)
