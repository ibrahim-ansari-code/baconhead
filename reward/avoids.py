"""
Preset avoids for the reward model: losing health, falling off map, death screen.
Detects bad states from frame (via caption keywords) and returns a penalty in [0, 1].
"""

import re
from typing import List, Optional

import numpy as np

# Default trigger phrases for "avoid" states (case-insensitive)
PRESET_AVOIDS = {
    "losing_health": [
        "low health", "no health", "health bar", "died", "death", "dead",
        "game over", "you died", "respawn", "failed", "lose life",
    ],
    "falling_off_map": [
        "falling", "fell", "void", "respawn", "out of bounds", "fell off",
        "death", "dead", "game over",
    ],
    "death_screen": [
        "game over", "you died", "dead", "respawn", "defeat", "failed",
    ],
}


def get_avoid_penalty(
    frame: np.ndarray,
    caption: Optional[str] = None,
    use_vision: bool = True,
    triggers: Optional[dict] = None,
) -> float:
    """
    Return penalty in [0, 1] for preset avoids (losing health, falling off map, death).
    If caption is None and use_vision True, we run BLIP to get a caption, then check triggers.
    """
    triggers = triggers or PRESET_AVOIDS
    if caption is None and use_vision:
        from vision.report import describe_frame
        caption = describe_frame(frame, max_new_tokens=60)
    if not caption:
        return 0.0
    caption_lower = caption.lower()
    for category, phrases in triggers.items():
        for phrase in phrases:
            if phrase.lower() in caption_lower:
                print(f"[avoids] trigger {category!r} (phrase {phrase!r}) -> penalty 1.0", flush=True)
                return 1.0
    return 0.0


def get_avoid_penalty_from_caption(caption: str, triggers: Optional[dict] = None) -> float:
    """Penalty from precomputed caption (no vision call)."""
    triggers = triggers or PRESET_AVOIDS
    caption_lower = caption.lower()
    for _category, phrases in triggers.items():
        for phrase in phrases:
            if phrase.lower() in caption_lower:
                return 1.0
    return 0.0
