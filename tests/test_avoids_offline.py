#!/usr/bin/env python3
"""Offline tests for reward/avoids. No BLIP/vision required for caption-based tests."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward.avoids import (
    get_avoid_penalty_from_caption,
    get_avoid_penalty,
    PRESET_AVOIDS,
)


def test_avoid_caption_empty():
    assert get_avoid_penalty_from_caption("") == 0.0


def test_avoid_caption_safe():
    assert get_avoid_penalty_from_caption("player running on platform") == 0.0


def test_avoid_caption_game_over():
    assert get_avoid_penalty_from_caption("Game Over") == 1.0


def test_avoid_caption_you_died():
    assert get_avoid_penalty_from_caption("You died") == 1.0


def test_avoid_caption_low_health():
    assert get_avoid_penalty_from_caption("low health bar") == 1.0


def test_avoid_caption_falling():
    assert get_avoid_penalty_from_caption("falling off the map") == 1.0


def test_avoid_caption_respawn():
    assert get_avoid_penalty_from_caption("respawn screen") == 1.0


def test_avoid_caption_death():
    assert get_avoid_penalty_from_caption("death screen") == 1.0


def test_avoid_caption_case_insensitive():
    assert get_avoid_penalty_from_caption("GAME OVER") == 1.0
    assert get_avoid_penalty_from_caption("Dead") == 1.0


def test_avoid_caption_void():
    assert get_avoid_penalty_from_caption("void") == 1.0


def test_avoid_caption_failed():
    assert get_avoid_penalty_from_caption("failed") == 1.0


def test_avoid_caption_defeat():
    assert get_avoid_penalty_from_caption("defeat") == 1.0


def test_avoid_caption_no_health():
    assert get_avoid_penalty_from_caption("no health") == 1.0


def test_avoid_caption_health_bar():
    assert get_avoid_penalty_from_caption("health bar empty") == 1.0


def test_avoid_caption_out_of_bounds():
    assert get_avoid_penalty_from_caption("out of bounds") == 1.0


def test_avoid_caption_custom_triggers():
    custom = {"custom": ["bad thing"]}
    assert get_avoid_penalty_from_caption("bad thing happened", triggers=custom) == 1.0
    assert get_avoid_penalty_from_caption("good thing", triggers=custom) == 0.0


def test_avoid_caption_none_triggers_uses_preset():
    assert get_avoid_penalty_from_caption("game over", triggers=None) == 1.0


def test_avoid_frame_no_vision():
    """get_avoid_penalty with caption provided skips vision."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert get_avoid_penalty(frame, caption="game over", use_vision=False) == 1.0
    assert get_avoid_penalty(frame, caption="running", use_vision=False) == 0.0


def test_preset_avoids_keys():
    assert "losing_health" in PRESET_AVOIDS
    assert "falling_off_map" in PRESET_AVOIDS
    assert "death_screen" in PRESET_AVOIDS


def test_preset_avoids_nonempty():
    for k, v in PRESET_AVOIDS.items():
        assert isinstance(v, list)
        assert len(v) >= 1


if __name__ == "__main__":
    tests = [
        test_avoid_caption_empty, test_avoid_caption_safe, test_avoid_caption_game_over,
        test_avoid_caption_you_died, test_avoid_caption_low_health, test_avoid_caption_falling,
        test_avoid_caption_respawn, test_avoid_caption_death, test_avoid_caption_case_insensitive,
        test_avoid_caption_void, test_avoid_caption_failed, test_avoid_caption_defeat,
        test_avoid_caption_no_health, test_avoid_caption_health_bar, test_avoid_caption_out_of_bounds,
        test_avoid_caption_custom_triggers, test_avoid_caption_none_triggers_uses_preset,
        test_avoid_frame_no_vision, test_preset_avoids_keys, test_preset_avoids_nonempty,
    ]
    for t in tests:
        t()
        print(t.__name__, "OK")
    print("All avoids tests passed.")
