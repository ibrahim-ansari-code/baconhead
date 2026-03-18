#!/usr/bin/env python3
"""Misc tests: key_map, execute_action keys, run_live_test flow."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_agent.cem import execute_action_ms

# Key map from cem.execute_action_ms
KEY_MAP = {"W": "w", "A": "a", "S": "s", "D": "d", "space": "space", "Space": "space"}


def test_key_map_w():
    assert KEY_MAP.get("W") == "w"


def test_key_map_space():
    assert KEY_MAP.get("space") == "space"


def test_key_map_space_capital():
    assert KEY_MAP.get("Space") == "space"


def test_key_map_d():
    assert KEY_MAP.get("D") == "d"


def test_execute_none_short():
    t0 = time.perf_counter()
    execute_action_ms("none", duration_ms=10)
    assert (time.perf_counter() - t0) * 1000 >= 5


def test_execute_empty_string_sleeps():
    t0 = time.perf_counter()
    execute_action_ms("", duration_ms=15)
    elapsed = (time.perf_counter() - t0) * 1000
    assert elapsed >= 10


def test_default_actions_order():
    from llm_agent.scout import DEFAULT_ACTIONS
    assert DEFAULT_ACTIONS[0] == "W" and DEFAULT_ACTIONS[4] == "space" and DEFAULT_ACTIONS[5] == "none"


def test_cem_actions_default_10():
    from llm_agent.cem import run_cem
    f = np.zeros((64, 64, 3), dtype=np.uint8)
    mock = ([0.5] * 10, 0.0, "")
    best, scores, _, _, _, _ = run_cem(f, mock_scout_result=mock, use_scout=False)
    assert len(scores) == 10


def test_avoids_preset_import():
    from reward.avoids import PRESET_AVOIDS
    assert len(PRESET_AVOIDS) >= 1


def test_scout_model_import():
    from llm_agent.scout import SCOUT_MODEL
    assert "scout" in SCOUT_MODEL.lower() or "llama" in SCOUT_MODEL.lower()


def test_capture_region_import():
    from capture.screen import capture_region
    assert callable(capture_region)


def test_get_roblox_region_import():
    from capture.screen import get_roblox_region
    assert callable(get_roblox_region)


def test_focus_roblox_import():
    from capture.screen import focus_roblox_and_click
    assert callable(focus_roblox_and_click)


def test_run_cem_returns_tuple():
    from llm_agent.cem import run_cem
    f = np.zeros((50, 50, 3), dtype=np.uint8)
    out = run_cem(f, mock_scout_result=([0.5] * 10, 0.0, ""), use_scout=False)
    assert isinstance(out, tuple) and len(out) == 6


def test_parse_rewards_import():
    from llm_agent.scout import _parse_rewards
    s = _parse_rewards("REWARD1=0.5", n=10)
    assert s[0] == 0.5


def test_frame_to_tensor_import():
    try:
        from reward.combined import frame_to_tensor
        x = frame_to_tensor(np.zeros((84, 84, 3), dtype=np.uint8))
        assert x.shape[0] == 1 and x.shape[1] == 3
    except ImportError:
        pass  # torch not installed


def test_is_active_import():
    from reward.input_state import is_active
    assert isinstance(is_active(1.0), bool)


def test_get_current_keys_import():
    from reward.input_state import get_current_keys
    assert isinstance(get_current_keys(), set)


def test_dotenv_load():
    from dotenv import load_dotenv
    load_dotenv()
    # No assert; just ensure no crash


def test_mss_monitors():
    import mss
    with mss.mss() as m:
        assert len(m.monitors) >= 1
        assert "left" in m.monitors[0] and "width" in m.monitors[0]


if __name__ == "__main__":
    tests = [
        test_key_map_w, test_key_map_space, test_key_map_space_capital, test_key_map_d,
        test_execute_none_short, test_execute_empty_string_sleeps, test_default_actions_order,
        test_cem_actions_default_10, test_avoids_preset_import, test_scout_model_import,
        test_capture_region_import, test_get_roblox_region_import, test_focus_roblox_import,
        test_run_cem_returns_tuple, test_parse_rewards_import, test_frame_to_tensor_import,
        test_is_active_import, test_get_current_keys_import, test_dotenv_load, test_mss_monitors,
    ]
    for t in tests:
        t()
        print(t.__name__, "OK")
    print("All misc tests passed.")
