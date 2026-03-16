#!/usr/bin/env python3
"""Offline tests for capture region parsing and shape. No Roblox required."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.screen import capture_region


def test_region_parse_run_takeover_style():
    """--region left,top,width,height should produce dict with correct keys."""
    region_str = "100,200,640,360"
    parts = [int(x.strip()) for x in region_str.split(",")]
    assert len(parts) == 4
    region = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    assert region["left"] == 100 and region["top"] == 200
    assert region["width"] == 640 and region["height"] == 360


def test_capture_region_shape():
    """capture_region(region=small_box) returns HxWx3 numpy array."""
    # Use a small region on primary monitor (0,0 might be menubar on Mac; use small box)
    with __import__("mss").mss() as m:
        mon = m.monitors[0]
        left = mon["left"]
        top = mon["top"]
    region = {"left": left, "top": top, "width": 40, "height": 30}
    frame = capture_region(region=region)
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[0] == 30 and frame.shape[1] == 40 and frame.shape[2] == 3
    assert frame.dtype in (np.uint8, np.float32) or frame.dtype == np.dtype("uint8")


def test_capture_region_values_in_range():
    """Captured pixel values should be in [0,255] for uint8."""
    with __import__("mss").mss() as m:
        mon = m.monitors[0]
    region = {"left": mon["left"], "top": mon["top"], "width": 20, "height": 20}
    frame = capture_region(region=region)
    assert frame.min() >= 0 and frame.max() <= 255


if __name__ == "__main__":
    for name in ("test_region_parse_run_takeover_style", "test_capture_region_shape", "test_capture_region_values_in_range"):
        fn = globals()[name]
        fn()
        print(name, "OK")
    print("All capture/config tests passed.")
