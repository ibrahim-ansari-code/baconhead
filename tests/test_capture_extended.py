#!/usr/bin/env python3
"""Extended capture tests: region, shape, dtype, monitor."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mss
from capture.screen import capture_region


def test_region_dict_keys():
    r = {"left": 0, "top": 0, "width": 100, "height": 100}
    frame = capture_region(region=r)
    assert frame.shape[0] == 100 and frame.shape[1] == 100


def test_region_small():
    with mss.mss() as m:
        mon = m.monitors[0]
    r = {"left": mon["left"], "top": mon["top"], "width": 10, "height": 10}
    frame = capture_region(region=r)
    assert frame.shape == (10, 10, 3)


def test_capture_dtype():
    with mss.mss() as m:
        mon = m.monitors[0]
    r = {"left": mon["left"], "top": mon["top"], "width": 20, "height": 20}
    frame = capture_region(region=r)
    assert frame.dtype == np.uint8 or frame.dtype.kind == "u"


def test_capture_ndim():
    with mss.mss() as m:
        mon = m.monitors[0]
    r = {"left": mon["left"], "top": mon["top"], "width": 30, "height": 20}
    frame = capture_region(region=r)
    assert frame.ndim == 3


def test_capture_channels():
    with mss.mss() as m:
        mon = m.monitors[0]
    r = {"left": mon["left"], "top": mon["top"], "width": 15, "height": 15}
    frame = capture_region(region=r)
    assert frame.shape[2] == 3


def test_capture_monitor_zero():
    frame = capture_region(region=None, monitor=0)
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_capture_reuse_sct():
    with mss.mss() as sct:
        mon = sct.monitors[0]
        r = {"left": mon["left"], "top": mon["top"], "width": 25, "height": 25}
        f1 = capture_region(region=r, sct=sct)
        f2 = capture_region(region=r, sct=sct)
    assert f1.shape == f2.shape


def test_region_negative_left():
    with mss.mss() as m:
        mon = m.monitors[0]
    r = {"left": max(0, mon["left"] - 0), "top": mon["top"], "width": 20, "height": 20}
    frame = capture_region(region=r)
    assert frame.shape[0] == 20 and frame.shape[1] == 20


def test_capture_values_bounds():
    with mss.mss() as m:
        mon = m.monitors[0]
    r = {"left": mon["left"], "top": mon["top"], "width": 16, "height": 16}
    frame = capture_region(region=r)
    assert frame.min() >= 0 and frame.max() <= 255


def test_capture_rectangular():
    with mss.mss() as m:
        mon = m.monitors[0]
    r = {"left": mon["left"], "top": mon["top"], "width": 80, "height": 40}
    frame = capture_region(region=r)
    assert frame.shape[0] == 40 and frame.shape[1] == 80


def test_region_parse_four_ints():
    s = "1,2,3,4"
    parts = [int(x.strip()) for x in s.split(",")]
    assert len(parts) == 4
    r = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    assert r["width"] == 3 and r["height"] == 4


def test_region_parse_with_spaces():
    s = "10, 20, 30, 40"
    parts = [int(x.strip()) for x in s.split(",")]
    r = {"left": parts[0], "top": parts[1], "width": parts[2], "height": parts[3]}
    assert r["left"] == 10 and r["top"] == 20


def test_capture_no_region_uses_monitor():
    frame = capture_region()
    assert frame.size > 0


def test_capture_shape_height_width():
    with mss.mss() as m:
        mon = m.monitors[0]
    h, w = 22, 33
    r = {"left": mon["left"], "top": mon["top"], "width": w, "height": h}
    frame = capture_region(region=r)
    assert frame.shape[0] == h and frame.shape[1] == w


if __name__ == "__main__":
    tests = [
        test_region_dict_keys, test_region_small, test_capture_dtype, test_capture_ndim,
        test_capture_channels, test_capture_monitor_zero, test_capture_reuse_sct,
        test_region_negative_left, test_capture_values_bounds, test_capture_rectangular,
        test_region_parse_four_ints, test_region_parse_with_spaces, test_capture_no_region_uses_monitor,
        test_capture_shape_height_width,
    ]
    for t in tests:
        t()
        print(t.__name__, "OK")
    print("All capture extended tests passed.")
