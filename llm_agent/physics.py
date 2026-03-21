"""
Verified motion constants for Roblox on macOS via Quartz drag.

Camera calibration (2026-03-19, revised):
  200px Quartz drag ≈ 100° turn (user observed 10° overshoot past 90°).
  Corrected to 180px = 90°.
  Previous: 200px=90° caused ~10° extra rotation per turn → 400° in 4 turns.

Movement calibration:
  Roblox default walk speed = 16 studs/sec → 62.5ms per stud.
"""

# ── Camera rotation ────────────────────────────────────────────────────────────
# 200px drag = 100° → 1° = 2.0px → 90° = 180px
LOOK_PX_PER_DEGREE: float = 180 / 90       # 2.0 px per degree
LOOK_PX_PER_MS: float     = 0.55           # drag speed: 220px over 400ms

# ── Movement ──────────────────────────────────────────────────────────────────
# Roblox default walk speed = 16 studs/sec
WALK_MS_PER_STUD: float = 1000 / 16        # 62.5 ms per stud

# No hardcoded upper cap — Claude decides how far to walk based on what it
# sees.  The only safety limit is MAX_MOVEMENT_MS (prevents runaway keys).
MAX_MOVEMENT_MS: int = 3000                 # absolute safety ceiling

# ── Quick reference (for prompts) ─────────────────────────────────────────────
#   45°  turn  →   90px  →  ~164ms
#   90°  turn  →  180px  →  ~327ms
#   180° U-turn →  360px  →  ~655ms
#   3 studs fwd →  ~190ms  (precise positioning / entering doors)
#   6 studs fwd →  ~375ms  (short walk)
#   10 studs fwd → ~625ms  (medium walk)
#   16 studs fwd → ~1000ms (cross a courtyard)
#   32 studs fwd → ~2000ms (cross a field)


def degrees_to_px(degrees: float) -> int:
    """Convert a camera rotation in degrees to Quartz drag pixel distance."""
    return max(10, int(abs(degrees) * LOOK_PX_PER_DEGREE))


def degrees_to_ms(degrees: float) -> int:
    """Convert a camera rotation in degrees to drag duration in ms."""
    px = degrees_to_px(degrees)
    return max(100, min(2000, int(px / LOOK_PX_PER_MS)))


def studs_to_ms(studs: float) -> int:
    """Convert a forward distance in Roblox studs to W-press duration in ms."""
    return max(100, min(MAX_MOVEMENT_MS, int(studs * WALK_MS_PER_STUD)))
