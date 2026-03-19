# RISKS.md — Known risks and hard constraints

## Hard constraints
These must never be violated under any circumstances:

- Never access Roblox game memory, internal APIs, or any data not visible on screen
- Never use direct memory access (DMA), memory hooks, or injection of any kind
- All inputs must go through OS-level emulation via pynput only
- The agent must solve the same perceptual problem a human player does

---

## Known risks

### Risk 1 — OCR reliability (HIGH)
**Description:** The entire reward signal depends on correctly reading the stage counter from the screen via OCR. If OCR misreads the number, the agent receives false rewards and training is corrupted.

**Why it matters:** Roblox uses custom UI fonts that may not be in easyocr or pytesseract's default training data. Stage numbers displayed at small sizes or with drop shadows may read inconsistently.

**Mitigation:**
- Phase 1 gate requires >95% OCR accuracy before any other module is built
- Crop tightly to the HUD region only — do not run OCR on the full frame
- Test on the specific obby's font and size before committing
- If accuracy cannot reach 95%, consider training a small custom digit classifier on screenshots instead of relying on general-purpose OCR

**Status:** Unvalidated — must be resolved in Phase 1.

---

### Risk 2 — Roblox reset delay (MEDIUM)
**Description:** RL training requires thousands of episode resets. Every time the agent dies, Roblox takes several seconds to respawn the character. This bottlenecks training throughput compared to games with instant resets.

**Why it matters:** At 10 seconds per reset, 10,000 episodes = ~28 hours of wall-clock time just in resets, before accounting for actual play time.

**Mitigation:**
- Target obbies with faster respawn behavior if possible
- Accept that training will be slow and set expectations accordingly

**Status:** Known and accepted. Monitor episode throughput in Phase 4.

---

### Risk 3 — No stage counter fallback (LOW)
**Description:** The primary reward signal assumes a visible stage counter on the HUD. If the target obby does not display one, OCR produces no usable signal and the reward function collapses entirely.

**Why it matters:** Some obbies use custom UIs, hide the stage counter, or track progress via checkpoints with no numeric display. Without a fallback, the agent has no reward signal and cannot learn.

**Fallback reward strategy (not yet implemented):**
- Track the character sprite's forward screen displacement per frame as a proxy for progress
- Award a small continuous reward proportional to displacement magnitude each frame
- Keep death and stuck penalties unchanged from the primary reward design
- Combine with a frame differencing signal to estimate forward vs. backward movement

**Known limitations of the fallback:**
- Course directionality assumptions break on non-linear obbies (branching paths, vertical courses, loops) — forward screen displacement does not map to actual progress
- Character tracking via color matching is fragile if the avatar overlaps visually with the environment (similar colors, busy backgrounds)
- The fallback provides no discrete progress signal, which weakens reward shaping compared to the stage counter approach

**Status:** Not yet implemented. Only relevant if targeting obbies without a visible stage counter. Do not implement until the stage counter approach is confirmed insufficient for a target obby.

---

### Risk 4 — Curriculum learning dependency on OCR (HIGH)
**Description:** Curriculum learning tracks death clusters by reading stage numbers via OCR. If OCR misreads a stage number during training, the curriculum logic will compute the wrong death cluster and set an incorrect respawn checkpoint — potentially placing the agent in the wrong section of the course and making training worse rather than better.

**Why it matters:** Unlike reward corruption (Risk 1), a bad curriculum start point does not just add noise to a single episode — it systematically misdirects the training distribution until corrected.

**Mitigation:**
- OCR must remain above 95% accuracy throughout training, not just at Phase 1 validation. The Phase 1 gate is a prerequisite, not a one-time certificate.
- Spot-check OCR accuracy at the start of every training session by running the OCR pipeline on a known set of stage counter crops before allowing curriculum updates to take effect.
- If OCR accuracy drops below 95% mid-training, freeze curriculum updates and fall back to stage 1 start until accuracy is restored.

**Status:** Unvalidated — dependent on Risk 1 mitigation holding throughout Phase 4.

---

### Risk 5 — Demonstration quality (HIGH)
**Description:** Behavioral cloning copies what the demonstrator does. Inconsistent play during recording — hesitation, sloppy movement, replaying the same mistake multiple times — produces a noisy dataset that the model will faithfully imitate, including the mistakes.

**Why it matters:** Unlike RL, behavioral cloning has no mechanism to self-correct from bad demonstrations. Noise in the input data translates directly to noise in learned policy behavior, and more training will not fix it.

**Mitigation:**
- Record 25–30 clean, focused runs on a single scoped section (5–10 platforms, no moving obstacles)
- Play at a deliberately slower-than-normal pace during recording — deliberate movements produce cleaner training signal than fast, instinctive ones
- Include intentional edge-recovery sequences in every run so the model sees the correct response to near-fall states
- Discard and re-record any run where a significant mistake was made rather than including the recovery in the dataset

**Status:** Must be managed during Phase 3 data collection.

---

### Risk 6 — Distribution shift (MEDIUM)
**Description:** The CNN only knows what it saw during demonstrations. In any situation that looks meaningfully different from the training data, the model's output is undefined — it may produce a confident but completely wrong action with no signal that it is out of distribution.

**Why it matters:** Behavioral cloning models fail silently. The model does not know it is confused and will not flag uncertainty — it will just do something wrong, often repeatedly at the same spot.

**Mitigation:**
- After deploying the agent, note exactly which platforms cause repeated failures and what the failure behavior is
- Record targeted additional demonstrations specifically at those failure points (5–10 focused runs approaching that platform correctly)
- Add targeted data to the existing dataset and fine-tune for ~10 epochs rather than retraining from scratch
- Iterate: record → add → fine-tune → redeploy → observe → repeat until the failure point is resolved

**Status:** Will emerge during Phase 4. Expect 2–3 iteration cycles before the failure rate drops significantly.

---

### Risk 7 — Depth Anything V2 inference latency (MEDIUM)
**Description:** Depth Anything V2 Small adds a neural inference step to every frame in the perception layer. On a gaming PC without a dedicated inference budget, this may exceed the 30ms per-frame budget required to remain in the 20fps CNN tier.

**Why it matters:** If depth inference is too slow for the CNN tier, it cannot be dropped silently — the scene state dictionary (see vision/CLAUDE.md) will be missing depth signals, and any downstream logic that consumes those signals will break or produce undefined behavior.

**Mitigation:**
- Benchmark Depth Anything V2 Small on target hardware before integrating into the live loop
- If inference exceeds 30ms per frame, move depth estimation to the Gemini tier (called every 1.5–2 seconds as part of the high-level planner's context) rather than running per-frame
- If moved to the Gemini tier, the scene state must still be valid at CNN-tier call frequency — depth fields should carry the last known value with a staleness timestamp rather than blocking

**Threshold:** 30ms per frame (hard). Above this, depth estimation must not run in the CNN tier.

**Status:** Unvalidated — must be benchmarked on target hardware before Phase 2 integration.

---

## Open decisions
These have not been finalized and must be resolved before the relevant phase begins:

- Final choice between easyocr and pytesseract (resolve in Phase 1)
- Exact value of N seconds for stuck detection timeout (resolve before Phase 5)
- Whether to use PPO or DQN as the primary RL algorithm (resolve before Phase 5, default to PPO)
- Target hardware spec — affects CNN inference speed validation in Phase 4
