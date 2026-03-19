# training/CLAUDE.md — Reward function and RL training

## Responsibility
This module defines the reward function, stuck detection logic, PPO configuration, and training loop. It does not handle perception, input, or the game loop directly.

---

## Reward function
The reward at each step is the sum of all applicable signals below.

### Checkpoint reward (primary signal)
Triggered when `current_stage` increases.

| Stage range | Reward per checkpoint |
|---|---|
| 1 – 25 | +1.0 |
| 26 – 50 | +1.5 |
| 51 – 100 | +2.0 |
| 101+ | +2.5 |

### Milestone bonus
Every 25 stages passed: additional +3.0 on top of the checkpoint reward.
Milestones: stage 25, 50, 75, 100, 125, ...

### Death penalty
-1.0 when `death_event` is True. Applied once per death, not per frame.

### Step penalty
-0.001 per frame on every step. Keeps the agent moving and prevents dawdling.
Do not increase this value — if too large, the agent learns to die immediately rather than struggle through hard sections.

### Stuck penalty
-0.5 when stuck detection fires. See stuck detection section below.

### Total reward formula
```
R = checkpoint_reward + milestone_bonus - death_penalty - step_penalty - stuck_penalty
```

---

## Stuck detection
If `current_stage` has not increased in the last N seconds, the agent is considered stuck.

- Default N = 30 seconds. Tune based on observed play behavior.
- On stuck detection: apply -0.5 penalty, set `truncated = True` in the gym env (forces episode reset)
- This prevents the agent from finding a safe bouncing spot to avoid the step penalty

Implementation:
```python
if (current_time - last_stage_increase_time) > STUCK_TIMEOUT:
    stuck = True
```

Store `last_stage_increase_time` and update it every time `current_stage` increases.

---

## PPO configuration (Stable Baselines3)
Use PPO as the default algorithm. Do not switch to DQN unless PPO fails to converge after 50+ episodes.

```python
from stable_baselines3 import PPO

model = PPO(
    policy="CnnPolicy",
    env=obby_env,
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=32,
    n_epochs=4,
    gamma=0.99,
    clip_range=0.1,
    verbose=1,
)
model.learn(total_timesteps=1_000_000)
```

These are starting values. Adjust learning_rate and clip_range if training is unstable.

---

## Episode logging
Log the following at the end of each episode:
- Episode number
- Total reward
- Max stage reached
- Number of deaths
- Episode length in steps
- Whether episode ended via death or stuck timeout

Use these logs to validate that mean episode reward increases over time (Phase 5 gate condition).

---

## Curriculum learning

### Goal
Reduce wasted episode time on early stages the agent has already mastered by starting each episode near the agent's current hardest section.

### Death cluster tracking
- At the end of each episode, record the stage number where the agent died.
- Maintain a rolling window of the last 10 episode death stages.
- The **death cluster stage** is the mode of that window (most frequent death stage). Break ties by taking the lower stage number.

### Curriculum start point
- Set the respawn starting checkpoint to `death_cluster_stage - 1`.
- Do not set the start point below stage 1.
- The start point is applied at the beginning of each episode reset.

### Advancing the curriculum
- Track the agent's **max stage reached** across the last 10 episodes.
- When `max_stage_last_10` improves (strictly greater than the previous window's value), move the curriculum start point back by 1 stage toward stage 1.
- This gradually reintroduces earlier stages as the agent's competence expands.

### Regression reset
- Track mean episode reward across the last 3 episodes.
- If mean episode reward drops for 3 consecutive episodes (each episode's mean reward is strictly less than the previous episode's), reset the curriculum start point to stage 1.
- This prevents the agent from getting stuck in a local optimum caused by skipping early stages.

### Implementation notes
```python
# At episode end:
death_history.append(current_death_stage)   # rolling window, maxlen=10
reward_history.append(episode_total_reward) # rolling window, maxlen=3

death_cluster = mode(death_history)
curriculum_start = max(1, death_cluster - 1)

if len(reward_history) == 3 and all(
    reward_history[i] < reward_history[i - 1]
    for i in range(1, len(reward_history))
):
    curriculum_start = 1
```

---

## Training expectations
Training will be slow due to the Roblox reset delay (see RISKS.md). Expect:
- 10-30 seconds per episode reset
- Meaningful improvement visible after 200-500 episodes minimum
- Do not conclude the reward function is broken until at least 100 episodes have run
