import sys
sys.path.append('..')

import mforl.model
from mforl.basic import State, Policy
import numpy as np
import random

# model
grid = mforl.model.GridWorldModel(
    width=3,
    height=3,
    forbidden_states=[(0, 1), (0, 2), (2, 1)],
    terminal_states=[(2, 2)]
)

# action
action_up = grid.ACTION_UP
action_down = grid.ACTION_DOWN
action_left = grid.ACTION_LEFT
action_right = grid.ACTION_RIGHT
action_stay = grid.ACTION_STAY

policy = Policy(grid.states, grid.actions)

policy[action_right | State(1)] = np.float32(1.0)
policy[action_down | State(2)] = np.float32(1.0)
policy[action_left | State(3)] = np.float32(1.0)
policy[action_right | State(4)] = np.float32(1.0)
policy[action_down | State(5)] = np.float32(1.0)
policy[action_left | State(6)] = np.float32(1.0)
policy[action_right | State(7)] = np.float32(1.0)
policy[action_right | State(8)] = np.float32(1.0)
policy[action_stay | State(9)] = np.float32(1.0)


print(grid)

# TD learning of state value
# Given a policy, evaluate the state value vector v

# Guess initial value vector
v = np.zeros((len(grid.states)))

ITERATION_LIMIT = 1000
SAMPLE_LENGTH = 100
alpha = 0.1     # learning rate

for t in range(ITERATION_LIMIT):
    # Episode generation
    current_state = random.choice(grid.states)
    for _ in range(SAMPLE_LENGTH):
        current_action = policy.decide(current_state)
        next_state, reward = grid.step(current_state, current_action)
        # TD update
        v[current_state.uid - 1] += alpha * (reward.value + grid.gamma * v[next_state.uid - 1] - v[current_state.uid - 1])
        current_state = next_state


print("Final State Values:")
for s in grid.states:
    print(f"v({s}) = {v[s.uid - 1]}")
