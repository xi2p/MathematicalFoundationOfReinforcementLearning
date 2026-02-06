import sys
sys.path.append('..')

import mforl.model
from mforl.basic import Action, State, Reward, Policy
import numpy as np
import random

# model
grid = mforl.model.GridWorldModel(
    width=3,
    height=3,
    forbidden_states=[(0, 1), (0, 2), (2, 1)],
    terminal_states=[(2, 2)],
    r_boundary=Reward(np.float32(-1.0)),
    r_forbidden=Reward(np.float32(-10.0)),
    r_terminal=Reward(np.float32(1.0)),
    r_other=Reward(np.float32(0.0)),
)

# action
action_up = grid.ACTION_UP
action_down = grid.ACTION_DOWN
action_left = grid.ACTION_LEFT
action_right = grid.ACTION_RIGHT
action_stay = grid.ACTION_STAY

print(grid)

# Q-learning
policy_behavior = Policy(grid.states, grid.actions)
policy_behavior.fill_uniform()
policy_target = Policy(grid.states, grid.actions)
policy_target.fill_uniform()


ITERATION_LIMIT = 100
SAMPLE_LENGTH = 1000
alpha = 0.1     # learning rate


# Episode generation
q_dict: dict[tuple[State, Action], np.float32] = {}
for __i in range(ITERATION_LIMIT):
    # Use behavior policy to generate episodes
    trajectory : list[tuple[State, Action, Reward]] = []
    current_state = random.choice(grid.states)
    for _ in range(SAMPLE_LENGTH):
        current_action = policy_behavior.decide(current_state)
        next_state, reward = grid.step(current_state, current_action)
        trajectory.append((current_state, current_action, reward))
        current_state = next_state

    # Update q-values and target policy using the generated trajectory
    for i in range(len(trajectory)-1):
        current_state, current_action, reward = trajectory[i]
        next_state, _, _ = trajectory[i+1]
        key = (current_state, current_action)

        if key not in q_dict:
            q_dict[key] = np.float32(0.0)

        for a in grid.actions:
            if (next_state, a) not in q_dict:
                q_dict[(next_state, a)] = np.float32(0.0)

        # find max_a' Q(s', a')
        max_next_q = max(q_dict[(next_state, a)] for a in grid.actions)
        q_dict[key] += alpha * (reward.value + grid.gamma * max_next_q - q_dict[key])

        # update policy to be epsilon-greedy
        max_q = -np.inf
        best_action = None
        for a in grid.actions:
            key = (current_state, a)
            q_value = q_dict.get(key, -np.inf)
            if q_value > max_q:
                max_q = q_value
                best_action = a

        if best_action is None:
            best_action = random.choice(grid.actions)

        for a in grid.actions:
            if a == best_action:
                policy_target[a | current_state] = np.float32(1.0)
            else:
                policy_target[a | current_state] = np.float32(0.0)

    print(f"Iteration {__i} completed.")

# print final policy
print("Final Policy:")
for s in grid.states:
    for a in grid.actions:
        prob = policy_target[a | s]
        if prob > 0:
            print(f"pi({a}|{s}) = {prob}")

print("Final State Values:")
P_pi = grid.P_pi(policy_target)
R_pi = grid.R_pi(policy_target)
v = np.linalg.inv(np.eye(len(grid.states)) - grid.gamma * P_pi).dot(R_pi)
for s in grid.states:
    print(f"v({s}) = {v[s.uid - 1]}")
