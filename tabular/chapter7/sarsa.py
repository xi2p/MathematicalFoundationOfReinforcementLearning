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

policy = Policy(grid.states, grid.actions)
policy.fill_uniform()


print(grid)

# TD learning of action value, and policy improvement

ITERATION_LIMIT = 100
SAMPLE_LENGTH = 1000
alpha = 0.1     # learning rate
epsilon = 0.1


# Episode generation
q_dict: dict[tuple[State, Action], np.float32] = {}
for __i in range(ITERATION_LIMIT):
    current_state = random.choice(grid.states)
    current_action = policy.decide(current_state)
    for _ in range(SAMPLE_LENGTH):
        next_state, reward = grid.step(current_state, current_action)
        next_action = policy.decide(next_state)
        # TD update

        key = (current_state, current_action)
        if key not in q_dict:
            q_dict[key] = np.float32(0.0)

        if (next_state, next_action) not in q_dict:
            q_dict[(next_state, next_action)] = np.float32(0.0)

        q_dict[key] += alpha * (reward.value + grid.gamma * q_dict[(next_state, next_action)] - q_dict[key])

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
                policy[a | current_state] = np.float32(1.0 - epsilon + (epsilon / len(grid.actions)))
            else:
                policy[a | current_state] = np.float32(epsilon / len(grid.actions))



        current_state = next_state
        current_action = next_action
    print(f"Iteration {__i} completed.")

# print final policy
print("Final Policy:")
for s in grid.states:
    for a in grid.actions:
        prob = policy[a | s]
        if prob > 0:
            print(f"pi({a}|{s}) = {prob}")

print("Final State Values:")
P_pi = grid.P_pi(policy)
R_pi = grid.R_pi(policy)
v = np.linalg.inv(np.eye(len(grid.states)) - grid.gamma * P_pi).dot(R_pi)
for s in grid.states:
    print(f"v({s}) = {v[s.uid - 1]}")
