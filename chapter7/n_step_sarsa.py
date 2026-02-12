import sys
sys.path.append('..')

from mforl.tabular.model import GridWorldModel
from mforl.tabular.basic import Action, State, Reward, Policy
import numpy as np
import random
from copy import deepcopy


GAMMA = np.float32(0.9)
ITERATION_LIMIT = 100
SAMPLE_LENGTH = 1000
alpha = 0.1     # learning rate
epsilon = 0.1
n = 5   # n-step


# model
grid = GridWorldModel(
    width=3,
    height=3,
    forbidden_states=[(0, 1), (0, 2), (2, 1)],
    terminal_states=[(2, 2)],
    r_boundary=Reward(np.float32(-1.0)),
    r_forbidden=Reward(np.float32(-10.0)),
    r_terminal=Reward(np.float32(1.0)),
    r_other=Reward(np.float32(0.0)),
)


policy = Policy(grid.states, grid.actions)
policy.fill_uniform()


print(grid)

# n step sarsa learning of action value, and policy improvement

# Episode generation
q_dict: dict[tuple[State, Action], np.float32] = {}
for __i in range(ITERATION_LIMIT):
    current_state = random.choice(grid.states)
    current_action = policy.decide(current_state)

    n_step_trajectory = []

    for _ in range(n-1):
        next_state, reward = grid.step(current_state, current_action)
        n_step_trajectory.append((current_state, current_action, reward))
        current_state = next_state
        current_action = policy.decide(current_state)

    for _ in range(SAMPLE_LENGTH):
        next_state, reward = grid.step(current_state, current_action)
        n_step_trajectory.append((current_state, current_action, reward))
        # TD update
        key_to_update = n_step_trajectory[0][0], n_step_trajectory[0][1]  # (state, action)
        key_current = (current_state, current_action)
        if key_to_update not in q_dict:
            q_dict[key_to_update] = np.float32(0.0)
        if key_current not in q_dict:
            q_dict[key_current] = np.float32(0.0)

        # compute n-step return
        g = np.float32(0.0)
        for s, a, r in n_step_trajectory[::-1]:
            g = r.value + GAMMA * g

        g += GAMMA ** n * q_dict[key_current]

        q_dict[key_to_update] += alpha * (g - q_dict[key_to_update])

        # update policy to be epsilon-greedy
        max_q = -np.inf
        best_action = None
        for a in grid.actions:
            key = (key_to_update[0], a)
            q_value = q_dict.get(key, -np.inf)
            if q_value > max_q:
                max_q = q_value
                best_action = a

        if best_action is None:
            best_action = random.choice(grid.actions)

        for a in grid.actions:
            if a == best_action:
                policy[a | key_to_update[0]] = np.float32(1.0 - epsilon + (epsilon / len(grid.actions)))
            else:
                policy[a | key_to_update[0]] = np.float32(epsilon / len(grid.actions))


        n_step_trajectory.pop(0)
        current_state = next_state
        current_action = policy.decide(current_state)
        
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
v = np.linalg.inv(np.eye(len(grid.states)) - GAMMA * P_pi).dot(R_pi)
for s in grid.states:
    print(f"v({s}) = {v[s.uid - 1]}")
