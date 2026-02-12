import sys
sys.path.append('..')

from typing import Tuple, Dict, List

from mforl.tabular.model import GridWorldModel
from mforl.tabular.basic import Action, State, Reward, Policy
import numpy as np


GAMMA = np.float32(0.9)
ITERATION_LIMIT = 50
SAMPLE_LENGTH = 20000
EPSILON = 0.2


# model
grid = GridWorldModel(
    width=5,
    height=5,
    forbidden_states=[(1, 1), (2, 1), (2, 2), (1, 3), (3, 3), (1, 4)],
    terminal_states=[(2, 3)],
    r_boundary=Reward(np.float32(-1.0)),
    r_forbidden=Reward(np.float32(-10.0)),
    r_terminal=Reward(np.float32(1.0)),
    r_other=Reward(np.float32(0.0)),
)


policy = Policy(grid.states, grid.actions)
policy.fill_uniform()


print(grid)

# MC epsilon-Greedy policy iteration

for t in range(ITERATION_LIMIT):
    # Episode generation
    trajectory : List[Tuple[State, Action, Reward]] = []
    state_action_counts : Dict[Tuple[State, Action], int] = {}

    current_state = State(1)
    i = 0
    # while i<SAMPLE_LENGTH or len(state_action_counts) < len(grid.states) * len(grid.actions):
    while i < SAMPLE_LENGTH:
        current_action = policy.decide(current_state)
        next_state, reward = grid.step(current_state, current_action)
        trajectory.append((current_state, current_action, reward))

        key = (current_state, current_action)
        if key not in state_action_counts:
            state_action_counts[key] = 0
        state_action_counts[key] += 1

        current_state = next_state
        i += 1
    print(f"Iteration {t+1}, generated {len(trajectory)} steps, visited {len(state_action_counts)} state-action pairs.")

    # print(trajectory)

    # Policy Evaluation
    q_dict : Dict[Tuple[State, Action], np.float32] = {}
    num_dict : Dict[Tuple[State, Action], int] = {}
    g = np.float32(0.0)
    for (s, a, r) in trajectory[::-1]:  # reverse order
        key = (s, a)
        if key not in q_dict:
            q_dict[key] = np.float32(0.0)
            num_dict[key] = 0
        g = r.value + GAMMA * g
        num_dict[key] += 1
        q_dict[key] += g

    for key in q_dict.keys():
        q_dict[key] /= num_dict[key]
    # print(q_dict)
    # Policy Improvement
    for s in grid.states:
        # find best action
        q_best = None
        a_best = None
        for a in grid.actions:
            key = (s, a)
            if key in q_dict:
                q_s_a = q_dict[key]
                if (q_best is None) or (q_s_a > q_best):
                    q_best = q_s_a
                    a_best = a

        # update policy to be epsilon-greedy
        if a_best is not None:
            for a in grid.actions:
                if a == a_best:
                    policy.policy_dict[(s, a)] = np.float32(1.0 - EPSILON + EPSILON / len(grid.actions))
                else:
                    policy.policy_dict[(s, a)] = np.float32(EPSILON / len(grid.actions))

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
