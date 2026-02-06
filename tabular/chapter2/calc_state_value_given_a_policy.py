import sys
sys.path.append('..')

import mforl.model
from mforl.basic import State, Policy
import numpy as np


# grid = mforl.model.GridWorldModel(
#     width=3,
#     height=3,
#     forbidden_states=[(0, 1), (0, 2), (2, 1)],
#     terminal_states=[(2, 2)]
# )

# model
grid = mforl.model.GridWorldModel(
    width=2,
    height=2,
    forbidden_states=[(1, 0)],
    terminal_states=[(1, 1)]
)

# action
action_up = grid.ACTION_UP
action_down = grid.ACTION_DOWN
action_left = grid.ACTION_LEFT
action_right = grid.ACTION_RIGHT
action_stay = grid.ACTION_STAY

# policy
# {
#     (State(1), action_up): np.float32(0.0),
#     (State(1), action_down): np.float32(1.0),
#     (State(1), action_left): np.float32(0.0),
#     (State(1), action_right): np.float32(0.0),
#     (State(1), action_stay): np.float32(0.0),
#
#     (State(2), action_up): np.float32(0.0),
#     (State(2), action_down): np.float32(1.0),
#     (State(2), action_left): np.float32(0.0),
#     (State(2), action_right): np.float32(0.0),
#     (State(2), action_stay): np.float32(0.0),
#
#     (State(3), action_up): np.float32(0.0),
#     (State(3), action_down): np.float32(0.0),
#     (State(3), action_left): np.float32(0.0),
#     (State(3), action_right): np.float32(1.0),
#     (State(3), action_stay): np.float32(0.0),
#
#     (State(4), action_up): np.float32(0.0),
#     (State(4), action_down): np.float32(0.0),
#     (State(4), action_left): np.float32(0.0),
#     (State(4), action_right): np.float32(0.0),
#     (State(4), action_stay): np.float32(1.0),
# }
policy = Policy(grid.states, grid.actions)
policy[action_down|State(1)] = np.float32(1.0)
policy[action_down|State(2)] = np.float32(1.0)
policy[action_right|State(3)] = np.float32(1.0)
policy[action_stay|State(4)] = np.float32(1.0)


print(grid)

# get the state transition probability matrix
P_pi = np.zeros((len(grid.states), len(grid.states)))

for s in grid.states:
    for s_next in grid.states:
        prob = np.float32(0.0) # p(s'|s)
        for a in grid.actions:
            prob += policy.pi(a|s) * grid.p(s_next|(s, a))
        P_pi[s.uid - 1, s_next.uid - 1] = prob

print(P_pi)

# get the reward expectation vector
R_pi = np.zeros((len(grid.states)))
for s in grid.states:
    expected_reward = np.float32(0.0)
    for a in grid.actions:
        for r in grid.rewards:
            expected_reward += policy.pi(a|s) * grid.p(r|(s, a)) * r.value
    R_pi[s.uid - 1] = expected_reward

print(R_pi)

# compute the state value matrix V_pi = np.linalg.inv(np.eye(len(grid.states)) - 0.9 * P_pi).dot(R_pi)
V_pi = np.linalg.inv(np.eye(len(grid.states)) - grid.gamma * P_pi).dot(R_pi)
print(V_pi)

# compute the state value matrix V_pi iteratively
V_pi_iter = np.zeros((len(grid.states)))
for _ in range(1000):
    V_pi_iter = R_pi + grid.gamma * np.matmul(P_pi, V_pi_iter)
print(V_pi_iter)
