import sys
sys.path.append('..')

import mforl.model
from mforl.basic import Policy
import numpy as np
from copy import deepcopy


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
policy.fill_uniform()


print(grid)

# MC basic policy iteration

# Guess initial value vector
v = np.zeros((len(grid.states)))

ITERATION_LIMIT = 10
SAMPLE_LENGTH = 100
EPISODE_LIMIT = 100

for t in range(ITERATION_LIMIT):
    policy_next = deepcopy(policy)

    for s in grid.states:
        # policy evaluation
        q_list = []
        a_list = []
        for a in grid.actions:
            # estimate q(s,a)
            q_s_a = np.float32(0.0)

            for _ in range(EPISODE_LIMIT):
                total_reward = np.float32(0.0)
                discount = np.float32(1.0)
                current_state = s
                current_action = a

                for _ in range(SAMPLE_LENGTH):
                    # take action and observe next state and reward
                    next_state, reward = grid.step(current_state, current_action)

                    total_reward += discount * reward.value
                    discount *= grid.gamma

                    current_state = next_state
                    # follow the current policy to choose next action
                    current_action = policy.decide(current_state)

                q_s_a += total_reward / EPISODE_LIMIT


            q_list.append(q_s_a)
            a_list.append(a)

        # find max q value and update v(s)
        max_q = max(q_list)
        v[s.uid - 1] = max_q
        # update policy to be greedy
        max_index = q_list.index(max_q)
        for a in grid.actions:
            if a == a_list[max_index]:
                policy_next[a | s] = np.float32(1.0)
            else:
                policy_next[a | s] = np.float32(0.0)
    policy = policy_next
    print(f"Iteration {t+1}: v = {v}")

# print final policy
print("Final Policy:")
for s in grid.states:
    for a in grid.actions:
        prob = policy[a | s]
        if prob > 0:
            print(f"pi({a}|{s}) = {prob}")
print("Final State Values:")
for s in grid.states:
    print(f"v({s}) = {v[s.uid - 1]}")
