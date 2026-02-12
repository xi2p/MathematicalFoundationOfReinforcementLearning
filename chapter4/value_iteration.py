import sys
sys.path.append('..')

from mforl.tabular.model import GridWorldModel
from mforl.tabular.basic import Policy
import numpy as np
from copy import deepcopy


GAMMA = np.float32(0.9)
ITERATION_LIMIT = 100


# model
grid = GridWorldModel(
    width=2,
    height=2,
    forbidden_states=[(1, 0)],
    terminal_states=[(1, 1)]
)


policy = Policy(grid.states, grid.actions)
policy.fill_uniform()


print(grid)

# Value Iteration
# Guess initial value vector
v = np.zeros((len(grid.states)))

for t in range(ITERATION_LIMIT):
    v_next = deepcopy(v)
    policy_next = deepcopy(policy)

    for s in grid.states:
        # update policy at every state by choosing max action value
        q_list = []
        a_list = []
        for a in grid.actions:
            # calculate q(s,a)
            q_s_a = np.float32(0.0)
            # first calculate reward part
            for r in grid.rewards:
                q_s_a += grid.p(r | (s, a)) * r.value

            # then calculate value part
            for s_next in grid.states:
                q_s_a += grid.p(s_next | (s, a)) * v[s_next.uid - 1] * GAMMA

            q_list.append(q_s_a)
            a_list.append(a)

        # find max q value and update v(s)
        max_q = max(q_list)
        v_next[s.uid - 1] = max_q
        # update policy to be greedy
        max_index = q_list.index(max_q)
        for a in grid.actions:
            if a == a_list[max_index]:
                policy_next[a | s] = np.float32(1.0)
            else:
                policy_next[a | s] = np.float32(0.0)
    v = v_next
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




