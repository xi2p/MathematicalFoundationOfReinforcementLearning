import mforl.model
from mforl.basic import Action, State, Reward, Policy
import numpy as np


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

policy = Policy(grid.states, grid.actions)
policy.fill_uniform()


print(grid)


def calc_state_value(grid_world, _policy, iteration_limit) -> np.ndarray:
    """
    Calculate state value given a policy.
    :param grid_world: The GridWorldModel instance.
    :param _policy: Given policy.
    :param iteration_limit: Number of iterations for evaluation.
    :return: State value vector as np.ndarray.
    """
    P_pi = np.zeros((len(grid_world.states), len(grid_world.states)))

    for s in grid_world.states:
        for s_next in grid_world.states:
            prob = np.float32(0.0)  # p(s'|s)
            for a in grid_world.actions:
                prob += _policy.pi(a | s) * grid_world.p(s_next | (s, a))
            P_pi[s.uid - 1, s_next.uid - 1] = prob

    R_pi = np.zeros((len(grid_world.states)))
    for s in grid_world.states:
        expected_reward = np.float32(0.0)
        for a in grid_world.actions:
            for r in grid_world.rewards:
                expected_reward += _policy.pi(a | s) * grid_world.p(r | (s, a)) * r.value
        R_pi[s.uid - 1] = expected_reward

    V_pi_iter = np.zeros((len(grid_world.states)))
    for _ in range(iteration_limit):
        V_pi_iter = R_pi + grid_world.gamma * np.matmul(P_pi, V_pi_iter)

    return V_pi_iter

# Policy Iteration
# Guess initial value vector
v = np.zeros((len(grid.states)))

ITERATION_LIMIT = 100
for t in range(ITERATION_LIMIT):
    # policy evaluation
    v = calc_state_value(grid, policy, 1000)
    # policy improvement
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
                q_s_a += grid.p(s_next | (s, a)) * v[s_next.uid - 1] * grid.gamma

            q_list.append(q_s_a)
            a_list.append(a)

        # find max q value and update policy to be greedy
        max_q = max(q_list)
        max_index = q_list.index(max_q)
        for a in grid.actions:
            if a == a_list[max_index]:
                policy[a | s] = np.float32(1.0)
            else:
                policy[a | s] = np.float32(0.0)


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




