import sys
sys.path.append('..')

from mforl.function.grid_world_model import GridWorldModel, PolicyTabular
from mforl.function.basic import State, Action, Reward
import torch
import random
from copy import deepcopy
import tqdm


# model
grid = GridWorldModel(
    width=3,
    height=3,
    forbidden_states=[State((0, 1)), State((0, 2)), State((2, 1))],
    terminal_states=[State((2, 2))],
    r_forbidden=Reward(torch.tensor(-10.0))
)

# action
action_up = grid.ACTION_UP
action_down = grid.ACTION_DOWN
action_left = grid.ACTION_LEFT
action_right = grid.ACTION_RIGHT
action_stay = grid.ACTION_STAY

policy = PolicyTabular(grid.states, grid.actions)
policy.fill_uniform()


print(grid)

# Define the state value function
net = torch.nn.Sequential(
    torch.nn.Linear(4, 16),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(16, 1)
)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

def feature_extractor(state: State, action: Action) -> torch.Tensor:
    x, y = state.value
    # normalize x, y to [0, 1]
    x = x / (grid.width - 1)
    y = y / (grid.height - 1)

    features = torch.tensor([x, y, action.value[0], action.value[1]], dtype=torch.float32)

    return features



# Sarsa with function approximation

# Generate episodes
NUM_EPISODES = 100
EPISODE_LENGTH = 100
alpha = 0.01  # learning rate
epsilon = 0.2
episodes = []

for episode_idx in tqdm.trange(NUM_EPISODES, desc="Training Episodes"):
    current_state = random.choice(grid.states)
    current_action = policy.decide(current_state)


    while not current_state in grid.terminal_states:

        next_state, reward = grid.step(current_state, current_action)
        next_action = policy.decide(next_state)

        # Update parameters w
        optimizer.zero_grad()
        v_t = net(feature_extractor(current_state, current_action))
        v_t1 = net(feature_extractor(next_state, next_action)).detach()  # detach to prevent backprop through next state
        td_error = torch.pow(reward.value + grid.gamma * v_t1 - v_t, 2)
        td_error.backward()
        optimizer.step()

        # update policy to be epsilon-greedy
        max_q = -torch.inf
        best_action = None
        for a in grid.actions:
            q_value = net(feature_extractor(current_state, a))
            if q_value > max_q:
                max_q = q_value
                best_action = a

        if best_action is None:
            best_action = random.choice(grid.actions)

        for a in grid.actions:
            if a == best_action:
                policy[a | current_state] = torch.tensor(1.0 - epsilon + (epsilon / len(grid.actions)),
                                                         dtype=torch.float32)
            else:
                policy[a | current_state] = torch.tensor(epsilon / len(grid.actions),
                                                         dtype=torch.float32)

        current_state = next_state
        current_action = next_action




# print final policy
print("Final Policy:")
for s in grid.states:
    for a in grid.actions:
        prob = policy[a | s]
        if prob > 0:
            print(f"pi({a}|{s}) = {prob}")


