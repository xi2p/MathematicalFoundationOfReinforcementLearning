import sys
sys.path.append('..')

from mforl.function.grid_world_model import GridWorldModel, PolicyTabular
from mforl.function.basic import State, Action, Reward
import torch
import random
from copy import deepcopy


# model
grid = GridWorldModel(
    width=3,
    height=3,
    forbidden_states=[State((0, 1)), State((0, 2)), State((2, 1))],
    terminal_states=[State((2, 2))]
)

# action
action_up = grid.ACTION_UP
action_down = grid.ACTION_DOWN
action_left = grid.ACTION_LEFT
action_right = grid.ACTION_RIGHT
action_stay = grid.ACTION_STAY

policy = PolicyTabular(grid.states, grid.actions)
policy[action_right | State((0, 0))] = torch.tensor(1.0)
policy[action_down | State((1, 0))] = torch.tensor(1.0)
policy[action_left | State((2, 0))] = torch.tensor(1.0)
policy[action_right | State((0, 1))] = torch.tensor(1.0)
policy[action_down | State((1, 1))] = torch.tensor(1.0)
policy[action_left | State((2, 1))] = torch.tensor(1.0)
policy[action_right | State((0, 2))] = torch.tensor(1.0)
policy[action_right | State((1, 2))] = torch.tensor(1.0)
policy[action_stay | State((2, 2))] = torch.tensor(1.0)


print(grid)

# Define the state value function
# Use polygonal linear function approximation
order = 10   # Take 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3 as features
# parameter
# w = torch.zeros(((order+1)**2,), dtype=torch.float32)
w = torch.zeros(((order+1)*(order+2)//2,), dtype=torch.float32)

def feature_extractor(state: State) -> torch.Tensor:
    x, y = state.value
    # normalize x, y to [0, 1]
    x = x / (grid.width - 1)
    y = y / (grid.height - 1)

    # features = torch.tensor([
    #     1.0,
    #     x,
    #     y,
    #     x**2,
    #     x*y,
    #     y**2,
    #     x**3,
    #     x**2 * y,
    #     x * y**2,
    #     y**3
    # ], dtype=torch.float32)

    # Polygonal feature
    tensor = []
    for o in range(0, order+1):
        for i in range(o+1):
            tensor.append(x**i * y**(o-i))

    # Fourier basis features
    # cos(c1 x + c2 y)
    # tensor = []
    # for i in range(order + 1):
    #     for j in range(order + 1):
    #         tensor.append(torch.cos(torch.tensor((i * x + j * y) * 3.1415926)))
    features = torch.tensor(tensor, dtype=torch.float32)

    return features

def net(input: torch.Tensor) -> torch.Tensor:
    return torch.dot(w, input)



# Use TD learning to optimize state value function

# Generate episodes
NUM_EPISODES = 1000
EPISODE_LENGTH = 1000
alpha = 0.01  # learning rate
episodes = []

for episode_idx in range(NUM_EPISODES):
    episode = []
    # random choice of starting state
    current_state = random.choice(grid.states)

    for t in range(EPISODE_LENGTH):
        current_action = policy.decide(current_state)
        next_state, reward = grid.step(current_state, current_action)
        episode.append((current_state, current_action, reward))
        current_state = next_state
    episodes.append(episode)

    # TD learning

    for t in range(len(episode)-1):
        state_t, action_t, reward_t = episode[t]
        state_t1, action_t1, reward_t1 = episode[t+1]
        # print(f"t: {t}", end=", ")
        # print(f"state_t: {state_t}, action_t: {action_t}, reward_t: {reward_t}", end=", ")
        # print(f"state_t1: {state_t1}", end=", ")


        delta = alpha * (reward_t.value + grid.gamma * feature_extractor(state_t1).dot(w) - feature_extractor(state_t).dot(w)) * feature_extractor(state_t)
        # print(f"delta: {delta}")
        w += delta

    print("Episode {} completed.".format(episode_idx + 1))

# Print learned state values
for state in grid.states:
    state_value = net(feature_extractor(state)).item()
    print(f"State: {state}, Learned Value: {state_value:.4f}")

print("Learned parameters w:", w)
