import sys

sys.path.append('')

from mforl.grid_world_model import GridWorldModel, PolicyTabular
from mforl.basic import State, Action, Reward
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

policy_behavior = PolicyTabular(grid.states, grid.actions)
policy_behavior.fill_uniform()

policy_target = PolicyTabular(grid.states, grid.actions)
policy_target.fill_uniform()


print(grid)

# Define the state value function
net = torch.nn.Sequential(
    torch.nn.Linear(4, 16),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(16, 16),
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


# Q learning with function approximation

# Generate episodes
EPISODES_NUM = 100      # number of episodes to train
TRAJECTORY_LENGTH = 1000   # length of trajectory in each episode
ITERATION_NUM = 100     # iteration number each episode
batch = 10              # number of samples in each batch for training
C = 5                   # number of iterations to let w_t <- w
alpha = 0.01  # learning rate


for episode_idx in tqdm.trange(EPISODES_NUM, desc="Training Episodes"):
    current_state = random.choice(grid.states)
    current_action = policy_behavior.decide(current_state)

    trajectory = []
    for _ in range(TRAJECTORY_LENGTH):
        next_state, reward = grid.step(current_state, current_action)
        trajectory.append((current_state, current_action, reward, next_state))

        next_action = policy_behavior.decide(next_state)
        current_state = next_state
        current_action = next_action


    net_t = deepcopy(net)
    i = 0

    for _ in range(ITERATION_NUM):
        batch_samples = random.sample(trajectory, min(batch, len(trajectory)))

        net.zero_grad()

        for s, a, r, s_ in batch_samples:
            # calculate y_target
            y_target = r.value + grid.gamma * torch.max(
                torch.tensor(
                    [net_t(feature_extractor(s_, a_)).detach() for a_ in grid.actions]
                )
            )

            y_pred = net(feature_extractor(s, a))

            loss = torch.pow(y_target - y_pred, 2) / batch

            loss.backward()

        optimizer.step()

        i += 1

        if i % C == 0:
            net_t.load_state_dict(net.state_dict())
            i = 0





# update policy to be greedy
for current_state in grid.states:
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
            policy_target[a | current_state] = torch.tensor(1.0,
                                                     dtype=torch.float32)
        else:
            policy_target[a | current_state] = torch.tensor(0.0,
                                                     dtype=torch.float32)



# print final policy
print("Final Policy:")
for s in grid.states:
    for a in grid.actions:
        prob = policy_target[a | s]
        if prob > 0:
            print(f"pi({a}|{s}) = {prob}")
