import sys
import torch
import random
from copy import deepcopy
from typing import List, Tuple
import tqdm

sys.path.append('..')

from mforl.function.grid_world_model import GridWorldModel, PolicyTabular
from mforl.function.basic import State, Action, Reward


GAMMA = torch.tensor(0.9, dtype=torch.float32)  # discount factor
EPISODES_NUM = 100  # number of episodes to train
TRAJECTORY_LENGTH = 1000  # length of trajectory in each episode
ITERATION_NUM = 100  # iteration number each episode
BATCH_SIZE = 10  # number of samples in each batch for training
UPDATE_FREQUENCY = 5  # number of iterations to let w_t <- w
LEARNING_RATE = 0.01  # learning rate


grid = GridWorldModel(
    width=3,
    height=3,
    forbidden_states=[State((0, 1)), State((0, 2)), State((2, 1))],
    terminal_states=[State((2, 2))],
    r_forbidden=Reward(torch.tensor(-10.0))
)

ACTIONS = {
    'up': grid.ACTION_UP,
    'down': grid.ACTION_DOWN,
    'left': grid.ACTION_LEFT,
    'right': grid.ACTION_RIGHT,
    'stay': grid.ACTION_STAY
}

policy_behavior = PolicyTabular(grid.states, grid.actions)
policy_behavior.fill_uniform()

policy_target = PolicyTabular(grid.states, grid.actions)
policy_target.fill_uniform()

print(grid)


class QNetwork(torch.nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


net = QNetwork()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)


def feature_extractor(state: State, action: Action) -> torch.Tensor:
    x, y = state.value
    # normalize
    x_norm = x / (grid.width - 1) if grid.width > 1 else 0
    y_norm = y / (grid.height - 1) if grid.height > 1 else 0

    return torch.tensor(
        [x_norm, y_norm, action.value[0], action.value[1]],
        dtype=torch.float32,
        device=next(net.parameters()).device
    )


def calculate_target_q(
        net_target: torch.nn.Module,
        state_next: State,
        reward: Reward,
        gamma: float
) -> torch.Tensor:
    with torch.no_grad():
        features = torch.stack([
            feature_extractor(state_next, action) for action in grid.actions
        ])
        q_values = net_target(features).squeeze()
        max_q = torch.max(q_values)
        return reward.value + gamma * max_q


def update_policy_greedy(network: torch.nn.Module, policy: PolicyTabular) -> None:
    for current_state in grid.states:
        q_values = []
        for action in grid.actions:
            q_value = network(feature_extractor(current_state, action))
            q_values.append((q_value.item(), action))

        # find best action
        best_action = max(q_values, key=lambda x: x[0])[1]

        # update policy
        for action in grid.actions:
            probability = 1.0 if action == best_action else 0.0
            policy[action | current_state] = torch.tensor(
                probability, dtype=torch.float32
            )


# Deep Q-Learning
for episode_idx in tqdm.trange(EPISODES_NUM, desc="Training Episodes"):
    current_state = random.choice(grid.states)
    current_action = policy_behavior.decide(current_state)

    # collect trajectory
    trajectory: List[Tuple[State, Action, Reward, State]] = []
    for _ in range(TRAJECTORY_LENGTH):
        next_state, reward = grid.step(current_state, current_action)
        trajectory.append((current_state, current_action, reward, next_state))

        current_state = next_state
        current_action = policy_behavior.decide(current_state)

    # let w_T = w
    net_target = deepcopy(net)


    for iteration in range(ITERATION_NUM):
        # sample
        if len(trajectory) >= BATCH_SIZE:
            batch_samples = random.sample(trajectory, BATCH_SIZE)
        else:
            batch_samples = trajectory

        losses = []
        optimizer.zero_grad()

        for state, action, reward, state_next in batch_samples:
            target_q = calculate_target_q(net_target, state_next, reward, GAMMA)

            features = feature_extractor(state, action)
            predicted_q = net(features)

            loss = torch.pow(target_q - predicted_q, 2)
            losses.append(loss)

        if losses:
            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            optimizer.step()

        # let w_T = w every C iterations
        if iteration % UPDATE_FREQUENCY == 0:
            net_target.load_state_dict(net.state_dict())


update_policy_greedy(net, policy_target)

# print final policy
print("\nFinal Policy:")
for state in grid.states:
    for action in grid.actions:
        prob = policy_target[action | state]
        if prob > 0:
            print(f"π({action}|{state}) = {prob.item():.3f}")
