import sys
import torch
import torch.nn.functional as F
import random
from copy import deepcopy
from typing import List, Tuple
import tqdm

sys.path.append('..')

from mforl.function.grid_world_model import GridWorldModel, PolicyTabular
from mforl.function.basic import State, Action, Reward


GAMMA = torch.tensor(0.9, dtype=torch.float32)  # discount factor
EPISODES_NUM = 500  # number of episodes to train
TRAJECTORY_LENGTH = 200  # length of trajectory in each episode
LEARNING_RATE = 0.01  # learning rate


grid = GridWorldModel(
    width=3,
    height=3,
    forbidden_states=[State((0, 1)), State((0, 2)), State((2, 1))],
    terminal_states=[State((2, 2))],
    r_forbidden=Reward(torch.tensor(-10.0))
)

print(grid)

ACTION_TO_INDEX_DICT = {
    grid.ACTION_UP: 0,
    grid.ACTION_DOWN: 1,
    grid.ACTION_LEFT: 2,
    grid.ACTION_RIGHT: 3,
    grid.ACTION_STAY: 4,
}


def feature_extractor(state: State) -> torch.Tensor:
    x, y = state.value
    # normalize
    x_norm = x / (grid.width - 1) if grid.width > 1 else 0
    y_norm = y / (grid.height - 1) if grid.height > 1 else 0

    return torch.tensor(
        [x_norm, y_norm],
        dtype=torch.float32
    )


class PolicyNetwork(torch.nn.Module):
    ACTIONS = [grid.ACTION_UP, grid.ACTION_DOWN, grid.ACTION_LEFT, grid.ACTION_RIGHT, grid.ACTION_STAY]
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        return F.softmax(output, dim=-1)

    def get_log_prob(self, state: State, action: Action) -> torch.Tensor:
        features = feature_extractor(state)
        output = self.net(features)
        log_probs = F.log_softmax(output, dim=-1)
        return log_probs[ACTION_TO_INDEX_DICT[action]]

    def decide(self, state: State) -> Action:
        with torch.no_grad():
            features = feature_extractor(state)
            probs = self(features)
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum()
            action_index = torch.multinomial(probs, num_samples=1).item()
            return self.ACTIONS[action_index]

    def prob(self, state: State, action: Action) -> torch.Tensor:
        with torch.no_grad():
            features = feature_extractor(state)
            probs = self(features)
            return probs[ACTION_TO_INDEX_DICT[action]]


policy = PolicyNetwork()
optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)


# REINFORCE
for episode_idx in tqdm.trange(EPISODES_NUM, desc="Training Episodes"):
    current_state = random.choice(grid.states)
    current_action = policy.decide(current_state)

    # collect trajectory
    trajectory: List[Tuple[State, Action, Reward]] = []
    for _ in range(TRAJECTORY_LENGTH):
        next_state, reward = grid.step(current_state, current_action)
        trajectory.append((current_state, current_action, reward))

        current_state = next_state
        current_action = policy.decide(current_state)

    optimizer.zero_grad()

    losses = []
    for i in range(len(trajectory)-1):
        current_state, current_action, _ = trajectory[i]

        # estimate q(s, a)
        with torch.no_grad():
            q = torch.tensor(0.0, dtype=torch.float32).detach()
            discount = torch.tensor(1.0, dtype=torch.float32)
            for j in range(i, len(trajectory)):
                _, _, reward = trajectory[j]
                q += discount * reward.value
                discount *= GAMMA



        j = -policy.get_log_prob(current_state, current_action) * q
        losses.append(j)

    if losses:
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        optimizer.step()


# print final policy
print("\nFinal Policy:")
for state in grid.states:
    for action in grid.actions:
        prob = policy.prob(state, action)
        if prob > 0:
            print(f"π({action}|{state}) = {prob.item():.3f}")
