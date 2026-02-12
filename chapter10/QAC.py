import sys

sys.path.append('..')

from mforl.function.grid_world_model import GridWorldModel, PolicyTabular
from mforl.function.basic import State, Action, Reward
import torch
import random
from copy import deepcopy
from typing import List, Tuple, Dict
import tqdm
import torch.nn.functional as F


EPISODES_NUM = 10000
TRAJECTORY_LENGTH = 100
LEARNING_RATE = 0.001


# model
grid = GridWorldModel(
    width=3,
    height=3,
    forbidden_states=[State((0, 1)), State((0, 2)), State((2, 1))],
    terminal_states=[State((2, 2))],
    gamma=torch.tensor(0.9),
    r_boundary=Reward(torch.tensor(-1.0)),
    r_forbidden=Reward(torch.tensor(-10.0)),
    r_terminal=Reward(torch.tensor(1.0)),
    r_other=Reward(torch.tensor(0.0))

)

print(grid)


ACTION_TO_INDEX_DICT = {
    grid.ACTION_UP: 0,
    grid.ACTION_DOWN: 1,
    grid.ACTION_LEFT: 2,
    grid.ACTION_RIGHT: 3,
    grid.ACTION_STAY: 4,
}


class PolicyNetwork(torch.nn.Module):
    ACTIONS = [grid.ACTION_UP, grid.ACTION_DOWN, grid.ACTION_LEFT,
               grid.ACTION_RIGHT, grid.ACTION_STAY]

    def __init__(self, input_dim=2, hidden_dim=8, output_dim=5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        return F.softmax(output, dim=-1)

    def get_log_prob(self, state: State, action: Action) -> torch.Tensor:
        features = self.feature_extractor(state)
        output = self.net(features)
        log_probs = F.log_softmax(output, dim=-1)
        return log_probs[ACTION_TO_INDEX_DICT[action]]

    def decide(self, state: State) -> Action:
        with torch.no_grad():
            features = self.feature_extractor(state)
            probs = self(features)
            action_index = torch.multinomial(probs, num_samples=1).item()
            return self.ACTIONS[action_index]

    def prob(self, state: State, action:Action) -> torch.Tensor:
        with torch.no_grad():
            features = self.feature_extractor(state)
            probs = self(features)
            return probs[ACTION_TO_INDEX_DICT[action]]

    @staticmethod
    def feature_extractor(state: State) -> torch.Tensor:
        x, y = state.value
        # normalize
        x_norm = x / (grid.width - 1)
        y_norm = y / (grid.height - 1)

        return torch.tensor(
            [x_norm, y_norm],
            dtype=torch.float32
        )


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

    def q(self, state: State, action: Action) -> torch.Tensor:
        features = self.feature_extractor(state, action)
        return self(features)

    @staticmethod
    def feature_extractor(state: State, action: Action) -> torch.Tensor:
        x, y = state.value
        # normalize
        x_norm = x / (grid.width - 1) if grid.width > 1 else 0
        y_norm = y / (grid.height - 1) if grid.height > 1 else 0

        return torch.tensor(
            [x_norm, y_norm, action.value[0], action.value[1]],
            dtype=torch.float32
        )

policy = PolicyNetwork()
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

q_net = QNetwork()
q_net_optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()


# train of QAC
for episode_idx in tqdm.trange(EPISODES_NUM, desc="Training Episodes"):
    current_state = random.choice(grid.states)

    for step_idx in range(TRAJECTORY_LENGTH):
        if current_state in grid.terminal_states:
            break

        current_action = policy.decide(current_state)
        next_state, reward = grid.step(current_state, current_action)

        q_value = q_net.q(current_state, current_action)

        with torch.no_grad():
            if next_state in grid.terminal_states:
                target = torch.tensor([reward.value], dtype=torch.float32)
            else:
                next_action = policy.decide(next_state)
                next_q = q_net.q(next_state, next_action)
                target = reward.value + grid.gamma * next_q

        # actor
        log_prob = policy.get_log_prob(current_state, current_action)
        policy_loss = -log_prob * q_value.detach()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        policy_optimizer.step()

        # critic
        # td_error = reward.value + grid.gamma * q_net.q(next_state, next_action).detach() - q_net.q(current_state, current_action)
        target = target.detach()
        critic_loss = criterion(q_value, target)
        q_net_optimizer.zero_grad()
        critic_loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
        q_net_optimizer.step()

        current_state = next_state




# print final policy
print("\nFinal Policy:")
for state in grid.states:
    print(f"State {state.value}:")
    for action in grid.actions:
        prob = policy.prob(state, action)
        print(f"    Action {action.value}:  {prob.item()}")
