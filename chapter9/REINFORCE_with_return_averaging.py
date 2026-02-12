"""
The REINFORCE algorithm introduced in the book use a single trajectory return to estimate
corresponding Q-value, which may lead to high variance in the policy gradient estimation.

Here I provide an alternative implementation that uses the average return of all visits
to (s, a) in the trajectory to estimate Q-value, which can reduce the variance of the estimation.

This implementation is NOT included in the book.
"""
import sys
import torch
import random
from typing import List, Tuple, Dict
import tqdm
import torch.nn.functional as F

sys.path.append('..')

from mforl.function.grid_world_model import GridWorldModel, PolicyTabular
from mforl.function.basic import State, Action, Reward


GAMMA = torch.tensor(0.9, dtype=torch.float32)  # discount factor
EPISODES_NUM = 2000
TRAJECTORY_LENGTH = 1000
LEARNING_RATE = 0.01


grid = GridWorldModel(
    width=5,
    height=5,
    forbidden_states=[State((1, 1)), State((2, 1)), State((2, 2)), State((1, 3)), State((3, 3)), State((1, 4))],
    terminal_states=[State((2, 3))],
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
    x_norm = x / (grid.width - 1) if grid.width > 1 else 0
    y_norm = y / (grid.height - 1) if grid.height > 1 else 0

    return torch.tensor(
        [x_norm, y_norm, 1.0],
        dtype=torch.float32
    )


class PolicyNetwork(torch.nn.Module):
    ACTIONS = [grid.ACTION_UP, grid.ACTION_DOWN, grid.ACTION_LEFT,
               grid.ACTION_RIGHT, grid.ACTION_STAY]

    def __init__(self, input_dim=3, hidden_dim=64, output_dim=5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Sigmoid(),
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


policy = PolicyNetwork(input_dim=3)
optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

# train
for episode_idx in tqdm.trange(EPISODES_NUM, desc="Training Episodes"):
    states = []
    actions = []
    rewards = []

    current_state = random.choice([s for s in grid.states if s not in grid.terminal_states])
    episode_ended = False

    # collect trajectory
    step_num = 0
    for t in range(TRAJECTORY_LENGTH):
        if episode_ended:
            break
        step_num += 1
        current_action = policy.decide(current_state)
        next_state, reward = grid.step(current_state, current_action)

        states.append(current_state)
        actions.append(current_action)
        rewards.append(reward)

        # check episode end
        if next_state in grid.terminal_states or step_num >= TRAJECTORY_LENGTH:
            episode_ended = True

        current_state = next_state

    # calculate return
    T = len(states)
    if T == 0:
        continue

    with torch.no_grad():
        q_dict : Dict[Tuple[State, Action], torch.Tensor] = {}
        num_dict : Dict[Tuple[State, Action], int] = {}

        g_t = torch.tensor(0.0, dtype=torch.float32)
        for t in range(T-1, -1, -1):
            g_t = g_t * GAMMA + rewards[t].value
            state = states[t]
            action = actions[t]
            if not (state, action) in q_dict:
                q_dict[(state, action)] = torch.tensor(0.0, dtype=torch.float32)
                num_dict[(state, action)] = 0
            q_dict[(state, action)] += g_t
            num_dict[(state, action)] += 1
        
        for key in q_dict:
            q_dict[key] /= num_dict[key]

    # update policy
    policy_loss = []
    for t in range(T):
        state = states[t]
        action = actions[t]
        Gt = q_dict[(state, action)]

        log_prob = policy.get_log_prob(state, action)

        loss = -log_prob * Gt
        policy_loss.append(loss)

    optimizer.zero_grad()
    total_loss = torch.stack(policy_loss).mean()
    total_loss.backward()

    # clip gradients
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

    optimizer.step()


# print final policy
print("\nFinal Policy:")
for state in grid.states:
    if state in grid.terminal_states:
        continue
    probs = []
    for action in grid.actions:
        prob = policy.prob(state, action)
        probs.append((action, prob.item()))

    if probs:
        print(f"State {state.value}:")
        for action, prob in probs:
            print(f"  {action}: {prob:.3f}")
