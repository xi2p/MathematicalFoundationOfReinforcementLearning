"""
The action that the deterministic policy network outputs is a continuous vector,
whereas the action space of the grid world environment is discrete.

This code is not a good implementation of DPG, so I make all the code in this file as a comment.
"""

# import sys
#
# sys.path.append('..')
#
# from mforl.function.grid_world_model import GridWorldModel, PolicyTabular
# from mforl.function.basic import State, Action, Reward
# import torch
# import random
# from copy import deepcopy
# from typing import List, Tuple, Dict
# import tqdm
# import torch.nn.functional as F
#
# EPISODES_NUM = 10000
# TRAJECTORY_LENGTH = 2000
# LEARNING_RATE = 0.001
#
# # model
# grid = GridWorldModel(
#     width=3,
#     height=3,
#     forbidden_states=[State((0, 1)), State((0, 2)), State((2, 1))],
#     terminal_states=[State((2, 2))],
#     gamma=torch.tensor(0.9),
#     r_boundary=Reward(torch.tensor(-1.0)),
#     r_forbidden=Reward(torch.tensor(-10.0)),
#     r_terminal=Reward(torch.tensor(1.0)),
#     r_other=Reward(torch.tensor(0.0))
#
# )
#
# print(grid)
#
# ACTION_TO_INDEX_DICT = {
#     grid.ACTION_UP: 0,
#     grid.ACTION_DOWN: 1,
#     grid.ACTION_LEFT: 2,
#     grid.ACTION_RIGHT: 3,
#     grid.ACTION_STAY: 4,
# }
#
#
# class DeterministicPolicyNetwork(torch.nn.Module):
#     def __init__(self, input_dim=2, hidden_dim=8, output_dim=2):
#         super().__init__()
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, hidden_dim),
#             torch.nn.LeakyReLU(),
#             # torch.nn.Linear(hidden_dim, hidden_dim),
#             # torch.nn.LeakyReLU(),
#             torch.nn.Linear(hidden_dim, hidden_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(hidden_dim, output_dim),
#         )
#
#         for name, param in self.named_parameters():
#             if 'weight' in name:
#                 torch.nn.init.xavier_uniform_(param)
#             elif 'bias' in name:
#                 torch.nn.init.constant_(param, 0)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)
#
#     def decide(self, state: State) -> torch.Tensor:
#         return self.net(self.feature_extractor(state))
#
#     @staticmethod
#     def feature_extractor(state: State) -> torch.Tensor:
#         x, y = state.value
#         # normalize
#         x_norm = x / (grid.width - 1)
#         y_norm = y / (grid.height - 1)
#
#         return torch.tensor(
#             [x_norm, y_norm],
#             dtype=torch.float32
#         )
#
#
# class QNetwork(torch.nn.Module):
#     def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
#         super().__init__()
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, hidden_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(hidden_dim, hidden_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(hidden_dim, hidden_dim),
#             torch.nn.Linear(hidden_dim, 1)
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)
#
#     def q(self, state: State, action: Action) -> torch.Tensor:
#         features = self.feature_extractor(state, action)
#         return self(features)
#
#     @staticmethod
#     def feature_extractor(state: State, action: Action) -> torch.Tensor:
#         x, y = state.value
#         # normalize
#         x_norm = x / (grid.width - 1) if grid.width > 1 else 0
#         y_norm = y / (grid.height - 1) if grid.height > 1 else 0
#
#         return torch.tensor(
#             [x_norm, y_norm, action.value[0], action.value[1]],
#             dtype=torch.float32
#         )
#
# target_policy = DeterministicPolicyNetwork()
# target_policy_optimizer = torch.optim.Adam(target_policy.parameters(), lr=LEARNING_RATE)
# behavior_policy = PolicyTabular(
#     state_space=grid.states, action_space=grid.actions
# )
# behavior_policy.fill_uniform()  # take any action at any state with equal probability
#
# q_net = QNetwork()
# q_net_optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
# criterion = torch.nn.MSELoss()
#
# # train of DPG
# for episode_idx in tqdm.trange(EPISODES_NUM, desc="Training Episodes"):
#     current_state = random.choice(grid.states)
#
#     for i in range(TRAJECTORY_LENGTH):
#         if current_state in grid.terminal_states:
#             break
#
#         current_action = behavior_policy.decide(current_state)
#         next_state, reward = grid.step(current_state, current_action)
#
#         current_q = q_net.q(current_state, current_action)
#
#         if next_state in grid.terminal_states:
#             target_q = torch.tensor([reward.value], dtype=torch.float32)  # gamma * next_value = 0
#
#         else:
#             next_action = target_policy.decide(next_state)
#             # Note that the next action decided by target policy is only used for critic update,
#             # and the behavior policy is still used to generate the trajectory, which is off-policy learning.
#             next_q = q_net.q(next_state, Action(next_action.tolist()))
#             target_q = reward.value + grid.gamma * next_q
#
#         with torch.no_grad():
#             advantage = target_q - current_q
#
#         # actor
#         policy_loss = -q_net.q(current_state, Action(target_policy.decide(current_state).tolist()))
#         target_policy_optimizer.zero_grad()
#         policy_loss.backward()
#         torch.nn.utils.clip_grad_norm_(target_policy.parameters(), max_norm=1.0)
#         target_policy_optimizer.step()
#
#         # critic
#         critic_loss = criterion(target_q, current_q)  # MSE
#         q_net_optimizer.zero_grad()
#         critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
#         q_net_optimizer.step()
#
#         current_state = next_state
#
#
# # print final policy
# print("\nFinal Policy:")
# for state in grid.states:
#     if state in grid.terminal_states:
#         continue
#
#     print(f"State {state.value}:")
#     action = target_policy.decide(state)
#     print(f"  Action: {action.detach().numpy()}")