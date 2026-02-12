"""
@author: xi2p
"""

from typing import Dict, List, Tuple, Set
from copy import deepcopy
import torch
from .basic import State, Action, Reward, ConditionExpr

"""
State values are designed as follows:
- State value is a tuple (x, y), where x and y are the coordinates of the state in the grid world.
- The top-left corner of the grid world is (0, 0), and the bottom-right corner is (width-1, height-1).

Action values are designed as follows:
- ACTION_UP: Action((0, -1))
- ACTION_DOWN: Action((0, 1))
- ACTION_LEFT: Action((-1, 0))
- ACTION_RIGHT: Action((1, 0))
- ACTION_STAY: Action((0, 0))
In another word, the action value represents the change in coordinates when the action is taken.
For the sake of simplicity, we define the action values as integers here.
"""



"""
Grid World Model: A simple grid world environment where an agent only moves in 4 directions or stays still.
"""
class GridWorldModel:
    ACTION_UP = Action((0, -1))
    ACTION_DOWN = Action((0, 1))
    ACTION_LEFT = Action((-1, 0))
    ACTION_RIGHT = Action((1, 0))
    ACTION_STAY = Action((0, 0))

    def __init__(self, width: int, height: int, forbidden_states: List[State], terminal_states: List[State],
                 gamma: torch.Tensor = torch.tensor(0.9),
                 r_boundary: Reward = Reward(torch.tensor(-1.0)),
                 r_forbidden: Reward = Reward(torch.tensor(-1.0)),
                 r_terminal: Reward = Reward(torch.tensor(1.0)),
                 r_other: Reward = Reward(torch.tensor(0.0))
                 ):
        """
        Initialize the Grid World Model.
        :param width: The width of the grid world.
        :param height: The height of the grid world.
        :param forbidden_states: Forbidden states in the grid world.
        :param terminal_states: Terminal states in the grid world.
        :param gamma: Discount factor.
        :param r_boundary: Reward for hitting the boundary.
        :param r_forbidden: Reward for hitting a wall.
        :param r_terminal: Reward for reaching a terminal state.
        :param r_other: Reward for other transitions.
        """
        self.width = width
        self.height = height
        self.gamma = gamma
        self.r_boundary = r_boundary
        self.r_forbidden = r_forbidden
        self.r_terminal = r_terminal
        self.r_other = r_other

        self.states = self._initialize_states()
        self.forbidden_states = forbidden_states
        self.terminal_states = terminal_states

        self.actions = {GridWorldModel.ACTION_UP, GridWorldModel.ACTION_DOWN,
                        GridWorldModel.ACTION_LEFT, GridWorldModel.ACTION_RIGHT,
                        GridWorldModel.ACTION_STAY}

        self.rewards = {r_boundary, r_forbidden, r_terminal, r_other}

    def _initialize_states(self) -> List[State]:
        states = []
        for y in range(self.width):
            for x in range(self.height):
                states.append(State((x, y)))
        return states


    def step(self, state: State, action: Action) -> Tuple[State, Reward]:
        """
        take a step in the grid world given a state and an action
        :param state: The current state
        :param action: The action to take
        :return: A tuple of (next_state, reward)
        """
        x, y = state.value

        if action == GridWorldModel.ACTION_UP:
            intended_pos = (x, y - 1)
        elif action == GridWorldModel.ACTION_DOWN:
            intended_pos = (x, y + 1)
        elif action == GridWorldModel.ACTION_LEFT:
            intended_pos = (x - 1, y)
        elif action == GridWorldModel.ACTION_RIGHT:
            intended_pos = (x + 1, y)
        elif action == GridWorldModel.ACTION_STAY:
            intended_pos = (x, y)
        else:
            raise ValueError("Unknown action.")

        # Determine next state
        if not ((0 <= intended_pos[0] < self.width) and (0 <= intended_pos[1] < self.height)):
            next_state = state
            reward = self.r_boundary
        else:
            next_state = State(intended_pos)
            if next_state in self.forbidden_states:
                reward = self.r_forbidden
            elif next_state in self.terminal_states:
                reward = self.r_terminal
            else:
                reward = self.r_other
        return next_state, reward

    def __str__(self):
        """
        render the grid world as a string
        for forbidden states, use "■"; for terminal states, use "○"; for other states, use "□"
        """
        grid_str = ""
        for y in range(self.height):
            for x in range(self.width):
                state = State((x, y))
                if state in self.forbidden_states:
                    grid_str += "■ "
                elif state in self.terminal_states:
                    grid_str += "○ "
                else:
                    grid_str += "□ "
            grid_str += "\n"
        return grid_str

    def __repr__(self):
        return self.__str__()


class PolicyFunction:
    def __init__(self, net: torch.nn.Module):
        """
        Use a neural network to represent the policy.
        The last layer should be a softmax layer to output action probabilities under the [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY] order.
        :param net: The neural network representing the policy.
        """
        self.net = net

    def pi(self, input: torch.Tensor) -> torch.Tensor:
        """
        Given the input state, output the action probabilities.
        :param input: The input state tensor.
        :return: The action probabilities tensor.
        """
        return self.net(input)

    def decide(self, state: State) -> Action:
        """
        Given the state, decide the action to take.
        :param state: The current state.
        :return: The action to take.
        """
        x, y = state.value
        state_tensor = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0)
        action_probs = self.pi(state_tensor).squeeze(0).detach()
        # randomly choose action according to the probabilities
        action_index = torch.multinomial(action_probs, num_samples=1).item()

        if action_index == 0:
            return GridWorldModel.ACTION_UP
        elif action_index == 1:
            return GridWorldModel.ACTION_DOWN
        elif action_index == 2:
            return GridWorldModel.ACTION_LEFT
        elif action_index == 3:
            return GridWorldModel.ACTION_RIGHT
        elif action_index == 4:
            return GridWorldModel.ACTION_STAY
        else:
            raise ValueError("Invalid action index.")

    def get_action_prob(self, state, action) -> torch.Tensor:
        """
        Given the state and action, get the probability of taking that action in that state.
        :param state: The current state.
        :param action: The action to take.
        :return: The probability of taking that action in that state.
        """
        x, y = state.value
        state_tensor = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0)
        action_probs = self.pi(state_tensor).squeeze(0)
        action_probs = action_probs / torch.sum(action_probs)  # normalize

        if action == GridWorldModel.ACTION_UP:
            return action_probs[0]
        elif action == GridWorldModel.ACTION_DOWN:
            return action_probs[1]
        elif action == GridWorldModel.ACTION_LEFT:
            return action_probs[2]
        elif action == GridWorldModel.ACTION_RIGHT:
            return action_probs[3]
        elif action == GridWorldModel.ACTION_STAY:
            return action_probs[4]
        else:
            raise ValueError("Invalid action.")


class StateValueFunction:
    def __init__(self, net: torch.nn.Module):
        """
        Use a neural network to represent the state value function.
        :param net: The neural network representing the state value function.
        """
        self.net = net

    def v(self, input: torch.Tensor) -> torch.Tensor:
        """
        Given the input state, output the state value.
        :param input: The input state tensor.
        :return: The state value tensor.
        """
        return self.net(input)

    def get_state_value(self, state: State) -> torch.Tensor:
        """
        Given the state, get the state value.
        :param state: The current state.
        :return: The state value.
        """
        x, y = state.value
        state_tensor = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0)
        state_value = self.v(state_tensor).squeeze(0).detach()
        return state_value


"""
A PolicyTabular represents a policy in reinforcement learning using a table (dictionary) mapping state-action pairs to probabilities.
Since chapter 8 still uses tabular policies, we define TabularPolicy here.
"""
class PolicyTabular:
    def __init__(self, state_space: Set[State], action_space: Set[Action]):
        self.policy_dict : Dict[Tuple[State, Action], torch.Tensor] = {}
        self.state_space = state_space
        self.action_space = action_space

        # initialize policy with all zero probabilities
        for s in state_space:
            for a in action_space:
                self.policy_dict[(s, a)] = torch.tensor(0.0, dtype=torch.float32)

    def pi(self, condition_expr: ConditionExpr) -> torch.Tensor:
        """
        Get the probability of taking action a in state s under this policy.
        :param condition_expr: The ConditionExpr instance representing a|s.
        :return: The probability.
        """
        if isinstance(condition_expr, ConditionExpr):
            if isinstance(condition_expr.outcome, Action) and isinstance(condition_expr.condition, State):
                key = (condition_expr.condition, condition_expr.outcome)
                return self.policy_dict[key]
            else:
                raise ValueError("ConditionExpr must be of the form a|s.")
        else:
            raise ValueError("Input must be a ConditionExpr instance.")

    def fill_uniform(self):
        """
        Fill the policy with uniform distribution over actions for each state.
        """
        state_action_count : Dict[State, int] = {}
        for (s, a) in self.policy_dict.keys():
            if s not in state_action_count:
                state_action_count[s] = 0
            state_action_count[s] += 1

        for (s, a) in self.policy_dict.keys():
            self.policy_dict[(s, a)] = torch.tensor(1.0 / state_action_count[s])

    def decide(self, state: State) -> Action:
        """
        Decide an action based on the current policy for a given state.
        :param state: The current state.
        :return: An action chosen according to the policy.
        """
        actions = []
        probabilities = []

        for a in self.action_space:
            actions.append(a)
            probabilities.append(self.pi(a | state))

        probabilities_tensor = torch.stack(probabilities)
        probabilities_tensor = probabilities_tensor / torch.sum(probabilities_tensor)
        action_index = int(torch.multinomial(probabilities_tensor, num_samples=1).item())
        return actions[action_index]


    def __getitem__(self, condition_expr: ConditionExpr) -> torch.Tensor:
        return self.pi(condition_expr)

    def __setitem__(self, condition_expr: ConditionExpr, value: torch.Tensor):
        if isinstance(condition_expr, ConditionExpr):
            if isinstance(condition_expr.outcome, Action) and isinstance(condition_expr.condition, State):
                key = (condition_expr.condition, condition_expr.outcome)
                self.policy_dict[key] = value
            else:
                raise ValueError("ConditionExpr must be of the form a|s.")
        else:
            raise ValueError("Input must be a ConditionExpr instance.")

    def __deepcopy__(self, memodict={}):
        new_policy = PolicyTabular(self.state_space, self.action_space)
        for key in self.policy_dict.keys():
            new_policy.policy_dict[key] = deepcopy(self.policy_dict[key])
        return new_policy
