"""
Defines the models.
@author: xi2p
"""

from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
from .basic import State, Action, Reward, ConditionExpr

"""
Grid World Model: A simple grid world environment where an agent only moves in 4 directions or stays still.
"""
class GridWorldModel:
    def __init__(self, width: int, height: int, forbidden_states: List[Tuple[int, int]], terminal_states: List[Tuple[int, int]],
                 gamma: np.float32 = 0.9,
                 r_boundary: Reward = Reward(np.float32(-1.0)),
                 r_forbidden: Reward = Reward(np.float32(-1.0)),
                 r_terminal: Reward = Reward(np.float32(1.0)),
                 r_other: Reward = Reward(np.float32(0.0))
                 ):
        """
        Initialize the Grid World Model.
        :param width: The width of the grid world.
        :param height: The height of the grid world.
        :param forbidden_states: Positions of forbidden states in the grid world.
        :param terminal_states: Positions of terminal states in the grid world.
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
        self.forbidden_states = set([self._position_to_state(*i) for i in forbidden_states])
        self.terminal_states = set([self._position_to_state(*i) for i in terminal_states])

        self.actions = [Action("UP"), Action("DOWN"), Action("LEFT"), Action("RIGHT"), Action("STAY")]
        self.transition_probabilities = self._initialize_transition_probabilities()
        self.rewards_probabilities = self._initialize_rewards_probabilities()



    def _state_to_position(self, s: State) -> Tuple[int, int]:
        uid = s.uid - 1
        x = uid % self.width
        y = uid // self.width
        return x, y

    def _position_to_state(self, x: int, y: int) -> State:
        return self.states[y * self.width + x]

    def _initialize_states(self) -> List[State]:
        states = []
        for x in range(self.width):
            for y in range(self.height):
                uid = y * self.width + x + 1
                states.append(State(uid))
        return states

    def _initialize_transition_probabilities(self) -> Dict[ConditionExpr, np.float32]:
        """
        Initialize the transition probabilities for the grid world,
        which is a dictionary mapping ConditionExpr (s'|s,a) instance to its probability.
        """
        transition_probabilities = {}
        for s in self.states:
            x, y = self._state_to_position(s)
            for a in self.actions:
                for s_next in self.states:
                    condition_expr = s_next|(s, a)
                    # check the probability of this transition here
                    x_next, y_next = self._state_to_position(s_next)
                    if (x, y) in self.terminal_states:
                        prob = np.float32(1.0) if (x_next, y_next) == (x, y) else np.float32(0.0)
                    else:
                        if a == self.actions[0]:  # UP
                            intended_pos = (x, y - 1)
                        elif a == self.actions[1]:  # DOWN
                            intended_pos = (x, y + 1)
                        elif a == self.actions[2]:  # LEFT
                            intended_pos = (x - 1, y)
                        elif a == self.actions[3]:  # RIGHT
                            intended_pos = (x + 1, y)
                        elif a == self.actions[4]:  # STAY
                            intended_pos = (x, y)
                        else:
                            raise ValueError("Unknown action.")

                        if intended_pos == (x_next, y_next):
                            if (0 <= x_next < self.width) and (0 <= y_next < self.height):
                                # ATTENTION: Here we assume that an agent can move into a forbidden state.
                                prob = np.float32(1.0)
                            else:
                                prob = np.float32(0.0)
                        else:
                            prob = np.float32(0.0)

                    transition_probabilities[condition_expr] = prob

        return transition_probabilities

    def _initialize_rewards_probabilities(self) -> Dict[ConditionExpr, np.float32]:
        """
        Initialize the reward probabilities for the grid world,
        which is a dictionary mapping ConditionExpr (r|s,a) instance to its probability.
        """
        rewards_probabilities = {}
        for s in self.states:
            x, y = self._state_to_position(s)
            for a in self.actions:
                if a == self.actions[0]:
                    intended_pos = (x, y - 1)
                elif a == self.actions[1]:
                    intended_pos = (x, y + 1)
                elif a == self.actions[2]:
                    intended_pos = (x - 1, y)
                elif a == self.actions[3]:
                    intended_pos = (x + 1, y)
                elif a == self.actions[4]:
                    intended_pos = (x, y)
                else:
                    raise ValueError("Unknown action.")

                rewards_probabilities[ConditionExpr(self.r_boundary, (s, a))] = np.float32(0.0)
                rewards_probabilities[ConditionExpr(self.r_forbidden, (s, a))] = np.float32(0.0)
                rewards_probabilities[ConditionExpr(self.r_terminal, (s, a))] = np.float32(0.0)
                rewards_probabilities[ConditionExpr(self.r_other, (s, a))] = np.float32(0.0)

                # 1. Check boundary
                if not ((0 <= intended_pos[0] < self.width) and (0 <= intended_pos[1] < self.height)):
                    rewards_probabilities[ConditionExpr(self.r_boundary, (s, a))] += np.float32(1.0)
                    rewards_probabilities[ConditionExpr(self.r_forbidden, (s, a))] += np.float32(0.0)
                    rewards_probabilities[ConditionExpr(self.r_terminal, (s, a))] += np.float32(0.0)
                    rewards_probabilities[ConditionExpr(self.r_other, (s, a))] += np.float32(0.0)

                else:
                    intended_state = self._position_to_state(intended_pos[0], intended_pos[1])

                    # 2. Check forbidden state
                    if intended_state in self.forbidden_states:
                        rewards_probabilities[ConditionExpr(self.r_boundary, (s, a))] += np.float32(0.0)
                        rewards_probabilities[ConditionExpr(self.r_forbidden, (s, a))] += np.float32(1.0)
                        rewards_probabilities[ConditionExpr(self.r_terminal, (s, a))] += np.float32(0.0)
                        rewards_probabilities[ConditionExpr(self.r_other, (s, a))] += np.float32(0.0)

                    # 3. Check terminal state
                    elif intended_state in self.terminal_states:
                        rewards_probabilities[ConditionExpr(self.r_boundary, (s, a))] += np.float32(0.0)
                        rewards_probabilities[ConditionExpr(self.r_forbidden, (s, a))] += np.float32(0.0)
                        rewards_probabilities[ConditionExpr(self.r_terminal, (s, a))] += np.float32(1.0)
                        rewards_probabilities[ConditionExpr(self.r_other, (s, a))] += np.float32(0.0)

                    else:
                        rewards_probabilities[ConditionExpr(self.r_boundary, (s, a))] += np.float32(0.0)
                        rewards_probabilities[ConditionExpr(self.r_forbidden, (s, a))] += np.float32(0.0)
                        rewards_probabilities[ConditionExpr(self.r_terminal, (s, a))] += np.float32(0.0)
                        rewards_probabilities[ConditionExpr(self.r_other, (s, a))] += np.float32(1.0)

        return rewards_probabilities

    # def __str__(self):
    #     return f"GridWorldModel(width={self.width}, height={self.height}, forbidden_states={self.forbidden_states}, terminal_states={self.terminal_states})"

    def __str__(self):
        """
        render the grid world as a string
        for forbidden states, use "■"; for terminal states, use "○"; for other states, use "□"
        """
        grid_str = ""
        for y in range(self.height):
            for x in range(self.width):
                state = self._position_to_state(x, y)
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