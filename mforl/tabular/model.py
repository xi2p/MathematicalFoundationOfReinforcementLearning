"""
Defines the models.
@author: xi2p
"""

from typing import Dict, List, Tuple

import numpy as np

from .basic import State, Action, Reward, ConditionExpr, Policy

"""
Grid World Model: A simple grid world environment where an agent only moves in 4 directions or stays still.
"""
class GridWorldModel:
    ACTION_UP = Action("UP")
    ACTION_DOWN = Action("DOWN")
    ACTION_LEFT = Action("LEFT")
    ACTION_RIGHT = Action("RIGHT")
    ACTION_STAY = Action("STAY")

    def __init__(self, width: int, height: int, forbidden_states: List[Tuple[int, int]], terminal_states: List[Tuple[int, int]],
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
        :param r_boundary: Reward for hitting the boundary.
        :param r_forbidden: Reward for hitting a wall.
        :param r_terminal: Reward for reaching a terminal state.
        :param r_other: Reward for other transitions.
        """
        self.width = width
        self.height = height
        self.r_boundary = r_boundary
        self.r_forbidden = r_forbidden
        self.r_terminal = r_terminal
        self.r_other = r_other

        self.states = self._initialize_states()
        self.forbidden_states = set([self._position_to_state(*i) for i in forbidden_states])
        self.terminal_states = set([self._position_to_state(*i) for i in terminal_states])

        self.actions = {GridWorldModel.ACTION_UP, GridWorldModel.ACTION_DOWN,
                        GridWorldModel.ACTION_LEFT, GridWorldModel.ACTION_RIGHT,
                        GridWorldModel.ACTION_STAY}
        self.rewards = {r_boundary, r_forbidden, r_terminal, r_other}

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
        for y in range(self.width):
            for x in range(self.height):
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
                        if a == GridWorldModel.ACTION_UP:  # UP
                            intended_pos = (x, y - 1)
                        elif a == GridWorldModel.ACTION_DOWN:  # DOWN
                            intended_pos = (x, y + 1)
                        elif a == GridWorldModel.ACTION_LEFT:  # LEFT
                            intended_pos = (x - 1, y)
                        elif a == GridWorldModel.ACTION_RIGHT:  # RIGHT
                            intended_pos = (x + 1, y)
                        elif a == GridWorldModel.ACTION_STAY:  # STAY
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
                if a == GridWorldModel.ACTION_UP:
                    intended_pos = (x, y - 1)
                elif a == GridWorldModel.ACTION_DOWN:
                    intended_pos = (x, y + 1)
                elif a == GridWorldModel.ACTION_LEFT:
                    intended_pos = (x - 1, y)
                elif a == GridWorldModel.ACTION_RIGHT:
                    intended_pos = (x + 1, y)
                elif a == GridWorldModel.ACTION_STAY:
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

    def p(self, condition_expr) -> np.float32:
        """
        Get the probability of a given condition expression.
        :param condition_expr: The ConditionExpr instance.
        :return: The probability as a np.float32.
        """
        if isinstance(condition_expr, ConditionExpr):
            if isinstance(condition_expr.outcome, State):
                return self.transition_probabilities.get(condition_expr, np.float32(0.0))
            elif isinstance(condition_expr.outcome, Reward):
                return self.rewards_probabilities.get(condition_expr, np.float32(0.0))
            else:
                raise ValueError("Unknown outcome type in ConditionExpr.")
        else:
            raise ValueError("Input must be a ConditionExpr instance.")
    # def __str__(self):
    #     return f"GridWorldModel(width={self.width}, height={self.height}, forbidden_states={self.forbidden_states}, terminal_states={self.terminal_states})"

    def P_pi(self, policy: Policy) -> np.ndarray:
        """
        Get the state transition probability matrix under a given policy.
        :param policy: Given policy.
        :return: The state transition probability matrix as a 2D numpy array.
        """
        P_pi = np.zeros((len(self.states), len(self.states)))

        for s in self.states:
            for s_next in self.states:
                prob = np.float32(0.0)  # p(s'|s)
                for a in self.actions:
                    prob += policy.pi(a | s) * self.p(s_next | (s, a))
                P_pi[s.uid - 1, s_next.uid - 1] = prob

        return P_pi

    def R_pi(self, policy: Policy) -> np.ndarray:
        """
        Get the expected reward vector under a given policy.
        :param policy: Given policy.
        :return: The expected reward vector as a 1D numpy array.
        """
        R_pi = np.zeros((len(self.states)))
        for s in self.states:
            expected_reward = np.float32(0.0)
            for a in self.actions:
                for r in self.rewards:
                    expected_reward += policy.pi(a | s) * self.p(r | (s, a)) * r.value
            R_pi[s.uid - 1] = expected_reward

        return R_pi

    def step(self, state: State, action: Action) -> Tuple[State, Reward]:
        """
        take a step in the grid world given a state and an action
        :param state: The current state
        :param action: The action to take
        :return: A tuple of (next_state, reward)
        """
        x, y = self._state_to_position(state)
        # TODO: the process is stochastic in general. Edit later to add stochasticity.
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
            next_state = self._position_to_state(intended_pos[0], intended_pos[1])
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