"""
Define "action" class, which refers to the action taken by an agent in an environment.
Define "state" class, which refers to the state of the agent.
Define "reward" class, which refers to the reward received by an agent in an environment.

Probability expression like p(s'|s), p(s'|s,a), p(r|s,a) often appear in reinforcement learning.
Here, a class ConditionExpr is defined to represent the conditional part of these expressions.
In particular, s'|s, s'|s,a, r|s,a are all instances of ConditionExpr.

@author: xi2p
"""
import numpy as np
from typing import Union, Tuple, Dict, Set


"""
An action instance represents a type of action taken by an agent in an environment.
For example, in a grid world environment, an action could be "move up", "move down", "move left", or "move right".
Different action would differ from each other by their name ONLY.
Namely, two action instances with the same name are considered the same action.
"""
class Action:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, Action):
            raise TypeError("Cannot compare Action with non-Action type.")
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f"Action({self.name})"

    def __repr__(self):
        return self.__str__()

    def __or__(self, other):
        return ConditionExpr(self, other)


"""
Each state has a unique integer ID.
State instances differ from each other by their ID ONLY.
"""
class State:
    def __init__(self, uid: int):
        self.uid = uid

    def __eq__(self, other):
        if not isinstance(other, State):
            raise TypeError("Cannot compare State with non-State type.")
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def __str__(self):
        return f"State({self.uid})"

    def __repr__(self):
        return self.__str__()

    def __or__(self, other):
        return ConditionExpr(self, other)


"""
The reward instance has a float32 value representing the reward received by an agent in an environment.
"""
class Reward:
    def __init__(self, value: np.float32):
        self.value = value

    # Define equality based on the reward value
    def __eq__(self, other):
        if not isinstance(other, Reward):
            raise TypeError("Cannot compare Reward with non-Reward type.")
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return f"Reward({self.value})"

    def __repr__(self):
        return self.__str__()

    def __or__(self, other):
        return ConditionExpr(self, other)

    # Define operations for reward addition and multiplication
    def __add__(self, other):
        if not isinstance(other, Reward):
            raise TypeError("Cannot add Reward with non-Reward type.")
        return Reward(self.value + other.value)

    def __mul__(self, other):
        if not isinstance(other, Reward):
            raise TypeError("Cannot multiply Reward with non-Reward type.")
        return Reward(self.value * other.value)

    def __rmul__(self, other):
        return self.__mul__(other)



class ConditionExpr:
    def __init__(self, outcome: Union[Action, State, Reward], condition: Union[Action, State, Reward, Tuple[Union[Action, State, Reward], ...]]):
        # take s'|s,a as an example, here outcome is s', condition is (s,a)
        self.outcome = outcome
        self.condition = condition

    def __repr__(self):
        return f"{self.outcome} | {self.condition}"

    def __eq__(self, other):
        if not isinstance(other, ConditionExpr):
            raise TypeError("Cannot compare ConditionExpr with non-ConditionExpr type.")
        return self.outcome == other.outcome and self.condition == other.condition

    def __hash__(self):
        return hash((self.outcome, self.condition))


"""
A Policy instance stores the probability distribution over actions given states.
"""
class Policy:
    def __init__(self, state_space: Set[State], action_space: Set[Action]):
        self.policy_dict : Dict[Tuple[State, Action], np.float32] = {}

        # initialize policy with all zero probabilities
        for s in state_space:
            for a in action_space:
                self.policy_dict[(s, a)] = np.float32(0.0)

    def pi(self, condition_expr: ConditionExpr) -> np.float32:
        """
        Get the probability of taking action a in state s under this policy.
        :param condition_expr: The ConditionExpr instance representing a|s.
        :return: The probability as a np.float32.
        """
        if isinstance(condition_expr, ConditionExpr):
            if isinstance(condition_expr.outcome, Action) and isinstance(condition_expr.condition, State):
                key = (condition_expr.condition, condition_expr.outcome)
                return self.policy_dict[key]
            else:
                raise ValueError("ConditionExpr must be of the form a|s.")
        else:
            raise ValueError("Input must be a ConditionExpr instance.")

    def __getitem__(self, condition_expr: ConditionExpr) -> np.float32:
        return self.pi(condition_expr)

    def __setitem__(self, condition_expr: ConditionExpr, value: np.float32):
        if isinstance(condition_expr, ConditionExpr):
            if isinstance(condition_expr.outcome, Action) and isinstance(condition_expr.condition, State):
                key = (condition_expr.condition, condition_expr.outcome)
                self.policy_dict[key] = value
            else:
                raise ValueError("ConditionExpr must be of the form a|s.")
        else:
            raise ValueError("Input must be a ConditionExpr instance.")