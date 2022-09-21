from typing import Tuple, Callable

import gym.spaces
import numpy as np
from ray.rllib import env
from ray.rllib.utils.typing import MultiAgentDict

StepOutput = Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]


class GenericEnv(env.MultiAgentEnv):

    def __init__(self,
                 step_fn: Callable[[MultiAgentDict], StepOutput] =
                 lambda a: ({0: np.array([0])}, {0: 0}, {0: False}, {0: {}}),
                 reset_fn: Callable[[], MultiAgentDict] = lambda: {0: np.array([0])},
                 obs_space: gym.spaces.Space = gym.spaces.Box(0, 1, (1,)),
                 action_space: gym.spaces.Space = gym.spaces.Box(0, 1, (1,))):
        self._step_fn = step_fn
        self._reset_fn = reset_fn
        self.observation_space = obs_space
        self.action_space = action_space

    def step(self, action_dict: MultiAgentDict) -> StepOutput:
        return self._step_fn(action_dict)

    def reset(self) -> MultiAgentDict:
        return self._reset_fn()

    def render(self, mode=None) -> None:
        pass
